"""PRD-262 US-031: the smokes hold that a warm restart redoes no seed work.

Each boot seed stage stores the fingerprint of its inputs once it ran clean, and a
boot whose stored fingerprint matches skips the stage and logs ``seed_fingerprint.<stage>:
seeds unchanged (fingerprint …) — skipped``. The smokes (smoke-fresh-clone and
smoke-images) prove it on a real stack:

* ``scripts/ci/smoke_seeds.py`` reads the backend's log of every boot: the fresh boot
  starts the seed loader; each warm restart logs both stages' skip lines and starts no
  seed loader. It also compares two ``credential_types`` snapshots (the row count and a
  digest of every row's ``updated_at``), taken before and after the warm restarts;
* ``scripts/ci/smoke-fresh-clone.sh`` takes the snapshots around the warm restarts and
  runs the check after the last boot.

These tests hold the parts; the lanes run them only in CI.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from core.seeds.fingerprint import skipped_line

_REPO = Path(__file__).resolve().parents[2]
_ORCH = _REPO / "orchestrator"
_CI = _REPO / "scripts" / "ci"
_SMOKE = _CI / "smoke-fresh-clone.sh"
_WORKFLOWS = _REPO / ".github" / "workflows"
NS = 1_000_000_000
T0 = 1_792_100_000  # 2026-10-15 21:33:20 UTC, a whole epoch second
BACKEND, FRONTEND = "automatos_backend", "automatos_frontend"
CHECKING = "📦 Checking seed data..."
LOADING = "📥 Loading seed data (idempotent)..."
STARTED = "INFO:     Application startup complete."


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _CI / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


seeds = _load("smoke_seeds")


def _skip_line(stage: str) -> str:
    """The skip line exactly as the boot logs it (core/seeds/fingerprint.py)."""
    return skipped_line(stage, "0123456789abcdef" * 4)


def _at(offset_s: float) -> int:
    return T0 * NS + round(offset_s * 1000) * 1_000_000


def _started_at(ns: int) -> str:
    seconds, fraction = divmod(ns, NS)
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    digits = f"{fraction:09d}".rstrip("0")
    return f"{stamp}.{digits}Z" if digits else f"{stamp}Z"


def _log(at_s: float, text: str) -> str:
    seconds, fraction = divmod(_at(at_s), NS)
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{stamp}.{fraction:09d}Z {text}"


def _boot_lines(*, loader: str = "skipped", leader: str = "skipped") -> list[str]:
    """One boot's log. ``loader``/``leader``: ``skipped`` or ``ran``."""
    lines = [CHECKING]
    lines += [_skip_line("loader")] if loader == "skipped" else [LOADING, "Seed data loaded"]
    lines += ["Boot lock acquired — this worker will run seeds"]
    lines += [_skip_line("leader")] if leader == "skipped" else ["Boot seeds completed (leader worker)"]
    return lines + [STARTED]


def _record(label: str, container: str, start_s: float) -> dict:
    return {"label": label, "container": container, "started_at": _started_at(_at(start_s))}


RECORDS = [
    _record("fresh", BACKEND, 0.0),
    _record("frontend", FRONTEND, 50.0),
    _record("warm-1", BACKEND, 100.0),
    _record("warm-2", BACKEND, 200.0),
]


_RAN = {"loader": "ran", "leader": "ran"}


def _run_log(fresh: dict = _RAN, warm_2: dict | None = None) -> list[str]:
    boots = [(0.0, _boot_lines(**fresh)), (100.0, _boot_lines()), (200.0, _boot_lines(**(warm_2 or {})))]
    return [_log(start + index * 0.01, text) for start, lines in boots for index, text in enumerate(lines)]


def _check(log: list[str], records=RECORDS) -> dict:
    return seeds.check_boots(records, log, ["warm-1", "warm-2"], ["fresh"], BACKEND)


# --------------------------------------------------------------------------- the boots


def test_the_skip_line_the_boot_logs_is_the_one_the_smoke_reads():
    for stage in seeds.STAGES:
        assert seeds.skip_marker(stage) in _skip_line(stage)


def test_a_fresh_boot_that_loads_and_warm_restarts_that_skip_pass():
    result = _check(_run_log())
    assert result["failures"] == []
    assert [(b["label"], b["loader_skipped"], b["leader_skipped"], b["loader_started"]) for b in result["boots"]] == [
        ("fresh", 0, 0, 1), ("warm-1", 1, 1, 0), ("warm-2", 1, 1, 0)]


@pytest.mark.parametrize("stage", ["loader", "leader"])
def test_a_warm_restart_that_ran_a_stage_again_fails(stage):
    result = _check(_run_log(warm_2={stage: "ran"}))
    assert any(f"warm-2: no `seed_fingerprint.{stage}: seeds unchanged" in f for f in result["failures"]), result
    if stage == "loader":
        assert "warm-2: `Loading seed data` logged, so a seed-loader process started" in result["failures"]


def test_a_fresh_boot_that_never_started_the_loader_fails():
    """A check that always skipped would pass the warm restarts; the fresh boot shows it."""
    result = _check(_run_log(fresh={}))
    assert "fresh: no `Loading seed data` line; the first boot must run the seed loader" in result["failures"]


def test_an_expected_boot_that_was_never_timed_fails():
    result = _check(_run_log(), records=RECORDS[:3])
    assert result["failures"] == ["warm-2: no such backend boot was timed"]


# --------------------------------------------------------------------------- credential_types


@pytest.mark.parametrize("before,after,problem", [
    ("418 9e107d9d372bb6826bd81d3542a419d6", "418 9e107d9d372bb6826bd81d3542a419d6", None),
    ("418 9e107d9d372bb6826bd81d3542a419d6", "418 e4d909c290d0fb1ca068ffaddf22cbd0", "changed across the warm restarts"),
    ("", "", "empty or unreadable"),
    ("0 ", "0 ", "empty or unreadable"),
])
def test_credential_types_must_be_present_and_unchanged(before, after, problem):
    problems = seeds.credential_types_problems(before, after)
    assert (problems == []) if problem is None else any(problem in p for p in problems), problems


def _write_run(tmp_path: Path, log: list[str], after: str) -> list[str]:
    results, backend_log = tmp_path / "boots.jsonl", tmp_path / "backend.log"
    results.write_text("".join(json.dumps(record) + "\n" for record in RECORDS), encoding="utf-8")
    backend_log.write_text("\n".join(log) + "\n", encoding="utf-8")
    before_file, after_file = tmp_path / "before.txt", tmp_path / "after.txt"
    before_file.write_text("418 abc\n", encoding="utf-8")
    after_file.write_text(after, encoding="utf-8")
    return ["--results", str(results), "--backend-log", str(backend_log),
            "--expect-loaded", "fresh", "--expect-skipped", "warm-1", "--expect-skipped", "warm-2",
            "--credential-types", str(before_file), str(after_file),
            "--out", str(tmp_path / "seed-stages.json"), "--summary", str(tmp_path / "summary.md")]


@pytest.mark.parametrize("after,code,verdict", [("418 abc\n", 0, "passed"), ("418 abd\n", 1, "FAILED")])
def test_the_command_writes_its_record_and_summary(tmp_path, after, code, verdict):
    assert seeds.main(_write_run(tmp_path, _run_log(), after)) == code
    record = json.loads((tmp_path / "seed-stages.json").read_text(encoding="utf-8"))
    assert [boot["label"] for boot in record["boots"]] == ["fresh", "warm-1", "warm-2"]
    assert record["credential_types"]["before"] == "418 abc"
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert f"### {seeds.TITLE}: {verdict}" in summary
    assert "| warm-1 | 1 | 1 | 0 |" in summary and "| fresh | 0 | 0 | 1 |" in summary


# --------------------------------------------------------------------------- the wiring


def _function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.M | re.S)
    assert match, f"smoke-fresh-clone.sh has no {name}() function"
    return match.group(1)


def test_the_smoke_snapshots_credential_types_around_the_warm_restarts_and_checks_every_boot():
    text = _SMOKE.read_text(encoding="utf-8")
    assert 'SEEDS_TOOL="scripts/ci/smoke_seeds.py"' in text
    main = _function(text, "main")
    order = [main.index(call) for call in (
        'snapshot_credential_types "$CREDENTIAL_TYPES_BEFORE"', "warm_restarts",
        'snapshot_credential_types "$CREDENTIAL_TYPES_AFTER"', "check_seed_stages", 'outcome="pass"')]
    assert order == sorted(order)
    snapshot = _function(text, "snapshot_credential_types")
    assert "compose exec -T postgres psql" in snapshot and '"$CREDENTIAL_TYPES_SQL"' in snapshot
    assert "updated_at" in re.search(r'^CREDENTIAL_TYPES_SQL="(.*)"$', text, re.M).group(1)
    check = _function(text, "check_seed_stages")
    assert '--expect-loaded "fresh"' in check
    assert 'for i in $(seq 1 "$WARM_RESTARTS"); do' in check and 'expects+=(--expect-skipped "warm-${i}")' in check
    assert '--credential-types "$CREDENTIAL_TYPES_BEFORE" "$CREDENTIAL_TYPES_AFTER"' in check
    assert '"${expects[@]}"' in check and "|| fail " in check


def test_the_loader_line_the_smoke_reads_is_the_entrypoints():
    entrypoint = (_ORCH / "docker-entrypoint.sh").read_text(encoding="utf-8")
    assert f'echo "{LOADING}"' in entrypoint
    assert seeds.LOADER_LINE in LOADING


@pytest.mark.parametrize("workflow,events", [("smoke-fresh-clone.yml", ("push", "pull_request")),
                                             ("smoke-images.yml", ("pull_request",))])
def test_both_smokes_watch_the_seed_check(workflow, events):
    data = yaml.safe_load((_WORKFLOWS / workflow).read_text(encoding="utf-8"))
    triggers = data.get("on", data.get(True))  # PyYAML reads the bare `on:` key as True
    for event in events:
        assert "scripts/ci/smoke_seeds.py" in triggers[event]["paths"]
