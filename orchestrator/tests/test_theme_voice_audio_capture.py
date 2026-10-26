"""Theme voice (TV-C7) — core's review finds a theme's code capturing audio or speaking by itself.

``services/themes/audio_capture.py``: the code the browser runs (``dist/entry.js``, ``dist/chunks/*.js``)
naming ``getUserMedia`` (and its old prefixed names), ``SpeechRecognition`` / ``webkitSpeechRecognition``
or ``speechSynthesis`` is an error at the review stage's ``audioCapture`` check, one finding per name and
file in the LLM stage's finding shape and one sentence per name; the verdict is blocked, and the
uploader's view keeps the check. A longer identifier that only contains a name, the SDK's own voice
area, a style sheet and ``source/`` are not findings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from services.themes import audio_capture  # noqa: E402
from services.themes.audio_capture import AUDIO_CAPTURE_LEVEL, audio_capture_check  # noqa: E402
from services.themes.package import read_theme_package  # noqa: E402
from services.themes.review_stage import member_view, run_review_stage  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

ENTRY = "dist/entry.js"
CHUNK = "dist/chunks/voice.js"
FINDING_KEYS = {"category", "severity", "description", "file", "evidence"}
HELLO_ENTRY = (build.HELLO_DIR / "dist" / "entry.js").read_text(encoding="utf-8")
SURFACE = {"theme": ["defineTheme"], "voice": ["VoiceButton", "readVoiceLevels", "useVoice"]}


def _check(files):
    return audio_capture_check({path: text.encode("utf-8") for path, text in files.items()})


def _reply(score: int = 0):
    text = json.dumps({"risk_score": score, "findings": [], "summary": f"risk {score}"})
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=10, output_tokens=2))


def _review(entry: str, **manifest):
    fields = {"uses": ["theme"], **manifest}
    replace = {build.MANIFEST: build.changed_manifest(fields), build.ENTRY: entry.encode("utf-8")}
    data = build.build_package(replace=replace)
    return run_review_stage(read_theme_package(data), len(data), llm_call=lambda model, text: _reply(), surface=SURFACE)


# ---------------------------------------------------------------------------
# The check on its own
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("code", "api"), [
    ("const s = await navigator.mediaDevices.getUserMedia({ audio: true })", "getUserMedia"),
    ("const { getUserMedia } = navigator.mediaDevices", "getUserMedia"),
    ("navigator.webkitGetUserMedia({ audio: true }, ok, fail)", "getUserMedia"),
    ("const listener = new SpeechRecognition()", "SpeechRecognition"),
    ("const Listener = window.SpeechRecognition || window.webkitSpeechRecognition", "SpeechRecognition"),
    ("window.speechSynthesis.speak(new SpeechSynthesisUtterance('hi'))", "speechSynthesis"),
    ('const api = window["speechSynthesis"]', "speechSynthesis"),
])
def test_each_audio_api_in_the_bundle_is_an_error_naming_it(code, api):
    check = _check({ENTRY: code})

    assert (check["level"], check["status"], check["apis"]) == (AUDIO_CAPTURE_LEVEL, "checked", [api])
    assert AUDIO_CAPTURE_LEVEL == "error"
    assert check["problems"] == [audio_capture.PROBLEM.format(
        api=api, does=next(item.does for item in audio_capture.AUDIO_APIS if item.name == api), files=ENTRY,
    )]
    assert check["reason"] == audio_capture.REASON


def test_a_finding_per_name_and_file_in_the_llm_stages_shape():
    check = _check({ENTRY: "navigator.mediaDevices.getUserMedia({ audio: true })", CHUNK: "speechSynthesis.cancel()"})

    assert check["apis"] == ["getUserMedia", "speechSynthesis"]
    assert [(finding["file"], finding["category"]) for finding in check["findings"]] == [
        (CHUNK, audio_capture.FINDING_CATEGORY), (ENTRY, audio_capture.FINDING_CATEGORY),
    ]
    for finding in check["findings"]:
        assert set(finding) == FINDING_KEYS and finding["severity"] == audio_capture.FINDING_SEVERITY
        assert len(finding["evidence"]) <= audio_capture.EVIDENCE_MAX
    assert "getUserMedia" in check["findings"][1]["evidence"]
    assert check["findings"][0]["description"].startswith("Uses speechSynthesis itself")


def test_one_sentence_per_name_lists_every_file_that_names_it():
    check = _check({ENTRY: "new webkitSpeechRecognition()", CHUNK: "new SpeechRecognition()"})

    assert check["apis"] == ["SpeechRecognition"]
    assert len(check["problems"]) == 1 and f"in {CHUNK}, {ENTRY}." in check["problems"][0]
    assert len(check["findings"]) == 2


@pytest.mark.parametrize("files", [
    {ENTRY: HELLO_ENTRY},
    {ENTRY: "import { VoiceButton, useVoice, readVoiceLevels } from '@automatos/theme-sdk'\nexport const v = [VoiceButton, useVoice, readVoiceLevels]"},
    {ENTRY: "const mySpeechSynthesisShim = 1; const getUserMediaLater = 2; const SpeechRecognitionEvents = 3"},
    {ENTRY: HELLO_ENTRY, "dist/theme.css": "html.hello .speechSynthesis { color: red }"},
    {ENTRY: HELLO_ENTRY, "source/voice.ts": "navigator.mediaDevices.getUserMedia({ audio: true })"},
])
def test_no_finding_for_the_sdk_voice_area_a_lookalike_name_or_code_the_browser_never_runs(files):
    check = _check(files)

    assert (check["level"], check["apis"], check["problems"], check["findings"]) == ("pass", [], [], [])
    assert check["reason"] is None


# ---------------------------------------------------------------------------
# In the review stage
# ---------------------------------------------------------------------------


def test_the_review_blocks_a_theme_that_opens_the_microphone_itself_and_the_uploader_sees_why():
    entry = HELLO_ENTRY + "\nexport const listen = () => navigator.mediaDevices.getUserMedia({ audio: true })\n"
    report = _review(entry)
    check = report["checks"]["audioCapture"]

    assert (report["verdict"], check["level"], check["apis"]) == ("blocked", "error", ["getUserMedia"])
    assert report["summary"]["error"] == 1 and report["checks"]["llm"]["level"] == "pass"
    assert member_view(report)["checks"]["audioCapture"] == check


def test_the_review_passes_a_theme_on_the_sdks_voice_area():
    entry = HELLO_ENTRY.replace(
        "import { defineTheme } from '@automatos/theme-sdk'",
        "import { defineTheme, VoiceButton, useVoice } from '@automatos/theme-sdk'",
    ) + "\nexport const talk = () => [VoiceButton, useVoice]\n"
    report = _review(entry, uses=["theme", "voice"])

    assert (report["verdict"], report["checks"]["audioCapture"]["level"]) == ("pass", "pass")
    assert report["checks"]["uses"]["usedAreas"] == ["theme", "voice"]
