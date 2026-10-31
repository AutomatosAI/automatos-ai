"""Theme voice (TV-C7) — a theme's code that captures audio or speaks by itself, as a review finding.

A theme talks to Auto by voice only through the SDK's voice area (``VoiceButton``, ``useVoice``,
``readVoiceLevels``, SDK 1.2.0): the host draws the AI voice caption and the stop control, speaks the
AI notice once per session and runs the tab's one session. Code of a theme's own that opens the
microphone (``getUserMedia``), listens (``SpeechRecognition``, ``webkitSpeechRecognition``) or speaks
(``speechSynthesis``) could record the person using it, send their voice to the browser vendor's
speech service, or speak as Auto with no notice, and nothing of the host's would show it.

The check reads the code the browser runs (``sdk_uses.bundle_texts``: ``dist/entry.js`` and
``dist/chunks/*.js``) for those names as whole identifiers, in calls, property reads, strings and
comments alike, as the kit's rules read source (``JS_GET_USER_MEDIA``, ``JS_SPEECH_RECOGNITION``):
a theme has no reason to name them. Each name found in a file is one finding, in the shape of the
LLM stage's (category, severity, description, file, evidence), and each name one plain sentence in
``problems``. Any finding makes the check ``AUDIO_CAPTURE_LEVEL``, an error, so the review blocks
the version (approve refuses it); none makes it a pass. The uploader sees it too:
``review_stage.member_view`` keeps every check but the LLM's.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from services.themes.sdk_uses import bundle_texts

LEVEL_PASS, LEVEL_ERROR = "pass", "error"
STATUS_CHECKED = "checked"
# The kit refuses these in a theme's source; the host's review blocks them in its bundle. A warning
# instead is this one constant.
AUDIO_CAPTURE_LEVEL = LEVEL_ERROR
FINDING_CATEGORY = "Audio capture"
FINDING_SEVERITY = "high"
EVIDENCE_AROUND = 60
EVIDENCE_MAX = 300

# A name counts when it stands alone: not part of a longer identifier (``mySpeechSynthesisShim``).
_ALONE_BEFORE, _ALONE_AFTER = r"(?<![\w$])", r"(?![\w$])"


class AudioApi(NamedTuple):
    """One browser audio API a theme may not use itself: its name, how it is written, what it does."""

    name: str
    pattern: re.Pattern[str]
    does: str


AUDIO_APIS: Tuple[AudioApi, ...] = (
    AudioApi(
        "getUserMedia",
        re.compile(_ALONE_BEFORE + r"(?:getUserMedia|webkitGetUserMedia|mozGetUserMedia)" + _ALONE_AFTER),
        "opens the microphone",
    ),
    AudioApi(
        "SpeechRecognition",
        re.compile(_ALONE_BEFORE + r"(?:webkit)?SpeechRecognition" + _ALONE_AFTER),
        "listens to the microphone",
    ),
    AudioApi("speechSynthesis", re.compile(_ALONE_BEFORE + r"speechSynthesis" + _ALONE_AFTER), "speaks"),
)

REASON = (
    "A theme talks to Auto by voice only through the SDK's voice area (VoiceButton, useVoice), which "
    "draws the AI voice caption and the stop control. Its code can't open the microphone, listen or "
    "speak by itself."
)
PROBLEM = "The code uses {api} itself, which {does}, in {files}. Use the SDK's voice area instead."
DESCRIPTION = (
    "Uses {found} itself: it {does} outside the host's voice, with no AI voice caption and no stop control."
)


def _evidence(text: str, match: re.Match[str]) -> str:
    start = max(match.start() - EVIDENCE_AROUND, 0)
    return text[start:match.end() + EVIDENCE_AROUND][:EVIDENCE_MAX]


def _finding(path: str, text: str, api: AudioApi) -> Optional[Dict[str, str]]:
    """The finding for the first place ``api`` is named in ``text``, or None when it is not."""
    match = api.pattern.search(text)
    if match is None:
        return None
    return {
        "category": FINDING_CATEGORY, "severity": FINDING_SEVERITY,
        "description": DESCRIPTION.format(found=match.group(0), does=api.does),
        "file": path, "evidence": _evidence(text, match),
    }


def audio_capture_findings(texts: Mapping[str, str]) -> List[Dict[str, str]]:
    """One finding per audio API per file that names it, in file order, then API order."""
    found = (_finding(path, text, api) for path, text in texts.items() for api in AUDIO_APIS)
    return [finding for finding in found if finding is not None]


def _problems(texts: Mapping[str, str]) -> Tuple[List[str], List[str]]:
    """The APIs the code names, and one sentence for each, naming the files."""
    apis: List[str] = []
    problems: List[str] = []
    for api in AUDIO_APIS:
        files = [path for path, text in texts.items() if api.pattern.search(text)]
        if files:
            apis.append(api.name)
            problems.append(PROBLEM.format(api=api.name, does=api.does, files=", ".join(files)))
    return apis, problems


def audio_capture_check(files: Mapping[str, bytes]) -> Dict[str, Any]:
    """The ``audioCapture`` entry of the scan report: an error naming each audio API the bundle uses, else a pass."""
    texts = bundle_texts(files)
    apis, problems = _problems(texts)
    return {
        "level": AUDIO_CAPTURE_LEVEL if apis else LEVEL_PASS, "status": STATUS_CHECKED,
        "apis": apis, "problems": problems, "findings": audio_capture_findings(texts),
        "reason": REASON if apis else None,
    }
