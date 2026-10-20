#!/usr/bin/env python3
"""The running image refuses a voice or a language it does not carry (PRD-262 US-052).

Stdlib only: it runs on the runner, not in the image (the media-render CI job).
Against a container started with the internal token, it posts:

- to POST /tts, a voice outside ``voices.json`` and a language other than en-us or
  en-gb: each must answer 400 ``invalid_request`` with a message that starts with the
  field's name (``voice``, ``lang``);
- to POST /render, the fixture bundle with the same two changes: each must answer
  400 ``invalid_bundle`` naming ``audio.voice.voice`` or ``audio.voice.lang``;
- to POST /tts, a catalogue voice in en-gb, which must be spoken (200): the refusals
  above are the data's, not a blanket one.

Every answer is printed, so the job log is the evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Voices upstream's voices-v1.0.bin holds and the catalogue leaves out: an English
# voice and a Chinese one. The rewritten file has neither.
OUTSIDE_VOICES = ("am_santa", "zf_xiaoxiao")
OTHER_LANGUAGE = "fr-fr"
SPOKEN_VOICE, SPOKEN_LANGUAGE = "bf_emma", "en-gb"
LINE = [{"id": "l01", "text": "Rendered on brand."}]


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")


def _json(raw: bytes) -> Dict[str, Any]:
    try:
        answer = json.loads(raw or b"{}")
    except ValueError:
        return {"body": raw.decode("utf-8", "replace")[:500]}
    return answer if isinstance(answer, dict) else {"body": answer}


def _post(url: str, token: str, body: Any, timeout: float) -> Tuple[int, Dict[str, Any]]:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST")
    request.add_header("X-Internal-Token", token)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, _json(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _json(exc.read())


def _with_voice(bundle: Dict[str, Any], **changes: str) -> Dict[str, Any]:
    changed = copy.deepcopy(bundle)
    changed["audio"]["voice"].update(changes)
    return changed


def _cases(bundle: Dict[str, Any]) -> List[Tuple[str, Any, str, str]]:
    """(route, body, the error code, the field the message must start with)."""
    cases: List[Tuple[str, Any, str, str]] = []
    for voice in OUTSIDE_VOICES:
        cases.append(("/tts", {"lines": LINE, "voice": voice}, "invalid_request", "voice"))
        cases.append(("/render", _with_voice(bundle, voice=voice), "invalid_bundle", "audio.voice.voice"))
    cases.append(("/tts", {"lines": LINE, "lang": OTHER_LANGUAGE}, "invalid_request", "lang"))
    cases.append(("/render", _with_voice(bundle, lang=OTHER_LANGUAGE), "invalid_bundle", "audio.voice.lang"))
    return cases


def check(url: str, token: str, bundle: Dict[str, Any], catalogue: List[str], timeout: float) -> List[str]:
    """Every case answered as it must be: the failures, none when the image holds."""
    failures = [f"{voice} is in voices.json; pick a voice the catalogue leaves out" for voice in OUTSIDE_VOICES if voice in catalogue]
    for route, body, code, field in _cases(bundle):
        status, answer = _post(url + route, token, body, timeout)
        _say(f"POST {route} ({field}): {status} {json.dumps(answer)[:300]}")
        message = str(answer.get("message", ""))
        if status != 400 or answer.get("error") != code or not message.startswith(f"{field} must be one of "):
            failures.append(f"POST {route} with a {field} the image does not carry answered {status} {answer}")
    spoken = {"lines": LINE, "voice": SPOKEN_VOICE, "lang": SPOKEN_LANGUAGE}
    status, answer = _post(url + "/tts", token, spoken, timeout)
    _say(f"POST /tts ({SPOKEN_VOICE}, {SPOKEN_LANGUAGE}): {status} voice={answer.get('voice')} lang={answer.get('lang')}")
    if status != 200 or (answer.get("voice"), answer.get("lang")) != (SPOKEN_VOICE, SPOKEN_LANGUAGE):
        failures.append(f"POST /tts with {SPOKEN_VOICE} in {SPOKEN_LANGUAGE} answered {status} {answer}")
    return failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="the service, e.g. http://127.0.0.1:8094")
    parser.add_argument("--token", required=True)
    parser.add_argument("--bundle", type=Path, required=True, help="the fixture bundle (fixture-bundle's output)")
    parser.add_argument("--voices", type=Path, required=True, help="services/media-render/media_render/voices.json")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    catalogue = json.loads(args.voices.read_text(encoding="utf-8"))["voices"]
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    failures = check(args.url, args.token, bundle, catalogue, args.timeout)
    for failure in failures:
        _say(f"::error::{failure}")
    if not failures:
        _say("a voice outside voices.json and a language other than en-us or en-gb are each a 400 naming the field: PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
