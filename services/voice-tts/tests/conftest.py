"""The voice-tts suite. It runs INSIDE the image (.github/workflows/voice-tts.yml builds
the Dockerfile's `test` stage and runs `python -m pytest` in it), so the image-fact and
real-voice tests read the real layout: /opt/kokoro, /opt/espeak and the image's LAME.
The HTTP tests use stand-ins for the model and the encoder (helpers.py).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
for path in (TESTS.parent, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from helpers import TOKEN  # noqa: E402
from voice_tts.config import load_settings  # noqa: E402


@pytest.fixture
def settings():
    """The image's own settings with a token, two syntheses at once and two threads."""
    return replace(load_settings({}), token=TOKEN, max_concurrent=2, threads=2)
