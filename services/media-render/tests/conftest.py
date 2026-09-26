"""The media-render suite. It runs INSIDE the image (the media-render CI job
builds the Dockerfile's `test` stage and runs `python -m pytest` in it), so the
image-fact tests read the real layout: /opt/espeak, /opt/kokoro, /opt/chrome.
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

from helpers import STORAGE, TOKEN  # noqa: E402
from media_render.config import load_settings  # noqa: E402
from media_render.media_urls import parse_prefixes  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    """The image's own settings, with a scratch work dir, an empty library and a token."""
    return replace(
        load_settings(),
        work_dir=str(tmp_path / "work"),
        music_dir=str(tmp_path / "music"),
        internal_token=TOKEN,
        media_url_prefixes=parse_prefixes(STORAGE),
    )
