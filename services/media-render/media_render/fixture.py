"""The committed fixture as a render bundle: the image's end-to-end proof.

The media-render CI job posts this bundle to POST /render and asserts the MP4
with ffprobe (h264 1080x1920 at 30 fps, AAC, 3.0 s) and ebur128 (-14 LUFS).
The composition is a template like any other: its headline is a variable, its
colours are brand tokens with fallbacks, and its one script line is spoken by
Kokoro at render time. No media is committed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
COMPOSITION = FIXTURES / "fixture" / "index.html"
BUNDLE = FIXTURES / "fixture.bundle.json"


def fixture_bundle() -> Dict[str, Any]:
    """The fixture's bundle, with the composition's HTML read in."""
    bundle = json.loads(BUNDLE.read_text())
    return {**bundle, "composition": {**bundle.get("composition", {}), "html": COMPOSITION.read_text()}}
