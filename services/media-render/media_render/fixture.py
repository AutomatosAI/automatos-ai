"""The committed fixtures as render bundles: the image's end-to-end proofs.

The media-render CI job posts each bundle to POST /render and asserts what
comes back:

- ``fixture``: the MP4 with ffprobe (h264 1080x1920 at 30 fps, AAC, 3.0 s) and
  ebur128 (-14 LUFS). Its one script line is spoken by Kokoro at render time.
- ``script`` (US-111): a 9-second script of four Kokoro lines, each with its own
  window. The first line is longer than its window (the fixture's own line,
  2.09 s, in a 1.9 s window), so the render fits it (fit.py); the job then
  asserts speech in every script window of the rendered MP4.

Each composition is a template like any other: its headline is a variable, its
colours are brand tokens with fallbacks. No media is committed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
COMPOSITION = FIXTURES / "fixture" / "index.html"
BUNDLE = FIXTURES / "fixture.bundle.json"
SCRIPT_COMPOSITION = FIXTURES / "script" / "index.html"
SCRIPT_BUNDLE = FIXTURES / "script.bundle.json"


def _bundle(bundle_path: Path, composition_path: Path) -> Dict[str, Any]:
    bundle = json.loads(bundle_path.read_text())
    return {**bundle, "composition": {**bundle.get("composition", {}), "html": composition_path.read_text()}}


def fixture_bundle() -> Dict[str, Any]:
    """The fixture's bundle, with the composition's HTML read in."""
    return _bundle(BUNDLE, COMPOSITION)


def script_bundle() -> Dict[str, Any]:
    """The fixture script's bundle (US-111), with the composition's HTML read in."""
    return _bundle(SCRIPT_BUNDLE, SCRIPT_COMPOSITION)


BUNDLES: Dict[str, Callable[[], Dict[str, Any]]] = {"fixture": fixture_bundle, "script": script_bundle}
