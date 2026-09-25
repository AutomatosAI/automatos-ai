"""The media-render suite. It runs INSIDE the image (the media-render CI job
builds the Dockerfile's `test` stage and runs `python -m pytest` in it), so the
image-fact tests read the real layout: /opt/espeak, /opt/kokoro, /opt/chrome.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
