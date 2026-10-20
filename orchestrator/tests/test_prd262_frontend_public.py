"""PRD-262 US-049: public/ holds only what the app serves, at the size it's shown.

test.yml's frontend-ci runs frontend/scripts/check-public-assets.js as a hard step. It
fails when public/ grows past frontend/scripts/public-budget.json, when a file under
public/logos is not a real PNG or SVG by its magic bytes, or when a logo PNG is over
256 px. The script's logic is tested by
frontend/scripts/__tests__/check-public-assets.test.ts. This file pins the wiring, so the
step cannot drop out of CI or turn advisory unnoticed, and checks the committed logos
against the story's own bar (128 px, real PNGs under their old names) from the backend lane.

Pure file reads: no Node, no network.
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
LOGOS = FRONTEND / "public" / "logos"
BUDGET = FRONTEND / "scripts" / "public-budget.json"
TEST_YML = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text())
CHECK = "node scripts/check-public-assets.js"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
LOGO_MAX_PX = 128
# Names the backend seed and lib/logo-service.ts reach that were not images before US-049.
WERE_NOT_IMAGES = ("Microsoft.png", "Nginx.png", "Sentry.png")


def _frontend_ci() -> dict:
    return TEST_YML["jobs"]["frontend-ci"]


def _png_size(data: bytes) -> tuple[int, int] | None:
    """Width and height from the IHDR chunk, or None when the bytes are not a PNG."""
    if not data.startswith(PNG_SIGNATURE) or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


def test_frontend_ci_runs_the_public_check_as_a_hard_step():
    job = _frontend_ci()
    assert job["defaults"]["run"]["working-directory"] == "frontend"
    steps = [step for step in job["steps"] if step.get("run", "").strip() == CHECK]
    assert len(steps) == 1, f"expected one frontend-ci step running {CHECK!r}, found {len(steps)}"
    step = steps[0]
    assert "continue-on-error" not in step
    # It reports even when vitest or tsc failed first: a red job otherwise shows only its first failure.
    assert step["if"] == "${{ !cancelled() }}"


def test_a_change_under_public_or_to_the_budget_runs_frontend_ci():
    match = next(step for step in TEST_YML["jobs"]["changes"]["steps"] if step.get("id") == "match")
    paths = match["env"]["FRONTEND_PATHS"]
    for changed in ("frontend/public/logos/Nginx.png", "frontend/scripts/public-budget.json"):
        assert re.search(paths, changed), changed


def test_the_budget_is_a_whole_number_of_bytes_with_a_reason():
    budget = json.loads(BUDGET.read_text())
    assert isinstance(budget["max_bytes"], int) and budget["max_bytes"] > 0
    assert "US-049" in budget["reason"]


def test_every_logo_is_a_real_png_of_at_most_128_px():
    sizes = {path.name: _png_size(path.read_bytes()) for path in LOGOS.iterdir() if not path.name.startswith(".")}
    assert [name for name, size in sizes.items() if size is None] == []
    assert {name: size for name, size in sizes.items() if max(size) > LOGO_MAX_PX} == {}
    assert len(sizes) >= 283


def test_the_names_that_were_not_images_are_real_pngs_now():
    for name in WERE_NOT_IMAGES:
        assert _png_size((LOGOS / name).read_bytes()) is not None, name
