"""PRD-261 C-02a (PA-18) — ``services/themes/data/sdk-surface.json`` matches the checkout's frontend/.

The review stage checks a theme's ``uses`` and its SDK imports against the names this host's
SDK exports (``services/themes/sdk_uses.py``). Those names are planned per area in
``frontend/lib/theme-sdk/surface/*.json``; the running image has no ``frontend/`` folder, so
their union by area is committed, and this test holds it to the files under ``frontend/`` in
the checkout. A change to a surface file fails it until the JSON is regenerated, from the
repository root, with a bare python3 (nothing installed)::

    python3 orchestrator/tests/test_prd261_sdk_surface_drift.py --write

sdk_uses.py is loaded from its file (it uses the standard library only), so the command needs
none of the orchestrator's dependencies.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Sequence

ORCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ORCH_ROOT.parent
SURFACE_DIR = REPO_ROOT / "frontend/lib/theme-sdk/surface"
WRITE_FLAG = "--write"
COMMENT = (
    "PRD-261 PA-18: the runtime export names of @automatos/theme-sdk by area, from "
    "frontend/lib/theme-sdk/surface/*.json. Generated: "
    "python3 orchestrator/tests/test_prd261_sdk_surface_drift.py --write. Don't edit by hand."
)


def _load_sdk_uses() -> ModuleType:
    """``services/themes/sdk_uses.py`` loaded from its file (no package imports run)."""
    path = ORCH_ROOT / "services" / "themes" / "sdk_uses.py"
    spec = importlib.util.spec_from_file_location("prd261_drift_sdk_uses", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sdk_uses = _load_sdk_uses()


def surface_files() -> List[Path]:
    """Every area's surface file under frontend/, in name order."""
    return sorted(SURFACE_DIR.glob("*.json"))


def checkout_areas() -> Dict[str, List[str]]:
    """``{area: sorted names}`` from the checkout's surface files, in area order."""
    areas: Dict[str, List[str]] = {}
    for path in surface_files():
        document = json.loads(path.read_text(encoding="utf-8"))
        areas[document["area"]] = sorted(document["names"])
    return dict(sorted(areas.items()))


def render(areas: Dict[str, List[str]]) -> str:
    """sdk-surface.json's text for ``areas``."""
    return json.dumps({"_comment": COMMENT, "areas": areas}, indent=2) + "\n"


def test_the_surface_folder_under_frontend_has_area_files():
    assert surface_files(), f"no surface files under {SURFACE_DIR.relative_to(REPO_ROOT)}"


def test_no_name_is_in_two_areas():
    seen: Dict[str, str] = {}
    for area, names in checkout_areas().items():
        for name in names:
            assert name not in seen, f"{name} is in both {seen[name]} and {area}"
            seen[name] = area


def test_sdk_surface_json_matches_the_checkout():
    committed = sdk_uses.SDK_SURFACE_FILE.read_text(encoding="utf-8")
    assert committed == render(checkout_areas()), (
        "services/themes/data/sdk-surface.json is out of date with frontend/lib/theme-sdk/surface/. "
        "Regenerate it from the repository root: python3 orchestrator/tests/test_prd261_sdk_surface_drift.py --write"
    )


def test_the_reader_returns_the_committed_names():
    expected = {area: tuple(names) for area, names in checkout_areas().items()}
    assert sdk_uses.sdk_surface() == expected


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``--write`` regenerates sdk-surface.json from the checkout."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args != [WRITE_FLAG]:
        sys.stderr.write(f"usage: python3 orchestrator/tests/test_prd261_sdk_surface_drift.py {WRITE_FLAG}\n")
        return 2
    sdk_uses.SDK_SURFACE_FILE.write_text(render(checkout_areas()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
