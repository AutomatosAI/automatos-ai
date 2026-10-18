"""PRD-261 A-06a (PA-18) — ``services/themes/data/core-pages.json`` matches the checkout's frontend/.

A theme that clones a core page records the sha256 of the file it copied (``clonedFrom``).
When a host release changes that file, the install shows "needs update" (A-07b) and review
flags the clone (C-02a). The running image has no ``frontend/`` folder, so the hashes of every
``PAGE_SOURCES`` file are committed, and this test holds them to the files under
``frontend/`` in the checkout. A change to a tracked page file fails it until the JSON is
regenerated, from the repository root, with a bare python3 (nothing installed)::

    python3 orchestrator/tests/test_prd261_core_pages_drift.py --write

contract.py and digest.py are loaded from their files, so the command needs none of the
orchestrator's dependencies.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional, Sequence

ORCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ORCH_ROOT.parent
FRONTEND_PREFIX = "frontend/"
THEMES_DIR = ORCH_ROOT / "services" / "themes"
WRITE_FLAG = "--write"
COMMENT = (
    "PRD-261 PA-18: sha256 (CRLF read as LF) of every PAGE_SOURCES file under frontend/. "
    "Generated: python3 orchestrator/tests/test_prd261_core_pages_drift.py --write. Don't edit by hand."
)


def _load(name: str) -> ModuleType:
    """``services/themes/<name>.py`` loaded from its file (no package imports run)."""
    spec = importlib.util.spec_from_file_location(f"prd261_drift_{name}", THEMES_DIR / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load services/themes/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = _load("contract")
digest = _load("digest")


def checkout_hashes() -> Dict[str, str]:
    """``{path: sha256}`` of each PAGE_SOURCES file as the checkout holds it under frontend/."""
    return {
        path: digest.page_source_sha256((REPO_ROOT / f"{FRONTEND_PREFIX}{path}").read_bytes())
        for path in contract.page_source_files()
    }


def render(hashes: Dict[str, str]) -> str:
    """core-pages.json's text for ``hashes``."""
    return json.dumps({"_comment": COMMENT, "files": hashes}, indent=2) + "\n"


def test_every_page_source_is_a_file_under_frontend():
    missing = [path for path in contract.page_source_files() if not (REPO_ROOT / f"{FRONTEND_PREFIX}{path}").is_file()]
    assert missing == [], f"PAGE_SOURCES names files that aren't under frontend/: {missing}"


def test_core_pages_json_matches_the_checkout():
    committed = contract.CORE_PAGES_FILE.read_text(encoding="utf-8")
    assert committed == render(checkout_hashes()), (
        "services/themes/data/core-pages.json is out of date with frontend/. Regenerate it from the "
        "repository root: python3 orchestrator/tests/test_prd261_core_pages_drift.py --write"
    )


def test_the_reader_returns_the_committed_hashes():
    assert contract.core_page_hashes() == checkout_hashes()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``--write`` regenerates core-pages.json from the checkout."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args != [WRITE_FLAG]:
        sys.stderr.write(f"usage: python3 orchestrator/tests/test_prd261_core_pages_drift.py {WRITE_FLAG}\n")
        return 2
    contract.CORE_PAGES_FILE.write_text(render(checkout_hashes()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
