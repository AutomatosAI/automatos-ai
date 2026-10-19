"""PRD-261 A-07b — the "needs update" flag of a theme that clones core pages (PRD §4.6).

A theme page in ``mode: "clone"`` names the core file it was copied from and that file's
sha256 at the time (``pages[route].clonedFrom {path, sha}``). This host release's hashes
of those files are committed in ``services/themes/data/core-pages.json`` (PA-18; the
running image has no ``frontend/`` folder). When a host release changes a file a clone
copied, the two hashes differ and the install row and the active theme say
``needsUpdate: true``, so the workspace knows its theme's copy is behind the core page.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Mapping

from services.themes.contract import core_page_hashes


@lru_cache(maxsize=1)
def _release_hashes() -> Dict[str, str]:
    """``{path: sha256}`` of the core page files this release ships, read once per process."""
    return core_page_hashes()


def _clone_is_stale(entry: Any, release: Mapping[str, str]) -> bool:
    if not isinstance(entry, dict) or entry.get("mode") != "clone":
        return False
    cloned = entry.get("clonedFrom")
    if not isinstance(cloned, dict):
        return False
    # A path this release no longer ships is stale too: the copy has nothing to match.
    return release.get(cloned.get("path")) != cloned.get("sha")


def needs_update(manifest: Any) -> bool:
    """True when any cloned page's ``clonedFrom.sha`` differs from this release's file hash."""
    pages = manifest.get("pages") if isinstance(manifest, dict) else None
    if not isinstance(pages, dict):
        return False
    release = _release_hashes()
    return any(_clone_is_stale(entry, release) for entry in pages.values())
