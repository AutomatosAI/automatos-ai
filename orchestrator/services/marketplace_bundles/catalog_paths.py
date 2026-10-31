"""PRD-270 W2 (K11, K12, K13) — where the hosted catalogue serves an item's bundle.

A catalogue entry names its bundle by PATH (W1's rule for every catalogue URL, ``urls.py``): the bundle
route under the catalogue, then the bundle's revision. The hosted route is mounted at ``BUNDLES_ROUTE``;
a local install builds the URL from its configured origin and the revision, never from the entry.

A leaf: W1's ``urls.py`` (itself a leaf) and nothing else, so the hosted and the local side share it.
"""
from __future__ import annotations

from services.marketplace_cloud.urls import CATALOG_ROUTE

BUNDLES_ROUTE = f"{CATALOG_ROUTE}/bundles"


def bundle_path(revision: str) -> str:
    """Where the catalogue's bundle route serves the bundle whose canonical bytes hash to ``revision``."""
    return f"{BUNDLES_ROUTE}/{revision}"


__all__ = ["BUNDLES_ROUTE", "bundle_path"]
