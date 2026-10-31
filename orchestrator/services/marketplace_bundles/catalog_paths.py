"""PRD-270 W2 (K11, K12, K13, K15) — where the hosted catalogue serves an item's bundle and a plugin's package.

A catalogue entry names its bundle by PATH (W1's rule for every catalogue URL, ``urls.py``): the bundle
route under the catalogue, then the bundle's revision. The hosted route is mounted at ``BUNDLES_ROUTE``;
a local install builds the URL from its configured origin and the revision, never from the entry. A
plugin's files are a zip under ``PLUGINS_ROUTE``, at a path built from the plugin bundle's slug and
version (``plugin_package_path``), never from a URL a catalogue or a bundle carries.

A leaf: W1's ``urls.py`` (itself a leaf) and nothing else, so the hosted and the local side share it.
"""
from __future__ import annotations

from services.marketplace_cloud.urls import CATALOG_ROUTE

BUNDLES_ROUTE = f"{CATALOG_ROUTE}/bundles"
PLUGINS_ROUTE = f"{CATALOG_ROUTE}/plugins"
PLUGIN_PACKAGE_FILE = "package.zip"


def bundle_path(revision: str) -> str:
    """Where the catalogue's bundle route serves the bundle whose canonical bytes hash to ``revision``."""
    return f"{BUNDLES_ROUTE}/{revision}"


def plugin_package_path(slug: str, version: str) -> str:
    """Where the catalogue's plugin route serves the zip of the plugin ``slug`` at ``version``."""
    return f"{PLUGINS_ROUTE}/{slug}/{version}/{PLUGIN_PACKAGE_FILE}"


__all__ = ["BUNDLES_ROUTE", "PLUGINS_ROUTE", "PLUGIN_PACKAGE_FILE", "bundle_path", "plugin_package_path"]
