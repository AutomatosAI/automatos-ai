"""PRD-270 K8 — the catalogue's paths, as the hosted edition writes them and a local install reads them.

Every URL in the catalogue is a PATH on the origin it was fetched from: the hosted edition has no
setting that names its own public origin, so the reader resolves each path against the origin it
fetched the catalogue from. A file of a listed version is
``/api/marketplace/catalog/files/<digest>/<path>``, and its package is ``package.autotheme`` there.
"""

from __future__ import annotations

CATALOG_ROUTE = "/api/marketplace/catalog"
CATALOG_FILES_ROUTE = f"{CATALOG_ROUTE}/files"
PACKAGE_FILE = "package.autotheme"


def catalog_file_path(digest: str, path: str) -> str:
    """Where the catalogue's files route serves ``path`` of the version ``digest``."""
    return f"{CATALOG_FILES_ROUTE}/{digest}/{path}"


def package_path(digest: str) -> str:
    """Where the catalogue's files route serves the version ``digest`` as one ``.autotheme``."""
    return catalog_file_path(digest, PACKAGE_FILE)


__all__ = ["CATALOG_FILES_ROUTE", "CATALOG_ROUTE", "PACKAGE_FILE", "catalog_file_path", "package_path"]
