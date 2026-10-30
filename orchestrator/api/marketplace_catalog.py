"""PRD-270 US-270-01 and US-270-02 (§5.2, K8, K9) — the hosted catalogue, schema 1, and its public files.

The hosted edition's public list of the signed theme versions it lists
(``services/marketplace_cloud/catalog.py``), which every local install's Sync with Automatos Cloud
reads and the public ``/marketplace`` shows. Public by design: no credentials and no workspace, so
nothing in it depends on the caller, and nothing in it is a workspace's. It copies the revocation
feed's rules (``api/themes_revocations_feed.py``):

* the local edition answers 404: it syncs the catalogue, it doesn't serve one. The edition is checked
  first, so a local request never counts against the limit;
* ``Cache-Control: public, max-age=300`` and ``nosniff``, with a weak ETag over everything but
  ``generatedAt`` (``revocation_feed.feed_etag``), so a matching ``If-None-Match`` is a 304 until a
  listed version changes;
* 503, logged as an error, when the signing keys file doesn't load, when storage can't give back a
  listed version's theme.json, or when the catalogue is over ``MARKETPLACE_CATALOG_FEED_MAX_KB``:
  served, it would fail every Sync. The size is checked before the ETag, so such a catalogue is never
  confirmed with a 304 either. A refusal (404, 429, 503) carries ``Cache-Control: no-store``;
* each client IP gets ``MARKETPLACE_PUBLIC_RATE_PER_MINUTE`` reads a minute in its own bucket
  (``api/public_rate_limit.py``): over it is 429 with ``Retry-After``;
* the catalogue is built at most once per ``catalog.CATALOG_KEPT_SECONDS`` for every caller
  (``catalog.catalog_snapshot``), so a public request never rebuilds it on its own.

``GET /api/marketplace/catalog/files/{digest}/{path}`` serves what the catalogue's paths name, and
nothing else of any package (``services/marketplace_cloud/catalog_files.py``): a listed version's
rebuilt ``package.autotheme`` and the screenshots its theme.json lists. Its rules are the member files
route's (``api/themes_files.py``), made public:

* ``SERVING_HEADERS`` (nosniff and the sandboxed CSP) on every answer, a refusal included, and every
  refusal ``no-store``; ``Service-Worker: script`` is refused (403);
* ``Cache-Control: public, max-age=<THEME_FILE_CACHE_SECONDS>, immutable`` and a strong ETag, the file's
  sha256 (the package's: its digest), with 304: a stored file never changes. The ETag is known from the
  row, so a 304 reads nothing from storage and never rebuilds the package;
* 404 in the local edition (checked first, so it never counts against the limit), and for anything the
  catalogue doesn't serve, a package that can't be rebuilt included (logged); 503 when storage or the keys
  file fails; its own per-IP bucket (429 with ``Retry-After``).

Plain ``def`` routes: the session and the object store are synchronous. Registered through ``RouterSpec``
(``router_manifest.py``) and listed in ``reports/route-manifest.json``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Dict, Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy.orm import Session

from api.public_rate_limit import REFUSAL_HEADERS, per_ip_limit
from api.themes_files import (
    NO_SERVICE_WORKER, SERVICE_WORKER_HEADER, SERVICE_WORKER_SCRIPT, SERVING_HEADERS, get_theme_file_reader,
)
from config import config
from core.database.database import get_db
from services.marketplace_cloud.catalog import CatalogSnapshot, CatalogUnavailable, catalog_max_bytes, catalog_snapshot
from services.marketplace_cloud.catalog_files import (
    CatalogFileMissing, CatalogFilesUnavailable, listed_sha256, read_catalog_file, served_version,
)
from services.marketplace_cloud.urls import CATALOG_FILES_ROUTE, CATALOG_ROUTE
from services.themes.revocation_feed import etag_matches
from services.themes.serve import FileReader, file_cache_seconds
from services.themes.signature_verify import SigningKeysError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Marketplace"])

CATALOG_MAX_AGE_SECONDS = 300
CATALOG_HEADERS = {"Cache-Control": f"public, max-age={CATALOG_MAX_AGE_SECONDS}", "X-Content-Type-Options": "nosniff"}
CATALOG_MEDIA_TYPE = "application/json"
NOT_MODIFIED, FORBIDDEN, NOT_FOUND, SERVICE_UNAVAILABLE = 304, 403, 404, 503
CATALOG_RATE_BUCKET = "marketplace-catalog:ip"
CATALOG_RATE_LABEL = "marketplace catalogue rate limiter"
FILES_RATE_BUCKET = "marketplace-catalog-files:ip"
FILES_RATE_LABEL = "marketplace catalogue files rate limiter"
FILE_REFUSAL_HEADERS = {**SERVING_HEADERS, **REFUSAL_HEADERS}
NOT_HOSTED = "The Automatos Cloud catalogue is served by the hosted edition; this edition syncs it."
CATALOG_DOWN = "The Automatos Cloud catalogue can't be built right now. Try again in a moment."
CATALOG_TOO_LARGE = "The Automatos Cloud catalogue is too large to serve right now. Try again later."
FILE_NOT_SERVED = "That file isn't in the Automatos Cloud catalogue."
FILES_DOWN = "Automatos Cloud catalogue files can't be read right now. Try again in a moment."


def require_hosted_edition() -> None:
    """404 in the local edition, before anything else runs: a local install syncs the catalogue, never serves it."""
    if config.IS_LOCAL_EDITION:
        raise HTTPException(status_code=NOT_FOUND, detail=NOT_HOSTED, headers={**REFUSAL_HEADERS})


def require_hosted_files() -> None:
    """The files route's edition check: 404 in the local edition, with the files' serving headers."""
    if config.IS_LOCAL_EDITION:
        raise HTTPException(status_code=NOT_FOUND, detail=NOT_HOSTED, headers={**FILE_REFUSAL_HEADERS})


limit_catalog_reads = per_ip_limit(CATALOG_RATE_BUCKET, CATALOG_RATE_LABEL)
limit_file_reads = per_ip_limit(FILES_RATE_BUCKET, FILES_RATE_LABEL, FILE_REFUSAL_HEADERS)


def _unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=SERVICE_UNAVAILABLE, detail=detail, headers={**REFUSAL_HEADERS})


def _snapshot(db: Session) -> CatalogSnapshot:
    """The kept or freshly built catalogue, or the route's 503 (logged) when the keys file or a theme.json can't
    be read."""
    try:
        return catalog_snapshot(db)
    except SigningKeysError as broken:
        logger.error("marketplace catalogue: the signing keys file did not load: %s", broken)
        raise _unavailable(CATALOG_DOWN) from broken
    except CatalogUnavailable as unreadable:
        logger.error("marketplace catalogue not served: %s", unreadable)
        raise _unavailable(CATALOG_DOWN) from unreadable


@router.get(CATALOG_ROUTE)
def get_catalog(
    request: Request,
    _hosted: None = Depends(require_hosted_edition),
    _limited: None = Depends(limit_catalog_reads),
    db: Session = Depends(get_db),
) -> Response:
    """The catalogue (200), or 304 when ``If-None-Match`` names its ETag; 404 in the local edition; 503 when
    the keys file or a theme.json can't be read or the catalogue is over its size; 429 over the per-IP limit."""
    snapshot, cap = _snapshot(db), catalog_max_bytes()
    if len(snapshot.body) > cap:
        logger.error("marketplace catalogue not served: %d bytes, %d items, over the %d-byte cap",
                     len(snapshot.body), len(snapshot.catalog["items"]), cap)
        raise _unavailable(CATALOG_TOO_LARGE)
    headers = {**CATALOG_HEADERS, "ETag": snapshot.etag}
    if etag_matches(request.headers.get("if-none-match"), snapshot.etag):
        return Response(status_code=NOT_MODIFIED, headers=headers)
    return Response(content=snapshot.body, media_type=CATALOG_MEDIA_TYPE, headers=headers)


def _file_refusal(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail, headers={**FILE_REFUSAL_HEADERS})


@contextmanager
def _file_refusals(digest: str, path: str) -> Iterator[None]:
    """The files route's refusals: 404 for nothing to serve, 503 (logged) when storage or the keys file fails."""
    try:
        yield
    except CatalogFileMissing as missing:
        raise _file_refusal(NOT_FOUND, FILE_NOT_SERVED) from missing
    except SigningKeysError as broken:
        logger.error("marketplace catalogue files: the signing keys file did not load: %s", broken)
        raise _file_refusal(SERVICE_UNAVAILABLE, FILES_DOWN) from broken
    except CatalogFilesUnavailable as down:  # only after the digest and path passed the allowlist
        logger.error("marketplace catalogue files: storage failed reading %r of %s", path, digest)
        raise _file_refusal(SERVICE_UNAVAILABLE, FILES_DOWN) from down


def _file_headers(sha256: str) -> Dict[str, str]:
    return {
        "Cache-Control": f"public, max-age={file_cache_seconds()}, immutable", "ETag": f'"{sha256}"',
        **SERVING_HEADERS,
    }


@router.get(CATALOG_FILES_ROUTE + "/{digest}/{path:path}")
def get_catalog_file(
    digest: str,
    path: str,
    request: Request,
    _hosted: None = Depends(require_hosted_files),
    _limited: None = Depends(limit_file_reads),
    db: Session = Depends(get_db),
    reader: FileReader = Depends(get_theme_file_reader),
) -> Response:
    """A listed version's package or listed screenshot (200), or 304 when ``If-None-Match`` names its ETag; 403
    for a service worker; 404 in the local edition and for anything else; 503 when storage fails; 429 over the
    per-IP limit."""
    if request.headers.get(SERVICE_WORKER_HEADER, "").strip().lower() == SERVICE_WORKER_SCRIPT:
        raise _file_refusal(FORBIDDEN, NO_SERVICE_WORKER)
    with _file_refusals(digest, path):
        package = served_version(db, digest, path)
    known = listed_sha256(package, path)
    if known is not None and etag_matches(request.headers.get("if-none-match"), f'"{known}"'):
        return Response(status_code=NOT_MODIFIED, headers=_file_headers(known))
    with _file_refusals(digest, path):
        served = read_catalog_file(package, path, reader)
    return Response(content=served.data, media_type=served.content_type, headers=_file_headers(served.sha256))


__all__ = ["CATALOG_HEADERS", "router"]
