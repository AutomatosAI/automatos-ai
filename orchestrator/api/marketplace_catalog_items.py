"""PRD-270 W2 US-270-12c and 12d (§5.6, K12) — the hosted catalogue's item bundles and plugin packages, public.

``GET /api/marketplace/catalog/bundles/{revision}`` answers the canonical bytes of the bundle the hosted
catalogue lists under ``revision`` (``services/marketplace_bundles/catalog_items.bundle_bytes_for``): the
path every catalogue entry names, from which a local install's Sync fetches a new or changed item. It
answers nothing else: a revision no listed bundle has, or one that is not 64 lower-case hex characters, is
404.

``GET /api/marketplace/catalog/plugins/{slug}/{version}/package.zip`` answers the zip of a plugin's stored
files (``services/marketplace_bundles/plugin_files.py``), only for a plugin the export lists with that slug
at that version (approved, active, its scan safe) whose manifest.json names both. Anything else is 404: a
pending, unsafe or inactive plugin, another version, an unknown slug, files that can't be served (logged),
a slug or version off the bundle format's patterns (checked before the database). The route reads the
row, then ends its read transaction, then reads storage, so no pooled connection waits on S3. Its ETag is
the zip's sha256, the zip's bytes depending on the files alone. A version's zip is built once and kept
(``plugin_packages.kept_package``, keyed by the export's revision and scan too; a refusal kept a minute): a
repeat request, and a matching ``If-None-Match``, read nothing from storage. One request builds a version
at a time, in one of the process's ``plugin_packages.PLUGIN_PACKAGE_BUILDS`` slots, as W1's package rebuilds
do: a wait for either past the short limit is 503 with ``Retry-After`` (logged as a warning), storage never
read.

Both copy the rules of W1's catalogue files route (``api/marketplace_catalog.py``):

* 404 in the local edition (``require_hosted_bundles``), checked first, so a local request never counts
  against the limit;
* ``SERVING_HEADERS`` (nosniff and the sandboxed CSP) on every answer, a refusal included, and every
  refusal ``no-store``;
* ``Cache-Control: public, max-age=<THEME_FILE_CACHE_SECONDS>, immutable`` and a strong ETag with 304:
  ``"<revision>"`` for a bundle, named by its own sha256, so it never changes; the zip's sha256 for a plugin
  package, so a version whose stored files were rewritten (``harvest_plugins --repair-s3``) has another once
  its zip is built again (a kept zip is served until it is dropped or the process restarts; a rescan names
  another scan and builds anew);
* each route its own per-IP bucket (``api/public_rate_limit.py``): over it is 429 with ``Retry-After``;
* 503, logged, when the database can't give the listing back or storage fails, or when a plugin package's
  build doesn't start within the wait (with ``Retry-After``).

Public by design: no credentials and no workspace, so nothing in it depends on the caller. A plain ``def``:
the session is synchronous. Registered through ``RouterSpec`` (``router_manifest.py``) and listed in
``reports/route-manifest.json``.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.marketplace_catalog import NOT_HOSTED
from api.public_rate_limit import REFUSAL_HEADERS, per_ip_limit
from api.themes_files import SERVING_HEADERS
from config import config
from core.database.database import get_db
from services.marketplace_bundles.catalog_items import bundle_bytes_for
from services.marketplace_bundles.catalog_paths import BUNDLES_ROUTE, PLUGIN_PACKAGE_FILE, PLUGINS_ROUTE
from services.marketplace_bundles.plugin_files import (
    PluginFilesUnavailable, PluginPackage, StoredPlugin, exported_plugin,
)
from services.marketplace_bundles.plugin_packages import BUILD_RETRY_SECONDS, PluginBuildsBusy, kept_package
from services.marketplace_bundles.schema import PLUGIN_SLUG_PATTERN, REVISION_PATTERN, VERSION_PATTERN
from services.themes.revocation_feed import etag_matches
from services.themes.serve import file_cache_seconds

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Marketplace"])

BUNDLE_MEDIA_TYPE = "application/json"
PLUGIN_PACKAGE_MEDIA_TYPE = "application/zip"
NOT_MODIFIED, NOT_FOUND, SERVICE_UNAVAILABLE = 304, 404, 503
BUNDLES_RATE_BUCKET = "marketplace-catalog-bundles:ip"
BUNDLES_RATE_LABEL = "marketplace catalogue bundles rate limiter"
PLUGINS_RATE_BUCKET = "marketplace-catalog-plugins:ip"
PLUGINS_RATE_LABEL = "marketplace catalogue plugins rate limiter"
BUNDLE_REFUSAL_HEADERS = {**SERVING_HEADERS, **REFUSAL_HEADERS}
BUNDLE_NOT_LISTED = "That bundle isn't in the Automatos Cloud catalogue."
BUNDLES_DOWN = "Automatos Cloud bundles can't be read right now. Try again in a moment."
PLUGIN_NOT_LISTED = "That plugin package isn't in the Automatos Cloud catalogue."
PLUGINS_DOWN = "Automatos Cloud plugin packages can't be read right now. Try again in a moment."
PLUGINS_BUSY = "Automatos Cloud is busy preparing other plugin packages. Try again in a moment."
_REVISION = re.compile(REVISION_PATTERN)
_PLUGIN_SLUG = re.compile(PLUGIN_SLUG_PATTERN)
_VERSION = re.compile(VERSION_PATTERN)


def _refusal(status_code: int, detail: str, retry_after: Optional[int] = None) -> HTTPException:
    retry = {} if retry_after is None else {"Retry-After": str(retry_after)}
    return HTTPException(status_code=status_code, detail=detail, headers={**BUNDLE_REFUSAL_HEADERS, **retry})


def require_hosted_bundles() -> None:
    """404 in the local edition, before anything else runs, with the serving headers: a local install
    fetches bundles and plugin packages, it never serves them."""
    if config.IS_LOCAL_EDITION:
        raise _refusal(NOT_FOUND, NOT_HOSTED)


limit_bundle_reads = per_ip_limit(BUNDLES_RATE_BUCKET, BUNDLES_RATE_LABEL, BUNDLE_REFUSAL_HEADERS)
limit_plugin_reads = per_ip_limit(PLUGINS_RATE_BUCKET, PLUGINS_RATE_LABEL, BUNDLE_REFUSAL_HEADERS)


def _listed_bytes(db: Session, revision: str) -> Optional[bytes]:
    """The listed bundle's bytes, None for a revision off the pattern or not listed; 503 (logged) when the
    database fails."""
    if _REVISION.fullmatch(revision) is None:
        return None
    try:
        return bundle_bytes_for(db, revision)
    except SQLAlchemyError as failed:
        logger.exception("marketplace catalogue bundles: the listing could not be read")
        raise _refusal(SERVICE_UNAVAILABLE, BUNDLES_DOWN) from failed


def _served_headers(etag: str) -> Dict[str, str]:
    return {"Cache-Control": f"public, max-age={file_cache_seconds()}, immutable", "ETag": etag, **SERVING_HEADERS}


@router.get(BUNDLES_ROUTE + "/{revision}")
def get_catalog_bundle(
    revision: str,
    request: Request,
    _hosted: None = Depends(require_hosted_bundles),
    _limited: None = Depends(limit_bundle_reads),
    db: Session = Depends(get_db),
) -> Response:
    """A listed bundle's canonical bytes (200), or 304 when ``If-None-Match`` names its ETag; 404 in the
    local edition and for any revision the catalogue doesn't list; 503 when the database fails; 429 over
    the per-IP limit."""
    data = _listed_bytes(db, revision)
    if data is None:
        raise _refusal(NOT_FOUND, BUNDLE_NOT_LISTED)
    etag = f'"{revision}"'
    headers = _served_headers(etag)
    if etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=NOT_MODIFIED, headers=headers)
    return Response(content=data, media_type=BUNDLE_MEDIA_TYPE, headers=headers)


def _exported(db: Session, slug: str, version: str) -> Optional[StoredPlugin]:
    """The exported plugin, None for a slug or version off the format's patterns or not exported; 503 when
    the database fails (logged)."""
    if _PLUGIN_SLUG.fullmatch(slug) is None or _VERSION.fullmatch(version) is None:
        return None
    try:
        return exported_plugin(db, slug, version)
    except SQLAlchemyError as failed:
        logger.exception("marketplace catalogue plugins: the export could not be read")
        raise _refusal(SERVICE_UNAVAILABLE, PLUGINS_DOWN) from failed


def _built(plugin: StoredPlugin) -> Optional[PluginPackage]:
    """The plugin's zip (kept after its first build), None when its files can't be served; 503 when storage
    fails (logged), or with ``Retry-After`` when its build doesn't start within the wait (logged as a
    warning)."""
    try:
        return kept_package(plugin)
    except PluginBuildsBusy as busy:
        logger.warning("marketplace catalogue plugins: the package build of %s %s did not start within the wait",
                       plugin.slug, plugin.version)
        raise _refusal(SERVICE_UNAVAILABLE, PLUGINS_BUSY, BUILD_RETRY_SECONDS) from busy
    except PluginFilesUnavailable as down:  # plugin_files logged the storage failure
        logger.error("marketplace catalogue plugins: storage failed for %s %s", plugin.slug, plugin.version)
        raise _refusal(SERVICE_UNAVAILABLE, PLUGINS_DOWN) from down


@router.get(PLUGINS_ROUTE + "/{slug}/{version}/" + PLUGIN_PACKAGE_FILE)
def get_catalog_plugin_package(
    slug: str,
    version: str,
    request: Request,
    _hosted: None = Depends(require_hosted_bundles),
    _limited: None = Depends(limit_plugin_reads),
    db: Session = Depends(get_db),
) -> Response:
    """An exported plugin's zip (200), or 304 when ``If-None-Match`` names its ETag; 404 in the local
    edition and for any plugin or version the export doesn't list; 503 when the database or storage fails,
    or the build doesn't start within the wait; 429 over the per-IP limit."""
    plugin = _exported(db, slug, version)
    db.rollback()  # the read is over: the pooled connection goes back before the storage reads
    package = None if plugin is None else _built(plugin)
    if package is None:
        raise _refusal(NOT_FOUND, PLUGIN_NOT_LISTED)
    etag = f'"{package.sha256}"'
    headers = _served_headers(etag)
    if etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=NOT_MODIFIED, headers=headers)
    return Response(content=package.data, media_type=PLUGIN_PACKAGE_MEDIA_TYPE, headers=headers)


__all__ = [
    "BUNDLES_RATE_BUCKET", "PLUGINS_RATE_BUCKET", "limit_bundle_reads", "limit_plugin_reads",
    "require_hosted_bundles", "router",
]
