"""PRD-261 A-08a — theme files by digest, and an install's first-paint data.

* ``GET /api/themes/files/{digest}/{path}``: one file of a theme package, to exactly the
  callers allowed to load it (``services/themes/serve.py``: an approved listing, or an
  install in a workspace the caller belongs to, with their opt-in for a private package).
  Not workspace-scoped: a native ``import()`` and ``<video src>`` send no ``X-Workspace-ID``,
  so ``ctx.workspace_id`` is never read. The bytes are re-hashed against the package's file
  manifest on every read; one ``Range`` is answered with a 206 (Safari plays MP4 clips only
  that way) and an unsatisfiable one with a 416.
* ``GET /api/themes/boot/{install_id}``: the install's ``ThemeBoot`` (plan §2.2). The
  install's workspace comes from the install row; the caller must belong to it.

Every response carries ``X-Content-Type-Options: nosniff`` (a ``source/*.js`` file is served
as text/plain and must never run) and a sandboxed CSP, so a theme file opened as a page runs
nothing. A request carrying ``Service-Worker: script`` is refused (403): theme code never
registers a service worker, and ``Service-Worker-Allowed`` is never sent. Anything the caller
may not load is a 404, never a 403, so a digest or an install id of another workspace says
nothing about whether it exists.

Plain ``def`` routes: the database and the object store are synchronous (FastAPI runs these
in its threadpool). B2-wire registers this router in ``main.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from services.themes.membership import caller_membership
from services.themes.ranges import RangeNotSatisfiable, content_range, requested_range, unsatisfied_content_range
from services.themes.serve import (
    FileReader,
    ThemeFileMissing,
    ThemeStorageUnavailable,
    authorised_package,
    boot_target,
    file_cache_seconds,
    install_boot,
    read_stored_file,
    verified_file,
)

router = APIRouter(prefix="/api/themes", tags=["Themes"])

SERVICE_WORKER_HEADER = "service-worker"
SERVICE_WORKER_SCRIPT = "script"
SERVING_HEADERS: Dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; sandbox",
}
BOOT_HEADERS: Dict[str, str] = {"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"}
FILE_NOT_FOUND = "That theme file isn't available."
BOOT_NOT_FOUND = "That theme isn't installed in any workspace you belong to."
NO_SERVICE_WORKER = "Theme files can't be registered as a service worker."
STORAGE_DOWN = "Theme files can't be read right now. Try again in a moment."
WEAK_ETAG_PREFIX = "W/"


def get_theme_file_reader() -> FileReader:
    """Where stored theme files are read from (a dependency, so tests read from memory)."""
    return read_stored_file


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=404, detail=detail, headers=SERVING_HEADERS)


def _refuse_service_worker(request: Request) -> None:
    if request.headers.get(SERVICE_WORKER_HEADER, "").strip().lower() == SERVICE_WORKER_SCRIPT:
        raise HTTPException(status_code=403, detail=NO_SERVICE_WORKER, headers=SERVING_HEADERS)


def _etag_matches(if_none_match: Optional[str], etag: str) -> bool:
    if not if_none_match:
        return False
    tags = [tag.strip() for tag in if_none_match.split(",")]
    return any(tag == "*" or tag.removeprefix(WEAK_ETAG_PREFIX) == etag for tag in tags)


def _file_response(request: Request, data: bytes, content_type: str, sha256: str) -> Response:
    """200, 206, 304 or 416 for one verified file."""
    etag = f'"{sha256}"'
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": f"private, max-age={file_cache_seconds()}, immutable",
        "ETag": etag,
        **SERVING_HEADERS,
    }
    if _etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=headers)
    try:
        wanted = requested_range(request.headers.get("range"), len(data))
    except RangeNotSatisfiable:
        refused = {"Accept-Ranges": "bytes", "Content-Range": unsatisfied_content_range(len(data)), **SERVING_HEADERS}
        return Response(status_code=416, headers=refused)
    if wanted is None:
        return Response(content=data, media_type=content_type, headers=headers)
    partial = {**headers, "Content-Range": content_range(wanted, len(data))}
    return Response(content=data[wanted.start:wanted.end + 1], status_code=206, media_type=content_type, headers=partial)


@router.get("/files/{digest}/{path:path}")
def get_theme_file(
    digest: str,
    path: str,
    request: Request,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Any = Depends(get_db),
    reader: FileReader = Depends(get_theme_file_reader),
) -> Response:
    """One file of a theme package the caller may load, by the package digest."""
    _refuse_service_worker(request)
    package = authorised_package(db, caller_membership(db, ctx), digest)
    if package is None:
        raise _not_found(FILE_NOT_FOUND)
    try:
        data, content_type, sha256 = verified_file(package, path, reader)
    except ThemeFileMissing as exc:
        raise _not_found(FILE_NOT_FOUND) from exc
    except ThemeStorageUnavailable as exc:
        raise HTTPException(status_code=503, detail=STORAGE_DOWN, headers=SERVING_HEADERS) from exc
    return _file_response(request, data, content_type, sha256)


def _install_uuid(install_id: str) -> Optional[UUID]:
    try:
        return UUID(install_id)
    except ValueError:
        return None


@router.get("/boot/{install_id}")
def get_theme_boot(
    install_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Any = Depends(get_db),
    reader: FileReader = Depends(get_theme_file_reader),
) -> JSONResponse:
    """The install's first-paint data; the base boot when the caller hasn't opted in or it is revoked."""
    parsed = _install_uuid(install_id)
    membership = caller_membership(db, ctx)
    target = boot_target(db, membership, parsed) if parsed is not None else None
    if target is None:
        raise HTTPException(status_code=404, detail=BOOT_NOT_FOUND, headers=BOOT_HEADERS)
    try:
        boot = install_boot(db, membership, target, reader)
    except ThemeStorageUnavailable as exc:
        raise HTTPException(status_code=503, detail=STORAGE_DOWN, headers=BOOT_HEADERS) from exc
    return JSONResponse(content=boot, headers=BOOT_HEADERS)


__all__ = ["get_theme_file_reader", "router"]
