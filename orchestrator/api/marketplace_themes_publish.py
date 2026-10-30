"""PRD-261 — ``POST /api/marketplace/themes/publish``: the Design Kit's publish job lists, approves and signs
a theme version on the Marketplace, with no person in the loop (D2: the kit merge is the approval).

Its own credential, never hybrid's: ``Authorization: Bearer <THEME_PUBLISH_KEY>``, checked by
``require_publish_key`` against ``THEME_PUBLISH_KEY_SHA256`` (``services/themes/publish_key.py``):
503 ``PUBLISH_NOT_CONFIGURED`` while any of the three settings is unset, 401 for a missing or wrong
key. The authZ sweep classifies the route as own-auth (``OWN_AUTH_ROUTES``) and proves it carries no
``get_request_context_hybrid``; no other route accepts the key.

Gates, in order: the key, the Marketplace switch (403 while ``THEMES_MARKETPLACE`` is off), then
the multipart body, read only after the key passed (no body parameter, so FastAPI parses nothing
first). Its parts, each a file or a text field except the package:

* ``package``: the ``.autotheme`` (a file part), read up to one byte past the package cap, then
  checked as a submission is: 422 ``PACKAGE_REFUSED`` with the checks' codes, 422 ``SOURCE_REQUIRED``
  without ``source/``;
* ``signature``: the kit's ``.sig.json`` envelope text (``signed_upload``), at most 4 KB;
* ``review_summary``: the kit review's summary for this pack, UTF-8, at most the admin queue's note
  cap (``admin_themes.SUMMARY_MAX`` characters).

A missing or unreadable part is 422 ``PART_MISSING`` or ``SUMMARY_NOT_THIS_VERSION``. Everything
after that is ``services/themes/publish.py``: 200 ``{status: published|unchanged, id, version,
digest}``, else its refusal with ``{code, message, errors}``. ``main.py`` gives the route the
submit route's body limit. Refusals are logged by status and code only; nothing logs the key. A
plain ``def`` handler (a synchronous session and S3 client, in the threadpool); the body is read in
an async dependency, which touches no database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile

from api.admin_themes import SUMMARY_MAX
from api.marketplace_themes_submit import (
    SOURCE_REQUIRED, SOURCE_REQUIRED_CODE, SOURCE_REQUIRED_STATUS, marketplace_review, require_marketplace,
    ships_source,
)
from core.auth.bearer_registry import bearer_token
from core.database.database import get_db
from services.themes import publish
from services.themes.llm_stage import SOURCE_PREFIX
from services.themes.package import ThemePackage as CheckedThemePackage
from services.themes.package import ThemePackageError, cap_bytes, read_theme_package
from services.themes.publish_checks import (
    SUMMARY_CODE, SUMMARY_PATH, UNPROCESSABLE, PublishRefused, key_refused, not_configured,
)
from services.themes.publish_key import PublishSettings, key_matches, publish_settings
from services.themes.signed_upload import ENVELOPE_MAX_BYTES
from services.themes.submissions import ReviewStage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/marketplace/themes", tags=["Theme Marketplace"])
PUBLISH_PATH = publish.THEME_PUBLISH_ROUTE[len(router.prefix):]

PACKAGE_PART, SIGNATURE_PART, SUMMARY_PART = "package", "signature", "review_summary"
PARTS: Tuple[str, ...] = (PACKAGE_PART, SIGNATURE_PART, SUMMARY_PART)
TEXT_ENCODING = "utf-8"
UTF8_MAX_BYTES_PER_CHAR = 4
SUMMARY_MAX_BYTES = SUMMARY_MAX * UTF8_MAX_BYTES_PER_CHAR
PACKAGE_REFUSED_CODE = "PACKAGE_REFUSED"
PART_MISSING_CODE = "PART_MISSING"
PACKAGE_REFUSED = "This theme can't be published."
PART_MISSING = "The publish request needs the multipart parts package (a file), signature and review_summary."
SUMMARY_UNREADABLE = f"The review summary must be UTF-8 text of at most {SUMMARY_MAX} characters."


@dataclass(frozen=True)
class PublishParts:
    """The three parts as read, each up to one byte past its cap; None for a part that is missing."""

    package: Optional[bytes]
    signature: Optional[bytes]
    summary: Optional[bytes]


def _http(refused: PublishRefused) -> HTTPException:
    """A refusal as the route answers it, logged by status and code only."""
    logger.info("[Themes] Publish refused (%d): %s", refused.status_code, refused.code)
    return HTTPException(status_code=refused.status_code, detail=refused.detail, headers=refused.headers)


def require_publish_key(request: Request) -> PublishSettings:
    """503 while the publish settings are unset, 401 unless the bearer is the publish key; the settings."""
    settings = publish_settings()
    if settings is None:
        raise _http(not_configured())
    presented = bearer_token(request)
    if not presented or not key_matches(presented, settings.key_hashes):
        raise _http(key_refused())
    return settings


async def _read_part(value: Any, cap: int, text_ok: bool) -> Optional[bytes]:
    """A part's bytes up to ``cap``: a file part's, or a text field's when ``text_ok``; None otherwise."""
    if isinstance(value, UploadFile):
        return await value.read(cap)
    if text_ok and isinstance(value, str):
        return value.encode(TEXT_ENCODING)[:cap]
    return None


async def publish_parts(request: Request) -> PublishParts:
    """The multipart body, parsed only now (after the key) and bounded: three parts at most, each read to its cap."""
    limit = len(PARTS)
    async with request.form(max_files=limit, max_fields=limit, max_part_size=SUMMARY_MAX_BYTES + 1) as form:
        package = await _read_part(form.get(PACKAGE_PART), cap_bytes("package") + 1, text_ok=False)
        signature = await _read_part(form.get(SIGNATURE_PART), ENVELOPE_MAX_BYTES + 1, text_ok=True)
        summary = await _read_part(form.get(SUMMARY_PART), SUMMARY_MAX_BYTES + 1, text_ok=True)
    return PublishParts(package=package, signature=signature, summary=summary)


def publish_review_stage() -> ReviewStage:
    """Core's automated review stage on a published version: the submit route's. Tests override this."""
    return marketplace_review


def _checked_package(data: bytes) -> CheckedThemePackage:
    """The package checked as a submission is: 422 with the checks' codes, or without ``source/``."""
    try:
        checked = read_theme_package(data)
    except ThemePackageError as exc:
        raise PublishRefused(UNPROCESSABLE, PACKAGE_REFUSED_CODE, PACKAGE_REFUSED, errors=exc.errors) from exc
    if not ships_source(checked):
        raise PublishRefused(SOURCE_REQUIRED_STATUS, SOURCE_REQUIRED_CODE, SOURCE_REQUIRED, path=SOURCE_PREFIX)
    return checked


def _summary_text(data: bytes) -> str:
    """The summary part as text: UTF-8 and at most ``SUMMARY_MAX`` characters, else 422."""
    try:
        summary = data.decode(TEXT_ENCODING)
    except UnicodeDecodeError as bad:
        raise PublishRefused(UNPROCESSABLE, SUMMARY_CODE, SUMMARY_UNREADABLE, path=SUMMARY_PATH) from bad
    if len(summary) > SUMMARY_MAX:
        raise PublishRefused(UNPROCESSABLE, SUMMARY_CODE, SUMMARY_UNREADABLE, path=SUMMARY_PATH)
    return summary


def _upload(parts: PublishParts) -> publish.PublishUpload:
    """The parts checked at the boundary: all three present, the package read, the summary decoded."""
    missing = next((name for name, data in zip(PARTS, (parts.package, parts.signature, parts.summary))
                    if data is None), None)
    if missing is not None:
        raise PublishRefused(UNPROCESSABLE, PART_MISSING_CODE, PART_MISSING, path=missing)
    return publish.PublishUpload(
        checked=_checked_package(parts.package), package_bytes=len(parts.package),
        envelope=parts.signature, summary=_summary_text(parts.summary),
    )


@router.post(PUBLISH_PATH)
def publish_theme(
    settings: PublishSettings = Depends(require_publish_key),
    _marketplace_on: None = Depends(require_marketplace),
    parts: PublishParts = Depends(publish_parts),
    db: Session = Depends(get_db),
    stage: ReviewStage = Depends(publish_review_stage),
) -> Dict[str, Any]:
    """Publish a theme version: list it, approve it and store its signature, all or nothing."""
    try:
        return publish.publish_version(db, _upload(parts), settings, stage)
    except PublishRefused as refused:
        db.rollback()
        raise _http(refused) from refused
