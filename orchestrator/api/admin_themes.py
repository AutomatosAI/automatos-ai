"""PRD-261 C-03a — the admin review queue for Marketplace theme versions.

Routes (plan §2.2; C1-wire registers the router, lists the routes in
``reports/route-manifest.json`` and adds every mutating one to the authZ sweep's
``ADMIN_GATED_IN_HANDLER`` bucket):

=========================================================  ===========================
``GET  /api/admin/themes/pending``                          the queue, oldest first
``GET  /api/admin/themes/{packageId}/report``               scan report, note, history
``POST /api/admin/themes/{packageId}/report-note``          ``{summary, outcome}``
``POST /api/admin/themes/{packageId}/approve``              records a signature request
``POST /api/admin/themes/{packageId}/reject``               ``{reason}``
``POST /api/admin/themes/{packageId}/deactivate``           ``{reason}``; disable revocation
``POST /api/admin/themes/{packageId}/signature``            ``{requestId, signature}``
=========================================================  ===========================

Every handler checks ``caller_is_admin`` (``core/auth/roles.py:28-34``) first, through
``_require_admin(ctx)``; the local edition's anonymous owner is a super admin
(``hybrid.py:642``), so the queue works there too. A version that isn't a Marketplace
submission answers 404, as a missing one does. D3 (19 Oct): a person reviews every version;
the orchestrator never signs and never calls GitHub (``services/themes/signing.py``).

The four routes that publish, take down or vouch for a version (``report-note``, ``approve``,
``deactivate`` and ``signature``) also need a fresh step-up (A-07c, ``STEP_UP_MAX_AGE_MINUTES``):
``require_theme_admin`` runs first as a dependency, so a non-admin gets the plain 403 and is
never asked to reverify, then ``STEP_UP``. The handler still checks ``_require_admin(ctx)``
first (the authZ sweep reads it there). Plain ``def`` handlers over the synchronous session (AGENTS.md).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Literal, Optional, TypeVar
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.roles import caller_is_admin
from core.auth.step_up import require_step_up
from core.database.database import get_db
from services.themes import review
from services.themes.choices import CONFLICT, ThemeInstallError
from services.themes.signing import SIGNATURE_TEXT_MAX

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/themes", tags=["Admin Themes"])

FORBIDDEN = 403
ADMIN_REQUIRED = "Only an Automatos admin can review Marketplace themes."
CONCURRENT_CHANGE = "Someone changed this version at the same moment. Reload and try again."
REASON_MAX = 2000
SUMMARY_MAX = 20000

STEP_UP = require_step_up(config.STEP_UP_MAX_AGE_MINUTES)

T = TypeVar("T")


class ReasonBody(BaseModel):
    """``reject`` and ``deactivate``: the reason the submitter and the audit trail see."""

    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=1, max_length=REASON_MAX)


class ReportNoteBody(BaseModel):
    """The summary of the kit's isolated review job, pasted by the reviewer, and its outcome."""

    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1, max_length=SUMMARY_MAX)
    outcome: Literal["pass", "fail"]


class SignatureBody(BaseModel):
    """The base64 Ed25519 signature the kit's sign workflow returned, for one signature request."""

    model_config = ConfigDict(extra="forbid")
    requestId: UUID  # noqa: N815 — the wire name the approval answer carries
    signature: str = Field(min_length=1, max_length=SIGNATURE_TEXT_MAX)


def _require_admin(ctx: RequestContext) -> None:
    """403 unless the caller is a platform admin (``super_admin`` included)."""
    user = getattr(ctx, "user", None)
    if user is None or not caller_is_admin(user):
        raise HTTPException(status_code=FORBIDDEN, detail=ADMIN_REQUIRED)


def require_theme_admin(ctx: RequestContext = Depends(get_request_context_hybrid)) -> None:
    """``_require_admin`` as a dependency, listed before ``STEP_UP``: a non-admin is never asked to reverify."""
    _require_admin(ctx)


# The admin check, then a fresh step-up: on the routes that publish, take down or vouch for a version.
ADMIN_STEP_UP = [Depends(require_theme_admin), Depends(STEP_UP)]


def _run(db: Session, work: Callable[[], T], commit: bool) -> T:
    """Run one review call; commit a write once, and answer a refusal with its own status."""
    try:
        result = work()
        if commit:
            db.commit()
        return result
    except ThemeInstallError as refusal:
        db.rollback()
        raise HTTPException(status_code=refusal.status_code, detail=refusal.message) from refusal
    except IntegrityError as clash:
        db.rollback()
        logger.warning("theme review write lost a race: %s", clash.orig.__class__.__name__)
        raise HTTPException(status_code=CONFLICT, detail=CONCURRENT_CHANGE) from clash


@router.get("/pending")
def list_pending_themes(
    limit: Optional[int] = Query(None, ge=1, le=review.PENDING_LIMIT_MAX),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Marketplace theme versions waiting for review, oldest first."""
    _require_admin(ctx)
    return {"pending": review.list_pending(db, review.pending_limit(limit))}


@router.get("/{package_id}/report")
def get_theme_report(
    package_id: UUID, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """One version's scan report, the isolated-job note, its history and its signature request."""
    _require_admin(ctx)
    return _run(db, lambda: review.report(db, package_id), commit=False)


@router.post("/{package_id}/report-note", dependencies=ADMIN_STEP_UP)
def add_theme_report_note(
    package_id: UUID, body: ReportNoteBody,
    ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Store the isolated review job's summary the reviewer pasted (C-02b)."""
    _require_admin(ctx)
    actor = resolve_user_pk(db, ctx)
    return _run(db, lambda: review.add_report_note(db, package_id, body.summary, body.outcome, actor), commit=True)


@router.post("/{package_id}/approve", dependencies=ADMIN_STEP_UP)
def approve_theme(
    package_id: UUID, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Approve a reviewed version; refuses a blocked or not-run scan; answers the signature request."""
    _require_admin(ctx)
    actor = resolve_user_pk(db, ctx)
    answer = _run(db, lambda: review.approve(db, package_id, actor), commit=True)
    logger.info("theme %s@%s approved by user %s", answer["themeId"], answer["version"], actor)
    return answer


@router.post("/{package_id}/reject")
def reject_theme(
    package_id: UUID, body: ReasonBody,
    ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Reject a pending version with a reason."""
    _require_admin(ctx)
    actor = resolve_user_pk(db, ctx)
    return _run(db, lambda: review.reject(db, package_id, body.reason, actor), commit=True)


@router.post("/{package_id}/deactivate", dependencies=ADMIN_STEP_UP)
def deactivate_theme(
    package_id: UUID, body: ReasonBody,
    ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Take an approved version off the Marketplace; its digest gets a ``disable`` revocation."""
    _require_admin(ctx)
    actor = resolve_user_pk(db, ctx)
    answer = _run(db, lambda: review.deactivate(db, package_id, body.reason, actor), commit=True)
    logger.info("theme %s@%s deactivated by user %s", answer["themeId"], answer["version"], actor)
    return answer


@router.post("/{package_id}/signature", dependencies=ADMIN_STEP_UP)
def store_theme_signature(
    package_id: UUID, body: SignatureBody,
    ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Store the Ed25519 signature the kit's sign workflow returned, once an active Automatos key
    verifies it (``signature_verify``, C-05b); otherwise 400 and nothing is stored."""
    _require_admin(ctx)
    actor = resolve_user_pk(db, ctx)
    return _run(db, lambda: review.sign(db, package_id, body.requestId, body.signature, actor), commit=True)
