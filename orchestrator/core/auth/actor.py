"""The authenticated principal as an integer ``users.id``.

``RequestContext.user.id`` is a Clerk string id or an email address — never the
integer primary key. Every FK that records "who did this" (``audit_logs.user_id``,
``workspace_invitations.invited_by``, ``workspace_members.user_id``) is an
Integer FK to ``users.id``, so the row must be looked up before it can be
written. Treating ``ctx.user.id`` as ``users.id`` is a known 500 (#513).

Resolution order — Clerk id first (the stable identity), email second (covers a
principal whose Clerk id has not been backfilled onto its ``users`` row).

FAIL-CLOSED: an unresolvable principal returns ``None`` rather than a guess.
Callers that gate on identity must treat ``None`` as "not authorised"; callers
that only record it may store ``None`` (``AuditService.log`` accepts it and
carries ``actor_type`` instead).

Extracted 2026-09-11 from the byte-identical copies in ``api.team`` and
``api.harness`` — both now import from here.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.models.core import User


def resolve_recorded_person(db: Session, recorded: Optional[str]) -> Optional[int]:
    """The integer ``users.id`` for a person a row recorded as a string — a
    mission's ``created_by``, a watch's creator. The same order as below: the Clerk
    id first, then the email (a local operator has no Clerk id; the local REST API
    and chat record the email, F166). Never a bare number: a digit string there is
    an agent id wherever no person drove the action, so it resolves to nobody."""
    value = str(recorded or "").strip()
    if not value:
        return None
    row = db.query(User.id).filter(User.clerk_user_id == value).first()
    if row is None and "@" in value:
        row = db.query(User.id).filter(User.email == value).first()
    return int(row[0]) if row else None


def resolve_internal_user_id(db: Session, ctx: RequestContext) -> Optional[int]:
    """The integer ``users.id`` for ``ctx``'s principal, or None if unresolvable."""
    if not ctx.user:
        return None
    user = None
    if ctx.user.clerk_user_id:
        user = db.query(User).filter(User.clerk_user_id == ctx.user.clerk_user_id).first()
    if not user and ctx.user.email:
        user = db.query(User).filter(User.email == ctx.user.email).first()
    return user.id if user else None
