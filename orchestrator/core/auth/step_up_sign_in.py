"""PRD-277 US-406 (§5.5): step-up, by signing in again at the organisation's identity provider.

A sensitive route refuses a session whose person last proved who they are more than
``STEP_UP_MAX_AGE_MINUTES`` ago (``core.auth.step_up.confirm_by_signing_in_again``), with a body
naming ``STEP_UP_START``. The web app sends the browser there:

1. **Start** (``GET /api/v1/auth/oidc/start?step_up=1``): the browser must present a LIVE session
   (its cookie) whose person is active (``session_asking``); else ``StepUpRefused`` and nothing is
   stored. The sign-in starts as any other, its row keeping that session's id, and the provider is
   sent ``max_age=0``: it must make the person sign in again and say when, in ``auth_time``.
2. **Callback**: the state, the code and the ID token are checked exactly as for a sign-in, and the
   access rule is applied. Then ``confirm_step_up``, BEFORE anything that finds or creates a person,
   so a step-up can never create one or link an identity. It locks the session that asked and
   refuses, in this order, when:

   * that session is no longer live (``session_ended``);
   * the person who signed in again is not that session's person: their ``(issuer, subject)`` is
     linked to someone else, or to nobody (``another_person``). Never an identity switch: the
     session keeps its person and its old ``auth_time``, and no session is issued;
   * that person has been deactivated (``deactivated``);
   * the ID token carries no ``auth_time`` (``no_auth_time``): fails closed;
   * its ``auth_time`` is older than the step-up's start, less ``TIME_LEEWAY_SECONDS``
     (``stale_auth_time``): the provider did not make the person sign in again.

   Otherwise the session's ``auth_time`` becomes the token's, committed with an ``auth:step_up``
   row (``confirmed: true``); when the row cannot be written, neither lands. Every refusal rolls
   back and writes ``auth:step_up`` with ``confirmed: false`` and its reason, naming the session's
   person, never the other one.

No session is issued and no cookie changes: the browser returns to where it was, on the same
session, now fresh. Nothing here logs or records a token, a code or a claim.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.auth import people, session_audit, sessions
from core.auth.oidc_sign_in import NO_AUTH_TIME
from core.auth.oidc_tokens import TIME_LEEWAY_SECONDS
from core.auth.sign_in_methods import Finished, VerifiedIdentity
from core.auth.user_link import user_id_for_identity
from core.models.auth_sessions import AuthSession

logger = logging.getLogger(__name__)

STEP_UP_START = "/api/v1/auth/oidc/start?step_up=1"

_REFRESH = text("UPDATE auth_sessions SET auth_time = :auth_time "
                "WHERE id = :id AND user_id = :user_id AND revoked_at IS NULL")


class StepUpRefused(Exception):
    """A step-up that cannot start or did not confirm the session's person. Its text is a reason."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def session_asking(db: Session, token: Optional[str], now: Optional[datetime] = None) -> UUID:
    """The id of the live session ``token`` names, whose person is active; else ``StepUpRefused``."""
    row = sessions.find_session(db, token)
    if sessions.session_state(row, now or _now()) != sessions.LIVE:
        raise StepUpRefused(session_audit.STEP_UP_SESSION_ENDED)
    if not people.person_is_active(db, row.user_id):
        raise StepUpRefused(session_audit.STEP_UP_DEACTIVATED)
    return row.id


def _locked_session(db: Session, session_id: UUID) -> Optional[AuthSession]:
    return db.query(AuthSession).filter(AuthSession.id == session_id).with_for_update().first()


def _refusal(db: Session, identity: VerifiedIdentity, row: Optional[AuthSession],
             started_at: Optional[datetime], now: datetime) -> Optional[str]:
    """Why this callback does not confirm the session ``row``, or None when it does."""
    if sessions.session_state(row, now) != sessions.LIVE:
        return session_audit.STEP_UP_SESSION_ENDED
    if user_id_for_identity(db, identity.issuer, identity.subject) != row.user_id:
        return session_audit.STEP_UP_ANOTHER_PERSON
    if not people.person_is_active(db, row.user_id):
        return session_audit.STEP_UP_DEACTIVATED
    if identity.auth_time <= NO_AUTH_TIME:
        return session_audit.STEP_UP_NO_AUTH_TIME
    if started_at is None or identity.auth_time < started_at - timedelta(seconds=TIME_LEEWAY_SECONDS):
        return session_audit.STEP_UP_STALE
    return None


def confirm_step_up(db: Session, finished: Finished, *, ip_address: Optional[str], user_agent: Optional[str],
                    now: Optional[datetime] = None) -> None:
    """Refresh the asking session's ``auth_time`` from this sign-in, audited; else ``StepUpRefused``.

    ``SessionAuditFailed`` when the confirmation's row cannot be written (nothing changed).
    """
    session_id = finished.step_up_session_id
    row = _locked_session(db, session_id)
    asked_by = row.user_id if row is not None else None
    reason = _refusal(db, finished.identity, row, finished.started_at, now or _now())
    if reason is not None:
        db.rollback()
        logger.warning("users.id=%s: step-up refused (%s)", asked_by, reason)
        session_audit.record_step_up(db, user_id=asked_by, session_id=session_id, reason=reason,
                                     ip_address=ip_address, user_agent=user_agent)
        raise StepUpRefused(reason)
    db.execute(_REFRESH, {"auth_time": finished.identity.auth_time, "id": session_id, "user_id": asked_by})
    session_audit.record_step_up(db, user_id=asked_by, session_id=session_id, reason=None,
                                 ip_address=ip_address, user_agent=user_agent)
    logger.info("users.id=%s: step-up confirmed", asked_by)


__all__ = ["STEP_UP_START", "StepUpRefused", "confirm_step_up", "session_asking"]
