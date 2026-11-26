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
   so a step-up can never create one or link an identity. It locks that session's person
   (``FOR SHARE``, as a sign-in does, so a deactivation and a step-up of the same person happen one
   after the other), then the session that asked, and refuses, in this order, when:

   * that session is no longer live (``session_ended``);
   * the callback's request does not present that session in its cookie: it presents another
     session, or none (``session_not_presented``). The browser that asked is the one that holds
     that session, so a callback without it is refused even while that session is live;
   * the person who signed in again is not that session's person: their ``(issuer, subject)`` is
     linked to someone else, or to nobody (``another_person``). Never an identity switch: the
     session keeps its person and its old ``auth_time``, and no session is issued;
   * that person has been deactivated (``deactivated``);
   * the ID token carries no ``auth_time`` (``no_auth_time``): fails closed, and the person is told
     so with a code of its own (``sign_in_flow.STEP_UP_NO_AUTH_TIME``): a provider that never sends
     it can never confirm a step-up;
   * its ``auth_time`` is later than now by more than ``CLOCK_SKEW`` (``future_auth_time``);
   * its ``auth_time`` is earlier than the step-up's start by more than ``CLOCK_SKEW``
     (``stale_auth_time``): the provider did not make the person sign in again. ``CLOCK_SKEW`` (the
     ID token's ``TIME_LEEWAY_SECONDS``) allows only for the two clocks disagreeing.

   Otherwise the browser gets a new session (``sessions.renew_session``): the session that asked
   ends (``sessions.REVOKED_STEP_UP``), so its token is refused from the next request on, and its
   successor carries the token's ``auth_time``. The successor keeps the person, the provider's
   session id (the token's ``sid``, else the old session's) and the old session's absolute expiry,
   so a step-up never lengthens a session. Both are committed with one ``auth:step_up`` row
   (``confirmed: true``, ``new_session_id``) on the session that asked; when the row cannot be
   written, nothing lands. Every refusal rolls back and writes ``auth:step_up`` with
   ``confirmed: false`` and its reason, naming the session's person, never the other one, and
   changes no session.

The callback sets the new session's cookie, and the browser returns to where it was, signed in
afresh. Nothing here logs or records a token, a code or a claim.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.auth import people, session_audit, sessions
from core.auth.oidc_sign_in import NO_AUTH_TIME
from core.auth.oidc_tokens import TIME_LEEWAY_SECONDS
from core.auth.sign_in_methods import Finished
from core.auth.user_link import user_id_for_identity
from core.models.auth_sessions import AuthSession

logger = logging.getLogger(__name__)

STEP_UP_START = "/api/v1/auth/oidc/start?step_up=1"
# How far the provider's clock and ours may disagree: the ID token's own leeway (§5.11).
CLOCK_SKEW = timedelta(seconds=TIME_LEEWAY_SECONDS)


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
    """The session that asked, locked ``FOR UPDATE`` after its person (``FOR SHARE``): a
    deactivation locks the person first too, so neither waits on the other in the opposite order."""
    owner = db.query(AuthSession.user_id).filter(AuthSession.id == session_id).first()
    if owner is not None:
        sessions.person_is_active_locked(db, owner[0])
    return db.query(AuthSession).filter(AuthSession.id == session_id).with_for_update().first()


def _refusal(db: Session, finished: Finished, row: Optional[AuthSession], presented: Optional[str],
             now: datetime) -> Optional[str]:
    """Why this callback does not confirm the session ``row``, or None when it does."""
    identity = finished.identity
    if sessions.session_state(row, now) != sessions.LIVE:
        return session_audit.STEP_UP_SESSION_ENDED
    if not sessions.token_opens(row, presented):
        return session_audit.STEP_UP_NOT_PRESENTED
    if user_id_for_identity(db, identity.issuer, identity.subject) != row.user_id:
        return session_audit.STEP_UP_ANOTHER_PERSON
    if not people.person_is_active(db, row.user_id):
        return session_audit.STEP_UP_DEACTIVATED
    return sign_in_time_refusal(identity.auth_time, finished.started_at, now)


def sign_in_time_refusal(auth_time: datetime, started_at: Optional[datetime],
                         now: Optional[datetime] = None) -> Optional[str]:
    """Why ``auth_time`` does not prove a sign-in made since ``started_at``, or None when it does.

    The one sign-in time rule: a step-up's, and a phone's sign-in's (PRD-277 US-603,
    ``core.auth.sign_in_flow``). ``no_auth_time`` when the provider reported none, ``future_auth_time``
    or ``stale_auth_time`` as ``_freshness_refusal`` says.
    """
    if auth_time <= NO_AUTH_TIME:
        return session_audit.STEP_UP_NO_AUTH_TIME
    return _freshness_refusal(auth_time, started_at, now or _now())


def _freshness_refusal(auth_time: datetime, started_at: Optional[datetime], now: datetime) -> Optional[str]:
    """Fresh: ``auth_time`` is not earlier than the step-up's start, nor later than now. ``CLOCK_SKEW``
    is allowed either way, for the provider's clock and ours, and for nothing else."""
    if auth_time > now + CLOCK_SKEW:
        return session_audit.STEP_UP_FUTURE
    if started_at is None or auth_time < started_at - CLOCK_SKEW:
        return session_audit.STEP_UP_STALE
    return None


def confirm_step_up(db: Session, finished: Finished, *, presented: Optional[str], ip_address: Optional[str],
                    user_agent: Optional[str], now: Optional[datetime] = None) -> sessions.IssuedSession:
    """The asking session ended and a new one issued with this sign-in's ``auth_time``, audited, in
    one commit; else ``StepUpRefused``, and no session changed.

    ``presented`` is the session token the callback's request carries in its cookie: it must be the
    asking session's own. ``now`` is read once both locks are held, so the session's clocks and the
    sign-in's time are judged when the session is locked, never before a wait for the locks (a
    session that expired while this waited is refused as ended). ``SessionAuditFailed`` when the
    confirmation's row cannot be written, and ``sessions.SignInFailed`` when the new session cannot
    be stored: nothing changed either way.
    """
    session_id = finished.step_up_session_id
    row = _locked_session(db, session_id)
    moment = now or _now()
    asked_by = row.user_id if row is not None else None
    reason = _refusal(db, finished, row, presented, moment)
    if reason is not None:
        db.rollback()
        logger.warning("users.id=%s: step-up refused (%s)", asked_by, reason)
        session_audit.record_step_up(db, user_id=asked_by, session_id=session_id, reason=reason,
                                     ip_address=ip_address, user_agent=user_agent)
        raise StepUpRefused(reason)
    identity = finished.identity
    issued = sessions.renew_session(db, row, auth_time=identity.auth_time, sid=identity.idp_session_id,
                                    ip_address=ip_address, user_agent=user_agent, now=moment)
    session_audit.record_step_up(db, user_id=asked_by, session_id=session_id, reason=None,
                                 ip_address=ip_address, user_agent=user_agent, new_session_id=issued.session_id)
    logger.info("users.id=%s: step-up confirmed with a new session", asked_by)
    return issued


__all__ = ["CLOCK_SKEW", "STEP_UP_START", "StepUpRefused", "confirm_step_up", "session_asking",
           "sign_in_time_refusal"]
