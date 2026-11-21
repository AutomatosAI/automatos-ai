"""PRD-277 Wave 3 (§5.3, US-301, US-303): the sessions Automatos issues, finds and ends.

A session is what a person holds after signing in, so they do not sign in on every request.

* **The token** is ``ams_`` and 32 random bytes, minted by the pairing kernel
  (``core.auth.pairing.mint_token``) as device tokens are. It is returned once, by
  ``issue_session`` or ``renew_session``; ``auth_sessions`` keeps only ``hash_secret`` of it, found
  by that digest and confirmed in constant time (``secrets_match``).
* **Two clocks** (D-15): idle (``SESSION_IDLE_MINUTES``, default 120) and absolute
  (``SESSION_ABSOLUTE_HOURS``, default 12). The idle clock moves forward when the session is used,
  written at most once every ``LAST_SEEN_WRITE_INTERVAL`` (five minutes, the device lane's own). So
  a session ends between its idle limit less five minutes and its idle limit after its last
  request, never later; start-up refuses an idle limit shorter than that interval, which would
  end the session of a person still at work (``core.edition_start_up``). It never moves past the
  absolute expiry, which never moves.
* **A new session at every sign-in.** ``issue_session`` always inserts a row; nothing upgrades an
  earlier one. Wave 4's OIDC callback calls it with the person's ``users.id``, the ID token's
  ``auth_time`` and ``sid``, and the browser's address and agent. A deactivated person is refused
  (``SignInRefused``), and the refusal is audited. The person's row is read ``FOR SHARE``, in the
  transaction that inserts the session, so a deactivation (which takes the row ``FOR UPDATE``
  first, ``core.auth.people``) and a sign-in of the same person happen one after the other: the
  sign-in sees the person deactivated, or the deactivation sees and revokes the new session.
* **A confirmed step-up is a new session too** (US-406, ``renew_session``): the session that asked
  ends (``REVOKED_STEP_UP``) and its successor keeps that session's person and absolute expiry, so
  a step-up never lengthens a session. The browser gets the successor's token in its cookie.
* **Ending one** keeps its hash and records when and why (``REVOKED_*``), so the next request with
  it is told the session was revoked.

Only where the edition issues sessions (``core.editions`` ``issues_sessions``: enterprise);
``issue_session`` refuses everywhere else. Nothing here logs a token or a hash. ``issue_session``
commits (with its audit row, ``core.auth.session_audit``); every other writer flushes or leaves
the transaction to its caller, except ``touch_session``, which is bookkeeping on the session lane's
own database session.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Union
from uuid import UUID, uuid4

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config import config
from core import editions
from core.auth import session_audit
from core.auth.device import LAST_SEEN_WRITE_INTERVAL
from core.auth.pairing import hash_secret, mint_token, secrets_match
from core.auth.system_role import UnknownPerson
from core.models.auth_sessions import IDP_SESSION_ID_MAX, AuthSession
from core.models.core import User

logger = logging.getLogger(__name__)

SESSION_TOKEN_PREFIX = "ams_"
SESSION_TOKEN_BYTES = 32

# What a session is, as the lane and the routes report it.
LIVE = "live"
EXPIRED = "expired"
REVOKED = "revoked"
UNKNOWN = "unknown"

# Why a session ended (auth_sessions.revoked_reason).
REVOKED_SIGN_OUT = "sign_out"
REVOKED_BY_ADMIN = "admin"
REVOKED_DEACTIVATED = "deactivated"
REVOKED_STEP_UP = "step_up"  # a confirmed step-up issued its successor (``renew_session``)

NOT_ISSUED_MESSAGE = "This edition does not issue sessions."
SIGN_IN_FAILED_MESSAGE = "You were not signed in: the session could not be stored. Try again."
UNKNOWN_PERSON_MESSAGE = "No such person."


class SessionsNotIssued(RuntimeError):
    """The running edition does not issue sessions (``issues_sessions`` is off)."""


class SignInFailed(RuntimeError):
    """The new session could not be stored; nothing was kept. Its text never holds a value."""


class SignInRefused(RuntimeError):
    """A sign-in that may not have a session; ``reason`` is one of ``session_audit.REFUSAL_REASONS``."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class IssuedSession:
    """A new session. ``token`` is the only copy there will ever be: it never appears in a repr."""

    session_id: UUID
    user_id: int
    token: str = field(repr=False)
    idle_expires_at: datetime
    absolute_expires_at: datetime


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """``value`` as a UTC-aware datetime (a naive value is taken to be UTC)."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def idle_lifetime() -> timedelta:
    """How long a session may go unused (``SESSION_IDLE_MINUTES``)."""
    return timedelta(minutes=config.SESSION_IDLE_MINUTES)


def absolute_lifetime() -> timedelta:
    """How long a session may last at all (``SESSION_ABSOLUTE_HOURS``)."""
    return timedelta(hours=config.SESSION_ABSOLUTE_HOURS)


def _checked_sign_in(auth_time: datetime, sid: Optional[str]) -> None:
    """Fail fast on what the caller passes: an aware ``auth_time``, and a ``sid`` that fits."""
    if not isinstance(auth_time, datetime) or auth_time.tzinfo is None:
        raise TypeError("auth_time must be a timezone-aware datetime")
    if sid is not None and (not isinstance(sid, str) or not sid or len(sid) > IDP_SESSION_ID_MAX):
        raise ValueError(f"sid must be a non-empty string of at most {IDP_SESSION_ID_MAX} characters")


def person_is_active_locked(db: Session, user_id: int) -> bool:
    """Whether ``user_id`` may hold a session, their row locked ``FOR SHARE``; ``UnknownPerson`` when none.

    The lock is held until the sign-in (or step-up) commits, so a deactivation waits for it, and a
    sign-in waits for a deactivation that locked the row first and then reads the person as
    deactivated. Take it before any ``auth_sessions`` row, as a deactivation does.
    """
    row = db.query(User.is_active).filter(User.id == user_id).with_for_update(read=True).first()
    if row is None:
        raise UnknownPerson(UNKNOWN_PERSON_MESSAGE)
    return row[0] is not False


def _store(db: Session, row: AuthSession) -> None:
    """Insert the new session; on a database error, roll back and raise ``SignInFailed``.

    Raised outside the handler, so the driver's error, whose text names the statement's values (the
    token's hash, the address, the agent), is neither the cause nor the context of what is raised.
    """
    user_id, failure = row.user_id, None
    try:
        db.add(row)
        db.flush()
    except SQLAlchemyError as error:
        failure = type(error).__name__
    if failure is None:
        return
    db.rollback()
    logger.error("users.id=%s: the new session could not be stored (%s)", user_id, failure)
    raise SignInFailed(SIGN_IN_FAILED_MESSAGE)


def issue_session(
    db: Session,
    *,
    user_id: int,
    auth_time: datetime,
    sid: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    workspace_id: Union[UUID, str, None] = None,
    now: Optional[datetime] = None,
) -> IssuedSession:
    """A new session for ``user_id``, written with its ``auth:sign_in`` audit row in one commit.

    ``auth_time`` is when the person proved who they are (the ID token's), ``sid`` the provider's
    session id when it sent one. The audit row goes in ``workspace_id``, or the install's default
    workspace (``session_audit.audit_workspace``). A deactivated person is ``SignInRefused`` with
    an ``auth:sign_in_refused`` row; when the sign-in row cannot be written, nothing is kept and
    ``session_audit.SessionAuditFailed`` is raised; when the session itself cannot be stored,
    nothing is kept and ``SignInFailed`` is raised. Call it with nothing else pending.
    """
    if not editions.current().issues_sessions:
        raise SessionsNotIssued(NOT_ISSUED_MESSAGE)
    _checked_sign_in(auth_time, sid)
    ip, agent = session_audit.browser_fields(ip_address, user_agent)
    if not person_is_active_locked(db, user_id):
        session_audit.record_sign_in_refused(db, reason=session_audit.REFUSED_DEACTIVATED, user_id=user_id,
                                             ip_address=ip, user_agent=agent, workspace_id=workspace_id)
        raise SignInRefused(session_audit.REFUSED_DEACTIVATED)
    moment = now or _now()
    token, row = _new_session(user_id, auth_time, sid, (ip, agent), moment, moment + absolute_lifetime())
    _store(db, row)
    session_audit.record_sign_in(db, user_id=user_id, session_id=row.id, ip_address=ip, user_agent=agent,
                                 workspace_id=workspace_id)
    return _issued(token, row)


def renew_session(db: Session, asking: AuthSession, *, auth_time: datetime, sid: Optional[str],
                  ip_address: Optional[str], user_agent: Optional[str], now: datetime) -> IssuedSession:
    """US-406: end ``asking`` (``REVOKED_STEP_UP``) and store its successor, a new session; never commits.

    The successor keeps ``asking``'s person and absolute expiry (a step-up never lengthens a
    session), and the provider's session id: ``sid`` when the provider sent one, else ``asking``'s.
    It has a new token, the new ``auth_time``, this browser's address and agent, and an idle clock
    started at ``now`` and never past that expiry. The caller commits both with the step-up's audit
    row; when the new row cannot be stored, everything is rolled back and ``SignInFailed`` raised.
    """
    _checked_sign_in(auth_time, sid)
    browser = session_audit.browser_fields(ip_address, user_agent)
    absolute = _as_utc(asking.absolute_expires_at)
    token, row = _new_session(asking.user_id, auth_time, sid or asking.idp_session_id, browser, now, absolute)
    revoke_session(db, asking.id, REVOKED_STEP_UP, now)
    _store(db, row)
    return _issued(token, row)


def _new_session(user_id: int, auth_time: datetime, sid: Optional[str],
                 browser: Tuple[Optional[str], Optional[str]], moment: datetime,
                 absolute: datetime) -> Tuple[str, AuthSession]:
    """A new token and the unsaved row that keeps only its hash, both clocks started at ``moment``."""
    token, digest = mint_token(SESSION_TOKEN_PREFIX, SESSION_TOKEN_BYTES)
    ip, agent = browser
    row = AuthSession(
        id=uuid4(), user_id=user_id, token_hash=digest, created_at=moment, last_seen_at=moment,
        idle_expires_at=min(moment + idle_lifetime(), absolute), absolute_expires_at=absolute,
        auth_time=auth_time, idp_session_id=sid, ip_address=ip, user_agent=agent,
    )
    return token, row


def _issued(token: str, row: AuthSession) -> IssuedSession:
    return IssuedSession(session_id=row.id, user_id=row.user_id, token=token,
                         idle_expires_at=row.idle_expires_at, absolute_expires_at=row.absolute_expires_at)


def find_session(db: Session, token: Optional[str]) -> Optional[AuthSession]:
    """The session this token opens, in any state, or ``None``: found by digest, confirmed in constant time."""
    if not isinstance(token, str) or not token.startswith(SESSION_TOKEN_PREFIX):
        return None
    row = db.query(AuthSession).filter(AuthSession.token_hash == hash_secret(token)).first()
    return row if row is not None and secrets_match(row.token_hash, token) else None


def session_state(row: Optional[AuthSession], now: datetime) -> str:
    """``LIVE``, ``REVOKED``, ``EXPIRED`` (either clock, at or after it) or ``UNKNOWN`` (no row)."""
    if row is None:
        return UNKNOWN
    if row.revoked_at is not None:
        return REVOKED
    if now >= _as_utc(row.absolute_expires_at) or now >= _as_utc(row.idle_expires_at):
        return EXPIRED
    return LIVE


def touch_session(db: Session, row: AuthSession, now: datetime) -> None:
    """Stamp ``last_seen_at`` and move the idle clock, when the last stamp is five minutes old or more.

    Bookkeeping, never fatal: a failed write rolls back and logs by type (never the statement's
    text), and the request goes on. Never touches a session revoked in the meantime.
    """
    seen = row.last_seen_at
    if seen is not None and now - _as_utc(seen) < LAST_SEEN_WRITE_INTERVAL:
        return
    idle_until = min(now + idle_lifetime(), _as_utc(row.absolute_expires_at))
    try:
        db.query(AuthSession).filter(AuthSession.id == row.id, AuthSession.revoked_at.is_(None)).update(
            {AuthSession.last_seen_at: now, AuthSession.idle_expires_at: idle_until}, synchronize_session=False,
        )
        db.commit()
    except SQLAlchemyError as failure:
        db.rollback()
        logger.warning("Session lane: last_seen_at write failed for %s: %s", row.id, type(failure).__name__)


def revoke_session(db: Session, session_id: UUID, reason: str, now: Optional[datetime] = None) -> bool:
    """End one session that has not ended yet; whether one was ended. Never commits."""
    ended = db.query(AuthSession).filter(AuthSession.id == session_id, AuthSession.revoked_at.is_(None)).update(
        {AuthSession.revoked_at: now or _now(), AuthSession.revoked_reason: reason}, synchronize_session=False,
    )
    return bool(ended)


def revoke_sessions_of(db: Session, user_id: int, reason: str, now: Optional[datetime] = None) -> int:
    """End every session of ``user_id`` that has not ended yet; how many. Never commits."""
    return db.query(AuthSession).filter(AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None)).update(
        {AuthSession.revoked_at: now or _now(), AuthSession.revoked_reason: reason}, synchronize_session=False,
    )


__all__ = [
    "EXPIRED",
    "LIVE",
    "REVOKED",
    "REVOKED_BY_ADMIN",
    "REVOKED_DEACTIVATED",
    "REVOKED_SIGN_OUT",
    "REVOKED_STEP_UP",
    "SESSION_TOKEN_BYTES",
    "SESSION_TOKEN_PREFIX",
    "UNKNOWN",
    "IssuedSession",
    "SessionsNotIssued",
    "SignInFailed",
    "SignInRefused",
    "absolute_lifetime",
    "find_session",
    "idle_lifetime",
    "issue_session",
    "person_is_active_locked",
    "renew_session",
    "revoke_session",
    "revoke_sessions_of",
    "session_state",
    "touch_session",
]
