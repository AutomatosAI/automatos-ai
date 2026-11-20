"""PRD-277 Wave 3 (§5.3, US-301, US-303): the sessions Automatos issues, finds and ends.

A session is what a person holds after signing in, so they do not sign in on every request.

* **The token** is ``ams_`` and 32 random bytes, minted by the pairing kernel
  (``core.auth.pairing.mint_token``) as device tokens are. It is returned once, by
  ``issue_session``; ``auth_sessions`` keeps only ``hash_secret`` of it, found by that digest and
  confirmed in constant time (``secrets_match``).
* **Two clocks** (D-15): idle (``SESSION_IDLE_MINUTES``, default 120) and absolute
  (``SESSION_ABSOLUTE_HOURS``, default 12). The idle clock moves forward when the session is used,
  written at most once every ``LAST_SEEN_WRITE_INTERVAL`` (five minutes, the device lane's own), so
  a session left alone ends between five minutes before and at its idle limit. It never moves
  past the absolute expiry, which never moves.
* **A new session at every sign-in.** ``issue_session`` always inserts a row; nothing upgrades an
  earlier one. Wave 4's OIDC callback calls it with the person's ``users.id``, the ID token's
  ``auth_time`` and ``sid``, and the browser's address and agent. A deactivated person is refused
  (``SignInRefused``), and the refusal is audited.
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
from typing import Optional, Union
from uuid import UUID, uuid4

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config import config
from core import editions
from core.auth import session_audit
from core.auth.device import LAST_SEEN_WRITE_INTERVAL
from core.auth.pairing import hash_secret, mint_token, secrets_match
from core.auth.system_role import UnknownPerson
from core.models.auth_sessions import IDP_SESSION_ID_MAX, IP_ADDRESS_MAX, USER_AGENT_MAX, AuthSession
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

NOT_ISSUED_MESSAGE = "This edition does not issue sessions."
UNKNOWN_PERSON_MESSAGE = "No such person."


class SessionsNotIssued(RuntimeError):
    """The running edition does not issue sessions (``issues_sessions`` is off)."""


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


def _clipped(value: Optional[str], limit: int) -> Optional[str]:
    """A browser-supplied value cut to its column's size; blank is none."""
    text = (value or "").strip()
    return text[:limit] or None


def _checked_sign_in(auth_time: datetime, sid: Optional[str]) -> None:
    """Fail fast on what the caller passes: an aware ``auth_time``, and a ``sid`` that fits."""
    if not isinstance(auth_time, datetime) or auth_time.tzinfo is None:
        raise TypeError("auth_time must be a timezone-aware datetime")
    if sid is not None and (not isinstance(sid, str) or not sid or len(sid) > IDP_SESSION_ID_MAX):
        raise ValueError(f"sid must be a non-empty string of at most {IDP_SESSION_ID_MAX} characters")


def _person_is_active(db: Session, user_id: int) -> bool:
    """Whether ``user_id`` may hold a session; ``UnknownPerson`` when there is no such row."""
    row = db.query(User.is_active).filter(User.id == user_id).first()
    if row is None:
        raise UnknownPerson(UNKNOWN_PERSON_MESSAGE)
    return row[0] is not False


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
    ``session_audit.SessionAuditFailed`` is raised. Call it with nothing else pending.
    """
    if not editions.current().issues_sessions:
        raise SessionsNotIssued(NOT_ISSUED_MESSAGE)
    _checked_sign_in(auth_time, sid)
    ip, agent = _clipped(ip_address, IP_ADDRESS_MAX), _clipped(user_agent, USER_AGENT_MAX)
    if not _person_is_active(db, user_id):
        session_audit.record_sign_in_refused(db, reason=session_audit.REFUSED_DEACTIVATED, user_id=user_id,
                                             ip_address=ip, user_agent=agent, workspace_id=workspace_id)
        raise SignInRefused(session_audit.REFUSED_DEACTIVATED)
    moment = now or _now()
    token, digest = mint_token(SESSION_TOKEN_PREFIX, SESSION_TOKEN_BYTES)
    absolute = moment + absolute_lifetime()
    row = AuthSession(
        id=uuid4(), user_id=user_id, token_hash=digest, created_at=moment, last_seen_at=moment,
        idle_expires_at=min(moment + idle_lifetime(), absolute), absolute_expires_at=absolute,
        auth_time=auth_time, idp_session_id=sid, ip_address=ip, user_agent=agent,
    )
    db.add(row)
    db.flush()
    session_audit.record_sign_in(db, user_id=user_id, session_id=row.id, ip_address=ip, user_agent=agent,
                                 workspace_id=workspace_id)
    return IssuedSession(session_id=row.id, user_id=user_id, token=token,
                         idle_expires_at=row.idle_expires_at, absolute_expires_at=absolute)


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
    "SESSION_TOKEN_BYTES",
    "SESSION_TOKEN_PREFIX",
    "UNKNOWN",
    "IssuedSession",
    "SessionsNotIssued",
    "SignInRefused",
    "absolute_lifetime",
    "find_session",
    "idle_lifetime",
    "issue_session",
    "revoke_session",
    "revoke_sessions_of",
    "session_state",
    "touch_session",
]
