"""The sessions Automatos issues after a person signs in (PRD-277 Wave 3, §5.3).

One row per session. The token itself (``ams_`` and 32 random bytes) is handed to the browser
once, by the call that issues it (``core.auth.sessions.issue_session``); this table keeps only its
SHA-256 (``token_hash``, the hashing kernel device tokens use), so a copy of the table opens no
session. A revoked row keeps its hash, so the next request with that token is told the session
was revoked rather than that it never existed.

* ``idle_expires_at`` moves forward when the session is used (at most once every five minutes,
  with ``last_seen_at``) and never past ``absolute_expires_at``, which never moves.
* ``auth_time`` is when the person last proved who they are at the identity provider (the ID
  token's ``auth_time``): step-up reads it (US-406). ``idp_session_id`` is the ID token's ``sid``
  when the provider sends one.
* ``revoked_at`` and ``revoked_reason`` are set together or not at all
  (``ck_auth_sessions_revoked_with_reason``).
* ``ip_address`` and ``user_agent`` are the browser's at sign-in, as ``audit_logs`` sizes them.

CASCADE: erasing a person erases their sessions. The Alembic revision
``prd277_w3_sessions_20261120`` declares the same names.
"""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from core.database.base import Base

SHA256_HEX_LENGTH = 64
# OpenID Connect Front-Channel Logout §3: ``sid`` is a string; 255 bounds what is kept.
IDP_SESSION_ID_MAX = 255
REVOKED_REASON_MAX = 32
# audit_logs.ip_address and audit_logs.user_agent (core/workspaces/audit.py).
IP_ADDRESS_MAX = 45
USER_AGENT_MAX = 500

TOKEN_HASH_INDEX = "uq_auth_sessions_token_hash"
USER_INDEX = "ix_auth_sessions_user_id"
REVOKED_CHECK_NAME = "ck_auth_sessions_revoked_with_reason"
REVOKED_CHECK = "(revoked_at IS NULL) = (revoked_reason IS NULL)"


class AuthSession(Base):
    """One session a person holds after signing in: its token's hash, its clocks and its end."""

    __tablename__ = "auth_sessions"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(SHA256_HEX_LENGTH), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    idle_expires_at = Column(DateTime(timezone=True), nullable=False)
    absolute_expires_at = Column(DateTime(timezone=True), nullable=False)
    auth_time = Column(DateTime(timezone=True), nullable=False)
    idp_session_id = Column(String(IDP_SESSION_ID_MAX), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revoked_reason = Column(String(REVOKED_REASON_MAX), nullable=True)
    ip_address = Column(String(IP_ADDRESS_MAX), nullable=True)
    user_agent = Column(String(USER_AGENT_MAX), nullable=True)

    __table_args__ = (
        CheckConstraint(REVOKED_CHECK, name=REVOKED_CHECK_NAME),
        # The resolver's one lookup: the presented token's hash names at most one session.
        Index(TOKEN_HASH_INDEX, "token_hash", unique=True),
        # Revoking every session of a person, and listing them for an admin.
        Index(USER_INDEX, "user_id"),
        {"extend_existing": True},
    )


__all__ = [
    "IDP_SESSION_ID_MAX",
    "IP_ADDRESS_MAX",
    "REVOKED_CHECK",
    "REVOKED_CHECK_NAME",
    "REVOKED_REASON_MAX",
    "SHA256_HEX_LENGTH",
    "TOKEN_HASH_INDEX",
    "USER_AGENT_MAX",
    "USER_INDEX",
    "AuthSession",
]
