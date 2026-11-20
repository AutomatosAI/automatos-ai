"""A sign-in that has started and not finished yet (PRD-277 US-402, §5.2).

``GET /api/v1/auth/oidc/start`` writes one row; the callback ends it. A row lives at most
``core.auth.oidc_pending.SIGN_IN_TTL_SECONDS`` (ten minutes) and is used once:

* ``state_hash`` and ``nonce_hash`` are SHA-256 digests (the pairing kernel's ``hash_secret``) of the
  ``state`` and ``nonce`` sent to the identity provider; neither value is kept. ``state_hash`` is
  unique: the callback finds its row by it, and deleting the row is what ends the sign-in, so two
  callbacks with one state cannot both use it.
* ``binding_hash`` is the SHA-256 of the browser binding: a random secret of its own, set in the
  browser's ``__Host-`` sign-in cookie at ``/start`` and never sent anywhere else. The callback must
  carry it, so a callback carried to another browser is not a sign-in there.
* ``code_verifier`` is the PKCE verifier (RFC 7636: 43 to 128 characters), kept until the code is
  exchanged, because the exchange must present it. It never leaves the server.
* ``return_to`` is where the person goes after signing in: a path on the install's own address,
  checked before it is stored (``core.auth.sign_in_methods.safe_return_path``).

No person or workspace is named: nobody is known until the callback. The Alembic revision
``prd277_w4_oidc_sign_ins_20261120`` declares the same names.
"""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import Column, DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from core.database.base import Base

SHA256_HEX_LENGTH = 64
# RFC 7636 §4.1: a verifier is 43 to 128 characters.
CODE_VERIFIER_MAX = 128
RETURN_TO_MAX = 1024

STATE_HASH_INDEX = "uq_oidc_pending_sign_ins_state_hash"
EXPIRES_INDEX = "ix_oidc_pending_sign_ins_expires_at"


class OidcPendingSignIn(Base):
    """One started sign-in: the digests of its state and nonce, its PKCE verifier and where it returns."""

    __tablename__ = "oidc_pending_sign_ins"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    state_hash = Column(String(SHA256_HEX_LENGTH), nullable=False)
    nonce_hash = Column(String(SHA256_HEX_LENGTH), nullable=False)
    binding_hash = Column(String(SHA256_HEX_LENGTH), nullable=False)
    code_verifier = Column(String(CODE_VERIFIER_MAX), nullable=False)
    return_to = Column(String(RETURN_TO_MAX), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        # The callback's one lookup: a state names at most one started sign-in.
        Index(STATE_HASH_INDEX, "state_hash", unique=True),
        # Clearing sign-ins that were started and never finished.
        Index(EXPIRES_INDEX, "expires_at"),
        {"extend_existing": True},
    )


__all__ = [
    "CODE_VERIFIER_MAX",
    "EXPIRES_INDEX",
    "RETURN_TO_MAX",
    "SHA256_HEX_LENGTH",
    "STATE_HASH_INDEX",
    "OidcPendingSignIn",
]
