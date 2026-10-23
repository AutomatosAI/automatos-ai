"""PRD-266 §4.2 — mobile devices: the phones (and paired browsers) of Automatos Go.

One row per device a person uses with one workspace. Two kinds:

* ``paired`` — the local edition. Studio issues a one-time code (only its SHA-256 is
  stored), the phone or browser pairs with it and receives a device token (only its
  SHA-256 is stored, returned exactly once). A rotation keeps the previous hash valid
  until ``prev_token_valid_until``; a revoke keeps ``token_hash`` so the next call can
  be told ``device_revoked``.
* ``clerk`` — SaaS. The phone signs in with Clerk and registers for push only, so it
  never holds a code or token hash (``ck_mobile_devices_clerk_holds_no_token``).

A pending row is a code waiting for a phone, so ``install_id`` (the id the app keeps
in SecureStore) is null until it pairs; it is unique per workspace and person.
``push_key`` is the device's HMAC key for signing push payloads (W2): the server needs
it, so it is stored as is and returned only on the first registration.

Why a table, and not ``cli_hosts`` (CLAUDE.md "no new tables" rule): ``cli_hosts``
registers CLI runtimes (capabilities, heartbeats, terminals, sessions, local only); a
phone carries a person, push credentials and digest preferences, and mixing the two
lifecycles would make every query branch on kind. The Alembic revision
``prd266_mobile_devices`` declares the same names.
"""
from __future__ import annotations

from enum import Enum
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    false,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base


class MobileDeviceKind(str, Enum):
    """``PAIRED`` — local edition, holds a device token.
    ``CLERK`` — SaaS, signed in with Clerk, push registration only."""

    PAIRED = "paired"
    CLERK = "clerk"


class MobileDeviceStatus(str, Enum):
    """``PENDING`` — a pairing code was issued, no device has claimed it yet.
    ``ACTIVE`` — the device is paired (or registered) and may call.
    ``REVOKED`` — the device is cut off; it must pair again."""

    PENDING = "pending"
    ACTIVE = "active"
    REVOKED = "revoked"


# A paired browser is a row with platform 'web' (PRD-266 K6).
MOBILE_DEVICE_PLATFORMS = ("ios", "android", "web")

ENUM_LENGTH = 16
INSTALL_ID_LENGTH = 128
NAME_LENGTH = 120
APP_VERSION_LENGTH = 32
LOCALE_LENGTH = 64
TIMEZONE_LENGTH = 64
DIGEST_TIME_LENGTH = 5  # 'HH:MM'
SHA256_HEX_LENGTH = 64
EXPO_PUSH_TOKEN_LENGTH = 255
REASON_LENGTH = 64
PUSH_KEY_LENGTH = 128

# Never leave the server through to_dict(): the hashes and the push signing key.
SECRET_COLUMNS = ("pairing_code_hash", "token_hash", "prev_token_hash", "push_key")

# PRD-266 §8: a clerk row never holds a code or token hash.
CLERK_HOLDS_NO_TOKEN = (
    "kind = 'paired' OR "
    "(token_hash IS NULL AND prev_token_hash IS NULL AND pairing_code_hash IS NULL)"
)


def _in_list(column: str, values: tuple) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({quoted})"


def _iso(value) -> str | None:
    return value.isoformat() if value else None


class MobileDevice(Base):
    """One phone or paired browser, bound to one workspace and one person."""

    __tablename__ = "mobile_devices"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "user_id", "install_id", name="uq_mobile_devices_workspace_user_install",
        ),
        UniqueConstraint("token_hash", name="uq_mobile_devices_token_hash"),
        UniqueConstraint("expo_push_token", name="uq_mobile_devices_expo_push_token"),
        # /pair finds the code with no workspace: one code leads to one pending row only.
        UniqueConstraint("pairing_code_hash", name="uq_mobile_devices_pairing_code_hash"),
        CheckConstraint(
            _in_list("kind", tuple(k.value for k in MobileDeviceKind)), name="ck_mobile_devices_kind",
        ),
        CheckConstraint(
            _in_list("status", tuple(s.value for s in MobileDeviceStatus)), name="ck_mobile_devices_status",
        ),
        CheckConstraint(
            f"platform IS NULL OR {_in_list('platform', MOBILE_DEVICE_PLATFORMS)}",
            name="ck_mobile_devices_platform",
        ),
        CheckConstraint(CLERK_HOLDS_NO_TOKEN, name="ck_mobile_devices_clerk_holds_no_token"),
        Index("ix_mobile_devices_workspace_status", "workspace_id", "status"),
        Index("ix_mobile_devices_prev_token_hash", "prev_token_hash"),
        {"extend_existing": True},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(
        PGUUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False,
    )
    # Integer, matching users.id; resolve it with core.auth.principal.resolve_user_pk.
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    kind = Column(String(ENUM_LENGTH), nullable=False)
    # Null on a pending row: no phone has claimed the code yet.
    install_id = Column(String(INSTALL_ID_LENGTH), nullable=True)
    name = Column(String(NAME_LENGTH), nullable=True)
    platform = Column(String(ENUM_LENGTH), nullable=True)
    app_version = Column(String(APP_VERSION_LENGTH), nullable=True)
    locale = Column(String(LOCALE_LENGTH), nullable=True)
    timezone = Column(String(TIMEZONE_LENGTH), nullable=True)
    digest_time = Column(String(DIGEST_TIME_LENGTH), nullable=True)
    status = Column(
        String(ENUM_LENGTH), nullable=False,
        default=MobileDeviceStatus.PENDING.value, server_default=MobileDeviceStatus.PENDING.value,
    )
    # SHA-256 of the one-time pairing code; cleared the moment the device pairs.
    pairing_code_hash = Column(String(SHA256_HEX_LENGTH), nullable=True)
    pairing_expires_at = Column(DateTime(timezone=True), nullable=True)
    # SHA-256 of the device token. The token itself is returned exactly once.
    token_hash = Column(String(SHA256_HEX_LENGTH), nullable=True)
    token_rotated_at = Column(DateTime(timezone=True), nullable=True)
    # The token before the last rotation, accepted until prev_token_valid_until.
    prev_token_hash = Column(String(SHA256_HEX_LENGTH), nullable=True)
    prev_token_valid_until = Column(DateTime(timezone=True), nullable=True)
    expo_push_token = Column(String(EXPO_PUSH_TOKEN_LENGTH), nullable=True)
    push_enabled = Column(Boolean, nullable=False, default=False, server_default=false())
    push_disabled_reason = Column(String(REASON_LENGTH), nullable=True)
    push_key = Column(String(PUSH_KEY_LENGTH), nullable=True)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    paired_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revoked_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    events = Column(JSONB, nullable=False, default=list, server_default=text("'[]'"))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False,
    )

    def to_dict(self) -> dict:
        """The device as the API shows it: never a hash, the push key or the push token."""
        return {
            "id": str(self.id),
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "user_id": self.user_id,
            "kind": self.kind,
            "install_id": self.install_id,
            "name": self.name,
            "platform": self.platform,
            "app_version": self.app_version,
            "locale": self.locale,
            "timezone": self.timezone,
            "digest_time": self.digest_time,
            "status": self.status,
            "pairing_expires_at": _iso(self.pairing_expires_at),
            "token_rotated_at": _iso(self.token_rotated_at),
            "push_registered": bool(self.expo_push_token),
            "push_enabled": bool(self.push_enabled),
            "push_disabled_reason": self.push_disabled_reason,
            "last_seen_at": _iso(self.last_seen_at),
            "paired_at": _iso(self.paired_at),
            "revoked_at": _iso(self.revoked_at),
            "revoked_by": self.revoked_by,
            "events": list(self.events or []),
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }
