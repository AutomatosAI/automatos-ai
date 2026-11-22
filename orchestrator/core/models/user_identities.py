"""People who sign in without Clerk (PRD-277 Wave 2, §5.4).

* :class:`UserIdentity` (US-201) links one ``users`` row to one ``(issuer, subject)`` pair: the
  identity provider's address and its fixed id for that person. Unique on the pair, so one pair
  names one person for good. ``users.clerk_user_id`` is untouched and stays Clerk's. CASCADE:
  erasing a person erases their links too.
* :class:`UserSystemRoleColumns` (US-203) gives ``users`` a nullable ``system_role``: the
  install-wide role (``super_admin``, ``admin`` or ``user``) an install whose people sign in
  through its own identity provider stores for itself. It is read only when the edition signs
  people in with OIDC (``core.auth.system_role``); in ``saas`` the role still comes from Clerk's
  claims. ``core.models.core`` is over the file-size limit, so ``User`` takes the column from this
  mixin, as ``BlogPost`` takes its review columns.

The Alembic revision ``prd277_w2_identities_20261120`` declares the same names.
"""
from __future__ import annotations

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import declared_attr

from core.database.base import Base

# OIDC Core §2: ``sub`` is at most 255 ASCII characters. An issuer is an https address; 512
# leaves room for a tenant path. Both bound the unique index's key well under Postgres's limit.
ISSUER_MAX = 512
SUBJECT_MAX = 255

IDENTITY_PAIR_INDEX = "uq_user_identities_issuer_subject"
IDENTITY_USER_INDEX = "ix_user_identities_user_id"

# The install-wide roles a person can hold (modules.policy.roles: SUPER_ADMIN, ADMIN, USER).
SYSTEM_ROLES = ("super_admin", "admin", "user")
SYSTEM_ROLE_MAX = 32
SYSTEM_ROLE_CHECK_NAME = "ck_users_system_role"
SYSTEM_ROLE_CHECK = (
    "system_role IS NULL OR system_role IN (" + ", ".join(f"'{role}'" for role in SYSTEM_ROLES) + ")"
)


class UserIdentity(Base):
    """One ``(issuer, subject)`` pair a person signs in with, linked to their ``users`` row."""

    __tablename__ = "user_identities"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    issuer = Column(String(ISSUER_MAX), nullable=False)
    subject = Column(String(SUBJECT_MAX), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        # The arbiter of two first sign-ins racing: one pair, one row.
        Index(IDENTITY_PAIR_INDEX, "issuer", "subject", unique=True),
        Index(IDENTITY_USER_INDEX, "user_id"),
        {"extend_existing": True},
    )


class UserSystemRoleColumns:
    """Declarative mixin: the install-wide role ``users`` stores, NULL unless one was set."""

    system_role = Column(String(SYSTEM_ROLE_MAX), nullable=True)

    @declared_attr
    def __table_args__(cls):
        """The CHECK that keeps ``system_role`` to the known roles, made fresh for the class."""
        return (CheckConstraint(SYSTEM_ROLE_CHECK, name=SYSTEM_ROLE_CHECK_NAME),)


__all__ = [
    "IDENTITY_PAIR_INDEX",
    "IDENTITY_USER_INDEX",
    "ISSUER_MAX",
    "SUBJECT_MAX",
    "SYSTEM_ROLES",
    "SYSTEM_ROLE_CHECK",
    "SYSTEM_ROLE_CHECK_NAME",
    "SYSTEM_ROLE_MAX",
    "UserIdentity",
    "UserSystemRoleColumns",
]
