"""PRD-261: installable themes — every table Phase A and the minimal Marketplace (C) need.

* ``theme_packages``: one row per theme version. A workspace's private upload
  (``owner_type`` 'workspace', ``workspace_id`` set) or a Marketplace submission
  (``owner_type`` 'marketplace', no workspace). ``digest`` is the sha256 of the
  package's file list; it has a plain index, never a global unique, because two
  workspaces may upload the same bytes and a private package may later be
  submitted to the Marketplace. The partial unique indexes keep a version and a
  digest unique within one workspace, and within the Marketplace.
* ``theme_publishers``: who owns a publisher name (``automatos`` is reserved).
* ``theme_installations``: a workspace's installs, by reference to a package.
  ``previous_package_id`` is where a rollback goes, ``pending_package_id`` an
  update that is waiting, ``pinned_version`` a version the workspace holds. At most
  one active row (``uninstalled_at`` IS NULL) per workspace and theme, and at most
  one active default per workspace.
* ``theme_choices``: a member's theme choice in one workspace.
* ``theme_looks``: a member's saved looks for one theme in one workspace.
* ``theme_opt_ins``: a member's opt-in to run one private package version.
* ``theme_workspace_settings``: one row per workspace (the PRD §5 admin controls).
  ``private_uploads`` NULL means the edition default.
* ``theme_revocations``: a digest, or a theme id with an optional version range,
  marked ``warn`` or ``disable``.
* ``theme_review_events``: the review history of one package.

JSON columns are ``JSON().with_variant(JSONB(), "postgresql")`` and id columns the
portable ``Uuid``, so the tables also build under SQLite in the unit tests. The
``prd261_themes`` migration creates them all, and ``tests/test_prd261_theme_models.py``
holds the migration and the models together.
"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    false,
    text,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from core.database.base import Base

__all__ = [
    "ThemePackage",
    "ThemePublisher",
    "ThemeInstallation",
    "ThemeChoice",
    "ThemeLook",
    "ThemeOptIn",
    "ThemeWorkspaceSettings",
    "ThemeRevocation",
    "ThemeReviewEvent",
    "THEME_OWNER_TYPES",
    "THEME_PACKAGE_STATUSES",
    "THEME_PACKAGE_STATUS_DEFAULT",
    "THEME_SCAN_VERDICTS",
    "THEME_SCAN_VERDICT_DEFAULT",
    "THEME_PRIVATE_UPLOAD_VALUES",
    "THEME_INSTALL_ROLE_DEFAULT",
    "THEME_REVOCATION_LEVELS",
    "THEME_REVIEW_EVENTS",
]

THEME_OWNER_TYPES = ("marketplace", "workspace")
THEME_PACKAGE_STATUSES = ("pending", "approved", "rejected", "deactivated")
THEME_PACKAGE_STATUS_DEFAULT = "pending"
THEME_SCAN_VERDICTS = ("pass", "review_required", "blocked", "not_run")
THEME_SCAN_VERDICT_DEFAULT = "not_run"
# NULL in the column means "the edition default" (THEMES_PRIVATE_UPLOAD, A-07e).
THEME_PRIVATE_UPLOAD_VALUES = ("on", "off")
THEME_INSTALL_ROLE_DEFAULT = "workspace:manage"
THEME_REVOCATION_LEVELS = ("warn", "disable")
# The events C-01..C-03 write. Not a CHECK: a later review stage adds one without a migration.
THEME_REVIEW_EVENTS = ("submitted", "scanned", "approved", "rejected", "deactivated")

THEME_ID_LENGTH = 128  # "publisher/slug"
THEME_VERSION_LENGTH = 64
THEME_DIGEST_LENGTH = 64  # sha256, lowercase hex
THEME_PUBLISHER_LENGTH = 64
THEME_CHOICE_LENGTH = 64  # 'classic' | 'studio' | 'builtin:<id>' | an install id
THEME_STATUS_LENGTH = 16
THEME_ROLE_LENGTH = 64
THEME_VERSION_RANGE_LENGTH = 128
THEME_EVENT_LENGTH = 32

# The partial indexes' predicates, the same on Postgres and SQLite.
WORKSPACE_OWNED = "owner_type = 'workspace'"
MARKETPLACE_OWNED = "owner_type = 'marketplace'"
ACTIVE_INSTALLATION = "uninstalled_at IS NULL"
ACTIVE_DEFAULT = "is_default AND uninstalled_at IS NULL"

OWNER_MATCHES_WORKSPACE = (
    "(owner_type = 'workspace' AND workspace_id IS NOT NULL) "
    "OR (owner_type = 'marketplace' AND workspace_id IS NULL)"
)
PRIVATE_UPLOADS_CHECK = "private_uploads IS NULL OR private_uploads IN ('on', 'off')"
REVOCATION_TARGET = (
    "(digest IS NOT NULL AND theme_id IS NULL AND version_range IS NULL) "
    "OR (digest IS NULL AND theme_id IS NOT NULL)"
)


def _in_list(column: str, values: tuple) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({quoted})"


def _json_type():
    return JSON().with_variant(JSONB(), "postgresql")


def _partial_unique(name: str, predicate: str, *columns: str) -> Index:
    where = text(predicate)
    return Index(name, *columns, unique=True, postgresql_where=where, sqlite_where=where)


def _id_column() -> Column:
    return Column(Uuid(as_uuid=True), primary_key=True, default=uuid4)


def _workspace_column(nullable: bool = False) -> Column:
    return Column(Uuid(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=nullable)


def _user_column(ondelete: str, nullable: bool = True) -> Column:
    # Integer, matching users.id; resolve it with core.auth.principal.resolve_user_pk.
    return Column(Integer, ForeignKey("users.id", ondelete=ondelete), nullable=nullable)


def _package_column(ondelete: str, nullable: bool) -> Column:
    return Column(Uuid(as_uuid=True), ForeignKey("theme_packages.id", ondelete=ondelete), nullable=nullable)


def _created_at() -> Column:
    return Column(DateTime(timezone=True), nullable=False, server_default=func.now())


def _updated_at() -> Column:
    return Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class ThemePackage(Base):
    """One version of a theme: a workspace's private upload or a Marketplace submission."""

    __tablename__ = "theme_packages"
    __table_args__ = (
        CheckConstraint(_in_list("owner_type", THEME_OWNER_TYPES), name="ck_theme_packages_owner_type"),
        CheckConstraint(OWNER_MATCHES_WORKSPACE, name="ck_theme_packages_owner_workspace"),
        CheckConstraint(_in_list("status", THEME_PACKAGE_STATUSES), name="ck_theme_packages_status"),
        CheckConstraint(_in_list("scan_verdict", THEME_SCAN_VERDICTS), name="ck_theme_packages_scan_verdict"),
        Index("ix_theme_packages_workspace_id", "workspace_id"),
        Index("ix_theme_packages_theme_id", "theme_id"),
        Index("ix_theme_packages_digest", "digest"),
        Index("ix_theme_packages_status", "status"),
        _partial_unique(
            "uq_theme_packages_workspace_version", WORKSPACE_OWNED, "workspace_id", "theme_id", "version",
        ),
        _partial_unique("uq_theme_packages_workspace_digest", WORKSPACE_OWNED, "workspace_id", "digest"),
        _partial_unique("uq_theme_packages_marketplace_version", MARKETPLACE_OWNED, "theme_id", "version"),
        _partial_unique("uq_theme_packages_marketplace_digest", MARKETPLACE_OWNED, "digest"),
        {"extend_existing": True},
    )

    id = _id_column()
    theme_id = Column(String(THEME_ID_LENGTH), nullable=False)
    version = Column(String(THEME_VERSION_LENGTH), nullable=False)
    owner_type = Column(String(THEME_STATUS_LENGTH), nullable=False)
    workspace_id = _workspace_column(nullable=True)
    digest = Column(String(THEME_DIGEST_LENGTH), nullable=False)
    manifest = Column(_json_type(), nullable=False)
    file_manifest = Column(_json_type(), nullable=False)
    storage_prefix = Column(String, nullable=False)
    status = Column(
        String(THEME_STATUS_LENGTH), nullable=False,
        default=THEME_PACKAGE_STATUS_DEFAULT, server_default=THEME_PACKAGE_STATUS_DEFAULT,
    )
    reviewed = Column(Boolean, nullable=False, default=False, server_default=false())
    scan_report = Column(_json_type())
    scan_verdict = Column(
        String(THEME_STATUS_LENGTH), nullable=False,
        default=THEME_SCAN_VERDICT_DEFAULT, server_default=THEME_SCAN_VERDICT_DEFAULT,
    )
    signature = Column(Text)
    submitted_by = _user_column("SET NULL")
    created_at = _created_at()
    updated_at = _updated_at()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<ThemePackage {self.theme_id}@{self.version} owner={self.owner_type!r}>"


class ThemePublisher(Base):
    """A publisher name and the user who owns it."""

    __tablename__ = "theme_publishers"
    __table_args__ = (
        Index("uq_theme_publishers_publisher", "publisher", unique=True),
        {"extend_existing": True},
    )

    id = _id_column()
    publisher = Column(String(THEME_PUBLISHER_LENGTH), nullable=False)
    owner_user_id = _user_column("SET NULL")
    reserved = Column(Boolean, nullable=False, default=False, server_default=false())
    created_at = _created_at()
    updated_at = _updated_at()


class ThemeInstallation(Base):
    """A theme installed in a workspace; at most one active row per theme and one active default."""

    __tablename__ = "theme_installations"
    __table_args__ = (
        Index("ix_theme_installations_workspace_id", "workspace_id"),
        Index("ix_theme_installations_package_id", "package_id"),
        _partial_unique("uq_theme_installations_active", ACTIVE_INSTALLATION, "workspace_id", "theme_id"),
        _partial_unique("uq_theme_installations_default", ACTIVE_DEFAULT, "workspace_id"),
        {"extend_existing": True},
    )

    id = _id_column()
    workspace_id = _workspace_column()
    theme_id = Column(String(THEME_ID_LENGTH), nullable=False)
    package_id = _package_column("CASCADE", nullable=False)
    previous_package_id = _package_column("SET NULL", nullable=True)
    pending_package_id = _package_column("SET NULL", nullable=True)
    pinned_version = Column(String(THEME_VERSION_LENGTH))
    installed_by = _user_column("SET NULL")
    is_default = Column(Boolean, nullable=False, default=False, server_default=false())
    installed_at = _created_at()
    updated_at = _updated_at()
    uninstalled_at = Column(DateTime(timezone=True))

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<ThemeInstallation {self.theme_id} workspace_id={self.workspace_id}>"


class ThemeChoice(Base):
    """A member's theme choice in one workspace."""

    __tablename__ = "theme_choices"
    __table_args__ = (
        Index("uq_theme_choices_workspace_user", "workspace_id", "user_id", unique=True),
        {"extend_existing": True},
    )

    id = _id_column()
    workspace_id = _workspace_column()
    user_id = _user_column("CASCADE", nullable=False)
    choice = Column(String(THEME_CHOICE_LENGTH), nullable=False)
    created_at = _created_at()
    updated_at = _updated_at()


class ThemeLook(Base):
    """A member's saved looks for one theme in one workspace."""

    __tablename__ = "theme_looks"
    __table_args__ = (
        Index("uq_theme_looks_workspace_user_theme", "workspace_id", "user_id", "theme_id", unique=True),
        {"extend_existing": True},
    )

    id = _id_column()
    workspace_id = _workspace_column()
    user_id = _user_column("CASCADE", nullable=False)
    theme_id = Column(String(THEME_ID_LENGTH), nullable=False)
    looks = Column(_json_type(), nullable=False)
    active = Column(_json_type())
    created_at = _created_at()
    updated_at = _updated_at()


class ThemeOptIn(Base):
    """A member's opt-in to run one private package version in one workspace."""

    __tablename__ = "theme_opt_ins"
    __table_args__ = (
        Index(
            "uq_theme_opt_ins_workspace_user_package", "workspace_id", "user_id", "package_id", unique=True,
        ),
        Index("ix_theme_opt_ins_package_id", "package_id"),
        {"extend_existing": True},
    )

    id = _id_column()
    workspace_id = _workspace_column()
    user_id = _user_column("CASCADE", nullable=False)
    package_id = _package_column("CASCADE", nullable=False)
    created_at = _created_at()


class ThemeWorkspaceSettings(Base):
    """One workspace's theme settings: private uploads, member choice, who may install."""

    __tablename__ = "theme_workspace_settings"
    __table_args__ = (
        CheckConstraint(PRIVATE_UPLOADS_CHECK, name="ck_theme_workspace_settings_private_uploads"),
        {"extend_existing": True},
    )

    workspace_id = Column(
        Uuid(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True,
    )
    private_uploads = Column(String(THEME_STATUS_LENGTH))
    members_may_choose = Column(Boolean, nullable=False, default=True, server_default=true())
    install_role = Column(
        String(THEME_ROLE_LENGTH), nullable=False,
        default=THEME_INSTALL_ROLE_DEFAULT, server_default=THEME_INSTALL_ROLE_DEFAULT,
    )
    updated_by = _user_column("SET NULL")
    created_at = _created_at()
    updated_at = _updated_at()


class ThemeRevocation(Base):
    """A revoked digest, or a revoked theme id (optionally a version range of it)."""

    __tablename__ = "theme_revocations"
    __table_args__ = (
        CheckConstraint(_in_list("level", THEME_REVOCATION_LEVELS), name="ck_theme_revocations_level"),
        CheckConstraint(REVOCATION_TARGET, name="ck_theme_revocations_target"),
        Index("ix_theme_revocations_digest", "digest"),
        Index("ix_theme_revocations_theme_id", "theme_id"),
        {"extend_existing": True},
    )

    id = _id_column()
    digest = Column(String(THEME_DIGEST_LENGTH))
    theme_id = Column(String(THEME_ID_LENGTH))
    version_range = Column(String(THEME_VERSION_RANGE_LENGTH))
    level = Column(String(THEME_STATUS_LENGTH), nullable=False)
    reason = Column(Text, nullable=False)
    created_by = _user_column("SET NULL")
    created_at = _created_at()
    lifted_at = Column(DateTime(timezone=True))


class ThemeReviewEvent(Base):
    """One step in a package's review: submitted, scanned, approved, rejected, deactivated."""

    __tablename__ = "theme_review_events"
    __table_args__ = (
        Index("ix_theme_review_events_package_id", "package_id"),
        {"extend_existing": True},
    )

    id = _id_column()
    package_id = _package_column("CASCADE", nullable=False)
    event = Column(String(THEME_EVENT_LENGTH), nullable=False)
    from_status = Column(String(THEME_STATUS_LENGTH))
    to_status = Column(String(THEME_STATUS_LENGTH))
    actor_user_id = _user_column("SET NULL")
    reason = Column(Text)
    detail = Column(_json_type())
    created_at = _created_at()
