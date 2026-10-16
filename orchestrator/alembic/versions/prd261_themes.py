"""PRD-261 Phase A (A-05): installable themes — the ONE migration of Phase A and C (17 Oct 2026).

Every table the theme routes need, created once (PA-4), so later stories add no head:

* ``theme_packages``: one row per theme version, a workspace's private upload or a
  Marketplace submission (``owner_type``). ``digest`` has a plain index; uniqueness
  is partial: a version and a digest are unique within one workspace, and within the
  Marketplace. Two workspaces may hold the same digest, and so may a workspace row
  and a Marketplace row.
* ``theme_publishers``: publisher names and their owners.
* ``theme_installations``: a workspace's installs, with the package a rollback goes
  back to, a pending update and a pinned version; one active row per workspace and
  theme, one active default per workspace.
* ``theme_choices``, ``theme_looks``, ``theme_opt_ins``: per member, per workspace.
* ``theme_workspace_settings``: one row per workspace.
* ``theme_revocations`` and ``theme_review_events``.

No existing table is altered. ``core/models/themes.py`` declares the same shape, and
``tests/test_prd261_theme_models.py`` holds the two together.

Create_all-first safe (the 89d89c250 rule): CREATE TABLE IF NOT EXISTS and
CREATE [UNIQUE] INDEX IF NOT EXISTS, so it is a no-op on a database that
``create_all`` built from the models, and running it twice changes nothing.

Revision ID: prd261_themes
Revises: prd258_voice_timings
Create Date: 2026-10-17
"""

from alembic import op


revision = "prd261_themes"
down_revision = "prd258_voice_timings"
branch_labels = None
depends_on = None

# (table, CREATE TABLE), in creation order: theme_packages before the tables that refer to it.
_TABLES = (
    ("theme_packages", """
CREATE TABLE IF NOT EXISTS theme_packages (
    id UUID PRIMARY KEY,
    theme_id VARCHAR(128) NOT NULL,
    version VARCHAR(64) NOT NULL,
    owner_type VARCHAR(16) NOT NULL,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    digest VARCHAR(64) NOT NULL,
    manifest JSONB NOT NULL,
    file_manifest JSONB NOT NULL,
    storage_prefix VARCHAR NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    reviewed BOOLEAN NOT NULL DEFAULT false,
    scan_report JSONB,
    scan_verdict VARCHAR(16) NOT NULL DEFAULT 'not_run',
    signature TEXT,
    submitted_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    CONSTRAINT ck_theme_packages_owner_type CHECK (owner_type IN ('marketplace', 'workspace')),
    CONSTRAINT ck_theme_packages_owner_workspace CHECK ((owner_type = 'workspace' AND workspace_id IS NOT NULL) OR (owner_type = 'marketplace' AND workspace_id IS NULL)),
    CONSTRAINT ck_theme_packages_status CHECK (status IN ('pending', 'approved', 'rejected', 'deactivated')),
    CONSTRAINT ck_theme_packages_scan_verdict CHECK (scan_verdict IN ('pass', 'review_required', 'blocked', 'not_run'))
);
"""),
    ("theme_publishers", """
CREATE TABLE IF NOT EXISTS theme_publishers (
    id UUID PRIMARY KEY,
    publisher VARCHAR(64) NOT NULL,
    owner_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    reserved BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);
"""),
    ("theme_installations", """
CREATE TABLE IF NOT EXISTS theme_installations (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    theme_id VARCHAR(128) NOT NULL,
    package_id UUID NOT NULL REFERENCES theme_packages(id) ON DELETE CASCADE,
    previous_package_id UUID REFERENCES theme_packages(id) ON DELETE SET NULL,
    pending_package_id UUID REFERENCES theme_packages(id) ON DELETE SET NULL,
    pinned_version VARCHAR(64),
    installed_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    is_default BOOLEAN NOT NULL DEFAULT false,
    installed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    uninstalled_at TIMESTAMP WITH TIME ZONE
);
"""),
    ("theme_choices", """
CREATE TABLE IF NOT EXISTS theme_choices (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    choice VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);
"""),
    ("theme_looks", """
CREATE TABLE IF NOT EXISTS theme_looks (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    theme_id VARCHAR(128) NOT NULL,
    looks JSONB NOT NULL,
    active JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);
"""),
    ("theme_opt_ins", """
CREATE TABLE IF NOT EXISTS theme_opt_ins (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    package_id UUID NOT NULL REFERENCES theme_packages(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);
"""),
    ("theme_workspace_settings", """
CREATE TABLE IF NOT EXISTS theme_workspace_settings (
    workspace_id UUID PRIMARY KEY REFERENCES workspaces(id) ON DELETE CASCADE,
    private_uploads VARCHAR(16),
    members_may_choose BOOLEAN NOT NULL DEFAULT true,
    install_role VARCHAR(64) NOT NULL DEFAULT 'workspace:manage',
    updated_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    CONSTRAINT ck_theme_workspace_settings_private_uploads CHECK (private_uploads IS NULL OR private_uploads IN ('on', 'off'))
);
"""),
    ("theme_revocations", """
CREATE TABLE IF NOT EXISTS theme_revocations (
    id UUID PRIMARY KEY,
    digest VARCHAR(64),
    theme_id VARCHAR(128),
    version_range VARCHAR(128),
    level VARCHAR(16) NOT NULL,
    reason TEXT NOT NULL,
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    lifted_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT ck_theme_revocations_level CHECK (level IN ('warn', 'disable')),
    CONSTRAINT ck_theme_revocations_target CHECK ((digest IS NOT NULL AND theme_id IS NULL AND version_range IS NULL) OR (digest IS NULL AND theme_id IS NOT NULL))
);
"""),
    ("theme_review_events", """
CREATE TABLE IF NOT EXISTS theme_review_events (
    id UUID PRIMARY KEY,
    package_id UUID NOT NULL REFERENCES theme_packages(id) ON DELETE CASCADE,
    event VARCHAR(32) NOT NULL,
    from_status VARCHAR(16),
    to_status VARCHAR(16),
    actor_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    reason TEXT,
    detail JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);
"""),
)

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS ix_theme_packages_workspace_id ON theme_packages (workspace_id);",
    "CREATE INDEX IF NOT EXISTS ix_theme_packages_theme_id ON theme_packages (theme_id);",
    "CREATE INDEX IF NOT EXISTS ix_theme_packages_digest ON theme_packages (digest);",
    "CREATE INDEX IF NOT EXISTS ix_theme_packages_status ON theme_packages (status);",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_theme_packages_workspace_version "
    "ON theme_packages (workspace_id, theme_id, version) WHERE owner_type = 'workspace';",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_theme_packages_workspace_digest "
    "ON theme_packages (workspace_id, digest) WHERE owner_type = 'workspace';",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_theme_packages_marketplace_version "
    "ON theme_packages (theme_id, version) WHERE owner_type = 'marketplace';",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_theme_packages_marketplace_digest "
    "ON theme_packages (digest) WHERE owner_type = 'marketplace';",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_theme_publishers_publisher ON theme_publishers (publisher);",
    "CREATE INDEX IF NOT EXISTS ix_theme_installations_workspace_id ON theme_installations (workspace_id);",
    "CREATE INDEX IF NOT EXISTS ix_theme_installations_package_id ON theme_installations (package_id);",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_theme_installations_active "
    "ON theme_installations (workspace_id, theme_id) WHERE uninstalled_at IS NULL;",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_theme_installations_default "
    "ON theme_installations (workspace_id) WHERE is_default AND uninstalled_at IS NULL;",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_theme_choices_workspace_user ON theme_choices (workspace_id, user_id);",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_theme_looks_workspace_user_theme "
    "ON theme_looks (workspace_id, user_id, theme_id);",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_theme_opt_ins_workspace_user_package "
    "ON theme_opt_ins (workspace_id, user_id, package_id);",
    "CREATE INDEX IF NOT EXISTS ix_theme_opt_ins_package_id ON theme_opt_ins (package_id);",
    "CREATE INDEX IF NOT EXISTS ix_theme_revocations_digest ON theme_revocations (digest);",
    "CREATE INDEX IF NOT EXISTS ix_theme_revocations_theme_id ON theme_revocations (theme_id);",
    "CREATE INDEX IF NOT EXISTS ix_theme_review_events_package_id ON theme_review_events (package_id);",
)


def upgrade() -> None:
    for _table, statement in _TABLES:
        op.execute(statement)
    for statement in _INDEXES:
        op.execute(statement)


def downgrade() -> None:
    # The indexes go with their tables; the tables that refer to theme_packages go first.
    for table, _statement in reversed(_TABLES):
        op.execute(f"DROP TABLE IF EXISTS {table};")
