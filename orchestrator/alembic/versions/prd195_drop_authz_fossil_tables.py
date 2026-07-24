"""PRD-195 S8 (P2-14) — drop the agent-tool RBAC fossil tables.

The fifth role vocabulary: ``api/permissions.py`` (708 lines, unmounted and
deleted in this PR) enforced a hardcoded ``agent_type → vendor-tool`` matrix
over ``agent_tool_permissions`` / ``permission_audit_logs`` — workspace-blind
by schema (no ``workspace_id`` column), disconnected from the real tool-gating
lane (Composio registry + tool_router), untouched since PR #303.

Writer-audit (in-PR, grep-verified):

- ``agent_tool_permissions`` / ``permission_audit_logs`` — referenced ONLY by
  the deleted router and their own model definitions. DROPPED.
- ``tool_configurations`` / ``tool_categories`` — zero readers/writers outside
  their model definitions (the similarly-named identifiers in
  ``modules/tools`` are the in-memory registry's Enum, not these tables), no
  seeds, no raw SQL, no frontend surface. ORPHANED → DROPPED.
- ``tools`` — KEPT: live reader in
  ``modules/tools/registry/tool_registry.py`` (PRD-123 tier resolution).

HUMAN-GATED by convention (Gerard applies prior DROPs) — NOTE the deploy
entrypoint runs ``alembic upgrade heads`` on boot, so this migration
self-applies on the first post-merge deploy.

``IF EXISTS`` drops: several of these tables were created via the legacy
``Base.metadata.create_all`` path, so some environments never materialised
them. ``downgrade()`` recreates structure from the deleted models' DDL
(FKs + single-column indexes; python-side defaults not encoded).

Revision ID: prd195_drop_authz_fossil_tables
Revises: prd191_agent_skills_unique_and_priority
"""

import sqlalchemy as sa

from alembic import op

revision = "prd195_drop_authz_fossil_tables"
down_revision = "prd191_agent_skills_unique_and_priority"
branch_labels = None
depends_on = None

# Children first (FKs into tools/agents), then the orphaned standalone pair.
FOSSIL_TABLES = (
    "agent_tool_permissions",
    "permission_audit_logs",
    "tool_configurations",
    "tool_categories",
)


def upgrade() -> None:
    for table in FOSSIL_TABLES:
        op.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')


def downgrade() -> None:
    op.create_table(
        "tool_categories",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False, unique=True),
        sa.Column("description", sa.Text()),
        sa.Column("icon", sa.String(length=50)),
        sa.Column("parent_id", sa.Integer(), sa.ForeignKey("tool_categories.id")),
        sa.Column("sort_order", sa.Integer()),
        sa.Column("is_active", sa.Boolean()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_table(
        "tool_configurations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tool_id", sa.Integer(), sa.ForeignKey("tools.id"), nullable=False, index=True),
        sa.Column("environment", sa.String(length=50), nullable=False, index=True),
        sa.Column("configuration", sa.JSON()),
        sa.Column("is_active", sa.Boolean(), index=True),
        sa.Column("last_health_check", sa.DateTime()),
        sa.Column("health_status", sa.String(length=50)),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_table(
        "agent_tool_permissions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("agent_id", sa.Integer(), sa.ForeignKey("agents.id"), nullable=False, index=True),
        sa.Column("tool_id", sa.Integer(), sa.ForeignKey("tools.id"), nullable=False, index=True),
        sa.Column("environment", sa.String(length=50), nullable=False, index=True),
        sa.Column("permissions", sa.ARRAY(sa.String())),
        sa.Column("is_active", sa.Boolean(), index=True),
        sa.Column("expires_at", sa.DateTime()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_table(
        "permission_audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("agent_id", sa.Integer(), sa.ForeignKey("agents.id"), nullable=False, index=True),
        sa.Column("tool_id", sa.Integer(), sa.ForeignKey("tools.id"), nullable=False, index=True),
        sa.Column("action", sa.String(length=100), nullable=False, index=True),
        sa.Column("environment", sa.String(length=50), index=True),
        sa.Column("user_id", sa.String(length=255), index=True),
        sa.Column("details", sa.JSON()),
        sa.Column("timestamp", sa.DateTime(), server_default=sa.func.now(), index=True),
    )
