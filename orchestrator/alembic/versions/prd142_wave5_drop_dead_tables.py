"""PRD-142 Wave 5 (WS-AA) — drop 10 verified-dead tables.

Standalone migration (down_revision = None), matching the PRD-142 wave
pattern (wave0_error_events, wave4_harness_store). HUMAN-GATED: authored by
the wave, applied on prod by Gerard only — the loop never runs migrations.

Evidence (docs/PRDS/PRD-142-WAVE5-PHASE0-VERDICTS.md): each table has zero
reads/writes outside its own model definition + migrations, no inbound FKs
from live tables, no raw-SQL or frontend/analytics references. Most were
created via the legacy ``Base.metadata.create_all`` path rather than
migrations, so the drops are ``IF EXISTS`` to behave on environments where a
table never materialised. The corresponding model classes are deleted in the
same commit, so fresh environments no longer create these tables either.

EXCLUDED (needs an explicit decision): ``intent_classification_cache`` — it
is FK'd by the live ``tool_execution_logs.intent_cluster_id`` (PRD-139).

``downgrade()`` recreates each table from its model-DDL snapshot (structure,
FKs, single-column indexes; python-side defaults are not encoded as server
defaults — they lived in the deleted models).

Revision ID: prd142_wave5_drop_dead_tables
Revises: None (standalone)
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "prd142_wave5_drop_dead_tables"
down_revision = None
branch_labels = None
depends_on = None

# Verified-dead tables (Phase-0 verdicts, re-verified 2026-06-09 post-merge).
DEAD_TABLES = (
    "benchmark_assessments",
    "component_metrics",
    "database_relationships",
    "evaluation_results",
    "external_knowledge",
    "integration_analyses",
    "tool_credentials",
    "tool_execution_cache",
    "tool_installation_requests",
    "tool_reviews",
)


def upgrade() -> None:
    # No CASCADE on purpose: zero inbound FKs were verified, so a dependency
    # error here would mean the verification was wrong — fail loudly.
    for table in DEAD_TABLES:
        op.execute(f'DROP TABLE IF EXISTS "{table}"')


def downgrade() -> None:
    op.create_table(
        "external_knowledge",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("content", sa.JSON(), nullable=False),
        sa.Column("source", sa.String(255), nullable=False),
        sa.Column("knowledge_metadata", sa.JSON(), nullable=True),
        sa.Column("access_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column(
            "created_by_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("is_shared", sa.Boolean(), nullable=True),
    )

    op.create_table(
        "evaluation_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("evaluation_id", sa.String(255), nullable=False, unique=True),
        sa.Column("evaluation_type", sa.String(100), nullable=False),
        sa.Column("scope", sa.String(100), nullable=False),
        sa.Column("target_id", sa.String(255), nullable=False),
        sa.Column("overall_score", sa.Float(), nullable=False),
        sa.Column("detailed_results", sa.JSON(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("execution_time_seconds", sa.Float(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "benchmark_assessments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("benchmark_id", sa.String(255), nullable=False),
        sa.Column("benchmark_name", sa.String(255), nullable=False),
        sa.Column("benchmark_type", sa.String(100), nullable=False),
        sa.Column("validity_score", sa.Float(), nullable=True),
        sa.Column("reliability_score", sa.Float(), nullable=True),
        sa.Column("discriminatory_power", sa.Float(), nullable=True),
        sa.Column("overall_quality", sa.Float(), nullable=True),
        sa.Column("quality_classification", sa.String(50), nullable=True),
        sa.Column("assessment_data", sa.JSON(), nullable=True),
        sa.Column("recommendations", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "component_metrics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("component_id", sa.String(255), nullable=False),
        sa.Column("component_type", sa.String(100), nullable=False),
        sa.Column("performance_score", sa.Float(), nullable=True),
        sa.Column("reliability_score", sa.Float(), nullable=True),
        sa.Column("readiness_score", sa.Float(), nullable=True),
        sa.Column("capability_rating", sa.Float(), nullable=True),
        sa.Column("complexity_index", sa.Float(), nullable=True),
        sa.Column("environment_factor", sa.Float(), nullable=True),
        sa.Column("assessment_details", sa.JSON(), nullable=True),
        sa.Column("assessment_timestamp", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "integration_analyses",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("system_id", sa.String(255), nullable=False),
        sa.Column("coherence_score", sa.Float(), nullable=True),
        sa.Column("efficiency_score", sa.Float(), nullable=True),
        sa.Column("emergence_score", sa.Float(), nullable=True),
        sa.Column("integration_score", sa.Float(), nullable=True),
        sa.Column("integration_classification", sa.String(50), nullable=True),
        sa.Column("analysis_data", sa.JSON(), nullable=True),
        sa.Column("recommendations", sa.JSON(), nullable=True),
        sa.Column("confidence_level", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "tool_credentials",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "tool_id", sa.Integer(), sa.ForeignKey("tools.id"), nullable=False
        ),
        sa.Column("credential_key", sa.String(100), nullable=False),
        sa.Column("credential_value", sa.Text(), nullable=False),
        sa.Column("environment", sa.String(50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_tool_credentials_tool_id", "tool_credentials", ["tool_id"])
    op.create_index(
        "ix_tool_credentials_environment", "tool_credentials", ["environment"]
    )
    op.create_index(
        "ix_tool_credentials_is_active", "tool_credentials", ["is_active"]
    )

    op.create_table(
        "tool_reviews",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "tool_id", sa.Integer(), sa.ForeignKey("tools.id"), nullable=False
        ),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("review_text", sa.Text(), nullable=True),
        sa.Column("reviewer_id", sa.String(255), nullable=True),
        sa.Column("reviewer_type", sa.String(50), nullable=True),
        sa.Column("is_verified", sa.Boolean(), nullable=True),
        sa.Column("is_featured", sa.Boolean(), nullable=True),
        sa.Column("helpful_votes", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_tool_reviews_tool_id", "tool_reviews", ["tool_id"])
    op.create_index("ix_tool_reviews_reviewer_id", "tool_reviews", ["reviewer_id"])

    op.create_table(
        "tool_installation_requests",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "tool_id", sa.Integer(), sa.ForeignKey("tools.id"), nullable=False
        ),
        sa.Column("requested_by", sa.String(255), nullable=False),
        sa.Column("environment", sa.String(50), nullable=False),
        sa.Column("justification", sa.Text(), nullable=True),
        sa.Column("status", sa.String(50), nullable=True),
        sa.Column("approved_by", sa.String(255), nullable=True),
        sa.Column("approval_notes", sa.Text(), nullable=True),
        sa.Column("installation_config", sa.JSON(), nullable=True),
        sa.Column("auto_configure", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("approved_at", sa.DateTime(), nullable=True),
        sa.Column("installed_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_tool_installation_requests_tool_id",
        "tool_installation_requests",
        ["tool_id"],
    )
    op.create_index(
        "ix_tool_installation_requests_requested_by",
        "tool_installation_requests",
        ["requested_by"],
    )
    op.create_index(
        "ix_tool_installation_requests_status",
        "tool_installation_requests",
        ["status"],
    )
    op.create_index(
        "ix_tool_installation_requests_approved_by",
        "tool_installation_requests",
        ["approved_by"],
    )

    op.create_table(
        "tool_execution_cache",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("agent_id", sa.Integer(), nullable=False),
        sa.Column("app_name", sa.String(100), nullable=False),
        sa.Column("action_name", sa.String(255), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("input_parameters", postgresql.JSONB(), nullable=True),
        sa.Column("cached_result", postgresql.JSONB(), nullable=False),
        sa.Column("cache_key", sa.String(255), nullable=True),
        sa.Column("cached_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("hit_count", sa.Integer(), nullable=True),
        sa.Column("last_hit_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_tool_execution_cache_agent_id", "tool_execution_cache", ["agent_id"]
    )
    op.create_index(
        "ix_tool_execution_cache_cache_key", "tool_execution_cache", ["cache_key"]
    )

    op.create_table(
        "database_relationships",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "source_id",
            sa.Integer(),
            sa.ForeignKey("database_knowledge_sources.id"),
            nullable=False,
        ),
        sa.Column("from_table", sa.String(255), nullable=False),
        sa.Column("from_column", sa.String(255), nullable=False),
        sa.Column("to_table", sa.String(255), nullable=False),
        sa.Column("to_column", sa.String(255), nullable=False),
        sa.Column("relationship_type", sa.String(50), nullable=True),
        sa.Column("is_inferred", sa.Boolean(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
