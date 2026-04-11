"""PRD-129: Workspace Outputs Hub — deliverables table

Stores metadata about agent outputs (reports, images, documents, code, slides, etc.).
Content lives in workspace filesystem / S3; this table is for discovery, filtering,
and the Gallery view. Soft-deletable via `deleted_at`.

Revision ID: prd129_deliverables
Revises: None (standalone — safe to run anytime)
Create Date: 2026-04-10
"""
from alembic import op
import sqlalchemy as sa

revision = "prd129_deliverables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE deliverables (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id      UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,

            -- Provenance
            source_type       VARCHAR(30) NOT NULL,
            source_id         VARCHAR(255) NULL,
            agent_id          INTEGER NULL REFERENCES agents(id) ON DELETE SET NULL,
            agent_name        VARCHAR(100) NULL,

            -- Classification
            artifact_type     VARCHAR(30) NOT NULL,
            title             VARCHAR(255) NOT NULL,
            summary           VARCHAR(500) NULL,

            -- Storage
            storage_type      VARCHAR(20) NOT NULL DEFAULT 'workspace',
            file_path         VARCHAR(1024) NOT NULL,
            file_name         VARCHAR(255) NULL,
            file_type         VARCHAR(50) NULL,
            file_size_bytes   BIGINT NULL,

            -- Preview
            preview_url       VARCHAR(1024) NULL,
            preview_type      VARCHAR(30) NULL,

            -- Extensibility (named `extra` to avoid SQLAlchemy Base.metadata conflict)
            extra             JSONB NOT NULL DEFAULT '{}'::jsonb,

            -- Lifecycle
            status            VARCHAR(20) NOT NULL DEFAULT 'ready',
            deleted_at        TIMESTAMPTZ NULL,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        -- Indices
        CREATE INDEX ix_deliverables_workspace
            ON deliverables(workspace_id);
        CREATE INDEX ix_deliverables_agent
            ON deliverables(agent_id);
        CREATE INDEX ix_deliverables_type
            ON deliverables(workspace_id, artifact_type);
        CREATE INDEX ix_deliverables_source
            ON deliverables(workspace_id, source_type);
        CREATE INDEX ix_deliverables_created
            ON deliverables(workspace_id, created_at DESC);

        -- Idempotent re-registration: one live row per (workspace_id, file_path)
        CREATE UNIQUE INDEX uq_deliverables_workspace_path
            ON deliverables(workspace_id, file_path)
            WHERE deleted_at IS NULL;
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS deliverables;")
