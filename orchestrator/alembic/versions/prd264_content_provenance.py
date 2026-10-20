"""content_provenance — one row per marked file Automatos generated (PRD-264 EUAI-10, 19 Oct 2026).

The public detection route answers "made by Automatos AI: yes or no" for a file's
SHA-256. Each marked file is one ``content_provenance`` row: its hash, its kind
(a CHECK holds it to video, image, audio or document), the C2PA manifest id when
signed, the generator and when it was made. ``workspace_id`` is for tenancy and
erasure only. One row per workspace and hash; the hash has its own index for the
lookup. The names match ``core.models.content_provenance.ContentProvenance``.

A new table only: no existing table or row changes. ``IF NOT EXISTS`` keeps the
upgrade a no-op on a database that ``create_all`` built from the models, which
then gains only a missing index.

Revision ID: prd264_content_provenance
Revises: prd263_harness_ledger_grants
Create Date: 2026-10-19
"""
from alembic import op

revision = "prd264_content_provenance"
down_revision = "prd263_harness_ledger_grants"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS content_provenance (
            id BIGSERIAL PRIMARY KEY,
            workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            sha256 VARCHAR(64) NOT NULL,
            kind VARCHAR(16) NOT NULL,
            manifest_id VARCHAR(255),
            generator VARCHAR(120) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_content_provenance_ws_sha256 UNIQUE (workspace_id, sha256),
            CONSTRAINT ck_content_provenance_kind CHECK (kind IN ('video', 'image', 'audio', 'document')),
            CONSTRAINT ck_content_provenance_sha256 CHECK (sha256 ~ '^[0-9a-f]{64}$')
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_content_provenance_sha256 "
        "ON content_provenance (sha256);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_content_provenance_sha256;")
    op.execute("DROP TABLE IF EXISTS content_provenance;")
