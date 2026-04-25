"""PRD-133 (corrected): Workspace Outputs via UNION view — no more shadow writes

This supersedes prd133_deliverables_lifecycle. Instead of `deliverables` being
a shadow registry that agent_reports / blog_posts write into as a second hop,
we expose a read-only VIEW ``v_workspace_outputs`` that UNIONs the native
source-of-truth tables plus any ad-hoc artifacts that still live in
``deliverables`` (code/images/etc. an agent writes with no native home yet).

Goals
-----
* One write path per artifact type. Blog → blog_posts. Report → agent_reports.
  Ad-hoc → deliverables. No double-writes, no drift.
* UI / API shape stays identical (same columns) so GalleryView & DeliverablePreview
  need zero changes.
* Status semantics unified at READ time:
    - blog_posts.status: draft/published/scheduled → lifecycle draft/published/review
    - agent_reports.status: ok/warning/error → lifecycle published/review/archived
    - deliverables.status: already lifecycle, passes through.
* Soft-delete supported on source tables via new `deleted_at` columns.

Migration steps
---------------
1. Add nullable ``deleted_at timestamptz`` to blog_posts + agent_reports (safe: no default).
2. Index ``(workspace_id, created_at)`` on both tables for view pagination perf.
3. CREATE VIEW ``v_workspace_outputs`` — UNION of the three sources, column-aligned.

Rollback
--------
DROP VIEW; drop the added columns. Existing deliverables rows untouched.

Revision ID: prd133b_outputs_view
Revises: None (standalone — alembic drift blocks linkage to prd133_deliverables_lifecycle)
Create Date: 2026-04-24
"""
from alembic import op

revision = "prd133b_outputs_view"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. Source-table columns for soft-delete
    # ------------------------------------------------------------------
    op.execute("""
        ALTER TABLE blog_posts
            ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
        ALTER TABLE agent_reports
            ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
    """)

    # Pagination-friendly indexes matching the view's default ORDER BY.
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_blog_posts_workspace_created
            ON blog_posts (workspace_id, created_at DESC)
         WHERE deleted_at IS NULL;

        CREATE INDEX IF NOT EXISTS ix_agent_reports_workspace_created
            ON agent_reports (workspace_id, created_at DESC)
         WHERE deleted_at IS NULL;
    """)

    # ------------------------------------------------------------------
    # 2. The unified read view
    # ------------------------------------------------------------------
    # Column order MUST match DeliverableService._row_to_dict consumption.
    # Every branch of the UNION must produce the exact same types or PG errors.
    op.execute("""
        CREATE OR REPLACE VIEW v_workspace_outputs AS
        -- ============== Blog posts ==============
        SELECT
            bp.id                                        AS id,
            bp.workspace_id                              AS workspace_id,
            'chat'::varchar                              AS source_type,
            bp.id::text                                  AS source_id,
            bp.author_agent_id                           AS agent_id,
            bp.author_name                               AS agent_name,
            'blog_post'::varchar                         AS artifact_type,
            bp.title                                     AS title,
            bp.excerpt                                   AS summary,
            'workspace'::varchar                         AS storage_type,
            bp.file_path                                 AS file_path,
            CASE WHEN bp.file_path IS NOT NULL
                 THEN regexp_replace(bp.file_path, '^.*/', '')
                 ELSE bp.slug || '.md'
            END                                          AS file_name,
            'md'::varchar                                AS file_type,
            NULL::bigint                                 AS file_size_bytes,
            CASE WHEN bp.file_path IS NOT NULL
                 THEN '/api/workspaces/' || bp.workspace_id::text ||
                      '/files/content?path=' || bp.file_path
                 ELSE NULL
            END                                          AS preview_url,
            'markdown'::varchar                          AS preview_type,
            jsonb_build_object(
                'slug',          bp.slug,
                'category',      bp.category,
                'tags',          COALESCE(to_jsonb(bp.tags), '[]'::jsonb),
                'published_at',  bp.published_at,
                'reading_time_minutes', bp.reading_time_minutes
            )                                            AS extra,
            CASE bp.status
                WHEN 'draft'     THEN 'draft'
                WHEN 'published' THEN 'published'
                WHEN 'scheduled' THEN 'review'
                WHEN 'archived'  THEN 'archived'
                ELSE 'published'
            END                                          AS status,
            bp.deleted_at                                AS deleted_at,
            bp.created_at::timestamptz                   AS created_at,
            bp.updated_at::timestamptz                   AS updated_at
        FROM blog_posts bp

        UNION ALL

        -- ============== Agent reports ==============
        SELECT
            ar.id                                        AS id,
            ar.workspace_id                              AS workspace_id,
            CASE
                WHEN ar.heartbeat_result_id IS NOT NULL   THEN 'heartbeat'
                WHEN ar.orchestration_task_id IS NOT NULL THEN 'task'
                ELSE 'chat'
            END                                          AS source_type,
            COALESCE(
                ar.heartbeat_result_id::text,
                ar.orchestration_task_id::text,
                ar.id::text
            )                                            AS source_id,
            ar.agent_id                                  AS agent_id,
            ar.agent_name                                AS agent_name,
            'report'::varchar                            AS artifact_type,
            ar.title                                     AS title,
            ar.summary                                   AS summary,
            'workspace'::varchar                         AS storage_type,
            ar.file_path                                 AS file_path,
            regexp_replace(ar.file_path, '^.*/', '')     AS file_name,
            ar.file_type                                 AS file_type,
            ar.file_size_bytes::bigint                   AS file_size_bytes,
            '/api/workspaces/' || ar.workspace_id::text ||
                '/files/content?path=' || ar.file_path   AS preview_url,
            'markdown'::varchar                          AS preview_type,
            COALESCE(ar.metrics, '{}'::jsonb) ||
                jsonb_build_object(
                    'report_type',     ar.report_type,
                    'grade',           ar.grade,
                    'grade_notes',     ar.grade_notes,
                    'attachments',     COALESCE(ar.attachments, '[]'::jsonb)
                )                                        AS extra,
            CASE ar.status
                WHEN 'ok'       THEN 'published'
                WHEN 'warning'  THEN 'review'
                WHEN 'error'    THEN 'archived'
                ELSE 'published'
            END                                          AS status,
            ar.deleted_at                                AS deleted_at,
            ar.created_at                                AS created_at,
            ar.updated_at                                AS updated_at
        FROM agent_reports ar

        UNION ALL

        -- ============== Ad-hoc artifacts from deliverables ==============
        -- Blog posts and reports are excluded so this branch never conflicts
        -- with the two above. Existing historical rows with those types stay
        -- dormant — they'll be removed when `deliverables` is renamed during
        -- the PRD-134 cleanup pass.
        SELECT
            d.id                                         AS id,
            d.workspace_id                               AS workspace_id,
            d.source_type                                AS source_type,
            d.source_id                                  AS source_id,
            d.agent_id                                   AS agent_id,
            d.agent_name                                 AS agent_name,
            d.artifact_type                              AS artifact_type,
            d.title                                      AS title,
            d.summary                                    AS summary,
            d.storage_type                               AS storage_type,
            d.file_path                                  AS file_path,
            d.file_name                                  AS file_name,
            d.file_type                                  AS file_type,
            d.file_size_bytes                            AS file_size_bytes,
            d.preview_url                                AS preview_url,
            d.preview_type                               AS preview_type,
            d.extra                                      AS extra,
            d.status                                     AS status,
            d.deleted_at                                 AS deleted_at,
            d.created_at                                 AS created_at,
            d.updated_at                                 AS updated_at
        FROM deliverables d
        WHERE d.artifact_type NOT IN ('blog_post', 'report');
    """)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS v_workspace_outputs;")
    op.execute("""
        DROP INDEX IF EXISTS ix_agent_reports_workspace_created;
        DROP INDEX IF EXISTS ix_blog_posts_workspace_created;
    """)
    op.execute("""
        ALTER TABLE agent_reports DROP COLUMN IF EXISTS deleted_at;
        ALTER TABLE blog_posts    DROP COLUMN IF EXISTS deleted_at;
    """)
