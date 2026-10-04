"""Deliverables: a CLI agent's heartbeat report is a heartbeat, not a chat report.

A Claude Code (CLI) agent's heartbeat files a board ticket, "Heartbeat: <agent>"
(``services/heartbeat_service.py``). Its completion writes the generic task report
"Task: Heartbeat: <agent>" (``api/board_tasks.py``), which carries neither a
``heartbeat_result_id`` nor an ``orchestration_task_id``. So ``v_workspace_outputs``
classed it as ``chat``, and the feed's ``source_type_exclude=heartbeat`` never hid it:
1,333 of them buried the real reports in c1 (4 Oct 2026).

* The writer now files such a report with ``report_type='heartbeat'``.
* The view classes ``report_type='heartbeat'`` as source ``heartbeat``. It is recreated
  with ``CREATE OR REPLACE`` from prd133b's body, with that one rule added; every column
  is unchanged.
* The existing ones are re-typed, matched by a linked board ticket whose
  ``source_type`` is ``heartbeat`` (never by title). This is idempotent.

Create_all-first safe: the data step runs only when both tables exist.

Revision ID: outputs_heartbeat_reports
Revises: prd251c_wave4
Create Date: 2026-10-04
"""
from __future__ import annotations

from alembic import op

revision = "outputs_heartbeat_reports"
down_revision = "prd251c_wave4"
branch_labels = None
depends_on = None

# The view as prd133b built it, with the report_type rule added.
VIEW_SQL = """
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
                WHEN ar.report_type = 'heartbeat'         THEN 'heartbeat'
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
"""

# prd133b's view, unchanged, for the downgrade.
PRIOR_VIEW_SQL = """
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
"""

# A report linked to a heartbeat ticket, re-typed between ``task`` and ``heartbeat``.
_RETYPE = """
DO $$
BEGIN
    IF to_regclass('agent_reports') IS NOT NULL AND to_regclass('board_tasks') IS NOT NULL THEN
        UPDATE agent_reports ar
           SET report_type = '{to}'
         WHERE ar.report_type = '{frm}'
           AND jsonb_typeof(ar.linked_task_ids) = 'array'
           AND EXISTS (
                SELECT 1 FROM board_tasks bt
                 WHERE bt.workspace_id = ar.workspace_id
                   AND bt.source_type = 'heartbeat'
                   AND ar.linked_task_ids @> to_jsonb(bt.id));
    END IF;
END $$;
"""


def upgrade() -> None:
    op.execute(VIEW_SQL)
    op.execute(_RETYPE.format(frm="task", to="heartbeat"))


def downgrade() -> None:
    op.execute(_RETYPE.format(frm="heartbeat", to="task"))
    op.execute(PRIOR_VIEW_SQL)
