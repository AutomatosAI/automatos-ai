"""Backfill base models into every zero-model workspace (Harbourline fix).

2026-08-29 (Gerard): a workspace with ZERO ``workspace_models`` rows breaks
Settings → Orchestrator ("Failed to load LLM settings") and leaves chat with no
sane default. New workspaces are seeded at provisioning from this deploy on
(``services/workspace_model_seeding.py``); this migration brings every EXISTING
zero-model workspace to the same floor: up to 4 active OpenRouter-served base
models (defaults first, then featured by popularity), ``source='default'``,
``approval_status='approved'``, plus ``settings.orchestrator.model`` set to the
top pick where no primary exists.

Idempotent (NOT EXISTS guards; second run matches nothing). Additive DML only —
no DDL, no constraint assumptions (the ``workspaces_plan_check`` lesson).

Revision ID: prd_workspace_models_backfill
Revises: prd230_marketplace_packages
Create Date: 2026-08-29
"""
from __future__ import annotations

from alembic import op

revision = "prd_workspace_models_backfill"
down_revision = "prd230_marketplace_packages"
branch_labels = None
depends_on = None

_BASE_PICKS = """
CREATE TEMPORARY TABLE _base_model_picks ON COMMIT DROP AS
SELECT id, model_id,
       ROW_NUMBER() OVER (
         ORDER BY is_default DESC, is_featured DESC,
                  COALESCE(popularity_score, 0) DESC, id ASC
       ) AS rank
FROM llm_models
WHERE status = 'active'
  AND (provider = 'openrouter' OR tier = 'openrouter' OR model_id LIKE '%/%')
LIMIT 50
"""

_BACKFILL_ROWS = """
INSERT INTO workspace_models (workspace_id, model_id, is_active, source, approval_status)
SELECT w.id, p.id, TRUE, 'default', 'approved'
FROM workspaces w
CROSS JOIN _base_model_picks p
WHERE p.rank <= 4
  AND NOT EXISTS (
    SELECT 1 FROM workspace_models wm WHERE wm.workspace_id = w.id
  )
"""

_SET_PRIMARY = """
UPDATE workspaces w
SET settings = jsonb_set(
      COALESCE(w.settings, '{}'::jsonb),
      '{orchestrator}',
      COALESCE(w.settings->'orchestrator', '{}'::jsonb)
        || jsonb_build_object(
             'model',
             (SELECT p.model_id FROM _base_model_picks p WHERE p.rank = 1)
           ),
      true
    )
WHERE (w.settings->'orchestrator'->>'model') IS NULL
  AND EXISTS (SELECT 1 FROM _base_model_picks)
"""


def upgrade() -> None:
    op.execute(_BASE_PICKS)
    op.execute(_BACKFILL_ROWS)
    op.execute(_SET_PRIMARY)


def downgrade() -> None:
    # Remove only rows this backfill could have created (source='default') from
    # workspaces whose ONLY rows are defaults — hand-installed rows untouched.
    op.execute(
        """
        DELETE FROM workspace_models wm
        WHERE wm.source = 'default'
          AND NOT EXISTS (
            SELECT 1 FROM workspace_models o
            WHERE o.workspace_id = wm.workspace_id AND o.source <> 'default'
          )
        """
    )
