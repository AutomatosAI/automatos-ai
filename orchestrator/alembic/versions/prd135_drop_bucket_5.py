"""PRD-135 §12 — Bucket 5: drop 15 built-but-never-wired admin features

Surfaces designed in the platform's expansion phase but never delivered or
replaced by alternatives. None have any inbound code edges or runtime
calls in the 12-hour pg_stat_statements observation window.

Tables dropped:
  dashboard_configs              — widgets are config-less today (auto-discovery)
  custom_metrics                 — not yet implemented
  alert_configs                  — not yet implemented
  compliance_events              — not yet implemented
  marketplace_submissions        — no submission flow live
  knowledge_collections          — replaced by documents + kb_types
  knowledge_collection_items     — same
  knowledge_usage                — same
  usage_logs                     — replaced by llm_usage (alive: 534+ calls)
  usage_summary                  — same
  search_analytics               — not implemented
  execution_contexts             — workflow-era; PRD-125 supersedes
  integration_analysis           — not implemented
  workspace_shares               — no sharing flow live
  api_keys                       — inbound-API-key feature (never wired);
                                  LLM provider keys are stored in `credentials`
                                  (alive: 256+ calls); confirmed via Settings →
                                  API/Credentials UI working post-cleanup

All 15: zero rows, zero runtime hits, all FKs outbound (dead → live),
so dropping cannot break any live table.

Smoke test after drop: marketplace browse, knowledge bases page,
billing/usage page, settings pages.

Rollback: schema-only snapshot at
graphify-out/snapshots/bucket-5-pre-drop.sql captures column types,
constraints, and indexes. Re-creating these tables would NOT restore
data (all zero rows when dropped). The snapshot is the canonical
recovery artifact, NOT this migration's downgrade().

Revision ID: prd135_drop_bucket_5
Revises: None  (independent head, matches alembic convention in this repo)
Create Date: 2026-04-25
"""

from alembic import op


# revision identifiers, used by Alembic.
# (Short ID — alembic_version.version_num is varchar(32))
revision = "prd135_drop_bucket_5"
down_revision = None
branch_labels = None
depends_on = None


_ADMIN_FEATURE_TABLES = (
    "dashboard_configs",
    "custom_metrics",
    "alert_configs",
    "compliance_events",
    "marketplace_submissions",
    "knowledge_collections",
    "knowledge_collection_items",
    "knowledge_usage",
    "usage_logs",
    "usage_summary",
    "search_analytics",
    "execution_contexts",
    "integration_analysis",
    "workspace_shares",
    "api_keys",
)


def upgrade() -> None:
    # CASCADE drops outbound FK constraints with the table itself.
    # IF EXISTS makes the migration idempotent.
    for table in _ADMIN_FEATURE_TABLES:
        op.execute(f'DROP TABLE IF EXISTS public."{table}" CASCADE;')


def downgrade() -> None:
    # See module docstring: no-op by design. Restore from
    # graphify-out/snapshots/bucket-5-pre-drop.sql if recovery needed.
    pass
