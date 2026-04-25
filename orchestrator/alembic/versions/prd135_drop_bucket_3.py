"""PRD-135 §12 — Bucket 3: drop 7 multi-agent reasoning experiment tables

Coordination/consensus features that were designed but never wired.
Agent-to-agent communication today goes through `messages` (alive: 150+
calls) and `board_tasks` (alive: 159+ calls). None of these 7 tables
have any inbound code edges or runtime calls in the 12-hour
pg_stat_statements observation window.

Tables dropped:
  agent_coordination
  multi_agent_reasoning
  agent_behavior_monitoring
  agent_performance
  collaboration_proposals
  consensus_votes
  message_broadcasts

All 7: zero rows, zero runtime hits, all FKs outbound (dead → live),
so dropping cannot break any live table.

Smoke test after drop: run a mission with ≥2 agents — verify they
hand off via the live path (messages + board_tasks).

Rollback: schema-only snapshot at
graphify-out/snapshots/bucket-3-pre-drop.sql captures column types,
constraints, and indexes. Re-creating these tables would NOT restore
data (all zero rows when dropped). The snapshot is the canonical
recovery artifact, NOT this migration's downgrade().

Revision ID: prd135_drop_bucket_3
Revises: None  (independent head, matches alembic convention in this repo)
Create Date: 2026-04-25
"""

from alembic import op


# revision identifiers, used by Alembic.
# (Short ID — alembic_version.version_num is varchar(32))
revision = "prd135_drop_bucket_3"
down_revision = None
branch_labels = None
depends_on = None


_MULTI_AGENT_TABLES = (
    "agent_coordination",
    "multi_agent_reasoning",
    "agent_behavior_monitoring",
    "agent_performance",
    "collaboration_proposals",
    "consensus_votes",
    "message_broadcasts",
)


def upgrade() -> None:
    # CASCADE drops outbound FK constraints with the table itself.
    # IF EXISTS makes the migration idempotent.
    for table in _MULTI_AGENT_TABLES:
        op.execute(f'DROP TABLE IF EXISTS public."{table}" CASCADE;')


def downgrade() -> None:
    # See module docstring: no-op by design. Restore from
    # graphify-out/snapshots/bucket-3-pre-drop.sql if recovery needed.
    pass
