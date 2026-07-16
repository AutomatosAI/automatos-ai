"""PRD-204 S1: watch registry — watches + watch_events

A Watch is a first-class, workspace-scoped row supervising one launched unit
of work (mission / playbook execution / scheduled playbook) from launch to a
verdict. watch_events is the append-only observation log with an idempotency
key per observation.

Invariants installed here:
- CHECK constraint on watches.status (9-value lifecycle)
- partial UNIQUE index: one NON-TERMINAL watch per
  (workspace_id, target_type, target_id)
- UNIQUE(watch_id, event_key) on watch_events for idempotent ingest
- partial index on next_check_at for the watcher tick's SKIP LOCKED claim

Additive only. Chains on prd204_merge_heads (the single head after merging
the prd201/prd203 lineages).

Revision ID: prd204_watch_registry
Revises: prd204_merge_heads
Create Date: 2026-07-16
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID

revision = "prd204_watch_registry"
down_revision = "prd204_merge_heads"
branch_labels = None
depends_on = None

# Frozen snapshots of core/models/watch_enums.py at this revision.
_WATCH_STATUSES = (
    "watching",
    "acting",
    "awaiting_approval",
    "needs_attention",
    "passed",
    "failed",
    "escalated",
    "expired",
    "cancelled",
)
_TERMINAL_STATUSES_SQL = "'cancelled', 'escalated', 'expired', 'failed', 'passed'"


def upgrade() -> None:
    op.create_table(
        "watches",
        sa.Column(
            "id",
            PGUUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "workspace_id",
            PGUUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column(
            "owner_agent_id",
            sa.Integer,
            sa.ForeignKey("agents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("watch_type", sa.String(32), nullable=False),
        sa.Column("target_type", sa.String(32), nullable=False),
        sa.Column("target_id", sa.String(255), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column(
            "status", sa.String(32), nullable=False, server_default="watching"
        ),
        sa.Column("success_criteria", sa.Text, nullable=True),
        sa.Column("failure_criteria", sa.Text, nullable=True),
        sa.Column(
            "quality_threshold", sa.Float, nullable=False, server_default="0.8"
        ),
        sa.Column(
            "check_interval_seconds",
            sa.Integer,
            nullable=False,
            server_default="300",
        ),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_check_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "policy", sa.String(32), nullable=False, server_default="run_and_report"
        ),
        sa.Column("allowed_actions", JSONB, nullable=True),
        sa.Column("action_budget", sa.Integer, nullable=False, server_default="2"),
        sa.Column("actions_taken", sa.Integer, nullable=False, server_default="0"),
        sa.Column("final_score", sa.Float, nullable=True),
        sa.Column("final_verdict", sa.Text, nullable=True),
        sa.Column("lineage", JSONB, nullable=False, server_default=sa.text("'[]'")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version_id", sa.Integer, nullable=False, server_default="1"),
        sa.CheckConstraint(
            "status IN ({})".format(", ".join(repr(s) for s in _WATCH_STATUSES)),
            name="ck_watches_status",
        ),
    )
    op.create_index("ix_watches_workspace_id", "watches", ["workspace_id"])
    # One non-terminal watch per (workspace_id, target_type, target_id).
    op.create_index(
        "uq_watches_live_target",
        "watches",
        ["workspace_id", "target_type", "target_id"],
        unique=True,
        postgresql_where=sa.text(
            f"status NOT IN ({_TERMINAL_STATUSES_SQL})"
        ),
    )
    # Tick claim path: due, claimable watches only.
    op.create_index(
        "ix_watches_due",
        "watches",
        ["next_check_at"],
        postgresql_where=sa.text("status IN ('watching', 'acting')"),
    )

    op.create_table(
        "watch_events",
        sa.Column(
            "id",
            PGUUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "watch_id",
            PGUUID(as_uuid=True),
            sa.ForeignKey("watches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("snapshot", JSONB, nullable=True),
        sa.Column("score", sa.Float, nullable=True),
        sa.Column("action_taken", sa.String(100), nullable=True),
        sa.Column(
            "requires_attention",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("event_key", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("watch_id", "event_key", name="uq_watch_events_key"),
    )
    op.create_index("ix_watch_events_watch_id", "watch_events", ["watch_id"])
    op.create_index(
        "ix_watch_events_watch_created",
        "watch_events",
        ["watch_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_watch_events_watch_created", table_name="watch_events")
    op.drop_index("ix_watch_events_watch_id", table_name="watch_events")
    op.drop_table("watch_events")
    op.drop_index("ix_watches_due", table_name="watches")
    op.drop_index("uq_watches_live_target", table_name="watches")
    op.drop_index("ix_watches_workspace_id", table_name="watches")
    op.drop_table("watches")
