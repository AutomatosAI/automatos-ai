"""PRD-142 Wave 4 (W4-S11): HARNESS structured store

Extends learning_outcomes with HARNESS OUTCOME fields (workspace_id, run_id,
change_type, risk_score, status, applied_at, rolled_back_at, current_value_before)
and adds the harness_prescriptions PRESCRIPTION table — the Role-2 (config /
diagnosis) learning store HARNESS dual-writes (W4-S12), replacing the flat
baseline-JSON records. §12.2.

Online-safe: only NULLABLE add_column (no table rewrite, no lock) + a brand-new
table. No backfill, no NOT NULL added to an existing table.

Standalone migration (down_revision = None) — the orchestrator alembic config has
MANY heads (~55), and this matches the established add-a-table convention here
(see prd142_wave0_error_events.py, add_job_title_to_agents.py). Prod apply is
HUMAN-GATED (W4-S2 sibling): resolve/merge heads as needed, and note this assumes
learning_outcomes already exists (it does — init_complete_schema.sql / create_all).

Revision ID: prd142_wave4_harness_store
Create Date: 2026-06-09
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID


revision = "prd142_wave4_harness_store"
down_revision = None
branch_labels = None
depends_on = None


# (column_name, type) — all added NULLABLE to learning_outcomes (online-safe).
_LEARNING_OUTCOME_COLS = [
    ("workspace_id", PGUUID(as_uuid=True)),
    ("run_id", sa.String(64)),
    ("change_type", sa.String(64)),
    ("risk_score", sa.Integer),
    ("status", sa.String(32)),
    ("applied_at", sa.DateTime),
    ("rolled_back_at", sa.DateTime),
    ("current_value_before", JSONB),
]


def upgrade() -> None:
    # 1. Extend learning_outcomes (the OUTCOME store) — all nullable, online-safe.
    for name, type_ in _LEARNING_OUTCOME_COLS:
        op.add_column("learning_outcomes", sa.Column(name, type_, nullable=True))
    op.create_index("idx_learning_outcomes_ws", "learning_outcomes", ["workspace_id"])
    op.create_index("idx_learning_outcomes_run", "learning_outcomes", ["run_id"])

    # 2. harness_prescriptions (the PRESCRIPTION store) — brand-new table, no FKs.
    op.create_table(
        "harness_prescriptions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("workspace_id", PGUUID(as_uuid=True), nullable=False),
        sa.Column("run_id", sa.String(64), nullable=True),
        sa.Column("prescription_id", sa.String(64), nullable=False),
        sa.Column("target_type", sa.String(32), nullable=True),
        sa.Column("target_id", sa.Integer, nullable=True),
        sa.Column("target_name", sa.String(255), nullable=True),
        sa.Column("change_type", sa.String(64), nullable=False),
        sa.Column("risk_score", sa.Integer, nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="proposed"),
        sa.Column("proposed_value", JSONB, nullable=True),
        sa.Column("current_value_before", JSONB, nullable=True),
        sa.Column("rationale", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_harness_rx_ws_created", "harness_prescriptions", ["workspace_id", "created_at"])
    op.create_index("idx_harness_rx_prescription_id", "harness_prescriptions", ["prescription_id"])


def downgrade() -> None:
    op.drop_index("idx_harness_rx_prescription_id", table_name="harness_prescriptions")
    op.drop_index("idx_harness_rx_ws_created", table_name="harness_prescriptions")
    op.drop_table("harness_prescriptions")
    op.drop_index("idx_learning_outcomes_run", table_name="learning_outcomes")
    op.drop_index("idx_learning_outcomes_ws", table_name="learning_outcomes")
    for name, _type in _LEARNING_OUTCOME_COLS:
        op.drop_column("learning_outcomes", name)
