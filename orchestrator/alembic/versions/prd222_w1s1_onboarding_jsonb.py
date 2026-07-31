"""PRD-222 W1S1: workspaces.onboarding JSONB — Auto-led onboarding state + funnel

Adds a single JSONB column ``onboarding`` (server_default ``'{}'``) to the
``workspaces`` table. This is the ONE migration for the entire PRD-222 Wave-1
branch: every later schema need — the $5 trial ledger (W1S9), the segment
answers, and the per-stage funnel timestamps — lives INSIDE this one JSONB
document. No second migration, no new table.

The document shape is owned by ``services/onboarding_state.py``:
``{stage, stages: {<stage>: iso}, segment: {business, goal, comfort},
started_at, updated_at, completed_at, trial: {...}}``.

Chains single-parent on prd207_su_capture (the current single head).

Revision ID: prd222_w1s1_onboarding_jsonb
Revises: prd207_su_capture
Create Date: 2026-07-31
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "prd222_w1s1_onboarding_jsonb"
down_revision = "prd207_su_capture"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "workspaces",
        sa.Column(
            "onboarding",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("workspaces", "onboarding")
