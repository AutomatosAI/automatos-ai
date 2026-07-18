"""PRD-207: capture the caller's privilege tier on the mint row

``system_role`` lives ONLY in Clerk token claims (no users column) — the
custom-LLM webhook has no auth context, so a runtime lookup is impossible
(the #581 attempt crashed every spoken turn: silence). The mint, which DOES
hold the caller's session, now stamps ``is_super_admin`` onto the
``voice_calls`` row; the webhook reads row-truth like everything else on it.

Chains single-parent on prd207_voice_live (the current single head).

Revision ID: prd207_su_capture
Revises: prd207_voice_live
Create Date: 2026-07-18
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "prd207_su_capture"
down_revision = "prd207_voice_live"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "voice_calls",
        sa.Column(
            "is_super_admin", sa.Boolean, nullable=False, server_default=sa.text("false")
        ),
    )


def downgrade() -> None:
    op.drop_column("voice_calls", "is_super_admin")
