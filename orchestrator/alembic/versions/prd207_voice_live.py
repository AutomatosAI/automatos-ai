"""PRD-207 S3/S4: voice_calls lifecycle table + voice settings plane

* ``voice_calls`` — one row per live call, BORN at mint (the call_id →
  workspace/user/chat registry the S2 trust boundary and the S4 cap gate key
  off); Retell lifecycle events stamp start/end/duration idempotently.
* ``voice_turns.call_id`` — links per-turn telemetry to its call (S8 parity).
* Seeds the ``voice`` system-settings rows (category ``voice``): the
  ``live_enabled`` master toggle (default OFF — merging never arms
  microphones or billing) and the three masked Retell credentials, so arming
  is a super-admin Settings-page act (PRD-143: DB settings, never env).

Chains single-parent on prd206_chat_summary (the current single head).

Revision ID: prd207_voice_live
Revises: prd206_chat_summary
Create Date: 2026-07-17
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID as PGUUID

revision = "prd207_voice_live"
down_revision = "prd206_chat_summary"
branch_labels = None
depends_on = None


_VOICE_SETTINGS_SEED = (
    {
        "key": "live_enabled",
        "value": "false",
        "value_type": "boolean",
        "description": (
            "Auto Live master switch (PRD-207). OFF = no web-call minting "
            "platform-wide. Flip from the Settings page to arm/disarm live "
            "voice — no redeploy."
        ),
        "is_sensitive": False,
        "is_required": True,
        "default_value": "false",
    },
    {
        "key": "retell_api_key",
        "value": "",
        "value_type": "string",
        "description": (
            "Retell API key — used server-side to mint web-call tokens. "
            "Never reaches the browser."
        ),
        "is_sensitive": True,
        "is_required": False,
        "default_value": "",
    },
    {
        "key": "retell_webhook_secret",
        "value": "",
        "value_type": "string",
        "description": (
            "Signing key for inbound Retell webhooks (x-retell-signature). "
            "Retell signs with your API key unless you configure a distinct "
            "secret — paste the matching value; empty = webhooks fail closed."
        ),
        "is_sensitive": True,
        "is_required": False,
        "default_value": "",
    },
    {
        "key": "retell_agent_id",
        "value": "",
        "value_type": "string",
        "description": "Retell agent id Auto Live mints web calls against.",
        "is_sensitive": False,
        "is_required": False,
        "default_value": "",
    },
)


def upgrade() -> None:
    op.create_table(
        "voice_calls",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("call_id", sa.String(128), nullable=False),
        sa.Column("provider", sa.String(20), nullable=False, server_default="retell"),
        sa.Column("workspace_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("user_id", sa.Integer, nullable=True),
        sa.Column("chat_id", sa.String(64), nullable=True),
        sa.Column("fallback_chat_id", sa.String(64), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="minted"),
        sa.Column("minted_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime, nullable=True),
        sa.Column("ended_at", sa.DateTime, nullable=True),
        sa.Column("duration_seconds", sa.Integer, nullable=True),
        sa.Column("disconnect_reason", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "idx_voice_calls_workspace_minted", "voice_calls", ["workspace_id", "minted_at"]
    )
    op.create_index("uq_voice_calls_call_id", "voice_calls", ["call_id"], unique=True)

    op.add_column("voice_turns", sa.Column("call_id", sa.String(128), nullable=True))

    # Seed the voice settings rows insert-if-absent (system_settings has no
    # (category, key) unique constraint — the API guards manually, so do we).
    conn = op.get_bind()
    for row in _VOICE_SETTINGS_SEED:
        exists = conn.execute(
            sa.text(
                "SELECT 1 FROM system_settings WHERE category = 'voice' AND key = :key"
            ),
            {"key": row["key"]},
        ).first()
        if exists:
            continue
        conn.execute(
            sa.text(
                "INSERT INTO system_settings "
                "(category, key, value, value_type, description, is_sensitive, "
                " is_required, default_value, created_by) "
                "VALUES ('voice', :key, :value, :value_type, :description, "
                "        :is_sensitive, :is_required, :default_value, 'prd207')"
            ),
            row,
        )


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text("DELETE FROM system_settings WHERE category = 'voice' AND created_by = 'prd207'")
    )
    op.drop_column("voice_turns", "call_id")
    op.drop_index("uq_voice_calls_call_id", table_name="voice_calls")
    op.drop_index("idx_voice_calls_workspace_minted", table_name="voice_calls")
    op.drop_table("voice_calls")
