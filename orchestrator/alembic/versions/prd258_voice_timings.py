"""PRD-258 Wave 1: where a voice turn's time went, and Retell's own latency (11 Oct 2026).

Auto Live is "just slow" and nothing stores a number: ``voice_turns.total_ms``
stops the clock at the END of the reply, and the time to the first word is only
logged. The ONE migration of the wave (D2) adds every column it measures with:

* ``voice_turns.first_frame_ms`` / ``first_auto_frame_ms``: response_required to
  the first content frame sent (the ack counts) and to Auto's own first frame;
* ``voice_turns.prep_ms``: context assembly, copied from the reply's context_trace;
* ``voice_turns.tools_before_speech``: tool calls Auto made before it first spoke;
* ``voice_turns.lead_in`` / ``prefetch_hit``: false until PRD-258 US-008 / US-009;
* ``voice_calls.vendor_latency``: Retell's per-call p50/p90/num (US-002).

The integers and the JSONB are nullable with no default, so every existing row
reads "not measured". ``IF NOT EXISTS`` keeps the upgrade a no-op on a database
that ``create_all`` built from the updated models.

Revision ID: prd258_voice_timings
Revises: user_api_keys_provider_workspace_id
Create Date: 2026-10-11
"""

from alembic import op


revision = "prd258_voice_timings"
down_revision = "user_api_keys_provider_workspace_id"
branch_labels = None
depends_on = None

_VOICE_TURN_COLUMNS = (
    ("first_frame_ms", "INTEGER"),
    ("first_auto_frame_ms", "INTEGER"),
    ("prep_ms", "INTEGER"),
    ("tools_before_speech", "INTEGER"),
    ("lead_in", "BOOLEAN NOT NULL DEFAULT false"),
    ("prefetch_hit", "BOOLEAN NOT NULL DEFAULT false"),
)


def upgrade() -> None:
    for name, ddl in _VOICE_TURN_COLUMNS:
        op.execute(f"ALTER TABLE voice_turns ADD COLUMN IF NOT EXISTS {name} {ddl};")
    op.execute("ALTER TABLE voice_calls ADD COLUMN IF NOT EXISTS vendor_latency JSONB;")


def downgrade() -> None:
    op.execute("ALTER TABLE voice_calls DROP COLUMN IF EXISTS vendor_latency;")
    for name, _ddl in reversed(_VOICE_TURN_COLUMNS):
        op.execute(f"ALTER TABLE voice_turns DROP COLUMN IF EXISTS {name};")
