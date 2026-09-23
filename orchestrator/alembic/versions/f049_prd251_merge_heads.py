"""Join the night-fix and PRD-251 lineages back into one head.

Both start at ``kb_multimodal_tables``:

    - llm_usage_agent_name -> f125_playbook_timeouts_seconds
                            (F049 then F125, the customer-night fixes on
                             test/customer-night since 2026-09-22)
    - prd251_socials        (PRD-251 Socials Wave 0, cut from main before F049)

With both in one tree, ``alembic heads`` returns two revisions and the from-zero
"exactly one head" bar (test_prd209_alembic_single_head) fails. This is a
**merge revision only**, with no schema operations (mirrors w3_post201_merge_heads).

When the night fixes and PRD-251 both reach main, carry this file with the same revision id
instead of writing a new merge revision: a database built from test/customer-night
already has it in alembic_version.

Revision ID: f049_prd251_merge_heads
Revises: f125_playbook_timeouts_seconds, prd251_socials
Create Date: 2026-09-23
"""

# A pure merge point — no operations.
revision = "f049_prd251_merge_heads"
down_revision = (
    "f125_playbook_timeouts_seconds",
    "prd251_socials",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    """No-op: this revision only merges lineages."""


def downgrade() -> None:
    """No-op: splitting back into two heads needs no schema change."""
