"""PRD-223 Wave 0 — chat-scale tool-iteration budget.

Reseeds chatbot.max_tool_iterations 25 → 8 on existing deployments, but ONLY
where the value is still the old default — an operator-tuned value is left
alone (the seed only ever updates metadata, never values, so a data migration
is the one honest way to move a shipped default).

The model_policy settings rows themselves are created by the boot seed
(insert-if-missing), not here.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "prd223_w0_model_policy"
down_revision = "prd207_su_capture"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "UPDATE system_settings SET value = '8', default_value = '8' "
        "WHERE category = 'chatbot' AND key = 'max_tool_iterations' "
        "AND value = '25'"
    )


def downgrade():
    op.execute(
        "UPDATE system_settings SET value = '25', default_value = '25' "
        "WHERE category = 'chatbot' AND key = 'max_tool_iterations' "
        "AND value = '8'"
    )
