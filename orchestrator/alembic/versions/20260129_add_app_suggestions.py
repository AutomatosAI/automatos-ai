"""Add app_suggestions column to composio_actions_cache for dynamic tool suggestions

Revision ID: 20260129_app_suggestions
Revises: 20260127_add_workspace_member_unique_constraint
Create Date: 2026-01-29

This migration adds the app_suggestions column to composio_actions_cache which stores
curated suggestion prompts for each Composio app. This enables:
- Dynamic, context-aware suggestion chips when users click tool icons
- Per-app curated prompts (e.g., "Summarize unread emails" for Gmail)
- Schema-driven fallback generation for uncurated apps
- Improved tool discoverability for users
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import json

# revision identifiers, used by Alembic.
revision = '20260129_app_suggestions'
down_revision = '20260127_workspace_member_uq'
branch_labels = None
depends_on = None


# Curated suggestions for popular apps
# Note: No {{placeholders}} - suggestions are sent directly to LLM as-is
INITIAL_SUGGESTIONS = {
    "GMAIL": [
        "Summarize unread emails from this morning",
        "Draft replies to urgent messages",
        "Find emails with attachments from last week",
        "Check my unread emails"
    ],
    "SLACK": [
        "Send a message to the team",
        "Summarize today's messages",
        "Check my unread DMs",
        "Show recent channel activity"
    ],
    "GITHUB": [
        "Show my open pull requests",
        "List issues assigned to me",
        "Check recent commits",
        "Show repository activity"
    ],
    "GOOGLECALENDAR": [
        "What's on my calendar today?",
        "Show my next meeting",
        "Find my next free slot this week",
        "Show meetings for tomorrow"
    ],
    "NOTION": [
        "Search my Notion workspace",
        "Show recent updates",
        "List all my pages",
        "Show recent notes"
    ]
}


def upgrade():
    """Add app_suggestions column and populate initial curated suggestions."""
    # Add column to composio_actions_cache table
    op.add_column(
        'composio_actions_cache',
        sa.Column('app_suggestions', postgresql.JSONB, server_default='[]')
    )

    # Populate initial suggestions for popular apps
    # Note: We update based on app_name, and all actions for an app get the same suggestions
    # This is intentional - suggestions are app-level, not action-level
    conn = op.get_bind()
    for app_name, suggestions in INITIAL_SUGGESTIONS.items():
        conn.execute(
            sa.text("""
                UPDATE composio_actions_cache
                SET app_suggestions = :suggestions
                WHERE app_name = :app_name
            """),
            {"suggestions": json.dumps(suggestions), "app_name": app_name}
        )


def downgrade():
    """Remove app_suggestions column."""
    op.drop_column('composio_actions_cache', 'app_suggestions')
