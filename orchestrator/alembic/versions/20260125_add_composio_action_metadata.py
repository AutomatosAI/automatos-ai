"""Add composio_action_metadata table for capability-based action filtering

Revision ID: 20260125_composio_metadata
Revises: 20260120_add_cached_metadata
Create Date: 2026-01-25

This migration creates the composio_action_metadata table which stores
LLM-classified capability tags for Composio actions. This enables:
- Sync-time classification (classify once, use forever)
- Runtime capability filtering (no API calls)
- Deterministic action eligibility based on intent
- Separation of semantic capabilities from safety/risk policies
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260125_composio_metadata'
down_revision = '20260120_add_cached_metadata'
branch_labels = None
depends_on = None


def upgrade():
    # Create composio_action_metadata table
    op.create_table(
        'composio_action_metadata',

        # Primary key
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),

        # Identity - unique action identification
        sa.Column('app_id', sa.String(255), nullable=False, index=True,
                  comment='Composio app identifier (e.g., "slack", "github", "gmail")'),
        sa.Column('action_id', sa.String(512), nullable=False, unique=True,
                  comment='Full Composio action ID (e.g., "SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL")'),
        sa.Column('action_name_normalized', sa.String(255), nullable=True,
                  comment='LLM-friendly normalized name (e.g., "slack_send_message")'),

        # SEMANTIC CAPABILITIES (what the action can do)
        sa.Column('capabilities', postgresql.ARRAY(sa.String), nullable=False, default=[],
                  comment='Capability tags like ["message.send", "channel.write"]'),
        sa.Column('intent_keywords', postgresql.ARRAY(sa.String), nullable=True, default=[],
                  comment='Keywords users might say: ["send", "post", "message"]'),

        # SAFETY POLICY (separate from capabilities per GPT-5.2 recommendation)
        sa.Column('risk_level', sa.String(50), nullable=False, default='safe',
                  comment='Risk classification: "safe", "cautious", "dangerous"'),
        sa.Column('destructive', sa.Boolean, nullable=False, default=False,
                  comment='True for delete/remove/revoke actions'),
        sa.Column('requires_confirmation', sa.Boolean, nullable=False, default=False,
                  comment='True for actions that need explicit user confirmation'),

        # Parameter metadata (for validation)
        sa.Column('required_params', postgresql.JSONB, nullable=True,
                  comment='Required parameters: ["channel", "message"]'),
        sa.Column('optional_params', postgresql.JSONB, nullable=True,
                  comment='Optional parameters: ["thread_ts", "attachments"]'),

        # Descriptions
        sa.Column('description_short', sa.String(255), nullable=True,
                  comment='LLM-summarized description, <100 chars'),
        sa.Column('description_original', sa.Text, nullable=True,
                  comment='Original description from Composio'),

        # Classification tracking
        sa.Column('classification_method', sa.String(50), nullable=False, default='heuristic',
                  comment='How this was classified: "heuristic" or "llm"'),
        sa.Column('classification_confidence', sa.Float, nullable=True,
                  comment='Confidence score: 1.0 for heuristic, model confidence for LLM'),
        sa.Column('classification_model', sa.String(100), nullable=True,
                  comment='LLM model used if classified by LLM'),
        sa.Column('manually_overridden', sa.Boolean, nullable=False, default=False,
                  comment='True if admin manually corrected classification'),

        # Sync tracking
        sa.Column('composio_hash', sa.String(64), nullable=True,
                  comment='Hash of original action spec for change detection'),
        sa.Column('synced_at', sa.DateTime, nullable=True,
                  comment='Last sync from Composio'),
        sa.Column('classified_at', sa.DateTime, nullable=True,
                  comment='When classification was performed'),

        # Standard timestamps
        sa.Column('created_at', sa.DateTime, nullable=False,
                  server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime, nullable=False,
                  server_default=sa.text('NOW()'),
                  onupdate=sa.text('NOW()')),
    )

    # Create indexes for common queries

    # Index for filtering by app
    op.create_index(
        'ix_composio_action_metadata_app_id',
        'composio_action_metadata',
        ['app_id']
    )

    # Index for capability-based filtering (GIN index for array overlap queries)
    op.execute("""
        CREATE INDEX ix_composio_action_metadata_capabilities
        ON composio_action_metadata USING GIN (capabilities)
    """)

    # Index for filtering safe/destructive actions
    op.create_index(
        'ix_composio_action_metadata_safety',
        'composio_action_metadata',
        ['destructive', 'risk_level']
    )

    # Composite index for common runtime query pattern
    op.create_index(
        'ix_composio_action_metadata_runtime_filter',
        'composio_action_metadata',
        ['app_id', 'destructive']
    )

    # Index for sync operations
    op.create_index(
        'ix_composio_action_metadata_sync',
        'composio_action_metadata',
        ['synced_at', 'manually_overridden']
    )


def downgrade():
    # Drop indexes first
    op.drop_index('ix_composio_action_metadata_sync')
    op.drop_index('ix_composio_action_metadata_runtime_filter')
    op.drop_index('ix_composio_action_metadata_safety')
    op.execute('DROP INDEX ix_composio_action_metadata_capabilities')
    op.drop_index('ix_composio_action_metadata_app_id')

    # Drop table
    op.drop_table('composio_action_metadata')
