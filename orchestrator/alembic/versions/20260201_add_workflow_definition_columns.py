"""Add workflow definition JSONB columns to workflow_recipes table

Revision ID: 20260201_workflow_definition
Revises: 20260201_marketplace_recipes
Create Date: 2026-02-01

This migration adds workflow definition fields to the workflow_recipes table:
- steps: Array of step definitions with agent_id, prompt_template, error_handling
- inputs: Input schema (like a function signature)
- outputs: Expected output format
- execution_config: Runtime behavior (mode, retries, timeouts, quality_threshold)
- schedule_config: Scheduling configuration (type: manual/cron/trigger)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260201_workflow_definition'
down_revision = '20260201_marketplace_recipes'
branch_labels = None
depends_on = None


def upgrade():
    """Add workflow definition JSONB columns to workflow_recipes table."""

    # Steps: Array of step definitions
    # Each step: { step_number, agent_id, prompt_template, pass_to, error_handling, ... }
    op.add_column('workflow_recipes', sa.Column('steps', postgresql.JSONB(), nullable=True))

    # Inputs: Input schema definition (like a function signature)
    # Example: { "order_id": { "type": "string", "required": true }, ... }
    op.add_column('workflow_recipes', sa.Column('inputs', postgresql.JSONB(), nullable=True))

    # Outputs: Expected output format definition
    # Example: { "summary": { "type": "string" }, "score": { "type": "number" } }
    op.add_column('workflow_recipes', sa.Column('outputs', postgresql.JSONB(), nullable=True))

    # Execution config: Runtime behavior settings
    # Example: { "mode": "sequential", "max_retries": 3, "timeout_per_step": 300, ... }
    op.add_column('workflow_recipes', sa.Column('execution_config', postgresql.JSONB(), nullable=True))

    # Schedule config: How the recipe is triggered
    # Example: { "type": "manual" } or { "type": "cron", "cron_expression": "0 9 * * 1-5" }
    op.add_column('workflow_recipes', sa.Column('schedule_config', postgresql.JSONB(), nullable=True))


def downgrade():
    """Remove workflow definition JSONB columns from workflow_recipes table."""

    op.drop_column('workflow_recipes', 'schedule_config')
    op.drop_column('workflow_recipes', 'execution_config')
    op.drop_column('workflow_recipes', 'outputs')
    op.drop_column('workflow_recipes', 'inputs')
    op.drop_column('workflow_recipes', 'steps')
