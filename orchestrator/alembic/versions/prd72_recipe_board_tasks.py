"""PRD-72: Recipe Board Tasks

Add source_type and source_id columns to board_tasks so recipe executions
auto-create kanban tasks that flow through the board lifecycle.
"""

from alembic import op
import sqlalchemy as sa

revision = "prd72_recipe_board_tasks"
down_revision = None  # Standalone — safe to run anytime
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('board_tasks', sa.Column('source_type', sa.String(30), nullable=False, server_default='user'))
    op.add_column('board_tasks', sa.Column('source_id', sa.String(255), nullable=True))

    op.create_index('ix_board_tasks_source', 'board_tasks', ['source_type', 'source_id'])
    op.execute(
        "CREATE UNIQUE INDEX uq_board_tasks_recipe_exec "
        "ON board_tasks(source_id) WHERE source_type = 'recipe'"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_board_tasks_recipe_exec")
    op.drop_index('ix_board_tasks_source', table_name='board_tasks')
    op.drop_column('board_tasks', 'source_id')
    op.drop_column('board_tasks', 'source_type')
