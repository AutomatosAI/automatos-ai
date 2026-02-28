"""Add workspace persistence columns for widget layouts (US-001)

Revision ID: 20260225_ws_persistence
Revises: 20260224_semantic_routing
Create Date: 2026-02-25

Adds to workspaces table:
- layout          JSONB   DEFAULT '{"columns":12,"rowHeight":100}'
- layout_mode     VARCHAR(20) DEFAULT 'grid' CHECK IN ('grid','freeform','split','focus')
- widgets         JSONB   DEFAULT '[]'
- description     TEXT
- is_template     BOOLEAN DEFAULT FALSE
- template_category VARCHAR(50)
- template_icon   VARCHAR(10)
- visibility      VARCHAR(20) DEFAULT 'private' CHECK IN ('private','team','organization')
- last_opened_at  TIMESTAMP WITH TIME ZONE
- Partial index idx_workspaces_template on is_template WHERE is_template = TRUE
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260225_ws_persistence'
down_revision = '20260224_semantic_routing'
branch_labels = None
depends_on = None

TABLE = 'workspaces'


def upgrade() -> None:
    # -- layout & widgets --
    op.add_column(
        TABLE,
        sa.Column(
            'layout',
            postgresql.JSONB(),
            server_default='{"columns":12,"rowHeight":100}',
            nullable=False,
        ),
    )
    op.add_column(
        TABLE,
        sa.Column(
            'layout_mode',
            sa.String(20),
            server_default='grid',
            nullable=False,
        ),
    )
    op.add_column(
        TABLE,
        sa.Column(
            'widgets',
            postgresql.JSONB(),
            server_default='[]',
            nullable=False,
        ),
    )
    op.add_column(TABLE, sa.Column('description', sa.Text(), nullable=True))

    # -- template fields --
    op.add_column(
        TABLE,
        sa.Column('is_template', sa.Boolean(), server_default='false', nullable=False),
    )
    op.add_column(
        TABLE,
        sa.Column('template_category', sa.String(50), nullable=True),
    )
    op.add_column(
        TABLE,
        sa.Column('template_icon', sa.String(10), nullable=True),
    )

    # -- visibility --
    op.add_column(
        TABLE,
        sa.Column(
            'visibility',
            sa.String(20),
            server_default='private',
            nullable=False,
        ),
    )

    # -- last_opened_at --
    op.add_column(
        TABLE,
        sa.Column('last_opened_at', sa.DateTime(timezone=True), nullable=True),
    )

    # -- CHECK constraints --
    op.create_check_constraint(
        'ck_workspaces_layout_mode',
        TABLE,
        "layout_mode IN ('grid', 'freeform', 'split', 'focus')",
    )
    op.create_check_constraint(
        'ck_workspaces_visibility',
        TABLE,
        "visibility IN ('private', 'team', 'organization')",
    )

    # -- Partial index on is_template --
    op.create_index(
        'idx_workspaces_template',
        TABLE,
        ['is_template'],
        postgresql_where=sa.text('is_template = TRUE'),
    )


def downgrade() -> None:
    op.drop_index('idx_workspaces_template', table_name=TABLE)
    op.drop_constraint('ck_workspaces_visibility', TABLE, type_='check')
    op.drop_constraint('ck_workspaces_layout_mode', TABLE, type_='check')

    op.drop_column(TABLE, 'last_opened_at')
    op.drop_column(TABLE, 'visibility')
    op.drop_column(TABLE, 'template_icon')
    op.drop_column(TABLE, 'template_category')
    op.drop_column(TABLE, 'is_template')
    op.drop_column(TABLE, 'description')
    op.drop_column(TABLE, 'widgets')
    op.drop_column(TABLE, 'layout_mode')
    op.drop_column(TABLE, 'layout')
