"""Merge heads: observability_evidence and prd008a_widget_event_log

Revision ID: 20260520_merge
Revises: observability_evidence, prd008a_widget_event_log
Create Date: 2026-05-20

The `observability_evidence` migration was committed on a hotfix that
branched from `prd140_team_lead_enabled`. Meanwhile main had already
advanced along the `prd008a_*` chain (sites + widget_event_log),
leaving the migration graph with two parallel heads sharing an
ancestor. `alembic upgrade heads` errored with:

  Requested revision observability_evidence overlaps with other
  requested revisions prd140_team_lead_enabled

This no-op merge declares the join point so alembic resolves to a
single head again.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20260520_merge'
down_revision = ('observability_evidence', 'prd008a_widget_event_log')
branch_labels = None
depends_on = None


def upgrade():
    """No-op merge migration."""
    pass


def downgrade():
    """No-op merge migration."""
    pass
