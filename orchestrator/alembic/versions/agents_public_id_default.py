"""Make agents.public_id auto-populate at the DB level.

The ORM column carries ``default=uuid4`` (Python-side), but any raw-SQL
INSERT (seed scripts, ad-hoc maintenance, future migrations) bypasses
that default and inserts NULL — which then breaks the agent details
modal and any widget that keys off the UUID.

Backfill any stragglers, then add ``DEFAULT gen_random_uuid()`` so the
column is impossible to leave NULL regardless of the insert path.
"""

from alembic import op
from sqlalchemy import text

revision = "agents_public_id_default"
down_revision = "expand_ws_member_role_check"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
    op.execute(text("UPDATE agents SET public_id = gen_random_uuid() WHERE public_id IS NULL"))
    op.execute(text("ALTER TABLE agents ALTER COLUMN public_id SET DEFAULT gen_random_uuid()"))
    op.execute(text("ALTER TABLE agents ALTER COLUMN public_id SET NOT NULL"))


def downgrade():
    op.execute(text("ALTER TABLE agents ALTER COLUMN public_id DROP NOT NULL"))
    op.execute(text("ALTER TABLE agents ALTER COLUMN public_id DROP DEFAULT"))
