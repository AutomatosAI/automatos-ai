"""PRD-120: Seed agent_catalog_templates from SKILL.md files

Populates the agent_catalog_templates table (created in prd120_agent_catalog_templates)
with entries parsed from automatos-skills/skills/**\/SKILL.md.

Idempotent — upserts on slug.

Revision ID: prd120_seed_agent_catalog
Revises: prd120_agent_catalog_templates
Create Date: 2026-03-24
"""

import sys
from pathlib import Path

from alembic import op

revision = "prd120_seed_agent_catalog"
down_revision = "prd120_agent_catalog_templates"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add scripts/ to path so we can import the seed helper
    scripts_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from seed_agent_catalog import seed_from_alembic  # noqa: E402
    seed_from_alembic(op)


def downgrade() -> None:
    # Remove all seeded rows (they have no workspace_id = global templates)
    op.execute("DELETE FROM agent_catalog_templates WHERE workspace_id IS NULL")
