"""PRD-262 US-030: the two columns main.py added on every boot become a revision (19 Oct 2026).

Until this revision ``_boot_phase_1_core`` (main.py) ran two ``ALTER TABLE … ADD COLUMN
IF NOT EXISTS`` statements on every boot of every worker. Each takes an ACCESS EXCLUSIVE
lock on its table, even when the column is long there:

* ``system_prompts.futureagi_eval_enabled``: the prompts the FutureAGI evaluation scores
  (core/services/futureagi_service.py);
* ``skills.content_hash``: the SHA-256 of a builtin skill's SKILL.md, compared on boot to
  refresh a stale row (core/seeds/seed_builtin_skills.py). ``add_content_hash_to_skills``
  adds it too, but that revision is a root of the forest (no ``down_revision``).

The models declare both columns, so ``create_all`` and ``scripts/init_fresh_db.py`` build
them, and every database the old main.py booted on has them. ``IF NOT EXISTS`` keeps the
upgrade a no-op there and adds them only where they are missing. The DDL is the boot
ALTERs' own: ``NOT NULL DEFAULT FALSE`` fills the flag on existing rows.

The downgrade leaves both columns. They predate this revision: the models, the code a
downgrade returns to (which added them on every boot) and ``add_content_hash_to_skills``
all expect them.

Revision ID: prd262_boot_columns
Revises: auto_haiku_default_20261019
Create Date: 2026-10-19
"""

from alembic import op


revision = "prd262_boot_columns"
down_revision = "auto_haiku_default_20261019"
branch_labels = None
depends_on = None

# (table, column, column DDL): the statements main.py ran on every boot.
BOOT_COLUMNS = (
    ("system_prompts", "futureagi_eval_enabled", "BOOLEAN NOT NULL DEFAULT FALSE"),
    ("skills", "content_hash", "VARCHAR(64)"),
)


def upgrade() -> None:
    for table, column, ddl in BOOT_COLUMNS:
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {ddl};")


def downgrade() -> None:
    """The columns predate this revision (see the module docstring), so they stay."""
