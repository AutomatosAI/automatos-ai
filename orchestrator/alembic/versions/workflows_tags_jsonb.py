"""Workflows — tags becomes JSONB so its GIN index can actually be built.

Issue #840: ``workflows.tags`` was declared ``JSON`` (added by
``128a785a7681_add_tool_models_and_audit_logs``, which re-added the column that
``20250812_150001_add_owner_tags_policy_to_workflows`` had already added as
``JSONB`` days earlier — one of the two silently loses on any given database,
whichever ``op.add_column`` runs first). Postgres has no default GIN operator
class for ``json``, only ``jsonb``, so ``ix_workflows_tags_gin``
(``20250812_150002_add_indexes_workflows_owner_tags``) was left commented out —
and build_schema's stage-5 raw-SQL scraper reads ``upgrade()`` bodies as plain
text, comments included, so it kept retrying that commented-out
``CREATE INDEX ... USING GIN (tags)`` and reporting it "still failing" on every
fresh build.

``api/workflows.py``'s tag filter already calls ``Workflow.tags.contains([tag])``
with a text-cast ``ILIKE`` fallback, because the generic SQLAlchemy ``JSON``
comparator doesn't implement containment — only ``postgresql.JSONB``'s comparator
maps ``.contains()`` to the native ``@>`` operator, which this index now serves.

``ALTER COLUMN ... TYPE jsonb USING tags::jsonb`` is idempotent on either
starting type: ``json::jsonb`` converts it, and ``jsonb::jsonb`` (a database a
create_all-with-the-updated-model already built) is a harmless no-op cast.
``CREATE INDEX IF NOT EXISTS`` is idempotent by construction.
"""
from alembic import op

revision = "workflows_tags_jsonb"
down_revision = "outputs_heartbeat_reports"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE workflows ALTER COLUMN tags TYPE JSONB USING tags::jsonb")
    op.execute("CREATE INDEX IF NOT EXISTS ix_workflows_tags_gin ON workflows USING GIN (tags)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_workflows_tags_gin")
    op.execute("ALTER TABLE workflows ALTER COLUMN tags TYPE JSON USING tags::json")
