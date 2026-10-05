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

The column is altered only while it is still ``json``: a database that
create_all built from the updated model already has ``jsonb``, and an
``ALTER ... TYPE ... USING`` would rewrite the table under an exclusive lock for
nothing. Both steps are skipped when the table is absent.
"""
from alembic import op

_TAGS_TO_JSONB = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'workflows'
          AND column_name = 'tags' AND data_type = 'json'
    ) THEN
        ALTER TABLE workflows ALTER COLUMN tags TYPE JSONB USING tags::jsonb;
    END IF;
    IF to_regclass('public.workflows') IS NOT NULL THEN
        CREATE INDEX IF NOT EXISTS ix_workflows_tags_gin ON workflows USING GIN (tags);
    END IF;
END $$;
"""

revision = "workflows_tags_jsonb"
down_revision = "outputs_heartbeat_reports"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(_TAGS_TO_JSONB)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_workflows_tags_gin")
    op.execute("ALTER TABLE workflows ALTER COLUMN tags TYPE JSON USING tags::json")
