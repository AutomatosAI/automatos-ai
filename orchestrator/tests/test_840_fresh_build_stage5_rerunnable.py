"""#840 — build_schema stage 5 (the idempotent raw-SQL re-run) must actually be
idempotent, not just intended to be.

``scripts/generate_schema_baseline.py``'s stage 5 (``_replay_idempotent_raw_sql``)
re-runs every ``upgrade()`` body it scrapes for an idempotent marker (``CREATE INDEX
IF NOT EXISTS`` etc.) at the end of a fresh build, to repair orphan-root ordering
losses (see its own module docstring). Two bodies could never actually be re-run
clean, so every fresh build ended by logging them as "still failing":

1. ``prd77_agent_scheduled_tasks``'s body ends with two bare
   ``ALTER TABLE ... ADD CONSTRAINT`` statements. ``CREATE INDEX IF NOT EXISTS``
   earlier in the SAME ``op.execute()`` call makes the whole body idempotent-marked,
   but the constraints are not themselves guarded, so the one ``conn.execute(text(sql))``
   that runs the whole body fails (and the whole transaction rolls back) the second
   time it is tried — PostgreSQL's ``ALTER TABLE ... ADD CONSTRAINT`` has no
   ``IF NOT EXISTS`` form. ``calendar_scheduled_board_tasks`` (also an
   ``agent_scheduled_tasks`` constraint) already guards its ``ADD CONSTRAINT`` with
   ``DO $$ ... IF NOT EXISTS (SELECT 1 FROM pg_constraint ...) ...END $$`` — prd77's
   two constraints are given the same guard here.

2. The scraper (``_upgrade_raw_sql``) reads ``upgrade()`` bodies as plain source
   text via regex, comments included — it does not parse Python, so a commented-out
   ``# op.execute("CREATE INDEX IF NOT EXISTS ix_workflows_tags_gin ...")`` in
   ``20250812_150002_add_indexes_workflows_owner_tags`` was scraped and retried as if
   live. It never could succeed: ``workflows.tags`` was plain ``JSON``, and Postgres
   has no default GIN operator class for ``json``, only ``jsonb``. The commented-out
   line is removed (a comment never executed real SQL, so removing it changes no
   database); the real, working index is built by the new ``workflows_tags_jsonb``
   migration once ``tags`` is JSONB.

Both are proven two ways: a static read of the migration source (fails on old
source, the cheap/obvious check) and a behavioural run of the REAL
``_replay_idempotent_raw_sql`` against the REAL migration forest (via Alembic's own
``ScriptDirectory`` — no database; style matches ``test_prd225_asks_model.py``), with
a fake connection that reproduces Postgres's actual failure mode: a bare
``ADD CONSTRAINT`` on a constraint that already exists raises; one guarded by a
``pg_constraint`` existence check does not.
"""
from __future__ import annotations

import pathlib
import re

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

from scripts import generate_schema_baseline as gsb

_ORCH = pathlib.Path(__file__).resolve().parents[1]
_VERSIONS = _ORCH / "alembic" / "versions"


def _script() -> ScriptDirectory:
    cfg = Config()
    cfg.set_main_option("script_location", str(_ORCH / "alembic"))
    return ScriptDirectory.from_config(cfg)


# ---------------------------------------------------------------------------
# Static: the migration source itself
# ---------------------------------------------------------------------------


def test_prd77_add_constraints_are_guarded_by_a_pg_constraint_check():
    src = (_VERSIONS / "prd77_agent_scheduled_tasks.py").read_text()
    for name in ("ck_scheduled_task_type", "ck_scheduled_task_status"):
        guard = re.search(
            r"IF NOT EXISTS\s*\(\s*SELECT 1 FROM pg_constraint WHERE conname = '"
            + re.escape(name)
            + r"'\s*\)\s*THEN\s*\n\s*ALTER TABLE agent_scheduled_tasks\s*\n\s*ADD CONSTRAINT "
            + re.escape(name),
            src,
        )
        assert guard, f"{name} must be added inside a DO $$ ... IF NOT EXISTS (pg_constraint) guard"


def test_workflows_owner_tags_migration_no_longer_hides_executable_sql_in_a_comment():
    src = (_VERSIONS / "20250812_150002_add_indexes_workflows_owner_tags.py").read_text()
    assert "op.execute(" not in src, (
        "stage 5 re-runs upgrade() bodies as plain text, comments included — a "
        "commented-out op.execute(...) call is retried as live SQL on every fresh "
        "build (#840); this revision's GIN index now lives in workflows_tags_jsonb"
    )


def test_workflows_tags_is_jsonb_so_the_gin_index_has_an_opclass():
    src = (_ORCH / "core" / "models" / "core.py").read_text()
    m = re.search(r"class Workflow\(Base\):.*?(?=\nclass \w+\(Base\):)", src, re.S)
    assert m, "Workflow model not found"
    assert re.search(r"tags\s*=\s*Column\(JSONB", m.group(0)), (
        "workflows.tags must be JSONB — Postgres has no default GIN operator class "
        "for plain json, so ix_workflows_tags_gin could never be built (#840)"
    )


def test_workflows_tags_jsonb_migration_chains_onto_the_prior_head_and_is_the_new_one():
    src = (_VERSIONS / "workflows_tags_jsonb.py").read_text()
    assert 'down_revision = "outputs_heartbeat_reports"' in src
    assert "ALTER TABLE workflows ALTER COLUMN tags TYPE JSONB USING tags::jsonb" in src
    assert "CREATE INDEX IF NOT EXISTS ix_workflows_tags_gin ON workflows USING GIN (tags)" in src


def test_workflows_tags_jsonb_rewrites_only_a_json_column_of_a_present_table():
    """An ALTER ... TYPE ... USING rewrites the table under an exclusive lock even
    when the type is already jsonb, so the migration alters only a json column,
    and touches nothing when the table is absent."""
    src = (_VERSIONS / "workflows_tags_jsonb.py").read_text()
    assert "data_type = 'json'" in src
    assert "to_regclass('public.workflows') IS NOT NULL" in src


def test_the_tag_filter_has_no_text_scan_fallback():
    src = (pathlib.Path(__file__).resolve().parents[1] / "api" / "workflows.py").read_text()
    assert "Workflow.tags.contains([tag])" in src
    assert "cast(Workflow.tags, String)" not in src


# ---------------------------------------------------------------------------
# Behavioural: the real stage-5 scraper + replay, against the real forest
# ---------------------------------------------------------------------------


class _AlreadyAppliedConnection:
    """Simulates re-running stage 5 against a database where stage 2 already
    applied every migration once: IF NOT EXISTS / IF EXISTS statements are
    no-ops (as real Postgres makes them); a bare ADD CONSTRAINT whose name is
    not re-checked against pg_constraint raises DuplicateObject, exactly as
    real Postgres does the second time it runs."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def begin(self):
        return self

    def commit(self):
        pass

    def rollback(self):
        pass

    def execute(self, statement):
        sql = str(statement)
        if "ADD CONSTRAINT" in sql and "pg_constraint" not in sql:
            raise Exception('psycopg2.errors.DuplicateObject: constraint already exists')
        return None


class _AlreadyAppliedEngine:
    def connect(self):
        return _AlreadyAppliedConnection()


def test_prd77_scraped_body_can_be_executed_against_an_already_applied_database():
    """Direct proof the body is re-runnable: take the REAL text
    ``_upgrade_raw_sql`` scrapes for prd77 out of the REAL migration file, and
    execute it against a connection standing in for a database stage 2 already
    built (every CREATE is a no-op; a constraint ``pg_constraint`` does not
    re-check for is a DuplicateObject, exactly as real Postgres raises it).
    Before the fix this body ends in two bare ADD CONSTRAINTs and raises here;
    the guard added for #840 must make the whole body a no-op instead."""
    bodies = dict(gsb._upgrade_raw_sql(_script()))
    sql = bodies["prd77_agent_scheduled_tasks"]
    assert "ADD CONSTRAINT" in sql, "fixture check: the constraints are still in the body"
    _AlreadyAppliedConnection().execute(sql)  # must not raise


def test_prd77_body_survives_a_second_run_of_the_real_stage5_replay(capsys):
    """End-to-end: the real ``_replay_idempotent_raw_sql`` must stop reporting
    prd77 as 'still failing'. (It now also drops out of stage 5's own
    idempotent-marker filter, since a guarded body contains ``DO $$`` with no
    ``ADD COLUMN`` — the prior test proves the body's SQL is re-runnable on its
    own merits, independent of that filter.)"""
    gsb._replay_idempotent_raw_sql(_AlreadyAppliedEngine(), _script())
    out = capsys.readouterr().out
    assert "prd77_agent_scheduled_tasks" not in out, out


def test_add_workflow_indexes_body_survives_a_second_run_of_the_real_stage5_replay(capsys):
    """The scraper must no longer find a live-looking CREATE INDEX for
    ix_workflows_tags_gin inside a comment in the old owner/tags migration."""
    gsb._replay_idempotent_raw_sql(_AlreadyAppliedEngine(), _script())
    out = capsys.readouterr().out
    assert "add_workflow_indexes" not in out, out


def test_prd77_body_without_the_fix_would_have_failed_the_same_harness():
    """Harness sanity: the fake connection really does reproduce the original
    failure for an unguarded ADD CONSTRAINT, so the two tests above are not
    vacuously true."""
    conn = _AlreadyAppliedConnection()
    with pytest.raises(Exception, match="DuplicateObject"):
        conn.execute("ALTER TABLE agent_scheduled_tasks ADD CONSTRAINT ck_scheduled_task_type "
                      "CHECK (task_type IN ('one_shot', 'recurring'))")
