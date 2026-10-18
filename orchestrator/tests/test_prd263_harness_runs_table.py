"""PRD-263 US-009 — HARNESS runs get their own table.

A run's state lived in JSON files under <WORKSPACE_VOLUME_PATH>/<ws>/harness. Each
run is now one ``harness_runs`` row: the ISO week it judges, when it started and
finished, its status and iteration, coverage, counts, verdicts, prescriptions,
and the audit report with its outcome. The ORM model and the Alembic revision
``prd263_harness_runs`` declare the same names, including a CHECK that holds
``status`` to the values the code writes, the revision survives a database that
``create_all`` built, and a read scoped to one workspace never returns another's run.
"""
from __future__ import annotations

import importlib.util
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy import text

MIGRATION = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "prd263_harness_runs.py"
COLUMNS = (
    "id", "workspace_id", "cur_window", "started_at", "finished_at", "status", "iteration",
    "coverage", "counts", "verdicts", "prescriptions", "report_md", "report_status",
)
INDEX = "ix_harness_runs_ws_window"
CHECK = "ck_harness_runs_status"
FINISHED = datetime(2026, 10, 19, 10, 5, tzinfo=timezone.utc)


def test_a_run_round_trips_and_a_workspace_read_never_returns_another_workspaces_run(db_session, seed_workspace):
    from core.models.harness import HarnessRun

    ws_a, ws_b = UUID(seed_workspace()), UUID(seed_workspace())
    db_session.add_all([
        HarnessRun(
            workspace_id=ws_a, cur_window="2026-W42", status="completed", iteration=1, coverage=0.75,
            counts={"prescriptions": 2}, verdicts={"7": {"success_rate": "WATCH"}},
            prescriptions=[{"prescription_id": "rx-1-001"}], report_md="# HARNESS Weekly Org Review",
            report_status="ok", finished_at=FINISHED,
        ),
        HarnessRun(workspace_id=ws_b, cur_window="2026-W42", status="running"),
    ])
    db_session.flush()
    db_session.expire_all()

    (run,) = db_session.query(HarnessRun).filter(HarnessRun.workspace_id == ws_a).all()
    assert (run.cur_window, run.status, run.iteration, run.coverage) == ("2026-W42", "completed", 1, 0.75)
    assert run.counts == {"prescriptions": 2}
    assert run.verdicts == {"7": {"success_rate": "WATCH"}}
    assert run.prescriptions == [{"prescription_id": "rx-1-001"}]
    assert (run.report_md, run.report_status) == ("# HARNESS Weekly Org Review", "ok")
    assert run.finished_at == FINISHED and run.started_at is not None

    (other,) = db_session.query(HarnessRun).filter(HarnessRun.workspace_id == ws_b).all()
    assert (other.status, other.iteration, other.finished_at) == ("running", 0, None)
    assert other.id != run.id


def _quoted(sql):
    return set(re.findall(r"'([a-z_]+)'", sql))


def test_the_model_and_the_migration_declare_one_shape():
    from sqlalchemy import CheckConstraint

    from core.models import harness

    table = harness.HarnessRun.__table__
    assert tuple(column.name for column in table.columns) == COLUMNS
    assert {index.name: [c.name for c in index.columns] for index in table.indexes} == {
        INDEX: ["workspace_id", "cur_window"]}
    (fk,) = table.c.workspace_id.foreign_keys
    assert (fk.target_fullname, fk.ondelete) == ("workspaces.id", "CASCADE")
    (check,) = [c for c in table.constraints if isinstance(c, CheckConstraint)]
    statuses = {harness.RUN_RUNNING, harness.RUN_COMPLETED, harness.RUN_FAILED,
                harness.RUN_DORMANT_AGENTS, harness.RUN_DORMANT_DATA}
    assert (check.name, _quoted(str(check.sqltext))) == (CHECK, statuses)

    src = MIGRATION.read_text()
    assert 'revision = "prd263_harness_runs"' in src
    assert 'down_revision = "prd258_voice_timings"' in src
    (ddl,) = re.findall(r"CREATE TABLE IF NOT EXISTS harness_runs \((.*?)\n\s*\)", src, re.S)
    columns, constraint = ddl.split(f"CONSTRAINT {CHECK} CHECK")
    assert tuple(line.split()[0] for line in columns.strip().splitlines()) == COLUMNS
    assert _quoted(constraint) == statuses
    assert "REFERENCES workspaces(id) ON DELETE CASCADE" in ddl
    assert f"CREATE INDEX IF NOT EXISTS {INDEX} " in src and "ON harness_runs (workspace_id, cur_window)" in src
    assert f"DROP INDEX IF EXISTS {INDEX};" in src and "DROP TABLE IF EXISTS harness_runs;" in src


def test_a_status_the_code_never_writes_is_rejected(db_session, seed_workspace):
    from sqlalchemy.exc import IntegrityError

    from core.models.harness import HarnessRun

    db_session.add(HarnessRun(workspace_id=UUID(seed_workspace()), cur_window="2026-W42", status="complete"))
    with pytest.raises(IntegrityError, match=CHECK):
        db_session.flush()


def _upgrade_statements():
    spec = importlib.util.spec_from_file_location("prd263_harness_runs_under_test", MIGRATION)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    statements = []
    migration.op = SimpleNamespace(execute=statements.append)
    migration.upgrade()
    return statements


def _index_def(db):
    return db.execute(text("SELECT indexdef FROM pg_indexes WHERE tablename = 'harness_runs' AND indexname = :name"),
                      {"name": INDEX}).scalar()


def test_the_upgrade_keeps_a_create_all_table_and_adds_only_its_missing_index(db_session):
    """The test database is built by create_all, so the table and index are there; the
    upgrade runs over it without error, and puts back an index that is missing.
    Everything here runs inside db_session's transaction and is rolled back."""
    assert "(workspace_id, cur_window)" in _index_def(db_session)
    db_session.execute(text("DROP INDEX ix_harness_runs_ws_window"))
    assert _index_def(db_session) is None

    for statement in _upgrade_statements():
        db_session.execute(text(statement))

    assert "(workspace_id, cur_window)" in _index_def(db_session)
    columns = db_session.execute(text(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'harness_runs' "
        "ORDER BY ordinal_position")).scalars().all()
    assert tuple(columns) == COLUMNS
