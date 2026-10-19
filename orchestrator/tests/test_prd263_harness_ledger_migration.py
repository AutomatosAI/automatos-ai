"""PRD-263 US-051 — the ledger tracks each proposal's grant, prior value and outcome.

``harness_task_ledger`` gains the governed apply rail's columns (``rx_id``, unique per
workspace, ``run_id``, ``grant_id``, ``kind``, ``target_agent_id``, the prior, proposed
and applied-over values, ``applied_at`` and the frozen reference), ``board_task_id``
becomes nullable, and the state CHECK allows the rail's states, with every F156
``held`` row becoming ``expired``. The ORM model and the Alembic revision
``prd263_harness_ledger_grants`` declare the same names, so the schema-drift and
alembic-from-zero lanes see one shape.

The upgrade is run for real on the F156 table, built by the f156 revision itself in a
schema of its own inside ``db_session``'s transaction (workspaces stays reachable
through the search path), and on the table ``create_all`` built. Everything is rolled
back.
"""
from __future__ import annotations

import importlib.util
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

VERSIONS = Path(__file__).resolve().parents[1] / "alembic" / "versions"
MIGRATION = VERSIONS / "prd263_harness_ledger_grants.py"
F156 = VERSIONS / "f156_harness_task_ledger.py"
CHECK = "ck_harness_task_ledger_state"
INDEX = "uq_harness_task_ledger_rx"
NEW_COLUMNS = (
    "rx_id", "run_id", "grant_id", "kind", "target_agent_id", "prior_value", "proposed_value",
    "applied_at", "value_at_apply", "frozen_ref_start", "frozen_ref_end",
)
STATES = {"pending", "applying", "applied", "failed", "denied", "expired", "drifted", "withdrawn",
          "raise_failed", "rolled_back"}
# The F156 table is built here, so the shared table is never reshaped.
F156_SCHEMA = "prd263_ledger_f156"
APPLIED_AT = datetime(2026, 10, 26, 9, 30, tzinfo=timezone.utc)
FROZEN_START, FROZEN_END = datetime(2026, 9, 28, tzinfo=timezone.utc), APPLIED_AT

_INSERT = text("INSERT INTO harness_task_ledger (workspace_id, board_task_id, state, rx_id) "
               "VALUES (CAST(:ws AS uuid), :task, :state, :rx)")
_SHAPE = {
    "columns": ("SELECT column_name, data_type, character_maximum_length, is_nullable, column_default "
                "FROM information_schema.columns WHERE table_schema = :schema AND table_name = 'harness_task_ledger'"),
    "indexes": ("SELECT indexname, indexdef FROM pg_indexes "
                "WHERE schemaname = :schema AND tablename = 'harness_task_ledger'"),
    "constraints": ("SELECT con.conname, pg_get_constraintdef(con.oid) FROM pg_constraint con "
                    "JOIN pg_class cls ON cls.oid = con.conrelid JOIN pg_namespace ns ON ns.oid = cls.relnamespace "
                    "WHERE ns.nspname = :schema AND cls.relname = 'harness_task_ledger'"),
}


def _statements(path: Path, step: str) -> list:
    """The SQL a revision's ``step`` runs, captured instead of executed."""
    spec = importlib.util.spec_from_file_location(f"{path.stem}_under_test", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    statements: list = []
    migration.op = SimpleNamespace(execute=statements.append)
    getattr(migration, step)()
    return statements


def _run(db, path: Path, step: str = "upgrade") -> None:
    for statement in _statements(path, step):
        db.execute(text(statement))


def _shape(db, schema: str) -> dict:
    """The schema's ledger: columns, index definitions and constraints, schema names removed."""
    qualifier = re.compile(rf"\b(?:{schema}|public)\.")
    return {
        kind: sorted(tuple(qualifier.sub("", str(value)) for value in row)
                     for row in db.execute(text(sql), {"schema": schema}))
        for kind, sql in _SHAPE.items()
    }


def _quoted(sql: str) -> set:
    return set(re.findall(r"'([a-z_]+)'", sql))


def test_the_model_and_the_migration_declare_one_shape():
    from sqlalchemy import CheckConstraint

    from core.models import harness

    table = harness.HarnessTaskLedger.__table__
    assert set(NEW_COLUMNS) <= set(table.c.keys())
    assert all(table.c[name].nullable for name in NEW_COLUMNS + ("board_task_id",))
    (check,) = [c for c in table.constraints if isinstance(c, CheckConstraint)]
    assert (check.name, _quoted(str(check.sqltext))) == (CHECK, STATES)
    assert {getattr(harness, name) for name in dir(harness) if name.startswith("LEDGER_")} == STATES
    assert {index.name: ([c.name for c in index.columns], index.unique) for index in table.indexes} == {
        INDEX: (["workspace_id", "rx_id"], True)}

    src = MIGRATION.read_text()
    assert 'revision = "prd263_harness_ledger_grants"' in src
    up = _statements(MIGRATION, "upgrade")
    # The order the maintainer issue states: the CHECK goes before the held rewrite.
    assert set(re.findall(r"ADD COLUMN IF NOT EXISTS (\w+)", up[0])) == set(NEW_COLUMNS)
    assert "ALTER COLUMN board_task_id DROP NOT NULL" in up[1]
    assert f"DROP CONSTRAINT IF EXISTS {CHECK}" in up[2]
    assert "SET state = 'expired' WHERE state = 'held'" in up[3]
    assert f"ADD CONSTRAINT {CHECK}" in up[4] and _quoted(up[4]) == STATES
    assert f"CREATE UNIQUE INDEX IF NOT EXISTS {INDEX} ON harness_task_ledger (workspace_id, rx_id)" in up[5]
    down = " ".join(_statements(MIGRATION, "downgrade"))
    assert f"DROP INDEX IF EXISTS {INDEX}" in down and f"DROP CONSTRAINT IF EXISTS {CHECK}" in down
    assert set(re.findall(r"DROP COLUMN IF EXISTS (\w+)", down)) == set(NEW_COLUMNS)


def test_a_held_row_becomes_expired_and_the_upgraded_f156_table_matches_the_model(db_session, seed_workspace):
    ws = seed_workspace()
    model = _shape(db_session, "public")
    db_session.execute(text("SET LOCAL lock_timeout = '5s'"))
    db_session.execute(text("CREATE SCHEMA " + F156_SCHEMA))
    db_session.execute(text("SET LOCAL search_path TO " + F156_SCHEMA + ", public"))
    _run(db_session, F156)
    db_session.execute(text("INSERT INTO harness_task_ledger (workspace_id, board_task_id, state) "
                            "VALUES (CAST(:ws AS uuid), 1, 'held'), (CAST(:ws AS uuid), 2, 'applied')"), {"ws": ws})

    _run(db_session, MIGRATION)

    states = db_session.execute(text("SELECT board_task_id, state FROM harness_task_ledger "
                                     "WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": ws}).all()
    assert dict(states) == {1: "expired", 2: "applied"}
    assert _shape(db_session, F156_SCHEMA) == model
    _run(db_session, MIGRATION)  # a second run changes nothing
    assert _shape(db_session, F156_SCHEMA) == model


def test_the_upgrade_keeps_a_create_all_table_as_it_is(db_session):
    db_session.execute(text("SET LOCAL lock_timeout = '5s'"))
    before = _shape(db_session, "public")

    _run(db_session, MIGRATION)

    assert _shape(db_session, "public") == before


def test_a_held_row_is_refused(db_session, seed_workspace):
    with pytest.raises(IntegrityError, match=CHECK):
        db_session.execute(_INSERT, {"ws": seed_workspace(), "task": 1, "state": "held", "rx": None})


def test_one_workspace_never_holds_an_rx_id_twice(db_session, seed_workspace):
    ws_a, ws_b = seed_workspace(), seed_workspace()
    db_session.execute(_INSERT, {"ws": ws_a, "task": None, "state": "pending", "rx": "rx-7-001"})
    db_session.execute(_INSERT, {"ws": ws_b, "task": None, "state": "pending", "rx": "rx-7-001"})
    db_session.execute(_INSERT, {"ws": ws_a, "task": 1, "state": "applied", "rx": None})
    db_session.execute(_INSERT, {"ws": ws_a, "task": 2, "state": "expired", "rx": None})  # legacy rows: no rx_id

    with pytest.raises(IntegrityError, match=INDEX):
        db_session.execute(_INSERT, {"ws": ws_a, "task": None, "state": "pending", "rx": "rx-7-001"})


def test_a_row_with_every_new_column_round_trips(db_session, seed_workspace):
    ws = seed_workspace()
    run_id, grant_id, agent_id = 41, 9001, 17
    db_session.execute(text("""
        INSERT INTO harness_task_ledger (workspace_id, board_task_id, state, rx_id, run_id, grant_id, kind,
            target_agent_id, prior_value, proposed_value, applied_at, value_at_apply, frozen_ref_start,
            frozen_ref_end)
        VALUES (CAST(:ws AS uuid), NULL, 'applied', 'rx-41-001', :run, :grant, 'heartbeat_tune', :agent,
            CAST(:prior AS jsonb), CAST(:proposed AS jsonb), :applied_at, CAST(:prior AS jsonb), :start, :end)
    """), {"ws": ws, "run": run_id, "grant": grant_id, "agent": agent_id, "prior": '{"interval_minutes": 60}',
           "proposed": '{"interval_minutes": 120}', "applied_at": APPLIED_AT, "start": FROZEN_START,
           "end": FROZEN_END})

    row = db_session.execute(text("SELECT * FROM harness_task_ledger WHERE workspace_id = CAST(:ws AS uuid)"),
                             {"ws": ws}).mappings().one()

    assert (row["board_task_id"], row["state"], row["rx_id"], row["kind"]) == (None, "applied", "rx-41-001",
                                                                              "heartbeat_tune")
    assert (row["run_id"], row["grant_id"], row["target_agent_id"]) == (run_id, grant_id, agent_id)
    assert row["prior_value"] == row["value_at_apply"] == {"interval_minutes": 60}
    assert row["proposed_value"] == {"interval_minutes": 120}
    assert (row["applied_at"], row["frozen_ref_start"], row["frozen_ref_end"]) == (APPLIED_AT, FROZEN_START,
                                                                                  FROZEN_END)


def test_a_workspace_read_never_returns_another_workspaces_row(db_session, seed_workspace):
    ws_a, ws_b = seed_workspace(), seed_workspace()
    db_session.execute(_INSERT, {"ws": ws_a, "task": None, "state": "pending", "rx": "rx-1-001"})
    db_session.execute(_INSERT, {"ws": ws_b, "task": None, "state": "pending", "rx": "rx-1-001"})
    db_session.execute(_INSERT, {"ws": ws_b, "task": 3, "state": "applied", "rx": None})

    read_a = db_session.execute(text("SELECT CAST(workspace_id AS text), rx_id FROM harness_task_ledger "
                                     "WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": ws_a}).all()

    assert [tuple(row) for row in read_a] == [(ws_a, "rx-1-001")]
