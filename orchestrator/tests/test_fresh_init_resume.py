"""An interrupted fresh-database build is resumed, never served half-built.

``build_schema`` creates ``alembic_version`` in its first statement and stage 2
stamps as it goes, so a build that dies partway (Postgres restarting, the host
stopping) used to look like an existing database on the next boot: the fresh
path was skipped and ``alembic upgrade heads`` either failed on the model-built
tables for ever or "succeeded" without the columns stages 3-6 add. The build now
carries a marker from its first statement to its last; these tests pin that
contract and the decision ``init_fresh_db`` makes from it. (The entrypoint's
side is in ``test_entrypoint_modes.py``; the real-Postgres interrupt-and-resume
run is the alembic-from-zero CI lane.)
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from scripts import generate_schema_baseline as gsb
from scripts import init_fresh_db
from scripts.generate_schema_baseline import INCOMPLETE_MARKER


class _Result:
    def scalar(self):
        return 154


class _Engine:
    """Records every statement, in order, alongside the stages' own events."""

    def __init__(self, events: list):
        self.events = events

    @contextmanager
    def begin(self):
        yield self

    connect = begin

    def execute(self, statement):
        self.events.append(str(statement))
        return _Result()


@pytest.fixture
def events(monkeypatch):
    log: list = []
    monkeypatch.setattr(gsb, "init_db", lambda: log.append("init_db"))
    monkeypatch.setattr(gsb, "AlembicConfig", lambda path: object())
    monkeypatch.setattr(gsb.ScriptDirectory, "from_config", staticmethod(lambda cfg: object()))
    monkeypatch.setattr(gsb, "_replay_forest", lambda cfg, script: log.append("replay") or (1, 0))
    monkeypatch.setattr(gsb, "_created_by", lambda script: {})
    monkeypatch.setattr(gsb, "_residual_passes", lambda engine, script, creators: log.append("residual"))
    monkeypatch.setattr(gsb, "_repair_passes", lambda engine, script: log.append("repair"))
    monkeypatch.setattr(gsb, "_missing_tables", lambda engine, names: set())
    return log


def _index(log: list, needle: str) -> int:
    return next(i for i, entry in enumerate(log) if needle in entry)


def test_the_marker_exists_before_alembic_version_and_goes_only_after_the_repairs(events):
    gsb.build_schema(_Engine(events))
    created = _index(events, f"CREATE TABLE IF NOT EXISTS {INCOMPLETE_MARKER}")
    assert created < _index(events, "CREATE TABLE IF NOT EXISTS alembic_version") < _index(events, "replay")
    assert _index(events, "repair") < _index(events, f"DROP TABLE IF EXISTS {INCOMPLETE_MARKER}")


def test_a_build_that_dies_partway_leaves_the_marker(events, monkeypatch):
    def interrupted(engine, script):
        raise RuntimeError("FATAL: the database system is shutting down")

    monkeypatch.setattr(gsb, "_repair_passes", interrupted)
    with pytest.raises(RuntimeError):
        gsb.build_schema(_Engine(events))
    assert any(f"CREATE TABLE IF NOT EXISTS {INCOMPLETE_MARKER}" in e for e in events)
    assert not any(f"DROP TABLE IF EXISTS {INCOMPLETE_MARKER}" in e for e in events)


@pytest.mark.parametrize(
    ("has_marker", "has_version", "tables", "action"),
    [
        (False, False, 0, init_fresh_db.BUILD),     # empty: build
        (True, True, 126, init_fresh_db.RESUME),    # interrupted mid-build: finish it
        (True, False, 1, init_fresh_db.RESUME),     # interrupted before alembic_version committed
        (False, True, 154, init_fresh_db.NOTHING),  # finished or migrated: incremental path
        (False, False, 12, init_fresh_db.REFUSE),   # tables, no history, no marker: unknown
    ],
)
def test_the_marker_decides_before_alembic_version_does(has_marker, has_version, tables, action):
    assert init_fresh_db.fresh_db_action(has_marker, has_version, tables) == action


def _main_with(monkeypatch, state: tuple) -> tuple[int, list]:
    built: list = []
    monkeypatch.setattr(init_fresh_db, "create_engine", lambda url: object())
    monkeypatch.setattr(init_fresh_db, "_inspect", lambda engine: state)
    monkeypatch.setattr(init_fresh_db, "build_schema", lambda engine: built.append(engine) or 154)
    return init_fresh_db.main(), built


def test_main_resumes_an_interrupted_build(monkeypatch, capsys):
    code, built = _main_with(monkeypatch, (True, True, 126))
    assert code == 0 and len(built) == 1
    assert "interrupted" in capsys.readouterr().out


def test_main_leaves_a_finished_database_alone(monkeypatch):
    code, built = _main_with(monkeypatch, (False, True, 154))
    assert code == 0 and built == []
