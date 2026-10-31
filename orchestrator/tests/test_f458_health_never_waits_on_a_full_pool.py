"""F458 (night 18): a full connection pool never makes GET /health miss its probe.

08:01Z: the pool (10 + 20, 30 s wait) ran dry. /health's SELECT 1 asked the pool for a
connection and waited past curl's 10 s, three probes in a row; the entrypoint's
liveness watchdog (docker-entrypoint.sh, F416) took that for a dead backend, sent
TERM, then KILL (`line 461: 50 Killed`), and every turn and agent run in flight was
lost. The backend was serving the whole time.

Now the database probe reads the pool's counters first: a full pool is ``busy`` at
once, with no checkout, and /health answers degraded (200). Only a pool seen full at
every probe for HEALTH_POOL_STUCK_SECONDS is ``stuck`` and answers 503, the one state
a restart cures. Driven on a real QueuePool (SQLite file, one connection, held).
"""
from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from config import config
from core import health_probe
from core.database import pool_watch

POOL_TIMEOUT_S = 3      # the test pool's own wait: what the old probe sat through
ANSWERS_WITHIN_S = 0.5  # what the probe may take with the pool full
MAIN = Path(__file__).resolve().parents[1] / "main.py"


@pytest.fixture
def one_connection_pool(tmp_path, monkeypatch):
    """The app's engine and sessions, swapped for a pool of exactly one connection."""
    import core.database.database as database

    engine = sa.create_engine(
        f"sqlite:///{tmp_path / 'health.db'}", poolclass=QueuePool,
        pool_size=1, max_overflow=0, pool_timeout=POOL_TIMEOUT_S,
    )
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=engine))
    yield engine
    engine.dispose()


def _timed_status(clock):
    started = time.monotonic()
    status = health_probe.database_status(clock)
    return status, time.monotonic() - started


def test_a_full_pool_is_busy_at_once_and_no_connection_is_asked_for(one_connection_pool):
    held = one_connection_pool.connect()  # night 18: every connection checked out
    try:
        assert pool_watch.pool_has_room() is False
        assert pool_watch.pool_numbers() == {"checked_out": 1, "limit": 1}
        status, took = _timed_status(health_probe.FullPoolClock())
    finally:
        held.close()
    assert status == health_probe.DATABASE_BUSY
    assert took < ANSWERS_WITHIN_S, f"the probe waited {took:.1f}s on a full pool (its timeout is {POOL_TIMEOUT_S}s)"


def test_with_a_free_connection_the_probe_still_runs_its_select(one_connection_pool):
    status, _ = _timed_status(health_probe.FullPoolClock())
    assert status == health_probe.HEALTHY
    assert pool_watch.pool_numbers()["checked_out"] == 0, "the probe gave its connection back"


def test_a_pool_full_at_every_probe_past_the_threshold_is_stuck(one_connection_pool, monkeypatch):
    monkeypatch.setattr(config, "HEALTH_POOL_STUCK_SECONDS", 0.2)
    clock = health_probe.FullPoolClock()
    held = one_connection_pool.connect()
    try:
        first, _ = _timed_status(clock)
        time.sleep(0.25)
        second, took = _timed_status(clock)
    finally:
        held.close()
    assert (first, second) == (health_probe.DATABASE_BUSY, health_probe.DATABASE_STUCK)
    assert took < ANSWERS_WITHIN_S


def test_a_free_sighting_restarts_the_count(one_connection_pool, monkeypatch):
    """Night 18's shape was a burst: a pool that frees between probes is never stuck."""
    monkeypatch.setattr(config, "HEALTH_POOL_STUCK_SECONDS", 0.2)
    clock = health_probe.FullPoolClock()
    held = one_connection_pool.connect()
    assert _timed_status(clock)[0] == health_probe.DATABASE_BUSY
    held.close()
    time.sleep(0.25)
    assert _timed_status(clock)[0] == health_probe.HEALTHY
    held = one_connection_pool.connect()
    try:
        assert _timed_status(clock)[0] == health_probe.DATABASE_BUSY
    finally:
        held.close()


@pytest.mark.parametrize("database,overall,http", [
    (health_probe.HEALTHY, health_probe.HEALTHY, 200),
    (health_probe.DATABASE_BUSY, health_probe.DEGRADED, 200),   # serving: never a watchdog miss
    (health_probe.DATABASE_STUCK, health_probe.UNHEALTHY, 503),  # a restart frees the pool
    (health_probe.UNHEALTHY, health_probe.UNHEALTHY, 200),       # the database is down: a restart won't help
])
def test_what_each_database_state_answers(database, overall, http):
    components = {"api_server": health_probe.HEALTHY, "database": database, "config": health_probe.HEALTHY}
    assert health_probe.overall_status(components) == overall
    assert health_probe.http_status(components) == http


def test_a_degraded_component_still_degrades_the_answer():
    components = {"api_server": "healthy", "database": "healthy", "trust_gate": "degraded"}
    assert health_probe.overall_status(components) == health_probe.DEGRADED


def test_the_route_reads_the_database_and_its_status_code_through_the_probes():
    """Importing main.py builds the whole app, so the wiring is pinned statically."""
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    (route,) = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "health_check"]
    called = {
        f"{node.func.value.id}.{node.func.attr}" for node in ast.walk(route)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
    }
    assert {"health_probe.database_status", "health_probe.overall_status", "health_probe.http_status"} <= called
    assert not any(isinstance(node, ast.Name) and node.id == "SessionLocal" for node in ast.walk(route)), (
        "the route never asks the pool for a connection itself"
    )
