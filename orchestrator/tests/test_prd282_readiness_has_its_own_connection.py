"""#1651 (PRD-282 US-201): readiness's SELECT 1 runs on a connection of its own, which a
database that stops answering cannot hold for minutes.

* ``core/database/readiness_engine.py`` builds it through ``create_app_engine`` from the main
  engine's address, so PRD-278's TLS keywords (``gssencmode`` included) apply, with no pool
  and with libpq's ``connect_timeout``, ``tcp_user_timeout`` (ms) and keepalives set from
  HEALTH_READY_CHECK_TIMEOUT_SECONDS.
* A check against a server that accepts and never answers ends near that limit.
* GET /health still checks on the app's pool (unchanged).
* A ``BaseException`` in a watch's check thread resolves its future, so the watch checks again.
* HEALTH_READY_CHECK_SECONDS and HEALTH_READY_CHECK_TIMEOUT_SECONDS are at least 1 s.
"""
from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import event
from sqlalchemy.pool import NullPool

from config import config
from core import health_probe, health_readiness
from core.database import readiness_engine as readiness
from core.health_watch import Watch
from tests.helpers_prd278_w3_ca import write_ca_bundle

ORCH = Path(__file__).resolve().parents[1]
LIMIT_S = 0.2        # a test watch's time limit
HUNG_LIMIT_S = 1.0   # the readiness time limit against a server that never answers
# libpq may round a 1 s connect_timeout up to 2 s; what is left is the thread's start.
HUNG_BOUND_S = 6.0


class _Captured(Exception):
    """Raised from do_connect once the connection's keywords are recorded: nothing connects."""


def _address(host: str = "db.example.invalid", port: int = 5432, query: str = "") -> str:
    """A database address built from parts (no literal credentialed URL)."""
    return "postgresql" + "://" + "app" + "@" + host + f":{port}/" + "app" + query


@pytest.fixture
def fresh_engine(monkeypatch):
    """readiness_engine() rebuilt for this test, on an address and TLS settings it sets."""
    import core.database.database as database

    def use(url: str, sslmode=None, sslrootcert=None):
        monkeypatch.setattr(database, "DATABASE_URL", url)
        monkeypatch.setattr(config, "DATABASE_SSLMODE", sslmode)
        monkeypatch.setattr(config, "DATABASE_SSLROOTCERT", sslrootcert)
        readiness.readiness_engine.cache_clear()
        return readiness.readiness_engine()

    readiness.readiness_engine.cache_clear()
    yield use
    readiness.readiness_engine.cache_clear()


def _connect_keywords(engine) -> dict:
    """The keywords the engine would hand psycopg2, recorded without connecting."""
    captured: dict = {}

    @event.listens_for(engine, "do_connect")
    def record(_dialect, _record, _cargs, cparams):
        captured.update(cparams)
        raise _Captured

    with pytest.raises(_Captured):
        engine.connect()
    return captured


# --------------------------------------------------------------------------- the engine


@pytest.mark.parametrize("limit_s,connect_s,tcp_ms", [(3, 3, 3000), (1.5, 2, 1500), (1, 1, 1000)])
def test_the_connect_args_follow_the_time_limit(limit_s, connect_s, tcp_ms):
    assert readiness.readiness_connect_args(limit_s) == {
        "connect_timeout": connect_s, "tcp_user_timeout": tcp_ms, "keepalives": 1,
        "keepalives_idle": 1, "keepalives_interval": 1, "keepalives_count": connect_s,
    }


def test_the_readiness_engine_has_no_pool_and_libpq_timeouts_from_the_setting(fresh_engine, monkeypatch):
    monkeypatch.setattr(config, "HEALTH_READY_CHECK_TIMEOUT_SECONDS", 4.0)
    engine = fresh_engine(_address())
    assert isinstance(engine.pool, NullPool)
    keywords = _connect_keywords(engine)
    assert keywords["connect_timeout"] == 4 and keywords["tcp_user_timeout"] == 4000
    assert keywords["keepalives"] == 1 and keywords["keepalives_count"] == 4
    assert readiness.readiness_engine() is engine, "built once, reused by every check"


def test_the_readiness_engine_gets_the_main_engines_address_and_tls(fresh_engine, tmp_path):
    ca_file = str(write_ca_bundle(tmp_path / "ca.crt"))
    engine = fresh_engine(_address(query="?requiressl=1"), sslmode="verify-full", sslrootcert=ca_file)
    keywords = _connect_keywords(engine)
    assert keywords["sslmode"] == "verify-full" and keywords["sslrootcert"] == ca_file
    assert keywords["gssencmode"] == "disable"
    assert "requiressl" not in keywords, "the address's alias is dropped, as for the main engine"
    assert keywords["host"] == "db.example.invalid" and keywords["connect_timeout"] == 3


def test_production_keeps_the_main_engines_sslmode_default(fresh_engine, monkeypatch):
    from core.database.tls import DatabaseTLS, main_engine_url

    url = main_engine_url(_address(), "production", DatabaseTLS())
    keywords = _connect_keywords(fresh_engine(url))
    assert keywords["sslmode"] == "require" and "gssencmode" not in keywords


# --------------------------------------------------------------------------- the check uses it; /health does not


def test_readiness_checks_on_its_own_engine(fresh_engine, monkeypatch):
    engine = fresh_engine(_address())
    seen: dict = {}
    monkeypatch.setattr(health_probe, "database_status", lambda **kwargs: seen.update(kwargs) or "healthy")
    assert health_readiness._database_status() == "healthy"
    assert seen == {"raise_errors": True, "engine": engine}


def test_health_still_checks_on_the_apps_pool(monkeypatch):
    import core.database.database as database

    sessions = []

    def session():
        sessions.append(SimpleNamespace(execute=lambda _sql: None, close=lambda: None))
        return sessions[-1]

    monkeypatch.setattr(health_probe.pool_watch, "pool_has_room", lambda: True)
    monkeypatch.setattr(database, "SessionLocal", session)
    assert health_probe.database_status(health_probe.FullPoolClock()) == health_probe.HEALTHY
    assert len(sessions) == 1


def test_a_database_that_never_answers_ends_the_check_near_its_limit(fresh_engine, monkeypatch):
    """A server that takes the connection and never answers (a failover with no TCP reset):
    the check's thread ends near the time limit, not on the kernel's timeouts."""
    monkeypatch.setattr(config, "HEALTH_READY_CHECK_TIMEOUT_SECONDS", HUNG_LIMIT_S)
    monkeypatch.setattr(health_probe.pool_watch, "pool_has_room", lambda: True)
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(1)  # the kernel completes the handshake; nothing ever answers
        fresh_engine(_address("127.0.0.1", silent.getsockname()[1]))
        outcome: dict = {}

        def check():
            started = time.monotonic()
            try:
                health_readiness._database_status()
            except Exception as exc:  # noqa: BLE001 — the failure is what this test expects
                outcome["error"] = exc
            outcome["took"] = time.monotonic() - started

        thread = threading.Thread(target=check, daemon=True)
        thread.start()
        thread.join(HUNG_BOUND_S)
    assert not thread.is_alive(), "the check still hangs"
    assert "error" in outcome and outcome["took"] < HUNG_BOUND_S


# --------------------------------------------------------------------------- the watch


class _Abandoned(BaseException):
    """Not an Exception: what used to leave a check's future unresolved."""


def test_a_base_exception_in_the_check_thread_does_not_stop_the_watch():
    calls = []

    def check():
        calls.append(1)
        raise _Abandoned

    watch = Watch("test", check, every_s=60, limit_s=LIMIT_S, failed_state="down")

    async def two_ticks():
        first = await watch.check_now()
        second = await watch.check_now()
        return first, second

    first, second = asyncio.run(two_ticks())
    assert first.state == "down" and second.state == "down"
    assert len(calls) == 2, "the second tick started a check: the first one's future was resolved"


# --------------------------------------------------------------------------- the settings' floors


def _read_settings(**env: str) -> list:
    """The two settings as config reads them with ``env`` set, in a fresh interpreter."""
    code = "from config import config; print(config.HEALTH_READY_CHECK_SECONDS, config.HEALTH_READY_CHECK_TIMEOUT_SECONDS)"
    done = subprocess.run([sys.executable, "-c", code], cwd=ORCH, env={**os.environ, **env},
                          capture_output=True, text=True, timeout=60, check=True)
    return [float(value) for value in done.stdout.split()[-2:]]


@pytest.mark.parametrize("every,limit,expected", [
    ("0", "0", [1.0, 1.0]),
    ("-5", "-1", [1.0, 1.0]),
    ("0.01", "0.2", [1.0, 1.0]),
    ("7", "2.5", [7.0, 2.5]),
])
def test_the_readiness_timings_have_a_floor(every, limit, expected):
    assert _read_settings(HEALTH_READY_CHECK_SECONDS=every, HEALTH_READY_CHECK_TIMEOUT_SECONDS=limit) == expected
