"""The seed loader connects to the database the app serves.

``load_seed_data`` used to read only POSTGRES_* and fall back to localhost, so on
Railway (and in a Kubernetes migration Job), where only DATABASE_URL is set, it
failed to connect and seeded nothing: hosted boots logged ``Error loading seed
data: connection to server at "localhost" ... refused`` on every deploy. It now
connects through ``core.database.database.get_database_url()``, like the app.
"""
from __future__ import annotations

import sys
import types

import pytest

import core.database.load_seed_data as seed_loader

_URL = "postgresql://seed:secret@db.internal:5432/app"


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rowcount = 1

    def execute(self, sql, params=None):
        self.conn.executed.append(sql)

    def fetchone(self):
        return (7,)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self):
        self.executed: list[str] = []
        self.closed = False
        self.committed = False

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


@pytest.fixture
def app_database_url(monkeypatch):
    """Stand in for core.database.database with a known get_database_url()."""
    fake = types.ModuleType("core.database.database")
    fake.get_database_url = lambda: _URL
    monkeypatch.setitem(sys.modules, "core.database.database", fake)


def test_connects_with_the_apps_database_url(monkeypatch, app_database_url):
    calls = []
    conn = _FakeConn()

    def fake_connect(*args, **kwargs):
        calls.append((args, kwargs))
        return conn

    monkeypatch.setattr(seed_loader.psycopg2, "connect", fake_connect)
    assert seed_loader.load_seed_data(load_credentials=False, load_platform_defaults=False)
    assert calls == [((_URL,), {})]
    assert conn.closed


def test_loads_credential_types_through_that_connection(monkeypatch, app_database_url):
    conn = _FakeConn()
    monkeypatch.setattr(seed_loader.psycopg2, "connect", lambda *a, **k: conn)
    assert seed_loader.load_seed_data(load_credentials=True, load_platform_defaults=False)
    assert any("INSERT INTO credential_types" in sql for sql in conn.executed)
    assert conn.committed


def test_connection_failure_returns_false(monkeypatch, app_database_url):
    def refuse(*args, **kwargs):
        raise seed_loader.psycopg2.OperationalError("connection refused")

    monkeypatch.setattr(seed_loader.psycopg2, "connect", refuse)
    assert seed_loader.load_seed_data(load_credentials=True, load_platform_defaults=False) is False


def test_platform_defaults_run_after_credential_types(monkeypatch, app_database_url):
    monkeypatch.setattr(seed_loader.psycopg2, "connect", lambda *a, **k: _FakeConn())
    ran = []
    monkeypatch.setattr(seed_loader, "_load_platform_defaults", lambda: ran.append(True))
    assert seed_loader.load_seed_data(load_credentials=False, load_platform_defaults=True)
    assert ran == [True]
