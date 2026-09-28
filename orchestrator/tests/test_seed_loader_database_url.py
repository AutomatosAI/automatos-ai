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
from unittest.mock import MagicMock

import pytest

import core.database.load_seed_data as seed_loader

_URL = "postgresql://seed:secret@db.internal:5432/app"


@pytest.fixture
def app_database_url(monkeypatch):
    """Stand in for core.database.database with a known get_database_url()."""
    fake = types.ModuleType("core.database.database")
    fake.get_database_url = lambda: _URL
    monkeypatch.setitem(sys.modules, "core.database.database", fake)


@pytest.fixture
def conn(monkeypatch):
    """A fake psycopg2 connection whose cursor reports one written row and 7 types."""
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.rowcount = 1
    cursor.fetchone.return_value = (7,)
    monkeypatch.setattr(seed_loader.psycopg2, "connect", MagicMock(return_value=connection))
    return connection


def _statements(connection) -> list[str]:
    """Every SQL statement the fake cursor executed, in order."""
    cursor = connection.cursor.return_value.__enter__.return_value
    return [call.args[0] for call in cursor.execute.call_args_list]


def test_connects_with_the_apps_database_url(app_database_url, conn):
    """The loader connects with get_database_url(), not POSTGRES_* or localhost."""
    assert seed_loader.load_seed_data(load_credentials=False, load_platform_defaults=False)
    seed_loader.psycopg2.connect.assert_called_once_with(_URL)
    conn.close.assert_called_once()


def test_loads_credential_types_through_that_connection(app_database_url, conn):
    """Credential types are upserted on that connection and committed."""
    assert seed_loader.load_seed_data(load_credentials=True, load_platform_defaults=False)
    assert any("INSERT INTO credential_types" in sql for sql in _statements(conn))
    conn.commit.assert_called_once()


def test_a_failed_upsert_rolls_back_only_its_own_savepoint(app_database_url, conn):
    """One bad row must not abort the transaction for the rows after it."""
    cursor = conn.cursor.return_value.__enter__.return_value

    def fail_first_insert(sql, params=None):
        """Raise on the first credential-type insert only, like a constraint error."""
        if "INSERT INTO credential_types" in sql and not getattr(fail_first_insert, "done", False):
            fail_first_insert.done = True
            raise seed_loader.psycopg2.IntegrityError("bad row")

    cursor.execute.side_effect = fail_first_insert
    assert seed_loader.load_seed_data(load_credentials=True, load_platform_defaults=False)
    statements = _statements(conn)
    first_rollback = statements.index("ROLLBACK TO SAVEPOINT credential_type")
    later_inserts = [s for s in statements[first_rollback:] if "INSERT INTO credential_types" in s]
    assert later_inserts, "rows after the bad one must still be upserted"
    assert "RELEASE SAVEPOINT credential_type" in statements[first_rollback:]


def test_connection_failure_returns_false(monkeypatch, app_database_url):
    """A database the loader can't reach is a failure, reported as False."""
    refuse = MagicMock(side_effect=seed_loader.psycopg2.OperationalError("connection refused"))
    monkeypatch.setattr(seed_loader.psycopg2, "connect", refuse)
    assert seed_loader.load_seed_data(load_credentials=True, load_platform_defaults=False) is False


def test_platform_defaults_run_after_credential_types(monkeypatch, app_database_url, conn):
    """load_platform_defaults=True runs the platform-default sections."""
    platform_defaults = MagicMock()
    monkeypatch.setattr(seed_loader, "_load_platform_defaults", platform_defaults)
    assert seed_loader.load_seed_data(load_credentials=False, load_platform_defaults=True)
    platform_defaults.assert_called_once_with()


def test_upsert_sets_the_timestamps_the_orm_would_have():
    """Raw SQL skips the model's default=func.now(); NULL timestamps made the API 500."""
    sql = seed_loader._UPSERT_CREDENTIAL_TYPE
    assert "created_at, updated_at)" in sql
    assert "NOW(), NOW())" in sql
    assert "created_at = COALESCE(credential_types.created_at, NOW())" in sql
    assert "updated_at = NOW()" in sql
