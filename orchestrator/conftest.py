"""Root pytest fixtures for the orchestrator suite (PRD-142 Wave 2, W2-S4).

This is the common ancestor of both ``tests/`` (the mock-based unit suite) and
``modules/*/tests/`` (the real-DB module suites), so the shared real-DB
transactional fixture lives here. The module conftests previously each carried
their own copy of this fixture plus a hardcoded database URL; that duplication
(and the in-source credential) is now centralised.

Credentials are resolved through the central config resolver
(``get_database_url``), never a literal in source. In CI the Postgres service
env vars drive it; for local runs the developer's ``DATABASE_URL`` /
``POSTGRES_*`` environment applies.

SQLAlchemy is imported lazily inside the DB fixtures (not at module level) so
that a pure-stdlib test tree — e.g. the operating-graph uplift self-tests, whose
CI job installs only pytest — can be COLLECTED without SQLAlchemy present. Only
tests that actually request ``test_engine`` / ``db_session`` pull it in.
"""

from typing import TYPE_CHECKING, Generator

import pytest

if TYPE_CHECKING:  # for type checkers only; never imported at runtime collection
    from sqlalchemy.orm import Session
else:  # runtime: keep the annotation valid without importing sqlalchemy at import time
    Session = "Session"


@pytest.fixture(scope="session")
def test_db_url() -> str:
    """Resolve the test database URL from central config — no hardcoded creds."""
    # Imported lazily so collecting the mock-only suite never forces a DB
    # config resolution it doesn't need.
    from core.database.database import get_database_url

    return get_database_url()


@pytest.fixture(scope="session")
def test_engine(test_db_url):
    """Session-scoped SQLAlchemy engine bound to the test database."""
    from sqlalchemy import create_engine

    engine = create_engine(test_db_url)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(test_engine) -> Generator["Session", None, None]:
    """Transactional session: each test runs inside a transaction that is rolled
    back on teardown, so tests never mutate the database for one another."""
    from sqlalchemy.orm import sessionmaker

    connection = test_engine.connect()
    transaction = connection.begin()

    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def seed_workspace(db_session):
    """Factory inserting a minimal ``workspaces`` row so FK-bound inserts
    (documents, chats, …) satisfy ``*_workspace_id_fkey``.

    Call ``seed_workspace()`` for a fresh id, or ``seed_workspace(ws_id)`` to back
    a specific one; returns the id (str). Rolled back with ``db_session``.
    """
    import uuid as _uuid

    from sqlalchemy import text as _t

    def _seed(workspace_id=None, name="test-ws"):
        ws_id = str(workspace_id or _uuid.uuid4())
        db_session.execute(
            _t(
                "INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :name) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {"id": ws_id, "name": name},
        )
        db_session.flush()
        return ws_id

    return _seed
