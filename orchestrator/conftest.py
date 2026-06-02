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
"""

from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


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
    engine = create_engine(test_db_url)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(test_engine) -> Generator[Session, None, None]:
    """Transactional session: each test runs inside a transaction that is rolled
    back on teardown, so tests never mutate the database for one another."""
    connection = test_engine.connect()
    transaction = connection.begin()

    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()

    yield session

    session.close()
    transaction.rollback()
    connection.close()
