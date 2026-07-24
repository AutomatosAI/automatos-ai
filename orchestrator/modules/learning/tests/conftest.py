"""
Pytest configuration for Learning module tests
Uses centralized credential system for database access
"""

import pytest
import os
from uuid import uuid4
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from core.models.workspaces import Workspace

# Set test environment
os.environ['ENVIRONMENT'] = 'development'

# Shared test workspace ID
TEST_WORKSPACE_ID = uuid4()

# ``test_engine`` (and its DB URL) come from the root orchestrator/conftest.py
# (PRD-142 W2-S4). This module overrides ``db_session`` only to seed a
# Workspace row so the learning tables' FK constraints are satisfied.


@pytest.fixture(scope="function")
def db_session(test_engine) -> Generator[Session, None, None]:
    """
    Provide a transactional database session for each test.
    Rolls back after test completes.
    """
    connection = test_engine.connect()
    transaction = connection.begin()

    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()

    # Create a workspace row so FK constraints are satisfied
    workspace = Workspace(id=TEST_WORKSPACE_ID, name="Test Workspace")
    session.add(workspace)
    session.flush()

    yield session

    session.close()
    transaction.rollback()
    connection.close()
