"""
Pytest configuration for Learning module tests
Uses centralized credential system for database access
"""

import pytest
import os
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from core.models.workspaces import Workspace

# Set test environment
os.environ['ENVIRONMENT'] = 'development'

# Use real database like CodeGraph/Memory
TEST_DB_URL = "postgresql://postgres:secure_password_123@127.0.0.1:5432/orchestrator_db"

# Shared test workspace ID
TEST_WORKSPACE_ID = uuid4()


@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine"""
    engine = create_engine(TEST_DB_URL)
    yield engine
    engine.dispose()


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
