"""
Pytest configuration for Learning module tests
Uses centralized credential system for database access
"""

import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

# Set test environment
os.environ['ENVIRONMENT'] = 'development'

# Use real database like CodeGraph/Memory
TEST_DB_URL = "postgresql://postgres:secure_password_123@127.0.0.1:5432/orchestrator_db"


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
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()
