"""
CodeGraph Test Configuration and Fixtures
==========================================

Pytest configuration and shared fixtures for CodeGraph testing.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from modules.codegraph import CodeGraphService


# Test database URL (use in-memory SQLite for fast tests)
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


@pytest.fixture
def codegraph_service(db_session: Session) -> CodeGraphService:
    """Provide CodeGraphService instance with test DB session"""
    return CodeGraphService(db=db_session)


@pytest.fixture
def test_repo_path() -> Generator[Path, None, None]:
    """
    Create a temporary test repository with known structure.
    
    Structure:
        test-repo/
        ├── python/
        │   ├── auth.py (login, logout, authenticate functions)
        │   ├── models.py (User, Session classes)
        │   └── utils.py (hash_password, validate_email functions)
        ├── typescript/
        │   ├── api.ts (fetchData, postData functions)
        │   └── types.ts (ApiResponse, User interfaces)
        └── README.md
    """
    temp_dir = tempfile.mkdtemp(prefix="codegraph_test_")
    repo_path = Path(temp_dir) / "test-repo"
    repo_path.mkdir()
    
    # Create Python files
    python_dir = repo_path / "python"
    python_dir.mkdir()
    
    (python_dir / "auth.py").write_text('''"""Authentication module"""

def login(username: str, password: str) -> bool:
    """Authenticate user and create session"""
    if not username or not password:
        return False
    # Authentication logic here
    return True

def logout(session_id: str) -> None:
    """End user session"""
    pass

def authenticate(token: str) -> dict:
    """Validate authentication token"""
    return {"valid": True, "user_id": 123}
''')
    
    (python_dir / "models.py").write_text('''"""Data models"""

class User:
    """User model"""
    def __init__(self, username: str, email: str):
        self.username = username
        self.email = email
    
    def save(self):
        """Save user to database"""
        pass

class Session:
    """Session model"""
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.active = True
    
    def invalidate(self):
        """Invalidate session"""
        self.active = False
''')
    
    (python_dir / "utils.py").write_text('''"""Utility functions"""
import hashlib
import re

def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
''')
    
    # Create TypeScript files
    ts_dir = repo_path / "typescript"
    ts_dir.mkdir()
    
    (ts_dir / "api.ts").write_text('''/**
 * API client functions
 */

export async function fetchData(url: string): Promise<ApiResponse> {
  const response = await fetch(url);
  return response.json();
}

export async function postData(url: string, data: any): Promise<ApiResponse> {
  const response = await fetch(url, {
    method: 'POST',
    body: JSON.stringify(data)
  });
  return response.json();
}
''')
    
    (ts_dir / "types.ts").write_text('''/**
 * Type definitions
 */

export interface ApiResponse {
  success: boolean;
  data: any;
  error?: string;
}

export interface User {
  id: number;
  username: string;
  email: string;
}
''')
    
    # Create README
    (repo_path / "README.md").write_text('''# Test Repository

This is a test repository for CodeGraph testing.

## Structure
- `python/`: Python modules
- `typescript/`: TypeScript modules
''')
    
    # Initialize as git repository so it can be cloned
    import subprocess
    subprocess.run(['git', 'init'], cwd=repo_path, capture_output=True)
    subprocess.run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo_path, capture_output=True)
    subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=repo_path, capture_output=True)
    subprocess.run(['git', 'add', '.'], cwd=repo_path, capture_output=True)
    subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=repo_path, capture_output=True)
    
    yield repo_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def cleanup_projects(db_session: Session):
    """Clean up test projects after tests"""
    yield
    
    # Delete all test projects
    db_session.execute(text("""
        DELETE FROM codegraph_projects 
        WHERE name LIKE 'test-%' OR name LIKE '%test%'
    """))
    db_session.commit()


@pytest.fixture
def sample_python_code():
    """Sample Python code for parsing tests"""
    return '''
def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two numbers"""
    return a + b

class Calculator:
    """Simple calculator class"""
    
    def add(self, x: float, y: float) -> float:
        """Add two numbers"""
        return x + y
    
    def multiply(self, x: float, y: float) -> float:
        """Multiply two numbers"""
        return x * y
'''


@pytest.fixture
def sample_typescript_code():
    """Sample TypeScript code for parsing tests"""
    return '''
/**
 * Add two numbers
 */
export function add(a: number, b: number): number {
  return a + b;
}

/**
 * Calculator class
 */
export class Calculator {
  /**
   * Multiply two numbers
   */
  multiply(x: number, y: number): number {
    return x * y;
  }
}
'''
