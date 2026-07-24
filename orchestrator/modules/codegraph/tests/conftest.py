"""
CodeGraph Test Configuration and Fixtures
==========================================

Pytest configuration and shared fixtures for CodeGraph testing.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import Generator

from modules.codegraph import CodeGraphService

# ``test_engine`` and the transactional ``db_session`` fixture come from the
# root orchestrator/conftest.py (PRD-142 W2-S4) — no per-module DB URL here.


# ---------------------------------------------------------------------------
# Environment capability guard (F056)
# ---------------------------------------------------------------------------
#
# The F056 orchestrator-module-tests job stands up stock ``postgres:15`` and
# configures no embedding provider. codegraph_service.index_github_project()
# calls _ensure_embedding_dimension(), which runs
# ``ALTER TABLE codegraph_symbols ALTER COLUMN embedding TYPE vector(N)`` — that
# needs the pgvector extension — and then generate_embeddings_batch(), which
# needs a live embedder. Those tests therefore CANNOT pass in this job; they are
# real tests gated on a service the job doesn't provide, so we SKIP them
# cleanly (never xfail/delete/weaken) with an honest reason. The table-only
# tests (list/delete-nothing) still run against the schema init_test_db.py now
# creates.
#
# The skip is decided by probing pgvector availability once per session against
# the real test engine. When pgvector IS present (e.g. a local pgvector DB or a
# future job upgrade to the pgvector image + an embedder), the guard is inert and
# every test runs.

# Tests whose body reaches the pgvector/embedder indexing path. Keyed by node
# name (class-qualified where needed) so the guard targets exactly these.
_PGVECTOR_DEPENDENT = frozenset({
    # test_codegraph_integration.py — every test indexes then searches.
    "test_index_and_verify",
    "test_symbol_search_finds_functions",
    "test_semantic_search_finds_relevant_code",
    "test_reindex_updates_data",
    "test_generate_call_graph",
    # test_codegraph_service.py — these index before asserting.
    "test_delete_project",
    "test_indexing_speed_small_repo",  # also needs the pytest-benchmark plugin
})


def _pgvector_available() -> bool:
    """True iff the test database can enable the pgvector extension.

    Probed against the same engine the DB fixtures use. Any failure (extension
    missing, insufficient privilege, DB unreachable) is treated as unavailable
    so the dependent tests skip rather than error.
    """
    try:
        # Import here so collecting this tree never forces a DB config resolution
        # or a sqlalchemy import that the environment might not have.
        from core.database.database import engine

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
            ).first()
            return row is not None
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip codegraph tests the running environment cannot support.

    * pgvector/embedder indexing tests skip when pgvector is unavailable.
    * ``test_indexing_speed_small_repo`` additionally needs the pytest-benchmark
      ``benchmark`` fixture; skip it if that plugin isn't installed (it is not in
      requirements.txt, so the F056 job lacks it).
    """
    has_pgvector = _pgvector_available()
    has_benchmark = config.pluginmanager.hasplugin("benchmark")

    pgvector_skip = pytest.mark.skip(
        reason="requires pgvector extension + an embedding provider (not stood "
        "up by the F056 orchestrator-module-tests job on stock postgres:15)"
    )
    benchmark_skip = pytest.mark.skip(
        reason="requires the pytest-benchmark 'benchmark' fixture (plugin not "
        "installed in the F056 job — absent from requirements.txt)"
    )

    for item in items:
        # Only touch this module's tests; leave every other tree alone.
        if "modules/codegraph/tests/" not in item.nodeid.replace("\\", "/"):
            continue
        if item.name == "test_indexing_speed_small_repo" and not has_benchmark:
            item.add_marker(benchmark_skip)
        elif item.name in _PGVECTOR_DEPENDENT and not has_pgvector:
            item.add_marker(pgvector_skip)


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
