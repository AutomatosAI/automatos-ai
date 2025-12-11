"""
CodeGraph Service Unit Tests
============================

Tests for CodeGraphService core functionality.
"""

import pytest
from pathlib import Path
from modules.codegraph import CodeGraphService


class TestServiceInitialization:
    """Test CodeGraphService initialization"""
    
    def test_service_creation(self, codegraph_service):
        """Test service can be created with DB session"""
        assert codegraph_service is not None
        assert codegraph_service.db is not None
        assert codegraph_service.embedding_manager is not None
    
    def test_language_extensions_configured(self, codegraph_service):
        """Test language extensions are properly configured"""
        assert 'python' in codegraph_service.language_extensions
        assert 'typescript' in codegraph_service.language_extensions
        assert 'javascript' in codegraph_service.language_extensions
        
        assert '.py' in codegraph_service.language_extensions['python']
        assert '.ts' in codegraph_service.language_extensions['typescript']
        assert '.js' in codegraph_service.language_extensions['javascript']
    
    def test_default_excludes_configured(self, codegraph_service):
        """Test default exclude patterns are set"""
        assert 'node_modules' in codegraph_service.default_excludes
        assert '__pycache__' in codegraph_service.default_excludes
        assert '.git' in codegraph_service.default_excludes


class TestFileDiscovery:
    """Test file discovery and filtering"""
    
    def test_discover_python_files(self, codegraph_service, test_repo_path):
        """Test discovering Python files"""
        files = list(codegraph_service._discover_files(
            test_repo_path, 
            codegraph_service.default_excludes
        ))
        
        python_files = [f for f in files if f.endswith('.py')]
        assert len(python_files) == 3  # auth.py, models.py, utils.py
    
    def test_discover_typescript_files(self, codegraph_service, test_repo_path):
        """Test discovering TypeScript files"""
        files = list(codegraph_service._discover_files(
            test_repo_path,
            codegraph_service.default_excludes
        ))
        
        ts_files = [f for f in files if f.endswith('.ts')]
        assert len(ts_files) == 2  # api.ts, types.ts
    
    def test_exclude_patterns_work(self, codegraph_service, test_repo_path):
        """Test that exclude patterns filter files correctly"""
        # Create a node_modules directory
        (test_repo_path / "node_modules").mkdir()
        (test_repo_path / "node_modules" / "test.js").write_text("// test")
        
        files = list(codegraph_service._discover_files(
            test_repo_path,
            codegraph_service.default_excludes
        ))
        
        # node_modules files should be excluded
        assert not any('node_modules' in str(f) for f in files)


class TestSymbolExtraction:
    """Test symbol extraction from code"""
    
    @pytest.mark.asyncio
    async def test_extract_python_functions(self, codegraph_service, test_repo_path):
        """Test extracting Python functions"""
        auth_file = test_repo_path / "python" / "auth.py"
        
        result = await codegraph_service._parse_file(str(auth_file), str(test_repo_path))
        
        assert result is not None
        assert len(result.symbols) >= 3  # login, logout, authenticate
        
        # Check function names
        function_names = [s.name for s in result.symbols if s.symbol_type == 'function']
        assert 'login' in function_names
        assert 'logout' in function_names
        assert 'authenticate' in function_names
    
    @pytest.mark.asyncio
    async def test_extract_python_classes(self, codegraph_service, test_repo_path):
        """Test extracting Python classes"""
        models_file = test_repo_path / "python" / "models.py"
        
        result = await codegraph_service._parse_file(str(models_file), str(test_repo_path))
        
        assert result is not None
        
        # Check class names
        class_names = [s.name for s in result.symbols if s.symbol_type == 'class']
        assert 'User' in class_names
        assert 'Session' in class_names
    
    @pytest.mark.asyncio
    async def test_extract_python_methods(self, codegraph_service, test_repo_path):
        """Test extracting Python class methods"""
        models_file = test_repo_path / "python" / "models.py"
        
        result = await codegraph_service._parse_file(str(models_file), str(test_repo_path))
        
        # Check for methods - they appear in qualified_name
        qualified_names = [s.qualified_name for s in result.symbols]
        assert any('save' in qn for qn in qualified_names), f"'save' method not found in {qualified_names}"
        assert any('invalidate' in qn for qn in qualified_names), f"'invalidate' method not found in {qualified_names}"
    
    @pytest.mark.asyncio
    async def test_extract_docstrings(self, codegraph_service, test_repo_path):
        """Test that docstrings are extracted"""
        auth_file = test_repo_path / "python" / "auth.py"
        
        result = await codegraph_service._parse_file(str(auth_file), str(test_repo_path))
        
        # Find login function
        login_symbol = next((s for s in result.symbols if s.name == 'login'), None)
        assert login_symbol is not None
        assert login_symbol.docstring is not None
        assert 'Authenticate user' in login_symbol.docstring


class TestProjectOperations:
    """Test project-level operations"""
    
    def test_list_projects_empty(self, codegraph_service, cleanup_projects):
        """Test listing projects when none exist"""
        projects = codegraph_service.list_projects()
        
        # Filter to only test projects
        test_projects = [p for p in projects if 'test' in p['name'].lower()]
        assert len(test_projects) == 0
    
    @pytest.mark.asyncio
    async def test_delete_project(self, codegraph_service, test_repo_path, cleanup_projects):
        """Test deleting a project"""
        # First index a project
        result = await codegraph_service.index_github_project(
            project_name="test-delete-project",
            github_url="file://" + str(test_repo_path),
            branch="main"
        )
        
        project_id = result['project_id']
        
        # Delete it
        delete_result = codegraph_service.delete_project(project_id)
        
        assert delete_result['message'] == 'Project deleted successfully'
        assert delete_result['project_name'] == 'test-delete-project'
        
        # Verify it's gone
        projects = codegraph_service.list_projects()
        assert not any(p['id'] == project_id for p in projects)


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks"""
    
    @pytest.mark.asyncio
    async def test_indexing_speed_small_repo(self, codegraph_service, test_repo_path, cleanup_projects, benchmark):
        """Benchmark indexing speed for small repository"""
        
        async def index_repo():
            return await codegraph_service.index_github_project(
                project_name="test-benchmark-small",
                github_url="file://" + str(test_repo_path),
                branch="main"
            )
        
        result = await index_repo()
        
        # Should complete in under 30 seconds for small repo
        assert result['duration_seconds'] < 30
        assert result['total_files'] > 0
        assert result['total_symbols'] > 0
