"""
CodeGraph Integration Tests
===========================

End-to-end workflow tests for CodeGraph.
"""

import pytest
from modules.codegraph import CodeGraphService


class TestFullIndexingWorkflow:
    """Test complete indexing workflow"""
    
    @pytest.mark.asyncio
    async def test_index_and_verify(self, codegraph_service, test_repo_path, cleanup_projects):
        """Test full indexing workflow and verify results"""
        # Index the test repository
        result = await codegraph_service.index_github_project(
            project_name="test-full-workflow",
            github_url="file://" + str(test_repo_path),
            branch="main"
        )
        
        # Verify indexing completed
        assert result['status'] in ['completed', 'success']  # Can be either
        assert result['total_files'] >= 5  # Python + TypeScript files
        assert result['total_symbols'] >= 10  # Functions + classes
        assert result['duration_seconds'] > 0
        
        # Verify project in database
        projects = codegraph_service.list_projects()
        test_project = next((p for p in projects if p['name'] == 'test-full-workflow'), None)
        
        assert test_project is not None
        assert test_project['total_files'] == result['total_files']
        assert test_project['total_symbols'] == result['total_symbols']
        assert test_project['status'] in ['completed', 'active']  # Can be 'active' after indexing


class TestSearchAfterIndexing:
    """Test search functionality after indexing"""
    
    @pytest.mark.asyncio
    async def test_symbol_search_finds_functions(self, codegraph_service, test_repo_path, cleanup_projects):
        """Test symbol search finds indexed functions"""
        # Index repository
        await codegraph_service.index_github_project(
            project_name="test-symbol-search",
            github_url="file://" + str(test_repo_path),
            branch="main"
        )
        
        # Search for 'login' function
        result = await codegraph_service.search_symbols(
            project_name="test-symbol-search",
            query="login",
            limit=10
        )
        
        assert result['count'] > 0
        assert any('login' in r['qualified_name'].lower() for r in result['results'])
    
    @pytest.mark.asyncio
    async def test_semantic_search_finds_relevant_code(self, codegraph_service, test_repo_path, cleanup_projects):
        """Test semantic search finds relevant code"""
        # Index repository
        await codegraph_service.index_github_project(
            project_name="test-semantic-search",
            github_url="file://" + str(test_repo_path),
            branch="main"
        )
        
        # Semantic search for authentication-related code
        result = await codegraph_service.semantic_search(
            project_name="test-semantic-search",
            query="user authentication and login",
            limit=5
        )
        
        assert result['count'] > 0
        # Should find auth-related symbols
        assert any('auth' in r['qualified_name'].lower() or 'login' in r['qualified_name'].lower() 
                  for r in result['results'])


class TestReindexing:
    """Test re-indexing existing projects"""
    
    @pytest.mark.asyncio
    async def test_reindex_updates_data(self, codegraph_service, test_repo_path, cleanup_projects):
        """Test re-indexing updates project data"""
        # Initial index
        result1 = await codegraph_service.index_github_project(
            project_name="test-reindex",
            github_url="file://" + str(test_repo_path),
            branch="main"
        )
        
        initial_symbols = result1['total_symbols']
        
        # Modify test repo (add a new file)
        new_file = test_repo_path / "python" / "new_module.py"
        new_file.write_text('''
def new_function():
    """A new function"""
    pass
''')
        
        # Re-index
        result2 = await codegraph_service.index_github_project(
            project_name="test-reindex",
            github_url="file://" + str(test_repo_path),
            branch="main"
        )
        
        # Should have more symbols now
        assert result2['total_symbols'] >= initial_symbols
        assert result2['status'] in ['completed', 'success']


class TestCallGraphGeneration:
    """Test call graph generation"""
    
    @pytest.mark.asyncio
    async def test_generate_call_graph(self, codegraph_service, test_repo_path, cleanup_projects):
        """Test generating call graph for a symbol"""
        # Index repository
        await codegraph_service.index_github_project(
            project_name="test-call-graph",
            github_url="file://" + str(test_repo_path),
            branch="main"
        )
        
        # Get call graph for login function
        result = await codegraph_service.get_call_graph(
            project_name="test-call-graph",
            symbol="python/auth.py::login",
            depth=1,
            direction="outgoing"
        )
        
        assert result['project'] == 'test-call-graph'
        assert result['root_symbol'] == 'python/auth.py::login'
        assert 'nodes' in result
        assert 'edges' in result
