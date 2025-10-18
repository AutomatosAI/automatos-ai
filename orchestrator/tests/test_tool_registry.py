"""
Unit Tests for Tool Registry - PRD-17
======================================

Tests for the centralized tool registry system including:
- Tool registration and retrieval
- Category-based queries
- OpenAI format export
- Tool prompts generation
- MCP tool integration
"""

import pytest
from unittest.mock import Mock, patch
from services.tool_registry import (
    ToolRegistry,
    ToolSpec,
    ToolParameter,
    ToolCategory,
    SecurityLevel,
    get_tool_registry,
    reset_tool_registry
)


class TestToolParameter:
    """Test ToolParameter dataclass"""
    
    def test_parameter_creation(self):
        """Test creating a tool parameter"""
        param = ToolParameter(
            name="query",
            type="string",
            description="Search query",
            required=True
        )
        
        assert param.name == "query"
        assert param.type == "string"
        assert param.required is True
    
    def test_parameter_to_openai_format(self):
        """Test converting parameter to OpenAI format"""
        param = ToolParameter(
            name="limit",
            type="number",
            description="Max results",
            required=False,
            default=5
        )
        
        openai_format = param.to_openai_format()
        
        assert openai_format["type"] == "number"
        assert openai_format["description"] == "Max results"
        assert openai_format["default"] == 5


class TestToolSpec:
    """Test ToolSpec dataclass"""
    
    def test_tool_spec_creation(self):
        """Test creating a tool specification"""
        tool = ToolSpec(
            name="search_knowledge",
            category=ToolCategory.RESEARCH,
            description="Search knowledge base",
            executor_class="AgentPlatformTools",
            executor_method="execute_tool",
            parameters=[
                ToolParameter("query", "string", "Search query", required=True)
            ],
            security_level=SecurityLevel.SAFE
        )
        
        assert tool.name == "search_knowledge"
        assert tool.category == ToolCategory.RESEARCH
        assert tool.security_level == SecurityLevel.SAFE
        assert len(tool.parameters) == 1
    
    def test_tool_to_openai_format(self):
        """Test converting tool to OpenAI function format"""
        tool = ToolSpec(
            name="read_file",
            category=ToolCategory.FILE_OPERATIONS,
            description="Read a file",
            executor_class="ActionExecutor",
            executor_method="read_file",
            parameters=[
                ToolParameter("file_path", "string", "Path to file", required=True),
                ToolParameter("encoding", "string", "File encoding", required=False, default="utf-8")
            ]
        )
        
        openai_format = tool.to_openai_format()
        
        assert openai_format["name"] == "read_file"
        assert openai_format["description"] == "Read a file"
        assert "file_path" in openai_format["parameters"]["properties"]
        assert "encoding" in openai_format["parameters"]["properties"]
        assert "file_path" in openai_format["parameters"]["required"]
        assert "encoding" not in openai_format["parameters"]["required"]
    
    def test_tool_to_markdown_doc(self):
        """Test generating markdown documentation"""
        tool = ToolSpec(
            name="execute_command",
            category=ToolCategory.SHELL_COMMANDS,
            description="Execute shell command",
            executor_class="ActionExecutor",
            executor_method="execute_command",
            parameters=[
                ToolParameter("command", "string", "Command to run", required=True)
            ],
            security_level=SecurityLevel.DANGEROUS
        )
        
        markdown = tool.to_markdown_doc()
        
        assert "execute_command" in markdown
        assert "shell" in markdown.lower()
        assert "dangerous" in markdown.lower()
        assert "command" in markdown
        assert "Command to run" in markdown


class TestToolRegistry:
    """Test ToolRegistry class"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        reset_tool_registry()
        yield
        reset_tool_registry()
    
    def test_registry_initialization(self):
        """Test registry initializes with core tools"""
        registry = ToolRegistry()
        
        # Should have registered core tools
        assert len(registry.tools) > 0
        
        # Should have research tools
        research_tools = registry.get_tools_by_category(ToolCategory.RESEARCH)
        assert len(research_tools) >= 3  # search_knowledge, semantic_search, search_codebase
        
        # Should have file operation tools
        file_tools = registry.get_tools_by_category(ToolCategory.FILE_OPERATIONS)
        assert len(file_tools) >= 5  # read, write, delete, list, create_directory
        
        # Should have shell tools
        shell_tools = registry.get_tools_by_category(ToolCategory.SHELL_COMMANDS)
        assert len(shell_tools) >= 1  # execute_command
    
    def test_register_tool(self):
        """Test registering a new tool"""
        registry = ToolRegistry()
        initial_count = len(registry.tools)
        
        new_tool = ToolSpec(
            name="test_tool",
            category=ToolCategory.API_TOOLS,
            description="Test tool",
            executor_class="TestExecutor",
            executor_method="execute"
        )
        
        registry.register_tool(new_tool)
        
        assert len(registry.tools) == initial_count + 1
        assert "test_tool" in registry.tools
        assert registry.get_tool("test_tool") == new_tool
    
    def test_get_tool_by_name(self):
        """Test retrieving tool by name"""
        registry = ToolRegistry()
        
        tool = registry.get_tool("search_knowledge")
        
        assert tool is not None
        assert tool.name == "search_knowledge"
        assert tool.category == ToolCategory.RESEARCH
    
    def test_get_tools_by_category(self):
        """Test retrieving tools by category"""
        registry = ToolRegistry()
        
        research_tools = registry.get_tools_by_category(ToolCategory.RESEARCH)
        
        assert len(research_tools) >= 3
        assert all(t.category == ToolCategory.RESEARCH for t in research_tools)
        
        # Verify specific tools exist
        tool_names = [t.name for t in research_tools]
        assert "search_knowledge" in tool_names
        assert "semantic_search" in tool_names
        assert "search_codebase" in tool_names
    
    def test_get_all_tools(self):
        """Test retrieving all tools"""
        registry = ToolRegistry()
        
        all_tools = registry.get_all_tools()
        
        assert len(all_tools) > 0
        assert all(isinstance(t, ToolSpec) for t in all_tools)
    
    def test_get_all_tools_active_only(self):
        """Test filtering active tools only"""
        registry = ToolRegistry()
        
        # Add an inactive tool
        inactive_tool = ToolSpec(
            name="inactive_tool",
            category=ToolCategory.API_TOOLS,
            description="Inactive tool",
            executor_class="Test",
            executor_method="test",
            is_active=False
        )
        registry.register_tool(inactive_tool)
        
        # Get active only
        active_tools = registry.get_all_tools(active_only=True)
        inactive_tools_in_result = [t for t in active_tools if t.name == "inactive_tool"]
        
        assert len(inactive_tools_in_result) == 0
        
        # Get all including inactive
        all_tools = registry.get_all_tools(active_only=False)
        inactive_tools_in_all = [t for t in all_tools if t.name == "inactive_tool"]
        
        assert len(inactive_tools_in_all) == 1
    
    def test_get_tools_for_categories(self):
        """Test getting tools for multiple categories"""
        registry = ToolRegistry()
        
        tools = registry.get_tools_for_categories(["research", "file_ops"])
        
        assert len(tools) > 0
        
        # Should have tools from both categories
        categories = set(t.category for t in tools)
        assert ToolCategory.RESEARCH in categories
        assert ToolCategory.FILE_OPERATIONS in categories
    
    def test_build_tool_prompt_markdown(self):
        """Test building markdown tool prompt"""
        registry = ToolRegistry()
        
        prompt = registry.build_tool_prompt(
            tool_categories=["research"],
            include_examples=True,
            format="markdown"
        )
        
        assert "Available Tools" in prompt
        assert "search_knowledge" in prompt
        assert "semantic_search" in prompt
        assert "Usage Instructions" in prompt
    
    def test_build_tool_prompt_empty(self):
        """Test building prompt with no matching categories"""
        registry = ToolRegistry()
        
        prompt = registry.build_tool_prompt(
            tool_categories=["nonexistent_category"],
            format="markdown"
        )
        
        assert prompt == ""
    
    def test_export_openai_functions(self):
        """Test exporting tools in OpenAI format"""
        registry = ToolRegistry()
        
        functions = registry.export_openai_functions(tool_categories=["research"])
        
        assert len(functions) >= 3
        assert all("name" in f for f in functions)
        assert all("description" in f for f in functions)
        assert all("parameters" in f for f in functions)
        
        # Check specific tool
        search_knowledge = next((f for f in functions if f["name"] == "search_knowledge"), None)
        assert search_knowledge is not None
        assert "query" in search_knowledge["parameters"]["properties"]
    
    def test_get_statistics(self):
        """Test getting registry statistics"""
        registry = ToolRegistry()
        
        stats = registry.get_statistics()
        
        assert "total_tools" in stats
        assert "active_tools" in stats
        assert "categories" in stats
        assert "security_levels" in stats
        assert stats["total_tools"] > 0
        assert stats["active_tools"] > 0
    
    def test_validate_tool_access_nonexistent_tool(self):
        """Test validating access to nonexistent tool"""
        registry = ToolRegistry()
        
        has_access, error = registry.validate_tool_access(
            agent_id=1,
            tool_name="nonexistent_tool"
        )
        
        assert has_access is False
        assert "not found" in error.lower()
    
    def test_validate_tool_access_inactive_tool(self):
        """Test validating access to inactive tool"""
        registry = ToolRegistry()
        
        # Add inactive tool
        inactive = ToolSpec(
            name="inactive",
            category=ToolCategory.API_TOOLS,
            description="Test",
            executor_class="Test",
            executor_method="test",
            is_active=False
        )
        registry.register_tool(inactive)
        
        has_access, error = registry.validate_tool_access(
            agent_id=1,
            tool_name="inactive"
        )
        
        assert has_access is False
        assert "not active" in error.lower()
    
    def test_validate_tool_access_safe_tool(self):
        """Test validating access to safe tool"""
        registry = ToolRegistry()
        
        has_access, error = registry.validate_tool_access(
            agent_id=1,
            tool_name="search_knowledge"
        )
        
        assert has_access is True
        assert error is None
    
    def test_singleton_get_tool_registry(self):
        """Test global registry singleton"""
        reset_tool_registry()
        
        registry1 = get_tool_registry()
        registry2 = get_tool_registry()
        
        assert registry1 is registry2  # Same instance
    
    def test_research_tools_registered(self):
        """Test that all research tools are registered correctly"""
        registry = ToolRegistry()
        
        # Check search_knowledge
        search_knowledge = registry.get_tool("search_knowledge")
        assert search_knowledge is not None
        assert search_knowledge.category == ToolCategory.RESEARCH
        assert search_knowledge.security_level == SecurityLevel.SAFE
        assert len(search_knowledge.parameters) == 2  # query, limit
        
        # Check semantic_search
        semantic_search = registry.get_tool("semantic_search")
        assert semantic_search is not None
        assert semantic_search.category == ToolCategory.RESEARCH
        
        # Check search_codebase
        search_codebase = registry.get_tool("search_codebase")
        assert search_codebase is not None
        assert search_codebase.category == ToolCategory.RESEARCH
    
    def test_file_operation_tools_registered(self):
        """Test that all file operation tools are registered correctly"""
        registry = ToolRegistry()
        
        # Check read_file
        read_file = registry.get_tool("read_file")
        assert read_file is not None
        assert read_file.category == ToolCategory.FILE_OPERATIONS
        assert read_file.security_level == SecurityLevel.SAFE
        
        # Check write_file
        write_file = registry.get_tool("write_file")
        assert write_file is not None
        assert write_file.security_level == SecurityLevel.CAUTIOUS
        
        # Check delete_file
        delete_file = registry.get_tool("delete_file")
        assert delete_file is not None
        assert delete_file.security_level == SecurityLevel.DANGEROUS
        
        # Check list_directory
        list_dir = registry.get_tool("list_directory")
        assert list_dir is not None
        
        # Check create_directory
        create_dir = registry.get_tool("create_directory")
        assert create_dir is not None
    
    def test_shell_command_tools_registered(self):
        """Test that shell command tools are registered correctly"""
        registry = ToolRegistry()
        
        execute_command = registry.get_tool("execute_command")
        
        assert execute_command is not None
        assert execute_command.category == ToolCategory.SHELL_COMMANDS
        assert execute_command.security_level == SecurityLevel.DANGEROUS
        assert "whitelisted_commands" in execute_command.metadata
    
    def test_tool_prompt_includes_all_categories(self):
        """Test that tool prompt includes tools from all requested categories"""
        registry = ToolRegistry()
        
        prompt = registry.build_tool_prompt(
            tool_categories=["research", "file_ops", "shell"],
            include_examples=True
        )
        
        # Should include tools from all categories
        assert "search_knowledge" in prompt
        assert "read_file" in prompt
        assert "execute_command" in prompt
        assert "Usage Instructions" in prompt


class TestToolRegistryWithDatabase:
    """Test ToolRegistry with database integration"""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session"""
        db = Mock()
        return db
    
    def test_registry_with_db_session(self, mock_db):
        """Test registry initialization with database session"""
        # Mock MCP tools query
        mock_query = Mock()
        mock_filter = Mock()
        mock_filter.all.return_value = []
        mock_query.filter.return_value = mock_filter
        mock_db.query.return_value = mock_query
        
        registry = ToolRegistry(db_session=mock_db)
        
        assert registry.db == mock_db
        # Should have attempted to load MCP tools
        mock_db.query.assert_called()


class TestToolRegistryStatistics:
    """Test registry statistics and reporting"""
    
    def test_statistics_structure(self):
        """Test statistics return correct structure"""
        registry = ToolRegistry()
        
        stats = registry.get_statistics()
        
        assert "total_tools" in stats
        assert "active_tools" in stats
        assert "categories" in stats
        assert "security_levels" in stats
        assert "last_updated" in stats
    
    def test_statistics_category_counts(self):
        """Test category counts in statistics"""
        registry = ToolRegistry()
        
        stats = registry.get_statistics()
        categories = stats["categories"]
        
        # Should have counts for each category
        assert categories["research"] >= 3
        assert categories["file_ops"] >= 5
        assert categories["shell"] >= 1
    
    def test_statistics_security_counts(self):
        """Test security level counts"""
        registry = ToolRegistry()
        
        stats = registry.get_statistics()
        security = stats["security_levels"]
        
        # Should have tools at different security levels
        assert security.get("safe", 0) > 0
        assert security.get("cautious", 0) > 0
        assert security.get("dangerous", 0) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

