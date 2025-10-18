"""
Integration Tests for Tool Registry System - PRD-17
===================================================

Tests integration between ToolRegistry and ToolCapabilityMapper,
ensuring the complete system works end-to-end.
"""

import pytest
from services.tool_registry import (
    ToolRegistry,
    ToolCategory,
    get_tool_registry,
    reset_tool_registry
)
from services.tool_capability_mapper import (
    ToolCapabilityMapper,
    get_tool_capability_mapper,
    reset_tool_capability_mapper
)


class TestToolSystemIntegration:
    """Test integration between registry and mapper"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        reset_tool_registry()
        reset_tool_capability_mapper()
        yield
        reset_tool_registry()
        reset_tool_capability_mapper()
    
    def test_end_to_end_code_review_task(self):
        """Test complete flow for code review task"""
        # Step 1: Get tool recommendations
        mapper = ToolCapabilityMapper()
        recommendation = mapper.get_recommendation(
            task_description="Review this pull request for bugs and security issues",
            task_type="code_review"
        )
        
        assert recommendation.task_type == "code_review"
        assert "research" in recommendation.required_categories
        assert "file_ops" in recommendation.required_categories
        
        # Step 2: Get actual tools from registry
        registry = ToolRegistry()
        tools = registry.get_tools_for_categories(recommendation.required_categories)
        
        assert len(tools) > 0
        
        # Step 3: Verify we have the right tools
        tool_names = [t.name for t in tools]
        assert "search_knowledge" in tool_names  # Research
        assert "read_file" in tool_names  # File ops
        
        # Step 4: Generate tool prompt
        prompt = registry.build_tool_prompt(
            tool_categories=recommendation.required_categories,
            include_examples=True
        )
        
        assert "search_knowledge" in prompt
        assert "read_file" in prompt
        assert len(prompt) > 100  # Should be substantial
    
    def test_end_to_end_bug_fix_task(self):
        """Test complete flow for bug fix task"""
        mapper = ToolCapabilityMapper()
        registry = ToolRegistry()
        
        # Analyze task
        analysis = mapper.analyze_task_requirements(
            task_description="Fix authentication bug in login.py",
            task_type="bug_fix"
        )
        
        assert analysis["task_type"] == "bug_fix"
        
        # Get tools
        categories = analysis["all_categories"]
        tools = registry.get_tools_for_categories(categories)
        
        # Should have file operations
        file_tools = [t for t in tools if t.category == ToolCategory.FILE_OPERATIONS]
        assert len(file_tools) > 0
        
        # Should have research tools
        research_tools = [t for t in tools if t.category == ToolCategory.RESEARCH]
        assert len(research_tools) > 0
        
        # Should have shell tools
        shell_tools = [t for t in tools if t.category == ToolCategory.SHELL_COMMANDS]
        assert len(shell_tools) > 0
    
    def test_end_to_end_server_restart_task(self):
        """Test complete flow for server restart task"""
        mapper = ToolCapabilityMapper()
        registry = ToolRegistry()
        
        # Get recommendation
        rec = mapper.get_recommendation(
            task_description="Restart the web server and check logs"
        )
        
        # Should infer as server_restart or infrastructure
        assert rec.task_type in ["server_restart", "infrastructure_management"]
        
        # Should require shell
        assert "shell" in rec.required_categories
        
        # Get tools
        tools = registry.get_tools_for_categories(rec.required_categories)
        
        # Should have execute_command
        tool_names = [t.name for t in tools]
        assert "execute_command" in tool_names
    
    def test_tool_availability_for_all_task_types(self):
        """Test that all predefined task types have available tools"""
        mapper = ToolCapabilityMapper()
        registry = ToolRegistry()
        
        task_types = mapper.get_all_task_types()
        
        for task_type in task_types:
            # Get required categories
            categories = mapper.get_tool_categories_for_task(task_type)
            
            # Get actual tools
            tools = registry.get_tools_for_categories(categories)
            
            # Every task type should have at least one tool
            assert len(tools) > 0, f"Task type '{task_type}' has no available tools"
    
    def test_security_levels_match_categories(self):
        """Test that security levels are appropriate for tool categories"""
        registry = ToolRegistry()
        
        # Research tools should be safe
        research_tools = registry.get_tools_by_category(ToolCategory.RESEARCH)
        for tool in research_tools:
            assert tool.security_level == SecurityLevel.SAFE
        
        # Shell commands should be dangerous
        shell_tools = registry.get_tools_by_category(ToolCategory.SHELL_COMMANDS)
        for tool in shell_tools:
            assert tool.security_level == SecurityLevel.DANGEROUS
        
        # Delete operations should be dangerous
        delete_tool = registry.get_tool("delete_file")
        assert delete_tool.security_level == SecurityLevel.DANGEROUS
    
    def test_openai_format_export_for_task(self):
        """Test exporting tools in OpenAI format for a specific task"""
        mapper = ToolCapabilityMapper()
        registry = ToolRegistry()
        
        # Get tools for code review
        categories = mapper.get_tool_categories_for_task("code_review")
        functions = registry.export_openai_functions(tool_categories=categories)
        
        # Should have multiple functions
        assert len(functions) > 0
        
        # All should follow OpenAI format
        for func in functions:
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"
            assert "properties" in func["parameters"]
    
    def test_prompt_generation_for_multiple_categories(self):
        """Test generating prompt with multiple tool categories"""
        registry = ToolRegistry()
        
        prompt = registry.build_tool_prompt(
            tool_categories=["research", "file_ops", "shell"],
            include_examples=True
        )
        
        # Should have sections for each category
        assert "research" in prompt.lower()
        assert "file" in prompt.lower() or "file_ops" in prompt.lower()
        assert "shell" in prompt.lower() or "command" in prompt.lower()
        
        # Should have usage instructions
        assert "usage instructions" in prompt.lower()
    
    def test_custom_mapping_integration(self):
        """Test adding and using custom mappings"""
        mapper = ToolCapabilityMapper()
        registry = ToolRegistry()
        
        # Add custom mapping
        mapper.add_custom_mapping(
            task_type="ml_training",
            required_categories=["file_ops", "shell"],
            optional_categories=["database"],
            specific_tools=["execute_command"],
            confidence=0.88
        )
        
        # Use the custom mapping
        rec = mapper.get_recommendation(
            task_description="Train ML model",
            task_type="ml_training"
        )
        
        assert rec.task_type == "ml_training"
        assert "file_ops" in rec.required_categories
        assert "shell" in rec.required_categories
        assert rec.confidence == 0.88
        
        # Get actual tools
        tools = registry.get_tools_for_categories(rec.required_categories)
        assert len(tools) > 0


class TestRealWorldScenarios:
    """Test realistic usage scenarios"""
    
    @pytest.fixture
    def mapper(self):
        return ToolCapabilityMapper()
    
    @pytest.fixture
    def registry(self):
        return ToolRegistry()
    
    def test_scenario_review_python_pr(self, mapper, registry):
        """Scenario: Review Python PR on GitHub"""
        task = "Review PR #456 in python-api repository for security vulnerabilities"
        
        # Analyze task
        analysis = mapper.analyze_task_requirements(
            task_description=task,
            additional_context={
                "repository": "github.com/company/python-api",
                "pr_number": 456,
                "focus": "security"
            }
        )
        
        # Should identify as code review or security audit
        assert analysis["task_type"] in ["code_review", "security_audit"]
        
        # Should recommend file and research tools
        assert "file_ops" in analysis["all_categories"]
        assert "research" in analysis["all_categories"]
    
    def test_scenario_fix_database_bug(self, mapper, registry):
        """Scenario: Fix database connection bug"""
        task = "Fix bug in database connection pool causing timeout errors"
        
        analysis = mapper.analyze_task_requirements(task)
        
        assert analysis["task_type"] == "bug_fix"
        
        # Should have file, shell, and research
        categories = analysis["all_categories"]
        assert "file_ops" in categories
        assert "shell" in categories
        assert "research" in categories
    
    def test_scenario_deploy_application(self, mapper, registry):
        """Scenario: Deploy application to production"""
        task = "Deploy the web application to AWS production environment"
        
        analysis = mapper.analyze_task_requirements(
            task_description=task,
            additional_context={"server": "aws-prod-01"}
        )
        
        assert analysis["task_type"] == "deployment"
        
        # Should require shell
        assert "shell" in analysis["all_categories"]
    
    def test_scenario_research_only(self, mapper, registry):
        """Scenario: Pure research task"""
        task = "Research best practices for API authentication"
        
        analysis = mapper.analyze_task_requirements(task)
        
        # Should be research task
        assert analysis["task_type"] in ["research", "competitive_analysis"]
        
        # Should only need research tools
        required = analysis["recommended_tools"]["required_categories"]
        assert required == ["research"]


class TestSecurityValidation:
    """Test security-related features"""
    
    def test_dangerous_tools_identified(self):
        """Test that dangerous tools are properly marked"""
        registry = ToolRegistry()
        
        execute_cmd = registry.get_tool("execute_command")
        delete_file = registry.get_tool("delete_file")
        
        assert execute_cmd.security_level == SecurityLevel.DANGEROUS
        assert delete_file.security_level == SecurityLevel.DANGEROUS
    
    def test_safe_tools_identified(self):
        """Test that safe tools are properly marked"""
        registry = ToolRegistry()
        
        search_knowledge = registry.get_tool("search_knowledge")
        read_file = registry.get_tool("read_file")
        list_dir = registry.get_tool("list_directory")
        
        assert search_knowledge.security_level == SecurityLevel.SAFE
        assert read_file.security_level == SecurityLevel.SAFE
        assert list_dir.security_level == SecurityLevel.SAFE
    
    def test_security_level_for_task_types(self):
        """Test security level recommendations for different task types"""
        mapper = ToolCapabilityMapper()
        
        # Safe tasks
        assert mapper.get_security_level_for_task("research") == "safe"
        assert mapper.get_security_level_for_task("documentation") == "safe"
        
        # Cautious tasks
        assert mapper.get_security_level_for_task("code_review") == "cautious"
        
        # Dangerous tasks
        assert mapper.get_security_level_for_task("server_restart") == "dangerous"
        assert mapper.get_security_level_for_task("deployment") == "dangerous"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

