"""
Unit Tests for Tool Capability Mapper - PRD-17
===============================================

Tests for task-to-tool mapping system including:
- Task type to tool category mapping
- Task inference from descriptions
- Custom mapping support
- Tool recommendations
- Security level determination
"""

import pytest
from unittest.mock import Mock
from services.tool_capability_mapper import (
    ToolCapabilityMapper,
    ToolRecommendation,
    get_tool_capability_mapper,
    reset_tool_capability_mapper
)


class TestToolCapabilityMapper:
    """Test ToolCapabilityMapper class"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        reset_tool_capability_mapper()
        yield
        reset_tool_capability_mapper()
    
    def test_mapper_initialization(self):
        """Test mapper initializes with predefined mappings"""
        mapper = ToolCapabilityMapper()
        
        task_types = mapper.get_all_task_types()
        
        assert len(task_types) > 0
        assert "code_review" in task_types
        assert "bug_fix" in task_types
        assert "security_audit" in task_types
        assert "general" in task_types
    
    def test_get_tool_categories_code_review(self):
        """Test getting tool categories for code review task"""
        mapper = ToolCapabilityMapper()
        
        categories = mapper.get_tool_categories_for_task("code_review")
        
        assert "research" in categories
        assert "file_ops" in categories
        # Should not include optional by default
        assert len(categories) == 2
    
    def test_get_tool_categories_with_optional(self):
        """Test getting tool categories including optional"""
        mapper = ToolCapabilityMapper()
        
        categories = mapper.get_tool_categories_for_task(
            "code_review",
            include_optional=True
        )
        
        assert "research" in categories
        assert "file_ops" in categories
        assert "mcp" in categories  # Optional for code review
    
    def test_get_tool_categories_bug_fix(self):
        """Test tool categories for bug fix task"""
        mapper = ToolCapabilityMapper()
        
        categories = mapper.get_tool_categories_for_task("bug_fix")
        
        assert "research" in categories
        assert "file_ops" in categories
        assert "shell" in categories
    
    def test_get_tool_categories_server_restart(self):
        """Test tool categories for server restart task"""
        mapper = ToolCapabilityMapper()
        
        categories = mapper.get_tool_categories_for_task("server_restart")
        
        assert "shell" in categories
        # Should not include research by default
        assert "research" not in categories
    
    def test_get_tool_categories_unknown_task(self):
        """Test fallback for unknown task type"""
        mapper = ToolCapabilityMapper()
        
        categories = mapper.get_tool_categories_for_task("completely_unknown_task_xyz")
        
        # Should fallback to general mapping
        assert "research" in categories
        assert len(categories) >= 1
    
    def test_infer_task_type_from_description(self):
        """Test inferring task type from description"""
        mapper = ToolCapabilityMapper()
        
        # Test code review inference
        task_type = mapper._infer_task_type("Please review this pull request for security issues")
        assert task_type == "code_review"
        
        # Test bug fix inference
        task_type = mapper._infer_task_type("Fix the authentication bug in login module")
        assert task_type == "bug_fix"
        
        # Test server restart inference
        task_type = mapper._infer_task_type("Restart the production web server")
        assert task_type == "server_restart"
        
        # Test database update inference
        task_type = mapper._infer_task_type("Update the user database schema")
        assert task_type == "database_update"
    
    def test_infer_task_type_ambiguous(self):
        """Test inference with ambiguous description"""
        mapper = ToolCapabilityMapper()
        
        task_type = mapper._infer_task_type("Do something with the system")
        
        # Should fallback to general
        assert task_type == "general"
    
    def test_get_recommendation_with_task_type(self):
        """Test getting recommendation with explicit task type"""
        mapper = ToolCapabilityMapper()
        
        recommendation = mapper.get_recommendation(
            task_description="Review code for security",
            task_type="code_review"
        )
        
        assert isinstance(recommendation, ToolRecommendation)
        assert recommendation.task_type == "code_review"
        assert "research" in recommendation.required_categories
        assert "file_ops" in recommendation.required_categories
        assert recommendation.confidence >= 0.9
        assert len(recommendation.rationale) > 0
    
    def test_get_recommendation_without_task_type(self):
        """Test getting recommendation with inferred task type"""
        mapper = ToolCapabilityMapper()
        
        recommendation = mapper.get_recommendation(
            task_description="Fix the authentication bug in the login system"
        )
        
        assert recommendation.task_type == "bug_fix"
        assert "research" in recommendation.required_categories
        assert "file_ops" in recommendation.required_categories
        assert "shell" in recommendation.required_categories
    
    def test_add_custom_mapping(self):
        """Test adding custom task-to-tool mapping"""
        mapper = ToolCapabilityMapper()
        
        mapper.add_custom_mapping(
            task_type="custom_task",
            required_categories=["research", "api"],
            optional_categories=["file_ops"],
            specific_tools=["search_knowledge"],
            confidence=0.85
        )
        
        categories = mapper.get_tool_categories_for_task("custom_task")
        
        assert "research" in categories
        assert "api" in categories
    
    def test_get_security_level_safe_tasks(self):
        """Test security level for safe tasks"""
        mapper = ToolCapabilityMapper()
        
        level = mapper.get_security_level_for_task("research")
        assert level == "safe"
        
        level = mapper.get_security_level_for_task("documentation")
        assert level == "safe"
    
    def test_get_security_level_cautious_tasks(self):
        """Test security level for cautious tasks"""
        mapper = ToolCapabilityMapper()
        
        level = mapper.get_security_level_for_task("code_review")
        assert level == "cautious"
        
        level = mapper.get_security_level_for_task("create_pr")
        assert level == "cautious"
    
    def test_get_security_level_dangerous_tasks(self):
        """Test security level for dangerous tasks"""
        mapper = ToolCapabilityMapper()
        
        level = mapper.get_security_level_for_task("server_restart")
        assert level == "dangerous"
        
        level = mapper.get_security_level_for_task("deployment")
        assert level == "dangerous"
        
        level = mapper.get_security_level_for_task("database_update")
        assert level == "dangerous"
    
    def test_validate_tool_combination_safe(self):
        """Test validating safe tool combinations"""
        mapper = ToolCapabilityMapper()
        
        is_valid, warning = mapper.validate_tool_combination(["research", "file_ops"])
        
        assert is_valid is True
        assert warning is None
    
    def test_validate_tool_combination_shell_ssh(self):
        """Test validating shell + SSH combination (dangerous)"""
        mapper = ToolCapabilityMapper()
        
        is_valid, warning = mapper.validate_tool_combination(["shell", "ssh"])
        
        assert is_valid is True
        assert warning is not None
        assert "remote command execution" in warning.lower()
    
    def test_validate_tool_combination_database_shell(self):
        """Test validating database + shell combination"""
        mapper = ToolCapabilityMapper()
        
        is_valid, warning = mapper.validate_tool_combination(["database", "shell"])
        
        assert is_valid is True
        assert warning is not None
        assert "database" in warning.lower()
    
    def test_get_minimal_tool_set(self):
        """Test getting minimal tool set (required only)"""
        mapper = ToolCapabilityMapper()
        
        minimal = mapper.get_minimal_tool_set("code_review")
        
        assert "research" in minimal
        assert "file_ops" in minimal
        # Should not include optional
        assert "mcp" not in minimal
    
    def test_get_extended_tool_set(self):
        """Test getting extended tool set (required + optional)"""
        mapper = ToolCapabilityMapper()
        
        extended = mapper.get_extended_tool_set("code_review")
        
        assert "research" in extended
        assert "file_ops" in extended
        assert "mcp" in extended  # Optional included
    
    def test_analyze_task_requirements_basic(self):
        """Test analyzing task requirements without context"""
        mapper = ToolCapabilityMapper()
        
        analysis = mapper.analyze_task_requirements(
            task_description="Review this code for security vulnerabilities",
            task_type="code_review"
        )
        
        assert analysis["task_type"] == "code_review"
        assert "required_categories" in analysis["recommended_tools"]
        assert "research" in analysis["recommended_tools"]["required_categories"]
        assert analysis["confidence"] >= 0.9
        assert len(analysis["rationale"]) > 0
    
    def test_analyze_task_requirements_with_context(self):
        """Test analyzing task requirements with additional context"""
        mapper = ToolCapabilityMapper()
        
        analysis = mapper.analyze_task_requirements(
            task_description="Review code",
            task_type="code_review",
            additional_context={
                "repository": "github.com/user/repo",
                "branch": "main"
            }
        )
        
        # Should suggest MCP tools based on repository context
        suggested = analysis["recommended_tools"].get("context_suggested", [])
        assert "mcp" in suggested or "file_ops" in suggested
    
    def test_analyze_task_requirements_database_context(self):
        """Test context analysis for database-related tasks"""
        mapper = ToolCapabilityMapper()
        
        analysis = mapper.analyze_task_requirements(
            task_description="Update system",
            task_type="general",
            additional_context={
                "database": "postgresql",
                "migration": "add_users_table"
            }
        )
        
        suggested = analysis["recommended_tools"]["context_suggested"]
        assert "database" in suggested
    
    def test_record_tool_usage(self):
        """Test recording tool usage for learning"""
        mapper = ToolCapabilityMapper()
        
        # Should not raise error
        mapper.record_tool_usage(
            task_type="code_review",
            tools_used=["research", "file_ops"],
            success=True
        )
    
    def test_singleton_get_mapper(self):
        """Test global mapper singleton"""
        reset_tool_capability_mapper()
        
        mapper1 = get_tool_capability_mapper()
        mapper2 = get_tool_capability_mapper()
        
        assert mapper1 is mapper2  # Same instance
    
    def test_all_predefined_task_types(self):
        """Test that all predefined task types have valid mappings"""
        mapper = ToolCapabilityMapper()
        
        for task_type in mapper.TASK_TOOL_MAPPINGS.keys():
            categories = mapper.get_tool_categories_for_task(task_type)
            
            # Each mapping should have at least one required category
            assert len(categories) > 0, f"Task type '{task_type}' has no required categories"
            
            # Each should have a rationale
            mapping = mapper.TASK_TOOL_MAPPINGS[task_type]
            assert "rationale" in mapping, f"Task type '{task_type}' missing rationale"
            assert len(mapping["rationale"]) > 0
    
    def test_task_type_normalization(self):
        """Test that task types are normalized correctly"""
        mapper = ToolCapabilityMapper()
        
        # Different formats should map to same result
        categories1 = mapper.get_tool_categories_for_task("code_review")
        categories2 = mapper.get_tool_categories_for_task("Code Review")
        categories3 = mapper.get_tool_categories_for_task("code-review")
        
        assert categories1 == categories2
        assert categories1 == categories3


class TestToolRecommendation:
    """Test ToolRecommendation dataclass"""
    
    def test_recommendation_creation(self):
        """Test creating a tool recommendation"""
        rec = ToolRecommendation(
            task_type="code_review",
            required_categories=["research", "file_ops"],
            optional_categories=["mcp"],
            specific_tools=["search_codebase"],
            confidence=0.95,
            rationale="Code review requires analysis and file access"
        )
        
        assert rec.task_type == "code_review"
        assert len(rec.required_categories) == 2
        assert len(rec.optional_categories) == 1
        assert rec.confidence == 0.95


class TestTaskTypeInference:
    """Test task type inference from descriptions"""
    
    @pytest.fixture
    def mapper(self):
        return ToolCapabilityMapper()
    
    def test_infer_code_review(self, mapper):
        """Test inferring code review tasks"""
        descriptions = [
            "Review this pull request",
            "Code review for PR #123",
            "Please review the code changes"
        ]
        
        for desc in descriptions:
            task_type = mapper._infer_task_type(desc)
            assert task_type == "code_review", f"Failed for: {desc}"
    
    def test_infer_bug_fix(self, mapper):
        """Test inferring bug fix tasks"""
        descriptions = [
            "Fix the login bug",
            "Debug the authentication error",
            "Resolve issue #456"
        ]
        
        for desc in descriptions:
            task_type = mapper._infer_task_type(desc)
            assert task_type == "bug_fix", f"Failed for: {desc}"
    
    def test_infer_security_audit(self, mapper):
        """Test inferring security audit tasks"""
        descriptions = [
            "Security audit of the codebase",
            "Check for vulnerabilities",
            "Penetration test the API"
        ]
        
        for desc in descriptions:
            task_type = mapper._infer_task_type(desc)
            assert task_type in ["security_audit", "penetration_test"], f"Failed for: {desc}"
    
    def test_infer_deployment(self, mapper):
        """Test inferring deployment tasks"""
        descriptions = [
            "Deploy to production",
            "Release version 2.0",
            "Publish the application"
        ]
        
        for desc in descriptions:
            task_type = mapper._infer_task_type(desc)
            assert task_type == "deployment", f"Failed for: {desc}"
    
    def test_infer_fallback_general(self, mapper):
        """Test fallback to general for unknown descriptions"""
        descriptions = [
            "Do something vague",
            "Handle this thing",
            "Complete the task"
        ]
        
        for desc in descriptions:
            task_type = mapper._infer_task_type(desc)
            assert task_type == "general", f"Failed for: {desc}"


class TestComprehensiveTaskCoverage:
    """Test that all major task types are covered"""
    
    def test_code_tasks_covered(self):
        """Test coverage of code-related tasks"""
        mapper = ToolCapabilityMapper()
        
        code_tasks = [
            "code_review", "bug_fix", "code_refactor", "code_analysis"
        ]
        
        for task in code_tasks:
            categories = mapper.get_tool_categories_for_task(task)
            assert len(categories) > 0, f"No tools for {task}"
    
    def test_security_tasks_covered(self):
        """Test coverage of security tasks"""
        mapper = ToolCapabilityMapper()
        
        security_tasks = [
            "security_audit", "vulnerability_scan", "penetration_test"
        ]
        
        for task in security_tasks:
            categories = mapper.get_tool_categories_for_task(task)
            assert len(categories) > 0, f"No tools for {task}"
    
    def test_infrastructure_tasks_covered(self):
        """Test coverage of infrastructure tasks"""
        mapper = ToolCapabilityMapper()
        
        infra_tasks = [
            "server_restart", "deployment", "infrastructure_management",
            "container_management"
        ]
        
        for task in infra_tasks:
            categories = mapper.get_tool_categories_for_task(task)
            assert "shell" in categories, f"Shell not included for {task}"
    
    def test_database_tasks_covered(self):
        """Test coverage of database tasks"""
        mapper = ToolCapabilityMapper()
        
        db_tasks = [
            "database_update", "database_migration", "data_backup"
        ]
        
        for task in db_tasks:
            categories = mapper.get_tool_categories_for_task(task)
            assert len(categories) > 0, f"No tools for {task}"
    
    def test_development_tasks_covered(self):
        """Test coverage of development tasks"""
        mapper = ToolCapabilityMapper()
        
        dev_tasks = [
            "create_pr", "api_design", "testing", "ci_cd_setup"
        ]
        
        for task in dev_tasks:
            categories = mapper.get_tool_categories_for_task(task)
            assert len(categories) > 0, f"No tools for {task}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

