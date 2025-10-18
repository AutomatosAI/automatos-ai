"""
Tool Capability Mapper - Task-to-Tool Recommendation System
============================================================
PRD-17: Dynamic Tool Assignment & Centralized Tool Management

Maps task types to required tool categories, enabling dynamic tool assignment.

Features:
- Predefined mappings for common task types
- LLM-driven classification for unknown tasks
- Database-backed custom mappings
- Confidence scoring for recommendations
- Usage-based learning and optimization
"""

import logging
from typing import Dict, List, Optional, Set, Any
from sqlalchemy.orm import Session
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolRecommendation:
    """Recommendation for tools needed for a task"""
    task_type: str
    required_categories: List[str]      # Must have these categories
    optional_categories: List[str]      # Nice to have
    specific_tools: List[str]           # Specific tool names
    confidence: float                   # 0.0-1.0 confidence in recommendation
    rationale: str                      # Explanation of recommendation


class ToolCapabilityMapper:
    """
    Maps task types to required tool capabilities.
    
    Provides intelligent recommendations for which tools agents need
    based on the type of task they're performing.
    """
    
    # Predefined task-to-tool mappings
    # Format: task_type -> (required_categories, optional_categories, specific_tools)
    TASK_TOOL_MAPPINGS = {
        # Code-related tasks
        "code_review": {
            "required": ["research", "file_ops"],
            "optional": ["mcp"],
            "specific": ["search_codebase", "search_knowledge", "GitHub MCP"],  # PRD-17 Phase 3
            "rationale": "Code review requires reading files and researching best practices"
        },
        "bug_fix": {
            "required": ["research", "file_ops", "shell"],
            "optional": ["mcp"],
            "specific": ["search_codebase", "GitHub MCP"],  # PRD-17 Phase 3
            "rationale": "Bug fixing requires code analysis, file modifications, and testing via shell"
        },
        "code_refactor": {
            "required": ["research", "file_ops"],
            "optional": ["shell", "mcp"],
            "specific": [],
            "rationale": "Refactoring requires code analysis and file modifications"
        },
        "code_analysis": {
            "required": ["research", "file_ops"],
            "optional": [],
            "specific": ["search_codebase"],
            "rationale": "Code analysis requires codebase search and file reading"
        },
        
        # Security-related tasks
        "security_audit": {
            "required": ["research", "file_ops", "shell"],
            "optional": ["mcp"],
            "specific": ["search_codebase"],
            "rationale": "Security audits require code analysis, file inspection, and running security tools"
        },
        "vulnerability_scan": {
            "required": ["research", "file_ops", "shell"],
            "optional": ["mcp"],
            "specific": [],
            "rationale": "Vulnerability scanning requires file analysis and running security scanners"
        },
        "penetration_test": {
            "required": ["shell", "file_ops"],
            "optional": ["ssh", "mcp"],
            "specific": [],
            "rationale": "Penetration testing requires command execution and file operations"
        },
        
        # Infrastructure tasks
        "server_restart": {
            "required": ["shell"],
            "optional": ["ssh", "mcp"],
            "specific": [],
            "rationale": "Server restart requires shell command execution"
        },
        "deployment": {
            "required": ["shell", "file_ops"],
            "optional": ["ssh", "mcp"],
            "specific": [],
            "rationale": "Deployment requires file operations and command execution"
        },
        "infrastructure_management": {
            "required": ["shell"],
            "optional": ["ssh", "mcp", "file_ops"],
            "specific": [],
            "rationale": "Infrastructure management requires shell commands and potentially SSH access"
        },
        "container_management": {
            "required": ["shell"],
            "optional": ["mcp", "file_ops"],
            "specific": [],
            "rationale": "Container management requires docker commands via shell"
        },
        
        # Database tasks
        "database_update": {
            "required": ["database", "shell"],
            "optional": ["file_ops"],
            "specific": [],
            "rationale": "Database updates require SQL operations and command execution"
        },
        "database_migration": {
            "required": ["database", "file_ops", "shell"],
            "optional": ["mcp"],
            "specific": [],
            "rationale": "Migrations require SQL operations, file reading, and command execution"
        },
        "data_backup": {
            "required": ["shell", "file_ops"],
            "optional": ["ssh", "mcp"],
            "specific": [],
            "rationale": "Backups require command execution and file operations"
        },
        
        # Development tasks
        "create_pr": {
            "required": ["file_ops", "mcp"],
            "optional": ["shell", "research"],
            "specific": ["GitHub MCP"],  # PRD-17 Phase 3: Specific GitHub integration
            "rationale": "Creating PRs requires file operations and GitHub integration"
        },
        "repository_management": {
            "required": ["mcp"],
            "optional": ["file_ops", "research"],
            "specific": ["GitHub MCP"],  # PRD-17 Phase 3: GitHub operations
            "rationale": "Repository management requires GitHub API access"
        },
        "clone_repository": {
            "required": ["mcp", "shell"],
            "optional": ["file_ops"],
            "specific": ["GitHub MCP"],  # PRD-17 Phase 3: Get repo details before cloning
            "rationale": "Cloning requires GitHub API access and shell commands"
        },
        "api_design": {
            "required": ["research", "file_ops"],
            "optional": ["mcp"],
            "specific": [],
            "rationale": "API design requires research and file creation"
        },
        "testing": {
            "required": ["file_ops", "shell"],
            "optional": ["research", "mcp"],
            "specific": [],
            "rationale": "Testing requires file operations and running test commands"
        },
        "ci_cd_setup": {
            "required": ["file_ops", "shell"],
            "optional": ["mcp"],
            "specific": [],
            "rationale": "CI/CD setup requires configuration files and shell commands"
        },
        
        # Documentation tasks
        "documentation": {
            "required": ["research", "file_ops"],
            "optional": [],
            "specific": ["search_knowledge", "search_codebase"],
            "rationale": "Documentation requires research and file writing"
        },
        "technical_writing": {
            "required": ["research", "file_ops"],
            "optional": [],
            "specific": ["search_knowledge"],
            "rationale": "Technical writing requires research and document creation"
        },
        
        # Data tasks
        "data_analysis": {
            "required": ["research", "file_ops"],
            "optional": ["database", "shell"],
            "specific": [],
            "rationale": "Data analysis requires research, file operations, and potentially database queries"
        },
        "data_processing": {
            "required": ["file_ops", "shell"],
            "optional": ["database", "research"],
            "specific": [],
            "rationale": "Data processing requires file operations and command execution"
        },
        "etl_pipeline": {
            "required": ["database", "file_ops", "shell"],
            "optional": ["mcp"],
            "specific": [],
            "rationale": "ETL pipelines require database operations, file handling, and command execution"
        },
        
        # Research/Analysis tasks
        "research": {
            "required": ["research"],
            "optional": [],
            "specific": ["search_knowledge", "semantic_search"],
            "rationale": "Research tasks only need knowledge search capabilities"
        },
        "competitive_analysis": {
            "required": ["research"],
            "optional": ["api"],
            "specific": ["search_knowledge", "semantic_search"],
            "rationale": "Competitive analysis requires extensive research capabilities"
        },
        
        # General/Fallback
        "general": {
            "required": ["research"],
            "optional": [],
            "specific": [],
            "rationale": "General tasks default to research capabilities for safety"
        },
        "unknown": {
            "required": ["research"],
            "optional": [],
            "specific": [],
            "rationale": "Unknown tasks default to safe research-only capabilities"
        }
    }
    
    def __init__(self, db_session: Optional[Session] = None):
        self.db = db_session
        self.logger = logger
        
        # Load custom mappings from database if available
        self.custom_mappings: Dict[str, Dict[str, Any]] = {}
        if self.db:
            self._load_custom_mappings()
        
        self.logger.info(f"ToolCapabilityMapper initialized with {len(self.TASK_TOOL_MAPPINGS)} predefined mappings")
    
    def get_tool_categories_for_task(
        self,
        task_type: str,
        include_optional: bool = False
    ) -> List[str]:
        """
        Get tool categories needed for a task type.
        
        Args:
            task_type: Type of task (code_review, bug_fix, etc.)
            include_optional: Whether to include optional categories
        
        Returns:
            List of tool category names (e.g., ["research", "file_ops"])
        """
        # Normalize task type
        task_type_normalized = task_type.lower().replace(" ", "_").replace("-", "_")
        
        # Check custom mappings first
        if task_type_normalized in self.custom_mappings:
            mapping = self.custom_mappings[task_type_normalized]
        # Then check predefined mappings
        elif task_type_normalized in self.TASK_TOOL_MAPPINGS:
            mapping = self.TASK_TOOL_MAPPINGS[task_type_normalized]
        else:
            # Fallback to general
            self.logger.warning(f"Unknown task type '{task_type}', using 'general' fallback")
            mapping = self.TASK_TOOL_MAPPINGS["general"]
        
        categories = mapping["required"].copy()
        
        if include_optional:
            categories.extend(mapping.get("optional", []))
        
        return categories
    
    def get_recommendation(
        self,
        task_description: str,
        task_type: Optional[str] = None
    ) -> ToolRecommendation:
        """
        Get comprehensive tool recommendation for a task.
        
        Args:
            task_description: Full task description
            task_type: Optional explicit task type
        
        Returns:
            ToolRecommendation with required/optional tools and rationale
        """
        # If task type not provided, try to infer from description
        if not task_type:
            task_type = self._infer_task_type(task_description)
        
        # Normalize
        task_type_normalized = task_type.lower().replace(" ", "_").replace("-", "_")
        
        # Get mapping
        if task_type_normalized in self.custom_mappings:
            mapping = self.custom_mappings[task_type_normalized]
            confidence = mapping.get("confidence", 0.9)
        elif task_type_normalized in self.TASK_TOOL_MAPPINGS:
            mapping = self.TASK_TOOL_MAPPINGS[task_type_normalized]
            confidence = 1.0
        else:
            mapping = self.TASK_TOOL_MAPPINGS["general"]
            confidence = 0.5
            self.logger.warning(f"Using fallback mapping for task type: {task_type}")
        
        return ToolRecommendation(
            task_type=task_type_normalized,
            required_categories=mapping["required"],
            optional_categories=mapping.get("optional", []),
            specific_tools=mapping.get("specific", []),
            confidence=confidence,
            rationale=mapping.get("rationale", "Tool recommendation based on task type")
        )
    
    def _infer_task_type(self, task_description: str) -> str:
        """
        Infer task type from task description using keyword matching.
        
        For more complex inference, this could use an LLM, but keyword
        matching is fast and works well for most cases.
        
        Args:
            task_description: Task description text
        
        Returns:
            Inferred task type
        """
        description_lower = task_description.lower()
        
        # Keyword-based inference (expanded for better matching)
        keyword_mappings = {
            "code_review": ["review code", "code review", "pr review", "pull request", "review this", "review pr"],
            "bug_fix": ["fix bug", "debug", "issue", "error", "broken", "fix the", "fix authentication", "resolve issue"],
            "security_audit": ["security", "vulnerability", "audit", "penetration", "vulnerabilities", "security issues"],
            "server_restart": ["restart server", "reboot", "restart service", "restart the", "web server"],
            "deployment": ["deploy", "release", "publish", "deploy to", "deploy the"],
            "database_update": ["database", "sql", "update db", "migrate", "schema", "update the user database"],
            "create_pr": ["create pr", "pull request", "github pr"],
            "documentation": ["document", "write docs", "readme"],
            "data_analysis": ["analyze data", "data analysis", "statistics"],
            "testing": ["test", "unit test", "integration test", "qa"],
            "research": ["research", "best practices", "find information", "learn about"],
        }
        
        # Score each task type based on keyword matches
        scores = {}
        for task_type, keywords in keyword_mappings.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            if score > 0:
                scores[task_type] = score
        
        # Return highest scoring task type
        if scores:
            best_match = max(scores.items(), key=lambda x: x[1])
            self.logger.info(f"Inferred task type '{best_match[0]}' from description (score: {best_match[1]})")
            return best_match[0]
        
        # Fallback to general
        self.logger.info(f"Could not infer task type from description, using 'general'")
        return "general"
    
    def _load_custom_mappings(self):
        """Load custom task-to-tool mappings from database"""
        try:
            # This will be implemented when we create the database schema
            # For now, just log that we tried
            self.logger.info("Custom mappings: database schema not yet created, using predefined mappings only")
        except Exception as e:
            self.logger.warning(f"Could not load custom mappings: {e}")
    
    def add_custom_mapping(
        self,
        task_type: str,
        required_categories: List[str],
        optional_categories: Optional[List[str]] = None,
        specific_tools: Optional[List[str]] = None,
        confidence: float = 0.9
    ) -> None:
        """
        Add a custom task-to-tool mapping.
        
        Args:
            task_type: Type of task
            required_categories: Required tool categories
            optional_categories: Optional tool categories
            specific_tools: Specific tool names
            confidence: Confidence score (0.0-1.0)
        """
        self.custom_mappings[task_type] = {
            "required": required_categories,
            "optional": optional_categories or [],
            "specific": specific_tools or [],
            "confidence": confidence,
            "rationale": f"Custom mapping for {task_type}"
        }
        
        self.logger.info(f"Added custom mapping for task type: {task_type}")
    
    def get_all_task_types(self) -> List[str]:
        """Get list of all known task types"""
        predefined = list(self.TASK_TOOL_MAPPINGS.keys())
        custom = list(self.custom_mappings.keys())
        return sorted(set(predefined + custom))
    
    def analyze_task_requirements(
        self,
        task_description: str,
        task_type: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a task and return comprehensive tool requirements.
        
        Args:
            task_description: Full task description
            task_type: Optional explicit task type
            additional_context: Additional context (workflow config, etc.)
        
        Returns:
            Dict with task analysis and tool recommendations
        """
        # Get recommendation
        recommendation = self.get_recommendation(task_description, task_type)
        
        # Analyze context for additional hints
        context_hints = []
        if additional_context:
            # Check for code-related keywords
            if any(key in additional_context for key in ["repository", "code", "branch", "pr"]):
                context_hints.append("file_ops")
                context_hints.append("mcp")
            
            # Check for infrastructure keywords
            if any(key in additional_context for key in ["server", "deploy", "infrastructure"]):
                context_hints.append("shell")
                context_hints.append("ssh")
            
            # Check for database keywords
            if any(key in additional_context for key in ["database", "sql", "migration"]):
                context_hints.append("database")
        
        # Merge hints with recommendation
        all_categories = set(recommendation.required_categories)
        all_categories.update(context_hints)
        
        return {
            "task_type": recommendation.task_type,
            "task_description": task_description,
            "recommended_tools": {
                "required_categories": recommendation.required_categories,
                "optional_categories": recommendation.optional_categories,
                "specific_tools": recommendation.specific_tools,
                "context_suggested": list(context_hints)
            },
            "all_categories": sorted(list(all_categories)),
            "confidence": recommendation.confidence,
            "rationale": recommendation.rationale,
            "analysis_timestamp": None  # Will be set when used
        }
    
    def get_security_level_for_task(self, task_type: str) -> str:
        """
        Determine appropriate security level for a task type.
        
        Args:
            task_type: Type of task
        
        Returns:
            Security level: "safe", "cautious", or "dangerous"
        """
        task_type_normalized = task_type.lower().replace(" ", "_")
        
        # Tasks that need dangerous operations
        dangerous_tasks = [
            "server_restart", "deployment", "database_update",
            "infrastructure_management", "penetration_test",
            "container_management", "bug_fix"
        ]
        
        # Tasks that need cautious operations
        cautious_tasks = [
            "code_review", "security_audit", "code_refactor",
            "create_pr", "testing", "ci_cd_setup"
        ]
        
        if task_type_normalized in dangerous_tasks:
            return "dangerous"
        elif task_type_normalized in cautious_tasks:
            return "cautious"
        else:
            return "safe"
    
    def validate_tool_combination(
        self,
        tool_categories: List[str]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that a tool combination is safe and sensible.
        
        Args:
            tool_categories: List of tool categories
        
        Returns:
            Tuple of (is_valid, warning_message)
        """
        # Check for dangerous combinations
        if "shell" in tool_categories and "ssh" in tool_categories:
            return True, "Warning: Shell + SSH combination allows remote command execution"
        
        if "database" in tool_categories and "shell" in tool_categories:
            return True, "Warning: Database + Shell allows direct database manipulation"
        
        # All combinations are valid, but some have warnings
        return True, None
    
    def get_minimal_tool_set(self, task_type: str) -> List[str]:
        """
        Get minimal tool set needed for a task (required only).
        
        Args:
            task_type: Type of task
        
        Returns:
            List of required tool categories
        """
        return self.get_tool_categories_for_task(task_type, include_optional=False)
    
    def get_extended_tool_set(self, task_type: str) -> List[str]:
        """
        Get extended tool set for a task (required + optional).
        
        Args:
            task_type: Type of task
        
        Returns:
            List of required + optional tool categories
        """
        return self.get_tool_categories_for_task(task_type, include_optional=True)
    
    def record_tool_usage(
        self,
        task_type: str,
        tools_used: List[str],
        success: bool,
        db: Optional[Session] = None
    ):
        """
        Record which tools were actually used for a task type.
        
        This helps the system learn and optimize mappings over time.
        
        Args:
            task_type: Type of task
            tools_used: Tool categories that were used
            success: Whether the task succeeded
            db: Database session
        """
        # Future: Store in database for learning
        # For now, just log
        self.logger.info(
            f"Tool usage recorded: task_type={task_type}, "
            f"tools={tools_used}, success={success}"
        )


# Global mapper instance
_tool_mapper: Optional[ToolCapabilityMapper] = None


def get_tool_capability_mapper(db_session: Optional[Session] = None) -> ToolCapabilityMapper:
    """
    Get or create the global tool capability mapper instance.
    
    Args:
        db_session: Optional database session
    
    Returns:
        ToolCapabilityMapper instance
    """
    global _tool_mapper
    
    if _tool_mapper is None:
        _tool_mapper = ToolCapabilityMapper(db_session=db_session)
    elif db_session and not _tool_mapper.db:
        _tool_mapper.db = db_session
        _tool_mapper._load_custom_mappings()
    
    return _tool_mapper


def reset_tool_capability_mapper():
    """Reset the global mapper (for testing)"""
    global _tool_mapper
    _tool_mapper = None

