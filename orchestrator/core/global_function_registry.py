"""
Global Function Registry - Central registry for ALL LLM functions
=================================================================

Consolidates functions from:
1. LLMAgentSelector functions (agent queries, performance, etc.)
2. Platform tools (file_ops, research, shell, etc.)
3. Future: Any other LLM-callable functions

This is the SINGLE SOURCE OF TRUTH for function calling across the platform.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FunctionCategory(Enum):
    """Categories of functions available to LLMs"""
    AGENT_QUERY = "agent_query"           # Query agent capabilities
    PERFORMANCE = "performance"           # Performance/metrics queries
    TASK_ANALYSIS = "task_analysis"       # Task requirement analysis
    TOOL_EXECUTION = "tool_execution"     # Platform tool execution
    COORDINATION = "coordination"         # Multi-agent coordination
    CUSTOM = "custom"                     # Custom functions


@dataclass
class FunctionParameter:
    """Function parameter definition"""
    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Optional[Any] = None


@dataclass
class FunctionSpec:
    """Complete function specification for LLM function calling"""
    name: str
    description: str
    parameters: List[FunctionParameter]
    category: FunctionCategory
    handler: Optional[Callable] = None  # Actual function to execute
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling schema"""
        properties = {}
        required = []
        
        for param in self.parameters:
            param_schema = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                param_schema["enum"] = param.enum
            
            properties[param.name] = param_schema
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


class GlobalFunctionRegistry:
    """
    CENTRAL REGISTRY for all LLM-callable functions.
    
    Provides:
    - Function registration from any source
    - Function lookup by name/category
    - OpenAI schema generation
    - Function execution routing
    
    Usage:
        registry = GlobalFunctionRegistry()
        registry.register_function(func_spec)
        schemas = registry.get_schemas_for_llm(category="agent_query")
    """
    
    def __init__(self):
        self.functions: Dict[str, FunctionSpec] = {}
        self._register_default_functions()
        logger.info("🌍 GlobalFunctionRegistry initialized with default functions")
    
    def register_function(self, spec: FunctionSpec):
        """Register a new function"""
        if spec.name in self.functions:
            logger.warning(f"⚠️ Overwriting function '{spec.name}'")
        
        self.functions[spec.name] = spec
        logger.debug(f"✅ Registered function '{spec.name}' (category: {spec.category.value})")
    
    def register_multiple(self, specs: List[FunctionSpec]):
        """Register multiple functions at once"""
        for spec in specs:
            self.register_function(spec)
    
    def get_function(self, name: str) -> Optional[FunctionSpec]:
        """Get function by name"""
        return self.functions.get(name)
    
    def get_functions_by_category(self, category: FunctionCategory) -> List[FunctionSpec]:
        """Get all functions in a category"""
        return [f for f in self.functions.values() if f.category == category]
    
    def get_all_functions(self) -> List[FunctionSpec]:
        """Get all registered functions"""
        return list(self.functions.values())
    
    def get_schemas_for_llm(
        self, 
        categories: Optional[List[FunctionCategory]] = None,
        names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get OpenAI function schemas for LLM.
        
        Args:
            categories: Filter by categories
            names: Filter by specific function names
        
        Returns: List of OpenAI function schemas
        """
        functions = self.functions.values()
        
        # Filter by categories
        if categories:
            functions = [f for f in functions if f.category in categories]
        
        # Filter by names
        if names:
            functions = [f for f in functions if f.name in names]
        
        schemas = [f.to_openai_schema() for f in functions]
        logger.debug(f"📋 Generated {len(schemas)} function schemas for LLM")
        
        return schemas
    
    def _register_default_functions(self):
        """Register default platform functions"""
        
        # ===== AGENT QUERY FUNCTIONS =====
        self.register_function(FunctionSpec(
            name="query_available_agents",
            description="Find agents matching specific skill or tool requirements. Returns agents with their capabilities.",
            parameters=[
                FunctionParameter(
                    name="skills",
                    type="array",
                    description="List of required skills (e.g., ['research', 'analysis'])",
                    required=False
                ),
                FunctionParameter(
                    name="tools",
                    type="array",
                    description="List of required tools (e.g., ['create_pdf', 'search_knowledge'])",
                    required=False
                ),
                FunctionParameter(
                    name="agent_type",
                    type="string",
                    description="Agent type filter (researcher, writer, analyst, etc.)",
                    required=False
                )
            ],
            category=FunctionCategory.AGENT_QUERY,
            handler=None  # Actual handler injected by LLMAgentSelector
        ))
        
        self.register_function(FunctionSpec(
            name="get_agent_performance_history",
            description="Get performance metrics for a specific agent including success rate, quality scores, and task history.",
            parameters=[
                FunctionParameter(
                    name="agent_id",
                    type="integer",
                    description="Agent ID to query performance for"
                ),
                FunctionParameter(
                    name="task_type",
                    type="string",
                    description="Filter by task type (research, code_review, etc.)",
                    required=False
                ),
                FunctionParameter(
                    name="time_window_days",
                    type="integer",
                    description="Number of days to look back (default: 30)",
                    required=False,
                    default=30
                )
            ],
            category=FunctionCategory.PERFORMANCE,
            handler=None
        ))
        
        self.register_function(FunctionSpec(
            name="check_agent_availability",
            description="Check if agents are available and their current workload.",
            parameters=[
                FunctionParameter(
                    name="agent_ids",
                    type="array",
                    description="List of agent IDs to check"
                )
            ],
            category=FunctionCategory.AGENT_QUERY,
            handler=None
        ))
        
        self.register_function(FunctionSpec(
            name="compare_agents",
            description="Compare multiple agents side-by-side on performance, skills, and suitability.",
            parameters=[
                FunctionParameter(
                    name="agent_ids",
                    type="array",
                    description="List of agent IDs to compare"
                ),
                FunctionParameter(
                    name="criteria",
                    type="array",
                    description="Comparison criteria (e.g., ['performance', 'skills', 'availability'])",
                    required=False
                )
            ],
            category=FunctionCategory.AGENT_QUERY,
            handler=None
        ))
        
        self.register_function(FunctionSpec(
            name="get_agent_collaboration_history",
            description="Get history of how agents have collaborated in the past.",
            parameters=[
                FunctionParameter(
                    name="agent_id",
                    type="integer",
                    description="Primary agent ID"
                ),
                FunctionParameter(
                    name="potential_collaborators",
                    type="array",
                    description="List of potential collaborator agent IDs",
                    required=False
                )
            ],
            category=FunctionCategory.COORDINATION,
            handler=None
        ))
        
        # ===== TASK ANALYSIS FUNCTIONS =====
        self.register_function(FunctionSpec(
            name="analyze_task_requirements",
            description="Analyze a task to determine required skills, tools, and complexity.",
            parameters=[
                FunctionParameter(
                    name="task_description",
                    type="string",
                    description="The task description to analyze"
                ),
                FunctionParameter(
                    name="priority",
                    type="string",
                    description="Task priority (high, medium, low)",
                    required=False
                )
            ],
            category=FunctionCategory.TASK_ANALYSIS,
            handler=None
        ))
        
        logger.info(f"✅ Registered {len(self.functions)} default functions")


# Singleton instance
_global_registry: Optional[GlobalFunctionRegistry] = None


def get_global_function_registry() -> GlobalFunctionRegistry:
    """Get or create singleton GlobalFunctionRegistry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = GlobalFunctionRegistry()
    return _global_registry

