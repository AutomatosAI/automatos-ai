"""
Function Registry - Manages available functions for LLM orchestration
======================================================================

PRD-16: Central registry for all functions that the LLM can call.
Implements the Tool-Augmented Reasoning pattern from Princeton ICML.

Design Principles:
- Atomic functions: Each does one thing well
- Observable: Functions log their actions
- Safe: Input validation, timeouts, error handling
- Composable: Functions can be chained
- Data, not decisions: Functions provide info, LLM decides
"""

import logging
import inspect
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class FunctionCategory(Enum):
    """Categories of functions for organization"""
    QUERY = "query"              # Database queries
    ANALYSIS = "analysis"        # Data analysis
    VALIDATION = "validation"    # Input/output validation
    MONITORING = "monitoring"    # System monitoring
    EXECUTION = "execution"      # Task execution
    COMMUNICATION = "communication"  # Inter-agent communication
    MEMORY = "memory"           # Memory operations
    META = "meta"               # Meta-cognitive functions


@dataclass
class FunctionParameter:
    """Specification for a function parameter"""
    name: str
    type: str                    # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    items: Optional[Dict[str, Any]] = None  # For array types
    properties: Optional[Dict[str, Any]] = None  # For object types


@dataclass
class FunctionSpec:
    """
    Complete specification for a callable function.
    
    This follows the OpenAI function calling format for compatibility,
    but can be extended for other providers.
    """
    name: str
    description: str
    category: FunctionCategory
    parameters: List[FunctionParameter] = field(default_factory=list)
    returns: Optional[str] = None  # Description of return value
    examples: List[Dict[str, Any]] = field(default_factory=list)
    implementation: Optional[Callable] = None
    timeout_seconds: float = 10.0
    retry_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop_def = {
                "type": param.type,
                "description": param.description
            }
            
            if param.enum:
                prop_def["enum"] = param.enum
            if param.minimum is not None:
                prop_def["minimum"] = param.minimum
            if param.maximum is not None:
                prop_def["maximum"] = param.maximum
            if param.items:
                prop_def["items"] = param.items
            if param.properties:
                prop_def["properties"] = param.properties
            if param.default is not None:
                prop_def["default"] = param.default
            
            properties[param.name] = prop_def
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate parameters against specification.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                return False, f"Missing required parameter: {param.name}"
            
            if param.name in params:
                value = params[param.name]
                
                # Type checking
                if not self._check_type(value, param.type):
                    return False, f"Parameter {param.name} has wrong type. Expected {param.type}"
                
                # Enum validation
                if param.enum and value not in param.enum:
                    return False, f"Parameter {param.name} must be one of {param.enum}"
                
                # Range validation
                if param.type == "number":
                    if param.minimum is not None and value < param.minimum:
                        return False, f"Parameter {param.name} must be >= {param.minimum}"
                    if param.maximum is not None and value > param.maximum:
                        return False, f"Parameter {param.name} must be <= {param.maximum}"
        
        return True, None
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_map = {
            "string": str,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)
        return True


class FunctionRegistry:
    """
    Central registry for all functions available to the LLM.
    
    This registry manages:
    - Function registration and discovery
    - Parameter validation
    - Function categorization
    - Usage tracking
    """
    
    def __init__(self):
        self.functions: Dict[str, FunctionSpec] = {}
        self.categories: Dict[FunctionCategory, List[str]] = {
            cat: [] for cat in FunctionCategory
        }
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        
        logger.info("FunctionRegistry initialized")
    
    def register(
        self,
        spec: Union[FunctionSpec, Dict[str, Any]],
        implementation: Optional[Callable] = None
    ) -> None:
        """
        Register a function specification.
        
        Args:
            spec: FunctionSpec or dict with function details
            implementation: Optional callable implementation
        """
        if isinstance(spec, dict):
            spec = self._spec_from_dict(spec)
        
        if implementation:
            spec.implementation = implementation
        
        # Validate spec
        if not spec.name:
            raise ValueError("Function must have a name")
        if not spec.description:
            raise ValueError(f"Function {spec.name} must have a description")
        
        # Register function
        self.functions[spec.name] = spec
        self.categories[spec.category].append(spec.name)
        
        # Initialize usage stats
        self.usage_stats[spec.name] = {
            "call_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "last_called": None
        }
        
        logger.info(f"Registered function: {spec.name} ({spec.category.value})")
    
    def register_decorator(
        self,
        name: str,
        description: str,
        category: FunctionCategory = FunctionCategory.QUERY,
        timeout: float = 10.0
    ) -> Callable:
        """
        Decorator for registering functions.
        
        Usage:
            @registry.register_decorator("get_agent", "Get agent by ID")
            async def get_agent(agent_id: int) -> Dict:
                ...
        """
        def decorator(func: Callable) -> Callable:
            # Extract parameters from function signature
            sig = inspect.signature(func)
            parameters = []
            
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                
                # Try to infer type from annotation
                param_type = "string"  # Default
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list:
                        param_type = "array"
                    elif param.annotation == dict:
                        param_type = "object"
                
                parameters.append(FunctionParameter(
                    name=param_name,
                    type=param_type,
                    description=f"Parameter {param_name}",
                    required=param.default == inspect.Parameter.empty
                ))
            
            spec = FunctionSpec(
                name=name,
                description=description,
                category=category,
                parameters=parameters,
                implementation=func,
                timeout_seconds=timeout
            )
            
            self.register(spec)
            return func
        
        return decorator
    
    def get(self, name: str) -> Optional[FunctionSpec]:
        """Get function specification by name"""
        return self.functions.get(name)
    
    def get_by_category(self, category: FunctionCategory) -> List[FunctionSpec]:
        """Get all functions in a category"""
        names = self.categories.get(category, [])
        return [self.functions[name] for name in names]
    
    def get_all(self) -> List[FunctionSpec]:
        """Get all registered functions"""
        return list(self.functions.values())
    
    def get_for_prompt(
        self,
        categories: Optional[List[FunctionCategory]] = None,
        max_functions: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get functions formatted for LLM prompt.
        
        Args:
            categories: Filter by categories
            max_functions: Maximum number to return
            
        Returns:
            List of functions in OpenAI format
        """
        functions = []
        
        if categories:
            for cat in categories:
                functions.extend(self.get_by_category(cat))
        else:
            functions = self.get_all()
        
        # Sort by usage (most used first)
        functions.sort(
            key=lambda f: self.usage_stats[f.name]["call_count"],
            reverse=True
        )
        
        if max_functions:
            functions = functions[:max_functions]
        
        return [f.to_openai_format() for f in functions]
    
    def update_usage(
        self,
        name: str,
        success: bool,
        execution_time: float
    ) -> None:
        """Update usage statistics for a function"""
        if name not in self.usage_stats:
            return
        
        stats = self.usage_stats[name]
        stats["call_count"] += 1
        
        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
        
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["call_count"]
        stats["last_called"] = datetime.now().isoformat()
    
    def get_usage_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics"""
        if name:
            return self.usage_stats.get(name, {})
        return self.usage_stats
    
    def _spec_from_dict(self, data: Dict[str, Any]) -> FunctionSpec:
        """Create FunctionSpec from dictionary"""
        parameters = []
        for param_data in data.get("parameters", []):
            parameters.append(FunctionParameter(**param_data))
        
        return FunctionSpec(
            name=data["name"],
            description=data["description"],
            category=FunctionCategory(data.get("category", "query")),
            parameters=parameters,
            returns=data.get("returns"),
            examples=data.get("examples", []),
            timeout_seconds=data.get("timeout_seconds", 10.0),
            retry_count=data.get("retry_count", 1),
            metadata=data.get("metadata", {})
        )
    
    def export_schema(self) -> Dict[str, Any]:
        """Export complete function schema"""
        return {
            "functions": {
                name: {
                    "spec": spec.to_openai_format(),
                    "category": spec.category.value,
                    "timeout": spec.timeout_seconds,
                    "examples": spec.examples
                }
                for name, spec in self.functions.items()
            },
            "categories": {
                cat.value: names
                for cat, names in self.categories.items()
            },
            "usage_stats": self.usage_stats
        }
    
    def import_schema(self, schema: Dict[str, Any]) -> None:
        """Import function schema"""
        for name, func_data in schema.get("functions", {}).items():
            # Reconstruct FunctionSpec
            spec_data = func_data["spec"]
            spec_data["category"] = func_data.get("category", "query")
            spec_data["timeout_seconds"] = func_data.get("timeout", 10.0)
            spec_data["examples"] = func_data.get("examples", [])
            
            self.register(spec_data)
        
        # Import usage stats if available
        if "usage_stats" in schema:
            self.usage_stats.update(schema["usage_stats"])


from datetime import datetime

# Create global registry instance
global_registry = FunctionRegistry()
