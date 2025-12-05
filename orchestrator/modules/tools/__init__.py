"""
Tools Module
============

Tool registry, execution, and management.

Usage:
    from modules.tools import ToolService, ToolRegistry
    
    service = ToolService(db_session)
    tools = service.list_tools(category='research')
    result = await service.execute('search_knowledge', {'query': '...'})

Sellable as: automatos-tools
"""

from .service import ToolService, ToolServiceConfig

# Registry exports
from .registry import (
    ToolRegistry,
    ToolCategory,
    SecurityLevel,
    ToolParameter,
    ToolSpec,
    ToolDefinition
)

# Execution exports
from .execution import (
    UnifiedToolExecutor,
    MCPToolExecutor
)

__all__ = [
    # Main Service
    "ToolService",
    "ToolServiceConfig",
    
    # Registry
    "ToolRegistry",
    "ToolCategory",
    "SecurityLevel",
    "ToolParameter",
    "ToolSpec",
    "ToolDefinition",
    
    # Execution
    "UnifiedToolExecutor",
    "MCPToolExecutor"
]
