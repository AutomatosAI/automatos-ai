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
)

# Tool Router exports (shared layer — PRD-50)
from .tool_router import (
    ToolRouter,
    get_tool_router,
    get_tools_for_agent,
    get_agent_tools,
    execute_tool,
    execute_tool_with_validation,
    validate_action_for_intent,
    get_filtered_composio_actions,
    get_capability_filter_stats,
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

    # Tool Router (shared layer — PRD-50)
    "ToolRouter",
    "get_tool_router",
    "get_tools_for_agent",
    "get_agent_tools",
    "execute_tool",
    "execute_tool_with_validation",
    "validate_action_for_intent",
    "get_filtered_composio_actions",
    "get_capability_filter_stats",
]
