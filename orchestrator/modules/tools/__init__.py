"""
Tools Module
============

Tool registry, execution, and management.

Usage:
    from modules.tools import ToolRegistry, get_tools_for_agent

    tools = get_tools_for_agent(agent_id, workspace_id)

Sellable as: automatos-tools
"""

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
    get_tools_for_agent_async,
    execute_tool,
    execute_tool_with_validation,
    validate_action_for_intent,
    get_filtered_composio_actions,
    get_capability_filter_stats,
)

__all__ = [
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
    "get_tools_for_agent_async",
    "execute_tool",
    "execute_tool_with_validation",
    "validate_action_for_intent",
    "get_filtered_composio_actions",
    "get_capability_filter_stats",
]
