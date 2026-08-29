"""
Agents Module
=============

Agent lifecycle, factory, and registry.

Usage:
    from modules.agents import AgentService, AgentFactory, AgentRegistry
    
    service = AgentService(db_session)
    capabilities = service.get_agent_capabilities(agent_id)
    result = await service.execute_task(agent_id, task)

Sellable as: automatos-agents
"""

from .service import AgentService, AgentServiceConfig

# Factory exports
from .factory import (
    AgentFactory,
    AgentRuntime,
    AgentMetadata,
    AgentLifecycle,
    ModelConfiguration,
)

# Registry exports
from .registry import (
    AgentRegistry,
    AgentCapabilities,
)

__all__ = [
    # Main Service
    "AgentService",
    "AgentServiceConfig",

    # Factory
    "AgentFactory",
    "AgentRuntime",
    "AgentMetadata",
    "AgentLifecycle",
    "ModelConfiguration",

    # Registry
    "AgentRegistry",
    "AgentCapabilities",
]
