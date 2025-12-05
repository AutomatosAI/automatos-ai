"""
Agents Module
=============

Agent lifecycle, execution, registry, and communication.

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
    create_specialized_agent,
)

# Registry exports
from .registry import (
    AgentRegistry,
    AgentCapabilities,
)

# Execution exports
from .execution import (
    AgentExecutionManager,
    ExecutionPlan,
    SubtaskExecution,
    SubtaskStatus,
)

# Communication exports
from .communication import (
    AgentCommunicationProtocol,
    Message,
    MessageType,
    MessagePriority,
    SharedContext,
)

# Multi-agent systems exports
from .multi_agent import (
    CoordinationManager,
    EmergentBehaviorMonitor,
    MultiAgentOptimizer,
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
    "create_specialized_agent",
    
    # Registry
    "AgentRegistry",
    "AgentCapabilities",
    
    # Execution
    "AgentExecutionManager",
    "ExecutionPlan",
    "SubtaskExecution",
    "SubtaskStatus",
    
    # Communication
    "AgentCommunicationProtocol",
    "Message",
    "MessageType",
    "MessagePriority",
    "SharedContext",
    
    # Multi-Agent Systems
    "CoordinationManager",
    "EmergentBehaviorMonitor",
    "MultiAgentOptimizer",
]
