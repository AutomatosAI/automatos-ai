
"""
Professional Agent Implementations
=================================

This module contains specialized agent implementations for different professional domains.
Each agent type has specific capabilities, skills, and workflows tailored to their domain.
"""

from .base_agent import BaseAgent, AgentCapability, AgentSkill
from .code_architect import CodeArchitectAgent
from .security_expert import SecurityExpertAgent
from .performance_optimizer import PerformanceOptimizerAgent
from .data_analyst import DataAnalystAgent
from .infrastructure_manager import InfrastructureManagerAgent
from .custom_agent import CustomAgent

__all__ = [
    'BaseAgent',
    'AgentCapability', 
    'AgentSkill',
    'CodeArchitectAgent',
    'SecurityExpertAgent',
    'PerformanceOptimizerAgent',
    'DataAnalystAgent',
    'InfrastructureManagerAgent',
    'CustomAgent'
]

# Agent type registry for dynamic instantiation
AGENT_REGISTRY = {
    'code_architect': CodeArchitectAgent,
    'security_expert': SecurityExpertAgent,
    'performance_optimizer': PerformanceOptimizerAgent,
    'data_analyst': DataAnalystAgent,
    'infrastructure_manager': InfrastructureManagerAgent,
    'custom': CustomAgent
}

def create_agent(agent_type: str, **kwargs):
    """Factory function to create agents by type"""
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent_class = AGENT_REGISTRY[agent_type]
    return agent_class(**kwargs)
