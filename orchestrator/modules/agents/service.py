"""
Agent Service
=============

Main service class for Agents module.
Coordinates agent factory, registry, execution, and communication.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from sqlalchemy.orm import Session

from config import config

logger = logging.getLogger(__name__)


@dataclass
class AgentServiceConfig:
    """Configuration for Agent service"""
    max_concurrent_agents: int = 10
    default_timeout_seconds: int = 300
    enable_communication: bool = True
    workspace_dir: str = "/tmp/automatos_workspace"


class AgentService:
    """Main Agent service for agent lifecycle management"""
    
    def __init__(
        self,
        db_session: Session,
        config: AgentServiceConfig = None
    ):
        self.db = db_session
        self.config = config or AgentServiceConfig()
        
        # Lazy-loaded components
        self._factory = None
        self._registry = None
        self._execution_manager = None
        self._communication = None
        
        logger.info("AgentService initialized")
    
    @property
    def factory(self):
        """Get AgentFactory (lazy-loaded)"""
        if self._factory is None:
            from .factory import AgentFactory
            self._factory = AgentFactory(self.db)
        return self._factory
    
    @property
    def registry(self):
        """Get AgentRegistry (lazy-loaded)"""
        if self._registry is None:
            from .registry import AgentRegistry
            self._registry = AgentRegistry(self.db)
        return self._registry
    
    @property
    def execution_manager(self):
        """Get AgentExecutionManager (lazy-loaded)"""
        if self._execution_manager is None:
            from .execution import AgentExecutionManager
            self._execution_manager = AgentExecutionManager(
                db_session=self.db,
                workspace_dir=self.config.workspace_dir
            )
        return self._execution_manager
    
    async def create_agent(
        self,
        name: str,
        agent_type: str,
        description: str = "",
        model_id: str = None,
        skills: List[str] = None
    ) -> Dict[str, Any]:
        """Create a new agent"""
        if model_id is None:
            model_id = config.LLM_MODEL

        # Use factory to create agent
        agent = await self.factory.create_agent(
            name=name,
            agent_type=agent_type,
            description=description,
            model_id=model_id,
            skills=skills or []
        )
        
        return {
            'id': agent.id,
            'name': agent.name,
            'agent_type': agent.agent_type,
            'status': 'active'
        }
    
    def get_agent_capabilities(self, agent_id: int) -> Optional[Dict[str, Any]]:
        """Get agent capabilities from registry"""
        
        capabilities = self.registry.get_agent_capabilities(agent_id)
        if not capabilities:
            return None
            
        return {
            'agent_id': capabilities.agent_id,
            'name': capabilities.name,
            'agent_type': capabilities.agent_type,
            'skills': capabilities.skills,
            'tools': capabilities.tools,
            'performance': capabilities.performance
        }
    
    def find_agents_for_task(self, task_description: str, required_tools: List[str] = None) -> List[Dict[str, Any]]:
        """Find agents suitable for a task"""
        
        # Find by tools first if specified
        if required_tools:
            agents = []
            for tool in required_tools:
                tool_agents = self.registry.find_agents_with_tool(tool)
                agents.extend(tool_agents)
            return list({a['agent_id']: a for a in agents}.values())
        
        # Otherwise return all active agents
        return self.registry.get_all_agents()
    
    async def execute_task(
        self,
        agent_id: int,
        task: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a task with an agent"""
        
        runtime = await self.factory.get_runtime(agent_id)
        if not runtime:
            return {
                'success': False,
                'error': f'Agent {agent_id} not found'
            }
        
        result = await runtime.execute(task, context or {})
        
        return {
            'success': True,
            'agent_id': agent_id,
            'response': result.response if hasattr(result, 'response') else str(result),
            'tokens_used': result.tokens_used if hasattr(result, 'tokens_used') else 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent service statistics"""
        
        return {
            'active_agents': len(self.registry.get_all_agents()) if self._registry else 0,
            'config': {
                'max_concurrent': self.config.max_concurrent_agents,
                'timeout': self.config.default_timeout_seconds,
                'communication_enabled': self.config.enable_communication
            }
        }

