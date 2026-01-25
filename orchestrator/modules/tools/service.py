"""
Tool Service
=============

Main service class for Tools module.
Coordinates tool registry, execution, and formatting.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class ToolServiceConfig:
    """Configuration for Tool service"""
    workspace_dir: str = "/tmp/automatos_workspace"
    enable_integrations: bool = True
    max_execution_time_seconds: int = 300


class ToolService:
    """Main Tool service for tool management and execution"""
    
    def __init__(
        self,
        db_session: Session,
        config: ToolServiceConfig = None
    ):
        self.db = db_session
        self.config = config or ToolServiceConfig()
        
        # Lazy-loaded components
        self._registry = None
        self._executor = None
        
        logger.info("ToolService initialized")
    
    @property
    def registry(self):
        """Get ToolRegistry (lazy-loaded)"""
        if self._registry is None:
            from .registry import ToolRegistry
            self._registry = ToolRegistry(self.db)
        return self._registry
    
    @property
    def executor(self):
        """Get UnifiedToolExecutor (lazy-loaded)"""
        if self._executor is None:
            from .execution import UnifiedToolExecutor
            self._executor = UnifiedToolExecutor(
                self.db,
                workspace_dir=self.config.workspace_dir
            )
        return self._executor
    
    def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool definition by name"""
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return None
        return {
            'name': tool.name,
            'description': tool.description,
            'category': tool.category.value,
            'security_level': tool.security_level.value,
            'parameters': [p.__dict__ for p in tool.parameters]
        }
    
    def list_tools(
        self,
        category: str = None,
    ) -> List[Dict[str, Any]]:
        """List available tools"""
        tools = []
        
        # Get from registry
        if category:
            from .registry import ToolCategory
            try:
                cat = ToolCategory(category)
                registry_tools = self.registry.get_tools_by_category(cat)
            except ValueError:
                registry_tools = []
        else:
            registry_tools = self.registry.get_all_tools()
        
        for tool in registry_tools:
            tools.append({
                'name': tool.name,
                'description': tool.description,
                'category': tool.category.value,
                'security_level': tool.security_level.value
            })
        
        return tools
    
    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int = None
    ) -> Dict[str, Any]:
        """Execute a tool"""
        
        result = await self.executor.execute_tool(
            tool_name=tool_name,
            parameters=parameters,
            agent_id=agent_id
        )
        
        return {
            'tool_name': tool_name,
            'success': result.success if hasattr(result, 'success') else True,
            'result': result.result if hasattr(result, 'result') else str(result),
            'error': result.error if hasattr(result, 'error') else None
        }
    
    def get_openai_tools_schema(self, tool_names: List[str] = None) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool schemas"""
        return self.registry.get_openai_tools_schema(tool_names)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool service statistics"""
        tools = self.registry.get_all_tools() if self._registry else []
        
        categories = {}
        for tool in tools:
            cat = tool.category.value
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_tools': len(tools),
            'categories': categories,
            'config': {
                'workspace': self.config.workspace_dir,
                'integrations_enabled': self.config.enable_integrations,
                'max_execution_time': self.config.max_execution_time_seconds
            }
        }

