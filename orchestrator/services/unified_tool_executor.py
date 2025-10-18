"""
Unified Tool Executor for PRD-17
=================================

Single entry point for all tool execution, routing to appropriate executors:
- Research tools (search_knowledge, semantic_search, search_codebase)
- File operations (read_file, write_file, list_directory)
- Shell commands (execute_command)
- MCP tools (third-party integrations)
"""

import logging
import os
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from services.agent_platform_tools import AgentPlatformTools
from services.agent_action_executor import ActionExecutor
from services.mcp_tool_executor import MCPToolExecutor
from services.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class UnifiedToolExecutor:
    """
    Unified tool executor that routes tool calls to the appropriate executor.
    
    Provides a single interface for all tool execution, simplifying agent code
    and making it easier to add new tools.
    """
    
    def __init__(self, db_session: Session, workspace_dir: str = "/tmp/automatos_workspace"):
        """
        Initialize unified tool executor.
        
        Args:
            db_session: Database session
            workspace_dir: Directory for file operations
        """
        self.db = db_session
        self.workspace_dir = workspace_dir
        
        # Lazy-loaded executors (only initialize when needed)
        self._platform_tools = None  # For research tools (RAG, CodeGraph)
        self._action_executor = None  # For file/shell operations
        self._mcp_executor = None  # For MCP tools
        self._tool_registry = None  # For tool metadata
        
        # Tool routing map
        self.tool_routes = {
            # Research tools
            'search_knowledge': self._execute_platform_tool,
            'semantic_search': self._execute_platform_tool,
            'search_codebase': self._execute_platform_tool,
            
            # File operations
            'read_file': self._execute_file_op,
            'write_file': self._execute_file_op,
            'list_directory': self._execute_file_op,
            'create_directory': self._execute_file_op,
            'delete_file': self._execute_file_op,
            
            # Shell commands
            'execute_command': self._execute_shell,
            
            # MCP tools handled dynamically
        }
        
        logger.info("🔧 UnifiedToolExecutor initialized (lazy-loading enabled)")
    
    @property
    def platform_tools(self):
        """Lazy-load platform tools (RAG, CodeGraph) only when needed."""
        if self._platform_tools is None:
            logger.info("  🔧 Initializing research tools (RAG, CodeGraph)...")
            self._platform_tools = AgentPlatformTools(self.db)
        return self._platform_tools
    
    @property
    def action_executor(self):
        """Lazy-load action executor (file/shell ops) only when needed."""
        if self._action_executor is None:
            logger.info("  🔧 Initializing file/shell executor...")
            self._action_executor = ActionExecutor(self.workspace_dir)
        return self._action_executor
    
    @property
    def mcp_executor(self):
        """Lazy-load MCP executor only when needed."""
        if self._mcp_executor is None:
            logger.info("  🔧 Initializing MCP executor...")
            self._mcp_executor = MCPToolExecutor(self.db)
        return self._mcp_executor
    
    @property
    def tool_registry(self):
        """Lazy-load tool registry only when needed."""
        if self._tool_registry is None:
            logger.info("  🔧 Initializing tool registry...")
            self._tool_registry = ToolRegistry(self.db)
        return self._tool_registry
    
    def _check_agent_permission(self, agent_id: int, tool_name: str) -> bool:
        """
        PRD-17: Check if agent has permission to use this tool.
        
        Args:
            agent_id: Agent ID
            tool_name: Tool name
            
        Returns:
            True if allowed, False otherwise
        """
        # For now, research tools are always allowed
        # File ops and shell require explicit permission via AgentToolPermission table
        research_tools = ['search_knowledge', 'semantic_search', 'search_codebase']
        if tool_name in research_tools:
            return True
            
        # Check AgentToolPermission table for other tools
        from models.tools import AgentToolPermission, Tool
        
        try:
            # Find tool in tools table
            tool = self.db.query(Tool).filter_by(name=tool_name).first()
            if not tool:
                logger.warning(f"⚠️  Tool '{tool_name}' not found in tools table")
                return False  # Unknown tools not allowed
            
            # Check if agent has permission
            permission = self.db.query(AgentToolPermission).filter_by(
                agent_id=agent_id,
                tool_id=tool.id,
                is_active=True
            ).first()
            
            return permission is not None
        except Exception as e:
            logger.warning(f"⚠️  Error checking permissions for tool '{tool_name}': {e}")
            return True  # Default to allow (fail open for now)
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int = 0
    ) -> Dict[str, Any]:
        """
        Execute a tool by name, routing to the appropriate executor.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            agent_id: ID of the agent calling the tool
            
        Returns:
            Tool execution result with standard format
        """
        try:
            logger.info(f"🔧 Executing tool '{tool_name}' for agent {agent_id}")
            logger.info(f"  📋 Parameters: {parameters}")
            
            # PRD-17: Check agent permissions (except for research tools which are always allowed)
            # if not self._check_agent_permission(agent_id, tool_name):
            #     logger.warning(f"⚠️  Agent {agent_id} does not have permission to use '{tool_name}'")
            #     return {
            #         "success": False,
            #         "error": f"Permission denied: Agent {agent_id} cannot use tool '{tool_name}'",
            #         "tool": tool_name
            #     }
            
            # Check if tool exists in registry
            tool_spec = self.tool_registry.get_tool(tool_name)
            if not tool_spec:
                logger.warning(f"⚠️  Tool '{tool_name}' not in registry, checking MCP tools...")
                # Check MCP tools
                mcp_tool = self.db.query(MCPTool).filter_by(name=tool_name).first()
                if mcp_tool:
                    return await self._execute_mcp_tool(tool_name, parameters, agent_id)
                else:
                    return {
                        "success": False,
                        "error": f"Unknown tool: {tool_name}",
                        "tool": tool_name
                    }
            
            # Route to appropriate executor
            executor_func = self.tool_routes.get(tool_name)
            if executor_func:
                result = await executor_func(tool_name, parameters, agent_id)
                logger.info(f"  ✅ Tool '{tool_name}' executed successfully")
                return result
            else:
                # Try MCP tool
                return await self._execute_mcp_tool(tool_name, parameters, agent_id)
                
        except Exception as e:
            logger.error(f"❌ Tool execution failed: {tool_name} - {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }
    
    async def _execute_platform_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """Execute research tools via AgentPlatformTools"""
        return await self.platform_tools.execute_tool(
            tool_name=tool_name,
            parameters=parameters,
            agent_id=agent_id
        )
    
    async def _execute_file_op(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """Execute file operations via AgentActionExecutor"""
        if tool_name == 'read_file':
            success, content = self.action_executor.read_file(
                parameters.get('path', '')
            )
            return {
                "success": success,
                "action": "read_file",
                "params": parameters,
                "result": content
            }
        elif tool_name == 'write_file':
            success, result = self.action_executor.write_file(
                parameters.get('path', ''),
                parameters.get('content', '')
            )
            return {
                "success": success,
                "action": "write_file",
                "params": parameters,
                "result": result
            }
        elif tool_name == 'list_directory':
            success, items = self.action_executor.list_directory(
                parameters.get('path', '.')
            )
            return {
                "success": success,
                "action": "list_directory",
                "params": parameters,
                "result": items
            }
        elif tool_name == 'create_directory':
            success, result = self.action_executor.create_directory(
                parameters.get('path', '')
            )
            return {
                "success": success,
                "action": "create_directory",
                "params": parameters,
                "result": result
            }
        elif tool_name == 'delete_file':
            success, result = self.action_executor.delete_file(
                parameters.get('path', '')
            )
            return {
                "success": success,
                "action": "delete_file",
                "params": parameters,
                "result": result
            }
        else:
            return {
                "success": False,
                "error": f"Unknown file operation: {tool_name}"
            }
    
    async def _execute_shell(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """Execute shell commands via AgentActionExecutor"""
        success, output = self.action_executor.execute_command(
            parameters.get('command', ''),
            timeout=parameters.get('timeout', 30)
        )
        return {
            "success": success,
            "action": "execute_command",
            "params": parameters,
            "result": output
        }
    
    async def _execute_mcp_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """
        Execute MCP tools via MCPToolExecutor.
        
        Connects to actual MCP servers via JSON-RPC protocol.
        """
        logger.info(f"  🌐 Routing to MCP server: {tool_name}")
        
        # Get tool from database
        mcp_tool = self.db.query(MCPTool).filter_by(name=tool_name).first()
        if not mcp_tool:
            return {"success": False, "error": f"MCP tool '{tool_name}' not found"}
        
        # Execute via MCP protocol (JSON-RPC to actual MCP server)
        method = parameters.get("method", "default")
        return await self.mcp_executor.execute_tool(
            agent_id=agent_id,
            tool_id=mcp_tool.id,
            method=method,
            params=parameters,
            execution_id=None
        )
    
    def get_available_tools(self, categories: Optional[list] = None) -> list:
        """
        Get list of available tools, optionally filtered by category.
        
        Args:
            categories: Optional list of categories to filter by
            
        Returns:
            List of tool specifications
        """
        if categories:
            tools = []
            for category in categories:
                tools.extend(self.tool_registry.get_tools_by_category(category))
            return tools
        else:
            return list(self.tool_registry.tools.values())


# Import MCPTool model at the bottom to avoid circular imports
from models import MCPTool

