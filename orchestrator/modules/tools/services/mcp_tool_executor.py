"""
MCP Tool Executor Service
=========================
Phase 3: Skills & Tools Integration

Handles execution of MCP (Model Context Protocol) tools for agents.
Provides a clean interface for tool invocation, result handling, and usage tracking.
"""

import asyncio
import httpx
import logging
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session

# Database models will be imported dynamically to avoid circular imports
from core.database.database import get_db
from core.models import MCPTool, AgentToolAssignment, ToolUsageLog, Agent

logger = logging.getLogger(__name__)

class MCPToolExecutionError(Exception):
    """Raised when MCP tool execution fails"""
    pass

class MCPToolNotFoundError(Exception):
    """Raised when a tool is not found or not assigned to agent"""
    pass

class MCPToolExecutor:
    """
    Executes MCP tools for agents with proper permission checking,
    logging, and error handling.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.http_client = httpx.AsyncClient(timeout=30.0)
        logger.info("MCPToolExecutor initialized")
    
    async def execute_tool(
        self,
        agent_id: int,
        tool_id: int,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        execution_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute an MCP tool method for a specific agent.
        
        Args:
            agent_id: ID of the agent executing the tool
            tool_id: ID of the MCP tool to execute
            method: The tool method to call (e.g., "repos.list", "chat.postMessage")
            params: Parameters to pass to the method
            execution_id: Optional workflow execution ID for tracking
        
        Returns:
            Dict containing the execution result
        
        Raises:
            MCPToolNotFoundError: If tool doesn't exist or agent doesn't have access
            MCPToolExecutionError: If execution fails
        """
        start_time = time.time()
        success = False
        error_message = None
        output_data = None
        
        try:
            # 1. Verify tool exists and is active
            tool = self.db.query(MCPTool).filter(
                MCPTool.id == tool_id,
                MCPTool.status == 'active'
            ).first()
            
            if not tool:
                raise MCPToolNotFoundError(f"Tool {tool_id} not found or inactive")
            
            # 2. Verify agent has permission to use this tool
            assignment = self.db.query(AgentToolAssignment).filter(
                AgentToolAssignment.agent_id == agent_id,
                AgentToolAssignment.tool_id == tool_id,
                AgentToolAssignment.enabled == True
            ).first()
            
            if not assignment:
                raise MCPToolNotFoundError(
                    f"Agent {agent_id} does not have access to tool {tool_id}"
                )
            
            # 3. Check permissions for the requested method
            permissions = assignment.permissions or {}
            if not self._check_method_permission(method, permissions):
                raise MCPToolExecutionError(
                    f"Agent {agent_id} does not have permission to execute method '{method}'"
                )
            
            # 4. Validate method exists in tool capabilities
            capabilities = tool.capabilities or {}
            available_methods = capabilities.get('methods', [])
            if method not in available_methods:
                raise MCPToolExecutionError(
                    f"Method '{method}' not available for tool {tool.name}. "
                    f"Available methods: {available_methods}"
                )
            
            # 5. Execute the tool via MCP server
            if tool.mcp_server_url:
                output_data = await self._call_mcp_server(
                    tool.mcp_server_url,
                    method,
                    params or {},
                    assignment.configuration
                )
            else:
                # If no server URL, return a simulated success (for tools that are metadata-only)
                output_data = {
                    "success": True,
                    "message": f"Tool {tool.name} executed (no server URL configured)",
                    "method": method,
                    "params": params
                }
            
            success = True
            logger.info(
                f"Tool executed successfully: agent={agent_id}, tool={tool.name}, method={method}"
            )
            
            return {
                "success": True,
                "tool_id": tool_id,
                "tool_name": tool.name,
                "method": method,
                "output": output_data,
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }
            
        except MCPToolNotFoundError as e:
            error_message = str(e)
            logger.warning(error_message)
            raise
            
        except MCPToolExecutionError as e:
            error_message = str(e)
            logger.error(error_message)
            raise
            
        except Exception as e:
            error_message = f"Unexpected error executing tool: {str(e)}"
            logger.exception(error_message)
            raise MCPToolExecutionError(error_message)
            
        finally:
            # Log the execution attempt
            self._log_tool_usage(
                execution_id=execution_id,
                agent_id=agent_id,
                tool_id=tool_id,
                method=method,
                input_data=params,
                output_data=output_data,
                success=success,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=error_message
            )
    
    async def _call_mcp_server(
        self,
        server_url: str,
        method: str,
        params: Dict[str, Any],
        configuration: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call the MCP server to execute the tool method.
        
        Args:
            server_url: Base URL of the MCP server
            method: Method to call
            params: Method parameters
            configuration: Agent-specific tool configuration
        
        Returns:
            Response data from the MCP server
        """
        try:
            # Build the request payload according to MCP protocol
            payload = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": f"{int(time.time() * 1000)}"
            }
            
            # Add any agent-specific configuration
            if configuration:
                payload["config"] = configuration
            
            # Make the HTTP request
            response = await self.http_client.post(
                server_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Check for JSON-RPC error
            if "error" in result:
                raise MCPToolExecutionError(
                    f"MCP server returned error: {result['error']}"
                )
            
            return result.get("result", result)
            
        except httpx.HTTPError as e:
            raise MCPToolExecutionError(f"HTTP error calling MCP server: {str(e)}")
        except json.JSONDecodeError as e:
            raise MCPToolExecutionError(f"Invalid JSON response from MCP server: {str(e)}")
    
    def _check_method_permission(self, method: str, permissions: Dict[str, Any]) -> bool:
        """
        Check if the agent has permission to execute the method.
        
        Args:
            method: Method name (e.g., "repos.create")
            permissions: Agent's permissions dict (e.g., {"read": true, "write": false})
        
        Returns:
            True if permission granted, False otherwise
        """
        # If no permissions specified, allow all
        if not permissions:
            return True
        
        # Map method types to permission categories
        # Methods with "create", "update", "delete", "post" require write permission
        # Methods with "list", "get", "read" require read permission
        # Methods with "execute", "run", "trigger" require execute permission
        
        method_lower = method.lower()
        
        if any(verb in method_lower for verb in ['create', 'update', 'delete', 'post', 'put']):
            return permissions.get('write', False)
        elif any(verb in method_lower for verb in ['execute', 'run', 'trigger', 'start']):
            return permissions.get('execute', False)
        else:
            # Default to read permission for listing/getting
            return permissions.get('read', False)
    
    def _log_tool_usage(
        self,
        execution_id: Optional[int],
        agent_id: int,
        tool_id: int,
        method: str,
        input_data: Optional[Dict[str, Any]],
        output_data: Optional[Dict[str, Any]],
        success: bool,
        execution_time_ms: int,
        error_message: Optional[str]
    ):
        """
        Log tool usage to the database for analytics and auditing.
        """
        try:
            usage_log = ToolUsageLog(
                execution_id=execution_id,
                agent_id=agent_id,
                tool_id=tool_id,
                method_called=method,
                input_data=input_data,
                output_data=output_data,
                success=success,
                execution_time_ms=execution_time_ms,
                error_message=error_message
            )
            self.db.add(usage_log)
            self.db.commit()
        except Exception as e:
            logger.error(f"Failed to log tool usage: {e}")
            # Don't raise - logging failure shouldn't break tool execution
    
    async def get_agent_tools(self, agent_id: int, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all tools available to an agent.
        
        Args:
            agent_id: ID of the agent
            enabled_only: If True, only return enabled tools
        
        Returns:
            List of tool dictionaries with details and permissions
        """
        query = self.db.query(AgentToolAssignment, MCPTool).join(
            MCPTool, AgentToolAssignment.tool_id == MCPTool.id
        ).filter(
            AgentToolAssignment.agent_id == agent_id
        )
        
        if enabled_only:
            query = query.filter(AgentToolAssignment.enabled == True)
        
        assignments = query.all()
        
        tools = []
        for assignment, tool in assignments:
            tools.append({
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "provider": tool.provider,
                "category": tool.category,
                "icon": tool.icon,
                "capabilities": tool.capabilities,
                "permissions": assignment.permissions,
                "configuration": assignment.configuration,
                "enabled": assignment.enabled
            })
        
        return tools
    
    async def close(self):
        """Clean up resources"""
        await self.http_client.aclose()
        logger.info("MCPToolExecutor closed")


# Convenience function for single-use executions
async def execute_tool_for_agent(
    db: Session,
    agent_id: int,
    tool_id: int,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    execution_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to execute a tool without managing the executor instance.
    
    Usage:
        result = await execute_tool_for_agent(
            db=db,
            agent_id=14,
            tool_id=1,
            method="repos.list",
            params={"org": "automatos-ai"}
        )
    """
    executor = MCPToolExecutor(db)
    try:
        return await executor.execute_tool(agent_id, tool_id, method, params, execution_id)
    finally:
        await executor.close()

