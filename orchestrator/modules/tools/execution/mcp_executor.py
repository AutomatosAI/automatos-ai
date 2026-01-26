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
from typing import Dict, Any, Optional, List, Union
from urllib.parse import urlparse
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

    def _sanitize_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            host = parsed.netloc or parsed.path
            return f"{parsed.scheme}://{host}"
        except Exception:
            return url
    
    async def execute_tool(
        self,
        agent_id: int,
        tool_id: Union[int, str],
        method: str,
        params: Optional[Dict[str, Any]] = None,
        execution_id: Optional[int] = None,
        trace_id: Optional[str] = None
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
        trace = trace_id or "no-trace"
        start_time = time.time()
        success = False
        error_message = None
        output_data = None
        
        tool_db_id: Optional[int] = None

        try:
            logger.info(
                f"[tool-trace {trace}] MCP execute start agent={agent_id} "
                f"tool_id={tool_id} method={method} params_keys={list((params or {}).keys())}"
            )
            # 1. Verify tool exists and is active
            tool = None
            if isinstance(tool_id, int):
                tool = self.db.query(MCPTool).filter(
                    MCPTool.id == tool_id,
                    MCPTool.status == 'active'
                ).first()
            else:
                tool = self.db.query(MCPTool).filter(
                    MCPTool.tool_id == str(tool_id),
                    MCPTool.status == 'active'
                ).first()
            
            if not tool:
                raise MCPToolNotFoundError(f"Tool {tool_id} not found or inactive")
            tool_db_id = tool.id
            
            # 2. Verify agent has permission to use this tool
            assignment_tool_id = tool.tool_id
            assignment = self.db.query(AgentToolAssignment).filter(
                AgentToolAssignment.agent_id == agent_id,
                AgentToolAssignment.tool_id == assignment_tool_id,
                AgentToolAssignment.enabled == True
            ).first()
            
            if not assignment:
                # Allow Chatbot (agent 1) to use tools assigned to any agent
                if agent_id == 1:
                    assignment = self.db.query(AgentToolAssignment).filter(
                        AgentToolAssignment.tool_id == assignment_tool_id,
                        AgentToolAssignment.enabled == True
                    ).first()
                if not assignment:
                    raise MCPToolNotFoundError(
                        f"Agent {agent_id} does not have access to tool {tool_id}"
                    )
            
            # 3. Check permissions for the requested method
            permissions = getattr(assignment, "permissions", None) or {}
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
                logger.info(
                    f"[tool-trace {trace}] MCP call server={self._sanitize_url(tool.mcp_server_url)} "
                    f"method={method}"
                )
                output_data = await self._call_mcp_server(
                    tool.mcp_server_url,
                    method,
                    params or {},
                    getattr(assignment, "configuration", None) or {},
                    trace_id=trace
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
                f"[tool-trace {trace}] MCP execute success agent={agent_id} tool={tool.name} "
                f"method={method} time_ms={int((time.time() - start_time) * 1000)}"
            )
            
            return {
                "success": True,
                "tool_id": tool_db_id,
                "tool_name": tool.name,
                "method": method,
                "output": output_data,
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }
            
        except MCPToolNotFoundError as e:
            error_message = str(e)
            logger.warning(f"[tool-trace {trace}] {error_message}")
            raise
            
        except MCPToolExecutionError as e:
            error_message = str(e)
            logger.error(f"[tool-trace {trace}] {error_message}")
            raise
            
        except Exception as e:
            error_message = f"Unexpected error executing tool: {str(e)}"
            logger.exception(f"[tool-trace {trace}] {error_message}")
            raise MCPToolExecutionError(error_message)
            
        finally:
            # Log the execution attempt
            if tool_db_id is not None:
                self._log_tool_usage(
                    execution_id=execution_id,
                    agent_id=agent_id,
                    tool_id=tool_db_id,
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
        configuration: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
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
        trace = trace_id or "no-trace"
        start_time = time.time()
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
            
            logger.info(
                f"[tool-trace {trace}] MCP request "
                f"url={self._sanitize_url(server_url)} method={method} "
                f"params_keys={list(params.keys())}"
            )
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
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[tool-trace {trace}] MCP response ok "
                f"status={response.status_code} time_ms={elapsed_ms} "
                f"result_keys={list(result.keys())}"
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
            MCPTool, AgentToolAssignment.tool_id == MCPTool.tool_id
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
                "permissions": getattr(assignment, "permissions", None) or {},
                "configuration": getattr(assignment, "configuration", None) or {},
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

