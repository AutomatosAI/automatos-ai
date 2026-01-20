from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import logging

import mcp.types
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

class JITMCPClient:
    """
    Just-in-Time MCP Client.
    
    Instantiated only when a tool execution is requested.
    Uses 'mcp' library directly (same as AgentScope).
    """
    
    def __init__(self, url: str, headers: Dict[str, str] = None, timeout: float = 30):
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        
        # Determine transport based on URL
        if url.endswith("/sse"):
            self.transport = "sse"
        else:
            self.transport = "streamable_http"
            
    @asynccontextmanager
    async def session(self):
        """Yields an active ClientSession"""
        client_gen = None
        if self.transport == "sse":
            client_gen = sse_client(self.url, headers=self.headers, timeout=self.timeout)
        else:
            client_gen = streamablehttp_client(self.url, headers=self.headers, timeout=self.timeout)
            
        async with client_gen as streams:
            read_stream, write_stream = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return the result"""
        async with self.session() as session:
            try:
                result = await session.call_tool(tool_name, arguments)
                # Parse result
                content = []
                for c in result.content:
                    if c.type == "text":
                        content.append(c.text)
                    elif c.type == "image":
                        content.append(f"[Image: {c.mimeType}]")
                    elif c.type == "resource":
                        content.append(f"[Resource: {c.uri}]")
                        
                return {
                    "success": not result.isError,
                    "result": "\n".join(content),
                    "tool": tool_name
                }
            except Exception as e:
                logger.error(f"MCP Tool execution failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "tool": tool_name
                }
