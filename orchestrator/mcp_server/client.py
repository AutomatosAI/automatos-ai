
"""
MCP Client for IDE Integration
==============================

Python client library for integrating with the MCP server from IDEs like
Cursor, DeepAgent, and other development environments.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

import aiohttp
import websockets

logger = logging.getLogger(__name__)

@dataclass
class MCPConfig:
    """MCP server configuration"""
    host: str = "localhost"
    port: int = 8002
    token: str = "default-token"
    use_ssl: bool = False
    timeout: int = 30

class MCPClient:
    """
    MCP Client for headless orchestration system integration
    
    Provides async methods for all orchestration functionality:
    - Workflow management
    - Document operations
    - Context engineering
    - SSH command execution
    - Progress tracking
    """
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.base_url = f"{'https' if config.use_ssl else 'http'}://{config.host}:{config.port}"
        self.headers = {
            "Authorization": f"Bearer {config.token}",
            "Content-Type": "application/json"
        }
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to MCP server"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                
                return await response.json()
        
        except aiohttp.ClientError as e:
            raise Exception(f"Request failed: {str(e)}")
    
    # Health and system methods
    async def health_check(self) -> Dict[str, Any]:
        """Check MCP server health"""
        return await self._request("GET", "/health")
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return await self._request("GET", "/system/info")
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        return await self._request("GET", "/system/metrics")
    
    # Workflow management methods
    async def create_workflow(
        self,
        repository_url: str,
        task_prompt: Optional[str] = None,
        security_level: str = "medium",
        ssh_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create and start a new workflow"""
        data = {
            "repository_url": repository_url,
            "task_prompt": task_prompt,
            "security_level": security_level,
            "ssh_config": ssh_config
        }
        return await self._request("POST", "/workflows", json=data)
    
    async def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows"""
        result = await self._request("GET", "/workflows")
        return result.get("workflows", [])
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status and details"""
        return await self._request("GET", f"/workflows/{workflow_id}")
    
    async def stop_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Stop a running workflow"""
        return await self._request("DELETE", f"/workflows/{workflow_id}")
    
    async def get_workflow_logs(self, workflow_id: str) -> List[str]:
        """Get workflow execution logs"""
        result = await self._request("GET", f"/workflows/{workflow_id}/logs")
        return result.get("logs", [])
    
    async def get_workflow_progress(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow progress"""
        return await self._request("GET", f"/progress/{workflow_id}")
    
    async def stream_workflow_progress(self, workflow_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream real-time workflow progress"""
        url = f"{self.base_url}/progress/{workflow_id}/stream"
        
        try:
            async with self.session.get(url) as response:
                if response.status >= 400:
                    raise Exception(f"HTTP {response.status}")
                
                async for line in response.content:
                    line = line.decode().strip()
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])  # Remove "data: " prefix
                            yield data
                        except json.JSONDecodeError:
                            continue
        
        except aiohttp.ClientError as e:
            raise Exception(f"Stream failed: {str(e)}")
    
    # Document management methods
    async def upload_document(
        self,
        filename: str,
        content: str,
        content_type: str = "text/plain"
    ) -> Dict[str, Any]:
        """Upload a document"""
        data = {
            "filename": filename,
            "content": content,
            "content_type": content_type
        }
        return await self._request("POST", "/documents", json=data)
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents"""
        result = await self._request("GET", "/documents")
        return result.get("documents", [])
    
    async def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Get document content"""
        return await self._request("GET", f"/documents/{doc_id}")
    
    async def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document"""
        return await self._request("DELETE", f"/documents/{doc_id}")
    
    # Context engineering methods
    async def search_context(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search context database"""
        data = {
            "query": query,
            "limit": limit,
            "filters": filters
        }
        result = await self._request("POST", "/context/search", json=data)
        return result.get("results", [])
    
    async def get_context_stats(self) -> Dict[str, Any]:
        """Get context database statistics"""
        return await self._request("GET", "/context/stats")
    
    async def configure_context(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure context manager settings"""
        return await self._request("POST", "/context/configure", json=config)
    
    # SSH command execution methods
    async def execute_ssh_command(
        self,
        command: str,
        security_level: str = "medium",
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Execute SSH command"""
        data = {
            "command": command,
            "security_level": security_level,
            "timeout": timeout
        }
        return await self._request("POST", "/ssh/execute", json=data)
    
    async def get_ssh_status(self) -> Dict[str, Any]:
        """Get SSH connection status"""
        return await self._request("GET", "/ssh/status")

# Convenience functions for common operations
async def quick_deploy(
    repository_url: str,
    task_prompt: Optional[str] = None,
    config: Optional[MCPConfig] = None
) -> str:
    """
    Quick deployment function for IDE integration
    
    Returns workflow_id for tracking
    """
    if not config:
        config = MCPConfig()
    
    async with MCPClient(config) as client:
        result = await client.create_workflow(
            repository_url=repository_url,
            task_prompt=task_prompt
        )
        return result["workflow_id"]

async def wait_for_completion(
    workflow_id: str,
    config: Optional[MCPConfig] = None,
    callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Wait for workflow completion with optional progress callback
    """
    if not config:
        config = MCPConfig()
    
    async with MCPClient(config) as client:
        if callback:
            # Stream progress with callback
            async for progress in client.stream_workflow_progress(workflow_id):
                callback(progress)
                if progress.get("status") in ["completed", "failed", "cancelled"]:
                    break
        
        # Get final status
        return await client.get_workflow_status(workflow_id)

# Example usage for IDE integration
async def example_ide_integration():
    """Example of how to integrate with IDEs"""
    config = MCPConfig(
        host="localhost",
        port=8002,
        token="your-api-token"
    )
    
    async with MCPClient(config) as client:
        # Check health
        health = await client.health_check()
        print(f"Server health: {health['status']}")
        
        # Deploy a repository
        workflow = await client.create_workflow(
            repository_url="https://github.com/user/repo.git",
            task_prompt="Deploy a secure web application"
        )
        
        workflow_id = workflow["workflow_id"]
        print(f"Started workflow: {workflow_id}")
        
        # Monitor progress
        async for progress in client.stream_workflow_progress(workflow_id):
            print(f"Progress: {progress}")
            if progress.get("status") in ["completed", "failed"]:
                break
        
        # Get final status
        final_status = await client.get_workflow_status(workflow_id)
        print(f"Final status: {final_status}")

if __name__ == "__main__":
    asyncio.run(example_ide_integration())
