"""
Minimal MCP Test Server
========================
PRD-17 Phase 3: Simple JSON-RPC 2.0 server to demonstrate MCP protocol

This is a minimal MCP server that implements the MCP protocol correctly:
- JSON-RPC 2.0 over HTTP
- Tool discovery
- Method execution
- Proper error handling

Run with: uvicorn services.minimal_mcp_server:app --host 0.0.0.0 --port 3001
"""

from fastapi import FastAPI, Request
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Minimal MCP Test Server")


# ===================================================================
# MCP PROTOCOL: Tool Definitions
# ===================================================================

MCP_TOOLS = {
    "test_echo": {
        "name": "test_echo",
        "description": "Simple echo test - returns the input message",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to echo back"
                }
            },
            "required": ["message"]
        }
    },
    "test_math": {
        "name": "test_math",
        "description": "Perform simple math operation",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "Math operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["operation", "a", "b"]
        }
    },
    "test_status": {
        "name": "test_status",
        "description": "Get status of the MCP server",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}


# ===================================================================
# MCP PROTOCOL: Method Implementations
# ===================================================================

def execute_test_echo(params: Dict[str, Any]) -> Dict[str, Any]:
    """Echo test implementation"""
    message = params.get("message", "")
    return {
        "success": True,
        "echo": message,
        "length": len(message),
        "server": "Minimal MCP Test Server"
    }


def execute_test_math(params: Dict[str, Any]) -> Dict[str, Any]:
    """Math operation implementation"""
    operation = params.get("operation")
    a = params.get("a")
    b = params.get("b")
    
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return {"success": False, "error": "Division by zero"}
        result = a / b
    else:
        return {"success": False, "error": f"Unknown operation: {operation}"}
    
    return {
        "success": True,
        "operation": operation,
        "a": a,
        "b": b,
        "result": result
    }


def execute_test_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Status check implementation"""
    return {
        "success": True,
        "status": "healthy",
        "server": "Minimal MCP Test Server",
        "protocol": "JSON-RPC 2.0 (MCP)",
        "available_tools": list(MCP_TOOLS.keys())
    }


# Map method names to implementations
METHOD_HANDLERS = {
    "test_echo": execute_test_echo,
    "test_math": execute_test_math,
    "test_status": execute_test_status,
}


# ===================================================================
# MCP PROTOCOL: JSON-RPC 2.0 Endpoint
# ===================================================================

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """
    MCP Protocol Endpoint (JSON-RPC 2.0)
    
    Accepts requests in format:
    {
        "jsonrpc": "2.0",
        "method": "test_echo",
        "params": {"message": "hello"},
        "id": "123"
    }
    
    Returns:
    {
        "jsonrpc": "2.0",
        "result": {...},
        "id": "123"
    }
    """
    try:
        body = await request.json()
        
        # Validate JSON-RPC 2.0 format
        if body.get("jsonrpc") != "2.0":
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: jsonrpc must be '2.0'"
                },
                "id": body.get("id")
            }
        
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")
        
        logger.info(f"🔧 MCP Request: method={method}, params={list(params.keys())}, id={request_id}")
        
        # Check if method exists
        if method not in METHOD_HANDLERS:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                    "data": {"available_methods": list(METHOD_HANDLERS.keys())}
                },
                "id": request_id
            }
        
        # Execute the method
        handler = METHOD_HANDLERS[method]
        result = handler(params)
        
        logger.info(f"✅ MCP Response: method={method}, success={result.get('success')}")
        
        # Return JSON-RPC 2.0 response
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id
        }
        
    except Exception as e:
        logger.error(f"❌ MCP Error: {e}")
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": "Internal error",
                "data": {"error": str(e)}
            },
            "id": body.get("id") if 'body' in locals() else None
        }


# ===================================================================
# MCP PROTOCOL: Tool Discovery
# ===================================================================

@app.get("/mcp/tools")
async def list_tools():
    """
    MCP Tool Discovery Endpoint
    
    Returns list of available tools with their schemas.
    This allows agents to dynamically discover what the MCP server can do.
    """
    return {
        "protocol": "MCP 1.0",
        "server": "Minimal MCP Test Server",
        "tools": list(MCP_TOOLS.values())
    }


@app.get("/")
async def root():
    """Health check"""
    return {
        "server": "Minimal MCP Test Server",
        "protocol": "JSON-RPC 2.0 (MCP)",
        "endpoint": "/mcp",
        "discovery": "/mcp/tools",
        "status": "healthy"
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Minimal MCP Test Server on port 3001...")
    print("   Endpoint: http://localhost:3001/mcp")
    print("   Discovery: http://localhost:3001/mcp/tools")
    print("   Available tools: test_echo, test_math, test_status")
    uvicorn.run(app, host="0.0.0.0", port=3001)

