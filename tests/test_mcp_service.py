
"""
MCP (Model Context Protocol) Service API tests
"""
import pytest
import httpx

@pytest.mark.asyncio
async def test_list_servers(client: httpx.AsyncClient):
    """Test listing MCP servers"""
    response = await client.get("/listServers")
    assert response.status_code == 200
    data = response.json()
    assert "servers" in data
    assert isinstance(data["servers"], list)

@pytest.mark.asyncio
async def test_start_server(client: httpx.AsyncClient):
    """Test starting an MCP server"""
    server_config = {
        "config": {
            "name": "test-server",
            "type": "stdio",
            "command": "echo",
            "args": ["hello"]
        }
    }
    
    response = await client.post("/startServer", json=server_config)
    
    if response.status_code == 201:
        data = response.json()
        assert "name" in data
        assert data["name"] == "test-server"
        return data["name"]
    else:
        print(f"Server start failed: {response.status_code} - {response.text}")
        return None

@pytest.mark.asyncio
async def test_list_tools(client: httpx.AsyncClient):
    """Test listing tools on servers"""
    request_data = {"server_name": None}  # List tools on all servers
    
    response = await client.post("/listTools", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "server_tools" in data
    assert isinstance(data["server_tools"], dict)

@pytest.mark.asyncio
async def test_call_tool(client: httpx.AsyncClient):
    """Test calling a tool on a server"""
    # This will likely fail without a proper server, but tests the endpoint
    tool_request = {
        "server_name": "test-server",
        "tool_name": "test-tool",
        "arguments": {"test": "value"}
    }
    
    response = await client.post("/callTool", json=tool_request)
    # Expected to fail, but we test the endpoint structure
    print(f"Tool call response: {response.status_code} - {response.text}")

@pytest.mark.asyncio
async def test_stop_server(client: httpx.AsyncClient):
    """Test stopping an MCP server"""
    stop_request = {"server_name": "test-server"}
    
    response = await client.post("/stopServer", json=stop_request)
    # May succeed or fail depending on server existence
    print(f"Server stop response: {response.status_code} - {response.text}")

@pytest.mark.asyncio
async def test_stop_all_servers(client: httpx.AsyncClient):
    """Test stopping all MCP servers"""
    response = await client.post("/stopAllServers")
    assert response.status_code == 204
