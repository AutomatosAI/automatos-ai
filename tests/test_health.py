
"""
Health check and basic connectivity tests
"""
import pytest
import httpx

@pytest.mark.asyncio
async def test_health_check(client: httpx.AsyncClient):
    """Test basic health check endpoint"""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

@pytest.mark.asyncio
async def test_root_endpoint(client: httpx.AsyncClient):
    """Test root endpoint"""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data

@pytest.mark.asyncio
async def test_openapi_docs(client: httpx.AsyncClient):
    """Test OpenAPI documentation endpoint"""
    response = await client.get("/docs")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_openapi_json(client: httpx.AsyncClient):
    """Test OpenAPI JSON schema endpoint"""
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "paths" in data
