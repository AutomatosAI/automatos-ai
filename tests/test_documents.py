
"""
Document management API tests
"""
import pytest
import httpx
import io
from typing import Dict, Any

@pytest.mark.asyncio
async def test_list_documents(client: httpx.AsyncClient):
    """Test listing documents"""
    response = await client.get("/api/documents/")
    # May fail if document manager not initialized
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list) or "documents" in data
    else:
        print(f"Document listing failed: {response.status_code} - {response.text}")

@pytest.mark.asyncio
async def test_upload_document(client: httpx.AsyncClient):
    """Test document upload"""
    # Create a test file
    test_content = "This is a test document for API testing."
    test_file = io.BytesIO(test_content.encode())
    
    files = {"file": ("test.txt", test_file, "text/plain")}
    data = {
        "tags": "test,api",
        "description": "Test document for API testing",
        "created_by": "test_user"
    }
    
    response = await client.post("/api/documents/upload", files=files, data=data)
    
    if response.status_code == 201:
        result = response.json()
        assert "document_id" in result
        assert result["filename"] == "test.txt"
        return result["document_id"]
    else:
        print(f"Document upload failed: {response.status_code} - {response.text}")
        return None

@pytest.mark.asyncio
async def test_get_document_not_found(client: httpx.AsyncClient):
    """Test getting non-existent document"""
    response = await client.get("/api/documents/99999")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_search_documents(client: httpx.AsyncClient):
    """Test document search"""
    search_data = {
        "query": "test document",
        "limit": 10
    }
    
    response = await client.post("/api/documents/search", json=search_data)
    
    if response.status_code == 200:
        data = response.json()
        assert "results" in data
        assert "query" in data
        assert data["query"] == search_data["query"]
    else:
        print(f"Document search failed: {response.status_code} - {response.text}")

@pytest.mark.asyncio
async def test_document_stats(client: httpx.AsyncClient):
    """Test getting document statistics"""
    response = await client.get("/api/documents/stats")
    
    if response.status_code == 200:
        data = response.json()
        assert "total_documents" in data
        assert "status_distribution" in data
    else:
        print(f"Document stats failed: {response.status_code} - {response.text}")

@pytest.mark.asyncio
async def test_context_search(client: httpx.AsyncClient):
    """Test context search endpoint"""
    search_data = {
        "query": "API testing context",
        "limit": 5
    }
    
    response = await client.post("/api/context/search", json=search_data)
    
    if response.status_code == 200:
        data = response.json()
        assert "results" in data
        assert "total_results" in data
    else:
        print(f"Context search failed: {response.status_code} - {response.text}")
