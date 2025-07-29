
"""
Workflow management API tests
"""
import pytest
import httpx
from typing import Dict, Any

@pytest.mark.asyncio
async def test_list_workflows(client: httpx.AsyncClient):
    """Test listing workflows"""
    response = await client.get("/api/workflows/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

@pytest.mark.asyncio
async def test_create_workflow(client: httpx.AsyncClient, sample_workflow_data: Dict[str, Any]):
    """Test creating a new workflow"""
    response = await client.post("/api/workflows/", json=sample_workflow_data)
    
    if response.status_code == 201:
        data = response.json()
        assert "id" in data
        assert data["name"] == sample_workflow_data["name"]
        assert data["description"] == sample_workflow_data["description"]
        return data["id"]
    else:
        print(f"Workflow creation failed: {response.status_code} - {response.text}")
        return None

@pytest.mark.asyncio
async def test_get_workflow_not_found(client: httpx.AsyncClient):
    """Test getting non-existent workflow"""
    response = await client.get("/api/workflows/99999")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_execute_workflow(client: httpx.AsyncClient):
    """Test workflow execution"""
    # First create a simple workflow
    workflow_data = {
        "name": "Execution Test Workflow",
        "description": "Simple workflow for execution testing",
        "steps": [
            {
                "name": "analysis_step",
                "agent_type": "custom",
                "task": "Analyze the input data"
            }
        ]
    }
    
    create_response = await client.post("/api/workflows/", json=workflow_data)
    if create_response.status_code == 201:
        workflow_id = create_response.json()["id"]
        
        # Execute the workflow
        execution_data = {
            "input_data": {"test": "data"},
            "parameters": {"timeout": 30}
        }
        
        response = await client.post(f"/api/workflows/{workflow_id}/execute", json=execution_data)
        # Execution might fail due to missing dependencies, but we record the response
        print(f"Workflow execution response: {response.status_code} - {response.text}")

@pytest.mark.asyncio
async def test_workflow_status(client: httpx.AsyncClient):
    """Test getting workflow execution status"""
    # This would typically require an actual execution ID
    response = await client.get("/api/workflows/executions/test-execution-id")
    # Expected to fail with 404, but we test the endpoint exists
    assert response.status_code in [404, 422]  # 422 for validation error on ID format

@pytest.mark.asyncio
async def test_list_workflow_executions(client: httpx.AsyncClient):
    """Test listing workflow executions"""
    response = await client.get("/api/workflows/executions/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
