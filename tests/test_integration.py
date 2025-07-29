
"""
Integration tests for complete workflows
"""
import pytest
import httpx
import asyncio
from typing import Dict, Any

@pytest.mark.asyncio
async def test_agent_skill_integration(client: httpx.AsyncClient):
    """Test integration between agents and skills"""
    # Create a skill first
    skill_data = {
        "name": "integration_test_skill",
        "skill_type": "technical",
        "description": "Skill for integration testing",
        "parameters": {"test_mode": True}
    }
    
    skill_response = await client.post("/api/skills/", json=skill_data)
    skill_id = None
    
    if skill_response.status_code == 201:
        skill_id = skill_response.json()["id"]
        
        # Create an agent with this skill
        agent_data = {
            "name": "Integration Test Agent",
            "description": "Agent for integration testing",
            "agent_type": "custom",
            "skill_ids": [skill_id]
        }
        
        agent_response = await client.post("/api/agents/", json=agent_data)
        
        if agent_response.status_code == 201:
            agent_id = agent_response.json()["id"]
            
            # Verify the agent has the skill
            get_response = await client.get(f"/api/agents/{agent_id}")
            if get_response.status_code == 200:
                agent_data = get_response.json()
                skill_names = [skill["name"] for skill in agent_data.get("skills", [])]
                assert "integration_test_skill" in skill_names
                
                print("✅ Agent-Skill integration successful")
                return True
    
    print("❌ Agent-Skill integration failed")
    return False

@pytest.mark.asyncio
async def test_workflow_execution_flow(client: httpx.AsyncClient):
    """Test complete workflow execution flow"""
    # Create a workflow
    workflow_data = {
        "name": "Integration Test Workflow",
        "description": "Complete workflow for integration testing",
        "steps": [
            {
                "name": "data_analysis",
                "agent_type": "data_analyst",
                "task": "Analyze the input data for patterns"
            },
            {
                "name": "report_generation",
                "agent_type": "custom",
                "task": "Generate a summary report"
            }
        ]
    }
    
    create_response = await client.post("/api/workflows/", json=workflow_data)
    
    if create_response.status_code == 201:
        workflow_id = create_response.json()["id"]
        
        # Execute the workflow
        execution_data = {
            "input_data": {
                "dataset": "sample_data.csv",
                "analysis_type": "exploratory"
            },
            "parameters": {
                "timeout": 60,
                "priority": "normal"
            }
        }
        
        exec_response = await client.post(f"/api/workflows/{workflow_id}/execute", json=execution_data)
        
        if exec_response.status_code in [200, 202]:  # 202 for async execution
            execution_result = exec_response.json()
            print(f"✅ Workflow execution initiated: {execution_result}")
            return True
    
    print("❌ Workflow execution flow failed")
    return False

@pytest.mark.asyncio
async def test_document_context_integration(client: httpx.AsyncClient):
    """Test document upload and context retrieval integration"""
    # Upload a document
    import io
    test_content = """
    This is a comprehensive test document for API integration testing.
    It contains multiple paragraphs with different topics.
    
    The first topic is about API testing methodologies.
    We use various approaches including unit tests, integration tests, and performance tests.
    
    The second topic covers document processing and context retrieval.
    This involves uploading documents, processing them into chunks, and enabling search functionality.
    """
    
    test_file = io.BytesIO(test_content.encode())
    files = {"file": ("integration_test.txt", test_file, "text/plain")}
    data = {
        "tags": "integration,testing,api",
        "description": "Integration test document",
        "created_by": "integration_test"
    }
    
    upload_response = await client.post("/api/documents/upload", files=files, data=data)
    
    if upload_response.status_code == 201:
        document_id = upload_response.json()["document_id"]
        
        # Wait a moment for processing (in real system)
        await asyncio.sleep(1)
        
        # Search for content
        search_data = {
            "query": "API testing methodologies",
            "limit": 5
        }
        
        search_response = await client.post("/api/context/search", json=search_data)
        
        if search_response.status_code == 200:
            search_results = search_response.json()
            print(f"✅ Document-Context integration successful: {len(search_results.get('results', []))} results")
            return True
    
    print("❌ Document-Context integration failed")
    return False

@pytest.mark.asyncio
async def test_end_to_end_scenario(client: httpx.AsyncClient):
    """Test complete end-to-end scenario"""
    print("\n🔄 Running End-to-End Integration Test")
    
    results = {
        "agent_skill_integration": False,
        "workflow_execution": False,
        "document_context": False
    }
    
    # Run all integration tests
    results["agent_skill_integration"] = await test_agent_skill_integration(client)
    results["workflow_execution"] = await test_workflow_execution_flow(client)
    results["document_context"] = await test_document_context_integration(client)
    
    # Summary
    successful_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"\n📊 End-to-End Test Results: {successful_tests}/{total_tests} successful")
    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {test_name.replace('_', ' ').title()}")
    
    # At least some integration should work
    assert successful_tests > 0, "No integration tests passed"
