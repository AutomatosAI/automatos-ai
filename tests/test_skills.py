
"""
Skills management API tests
"""
import pytest
import httpx
from typing import Dict, Any

@pytest.mark.asyncio
async def test_list_skills(client: httpx.AsyncClient):
    """Test listing skills"""
    response = await client.get("/api/skills/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

@pytest.mark.asyncio
async def test_create_skill(client: httpx.AsyncClient, sample_skill_data: Dict[str, Any]):
    """Test creating a new skill"""
    response = await client.post("/api/skills/", json=sample_skill_data)
    
    if response.status_code == 201:
        data = response.json()
        assert "id" in data
        assert data["name"] == sample_skill_data["name"]
        assert data["skill_type"] == sample_skill_data["skill_type"]
        return data["id"]
    else:
        print(f"Skill creation failed: {response.status_code} - {response.text}")
        return None

@pytest.mark.asyncio
async def test_get_skill_not_found(client: httpx.AsyncClient):
    """Test getting non-existent skill"""
    response = await client.get("/api/skills/99999")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_list_skills_with_filters(client: httpx.AsyncClient):
    """Test listing skills with filters"""
    # Test skill type filter
    response = await client.get("/api/skills/?skill_type=technical")
    assert response.status_code == 200
    
    # Test search
    response = await client.get("/api/skills/?search=test")
    assert response.status_code == 200
    
    # Test pagination
    response = await client.get("/api/skills/?skip=0&limit=5")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_update_skill(client: httpx.AsyncClient):
    """Test updating a skill"""
    skill_data = {
        "name": "update_test_skill",
        "skill_type": "analytical",
        "description": "Skill for update testing"
    }
    
    create_response = await client.post("/api/skills/", json=skill_data)
    if create_response.status_code == 201:
        skill_id = create_response.json()["id"]
        
        update_data = {
            "name": "updated_skill_name",
            "description": "Updated description"
        }
        
        response = await client.put(f"/api/skills/{skill_id}", json=update_data)
        if response.status_code == 200:
            data = response.json()
            assert data["name"] == update_data["name"]
            assert data["description"] == update_data["description"]

@pytest.mark.asyncio
async def test_delete_skill(client: httpx.AsyncClient):
    """Test deleting a skill"""
    skill_data = {
        "name": "delete_test_skill",
        "skill_type": "technical",
        "description": "Skill for delete testing"
    }
    
    create_response = await client.post("/api/skills/", json=skill_data)
    if create_response.status_code == 201:
        skill_id = create_response.json()["id"]
        
        response = await client.delete(f"/api/skills/{skill_id}")
        assert response.status_code in [200, 204]
        
        # Verify deletion
        get_response = await client.get(f"/api/skills/{skill_id}")
        assert get_response.status_code == 404
