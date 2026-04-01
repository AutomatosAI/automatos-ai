"""User Journey: Developer integrating tools and skills.

Simulates a developer connecting external tools, assigning skills to
agents, configuring credentials, and verifying the integration works.

Flow:
  1. Browse marketplace → discover available tools
  2. Check connected tools → current integrations
  3. List credentials → existing connections
  4. Browse skills → available capabilities
  5. Check skill sources → where skills come from
  6. Recommend skills for task → AI-driven matching
  7. Assign tools/skills to agent → configure agent
  8. Verify agent has capabilities → read back
  9. Check platform key status → provider health
  10. Refresh connections → sync state
"""

import pytest

from .helpers import pick, SEARCH_TERMS, SKILL_TASKS


def test_integration_01_browse_marketplace(client):
    """Step 1: Browse the tool marketplace."""
    r = client.get("/api/tools/marketplace", params={"limit": 20})
    assert r.status_code == 200
    data = r.json()
    assert "apps" in data or "items" in data or isinstance(data, list)


def test_integration_02_search_marketplace(client):
    """Step 2: Search for specific tool integrations."""
    r = client.get("/api/tools/marketplace", params={
        "search": pick(SEARCH_TERMS),
        "limit": 5,
    })
    assert r.status_code == 200


def test_integration_03_connected_tools(client):
    """Step 3: Check which tools are already connected."""
    r = client.get("/api/tools/connected")
    assert r.status_code == 200


def test_integration_04_list_credentials(client):
    """Step 4: Review existing credentials."""
    r = client.get("/api/credentials/")
    assert r.status_code == 200


def test_integration_05_tool_stats(client):
    """Step 5: Check tool usage statistics."""
    r = client.get("/api/tools/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_apps" in data


def test_integration_06_browse_skills(client):
    """Step 6: Browse available skills."""
    r = client.get("/api/v1/skills", params={"limit": 20})
    assert r.status_code == 200


def test_integration_07_skill_sources(client):
    """Step 7: Check skill sources."""
    r = client.get("/api/v1/skills/sources")
    assert r.status_code == 200


def test_integration_08_recommend_skill(client):
    """Step 8: Get AI skill recommendation for a task."""
    task = pick(SKILL_TASKS)
    r = client.post("/api/v1/skills/recommend", json={
        "task_description": task["description"],
        "task_type": task["type"],
        "limit": 3,
    })
    assert r.status_code == 200


def test_integration_09_agent_skills(client, first_agent_id):
    """Step 9: Check skills assigned to an agent."""
    if not first_agent_id:
        pytest.skip("No agent available")
    r = client.get(f"/api/v1/skills/agents/{first_agent_id}/skills")
    assert r.status_code == 200


def test_integration_10_platform_key_status(client):
    """Step 10: Verify platform API key health."""
    r = client.get("/api/keys/platform-status")
    assert r.status_code == 200
    data = r.json()
    assert "platform_keys" in data


def test_integration_11_marketplace_items(client):
    """Step 11: Check marketplace items catalog."""
    r = client.get("/api/marketplace/items")
    assert r.status_code == 200


def test_integration_12_refresh_connections(client):
    """Step 12: Refresh tool connections to sync state."""
    r = client.post("/api/tools/refresh-connections")
    assert r.status_code == 200


def test_integration_13_byok_preferences(client):
    """Step 13: Check BYOK preferences."""
    r = client.get("/api/workspaces/current/byok-preferences")
    assert r.status_code == 200
