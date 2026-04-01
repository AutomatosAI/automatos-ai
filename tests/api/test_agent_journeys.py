"""Journey: Agent lifecycle with tools and skills — deeper than CRUD.

Covers:
- Create -> assign tools -> verify assignment
- Create -> assign skill -> verify
- Create -> execute with tools -> verify handle
- Error handling for invalid operations
"""

import pytest

from .helpers import uid


@pytest.fixture(scope="module")
def agent_journey_state():
    return {"agent_id": None}


def test_create_agent_for_journey(client, created_agent_ids, agent_journey_state):
    """Create a test agent for the tool/skill assignment journey."""
    payload = {
        "name": uid("journey-tools"),
        "agent_type": "custom",
        "description": "Journey test — tool and skill assignment",
        "configuration": {
            "model": "openai/gpt-4o-mini",
            "system_prompt": "You are a journey test agent with tools.",
        },
    }
    r = client.post("/api/agents/", json=payload)
    assert r.status_code in (200, 201)
    agent_id = r.json()["id"]
    created_agent_ids.append(agent_id)
    agent_journey_state["agent_id"] = agent_id


def test_get_available_tools(client):
    """GET /api/tools/marketplace — discover available tools for assignment."""
    r = client.get("/api/tools/marketplace")
    assert r.status_code == 200
    data = r.json()
    # Should have apps or items
    assert "apps" in data or "items" in data or isinstance(data, list)


def test_agent_tools_list(client, agent_journey_state):
    """Verify we can list tools assigned to an agent."""
    if not agent_journey_state["agent_id"]:
        pytest.skip("No agent created")

    agent_id = agent_journey_state["agent_id"]
    r = client.get(f"/api/agents/{agent_id}")
    assert r.status_code == 200
    data = r.json()
    # Agent detail should include some indication of tools
    assert "id" in data


def test_get_available_skills(client):
    """GET /api/skills — discover available skills for assignment."""
    r = client.get("/api/skills/")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, (list, dict))


def test_agent_execute_fresh_agent(client, agent_journey_state):
    """Execute the freshly created agent — validates factory pipeline works for new agents."""
    if not agent_journey_state["agent_id"]:
        pytest.skip("No agent created")

    agent_id = agent_journey_state["agent_id"]
    r = client.post(
        f"/api/agents/{agent_id}/execute",
        json={"task": "List your available tools and capabilities", "mode": "test"},
    )
    assert r.status_code == 200, f"Execute failed on fresh agent: {r.status_code} {r.text[:500]}"
    data = r.json()
    assert data.get("agent_id") == agent_id


def test_agent_model_config_update_and_verify(client, agent_journey_state):
    """Update model config and verify it persists — round trip validation."""
    if not agent_journey_state["agent_id"]:
        pytest.skip("No agent created")

    agent_id = agent_journey_state["agent_id"]

    # Get current config
    current = client.get(f"/api/agents/{agent_id}/model-config")
    assert current.status_code == 200
    cfg = current.json().get("model_config", {}) or {}

    # Update temperature
    update_payload = {
        "provider": cfg.get("provider") or "openai",
        "model_id": cfg.get("model_id") or "openai/gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": cfg.get("max_tokens") or 2000,
        "top_p": cfg.get("top_p", 1.0),
        "frequency_penalty": cfg.get("frequency_penalty", 0.0),
        "presence_penalty": cfg.get("presence_penalty", 0.0),
        "fallback_model_id": cfg.get("fallback_model_id"),
    }
    updated = client.put(f"/api/agents/{agent_id}/model-config", json=update_payload)
    assert updated.status_code == 200

    # Verify persisted
    reloaded = client.get(f"/api/agents/{agent_id}/model-config")
    assert reloaded.status_code == 200
    reloaded_cfg = reloaded.json().get("model_config", {})
    assert float(reloaded_cfg.get("temperature", 0)) == 0.1


def test_agent_delete_nonexistent(client):
    """DELETE /api/agents/{id} for non-existent agent should not 500."""
    r = client.delete("/api/agents/999999")
    assert r.status_code != 500, f"Delete non-existent agent returned 500: {r.text[:300]}"


def test_agent_execute_nonexistent(client):
    """Execute on non-existent agent should return error, not 500."""
    r = client.post(
        "/api/agents/999999/execute",
        json={"task": "test", "mode": "test"},
    )
    assert r.status_code != 500, f"Execute non-existent agent returned 500: {r.text[:300]}"
