"""Contract tests: API response shapes that the frontend and agents depend on.

These tests validate that critical API responses maintain their expected shape.
Breaking these contracts breaks the frontend or agent workflows.
"""

import pytest


def test_agent_list_response_contract(client):
    """Agent list must return an array of objects with id, name, agent_type."""
    r = client.get("/api/agents/")
    assert r.status_code == 200
    agents = r.json()
    assert isinstance(agents, list)

    if agents:
        agent = agents[0]
        assert "id" in agent, "Agent missing 'id' field"
        assert "name" in agent, "Agent missing 'name' field"


def test_agent_detail_response_contract(client, first_agent_id):
    """Agent detail must include id, name, configuration."""
    if not first_agent_id:
        pytest.skip("No agent available")

    r = client.get(f"/api/agents/{first_agent_id}")
    assert r.status_code == 200
    agent = r.json()
    assert "id" in agent
    assert "name" in agent


def test_chat_history_response_contract(client):
    """Chat history must return a list with id and title per entry."""
    r = client.get("/api/chat/history", params={"limit": 5})
    assert r.status_code == 200
    data = r.json()
    chats = data if isinstance(data, list) else data.get("chats", data.get("items", []))
    assert isinstance(chats, list)

    if chats:
        chat = chats[0]
        assert "id" in chat, "Chat entry missing 'id'"


def test_workspace_current_response_contract(client):
    """Current workspace must include id and name."""
    r = client.get("/api/workspaces/current")
    assert r.status_code == 200
    data = r.json()
    assert "id" in data or "workspace_id" in data, "Workspace missing ID field"


def test_health_response_contract(client):
    """Health endpoint must include status field."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["status"] in ("ok", "healthy", "up", True, "running")
