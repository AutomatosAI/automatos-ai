"""User Journey: New user onboarding flow.

Simulates what happens when a brand-new user signs up and sets up their
workspace for the first time. Tests the critical path that determines
first-session retention.

Flow:
  1. Workspace current → confirms workspace exists
  2. List agents → empty or seeded
  3. Create first agent → with model config
  4. List models → pick appropriate model
  5. Chat with agent → first conversation
  6. Check memory → verify memory system is ready
  7. View analytics → dashboard should render (even if empty)
  8. Clean up
"""

import pytest

from .helpers import uid, pick, CHAT_MESSAGES, parse_sse_response


@pytest.fixture(scope="module")
def onboarding_state():
    return {
        "agent_id": None,
        "chat_id": None,
    }


def test_onboarding_01_workspace_exists(client):
    """Step 1: New user hits workspace endpoint — must return valid workspace."""
    r = client.get("/api/workspaces/current")
    assert r.status_code == 200
    data = r.json()
    assert "id" in data or "workspace" in data, (
        "Workspace current returned no ID — user would see blank screen"
    )


def test_onboarding_02_discover_agents(client):
    """Step 2: New user checks available agents."""
    r = client.get("/api/agents/")
    assert r.status_code == 200
    agents = r.json()
    assert isinstance(agents, list), "Agent list should be an array"


def test_onboarding_03_discover_models(client):
    """Step 3: User browses available models for their first agent."""
    r = client.get("/api/models/")
    assert r.status_code == 200


def test_onboarding_04_create_first_agent(client, created_agent_ids, onboarding_state):
    """Step 4: User creates their first agent."""
    r = client.post("/api/agents/", json={
        "name": uid("onboard-agent"),
        "agent_type": "custom",
        "description": "My first AI agent",
        "configuration": {
            "model": "openai/gpt-4o-mini",
            "system_prompt": "You are a helpful assistant for a new user.",
        },
    })
    assert r.status_code in (200, 201), (
        f"First agent creation failed: {r.status_code} {r.text[:300]}"
    )
    data = r.json()
    onboarding_state["agent_id"] = data["id"]
    created_agent_ids.append(data["id"])


def test_onboarding_05_first_chat(client, onboarding_state):
    """Step 5: User sends their first chat message."""
    agent_id = onboarding_state["agent_id"]
    body = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": "Hello! What can you help me with?"}],
        },
    }
    if agent_id:
        body["agentId"] = agent_id

    r = client.post("/api/chat", json=body, timeout=60.0)
    assert r.status_code == 200, (
        f"First chat failed: {r.status_code} {r.text[:300]}"
    )
    parsed = parse_sse_response(r)
    if parsed["chat_id"]:
        onboarding_state["chat_id"] = parsed["chat_id"]
    # Should get some text back
    assert parsed["text"] or parsed["data_events"], (
        "First chat returned no content — user would see blank response"
    )


def test_onboarding_06_check_memory_system(client):
    """Step 6: Memory system should be accessible (even if empty)."""
    r = client.get("/api/v1/memory/stats/real")
    assert r.status_code == 200
    data = r.json()
    assert "system_stats" in data, (
        "Memory stats missing system_stats — memory UI would fail"
    )


def test_onboarding_07_view_dashboard(client):
    """Step 7: Analytics dashboard should render without errors."""
    r = client.get("/analytics/dashboard/summary")
    assert r.status_code == 200


def test_onboarding_08_check_tools_marketplace(client):
    """Step 8: User browses tool marketplace to extend their agent."""
    r = client.get("/api/tools/marketplace", params={"limit": 10})
    assert r.status_code == 200
    data = r.json()
    assert "apps" in data or "items" in data or isinstance(data, list)


def test_onboarding_09_cleanup(client, onboarding_state):
    """Step 9: Clean up chat if created."""
    if onboarding_state["chat_id"]:
        r = client.delete(f"/api/chat/{onboarding_state['chat_id']}")
        assert r.status_code in (200, 204, 404)
