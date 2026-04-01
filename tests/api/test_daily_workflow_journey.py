"""User Journey: Power user daily workflow.

Simulates the daily routine of an active platform user who checks agent
status, reviews heartbeats, chats with agents, checks analytics, and
monitors their workspace.

Flow:
  1. Health check → platform is up
  2. Check heartbeat status → agents are healthy
  3. Run orchestrator heartbeat → daily check
  4. Review agent performance → identify issues
  5. Chat with an agent → daily work
  6. Check LLM usage/costs → budget awareness
  7. Review memory stats → knowledge health
  8. Check recent missions → work tracking
  9. Review workflow recipes → available automations
"""

import pytest

from .helpers import pick, CHAT_MESSAGES, parse_sse_response


@pytest.fixture(scope="module")
def daily_state():
    return {
        "chat_id": None,
    }


def test_daily_01_platform_health(client):
    """Step 1: User opens platform — health check fires."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)


def test_daily_02_heartbeat_status(client):
    """Step 2: Check if agents are healthy before interacting."""
    r = client.get("/api/heartbeat/status")
    assert r.status_code == 200


def test_daily_03_run_orchestrator(client):
    """Step 3: Trigger daily orchestrator heartbeat run."""
    r = client.post("/api/heartbeat/orchestrator/run")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)


def test_daily_04_review_agent_roster(client):
    """Step 4: Review agent roster and their stats."""
    r = client.get("/api/agents/")
    assert r.status_code == 200
    agents = r.json()
    assert isinstance(agents, list)

    # Check agent statistics summary
    r2 = client.get("/api/system/agent-statistics")
    assert r2.status_code == 200
    stats = r2.json()
    assert "total_agents" in stats


def test_daily_05_agent_performance(client, first_agent_id):
    """Step 5: Check specific agent performance."""
    if not first_agent_id:
        pytest.skip("No agent available")
    r = client.get(f"/api/agents/{first_agent_id}/performance", params={"period": "7d"})
    assert r.status_code == 200


def test_daily_06_chat_with_agent(client, first_agent_id, daily_state):
    """Step 6: Have a working conversation with an agent."""
    body = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": pick(CHAT_MESSAGES)}],
        },
    }
    if first_agent_id:
        body["agentId"] = first_agent_id

    r = client.post("/api/chat", json=body, timeout=60.0)
    assert r.status_code == 200
    parsed = parse_sse_response(r)
    if parsed["chat_id"]:
        daily_state["chat_id"] = parsed["chat_id"]


def test_daily_07_check_chat_history(client):
    """Step 7: Review recent chat history."""
    r = client.get("/api/chat/history", params={"limit": 10})
    assert r.status_code == 200
    data = r.json()
    chats = data if isinstance(data, list) else data.get("chats", data.get("items", []))
    assert isinstance(chats, list)


def test_daily_08_llm_usage_costs(client):
    """Step 8: Check LLM usage and costs — budget awareness."""
    r = client.get("/api/analytics/llm/summary", params={"period": "7d"})
    assert r.status_code == 200
    data = r.json()
    assert "total_requests" in data

    r2 = client.get("/api/analytics/llm/costs", params={"period": "7d", "breakdown": "model"})
    assert r2.status_code == 200


def test_daily_09_memory_health(client):
    """Step 9: Check memory system health."""
    r = client.get("/api/workspaces/current/memory-stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_memories" in data


def test_daily_10_recent_missions(client):
    """Step 10: Review recent missions for work tracking."""
    r = client.get("/api/missions", params={"limit": 5})
    assert r.status_code == 200


def test_daily_11_available_recipes(client):
    """Step 11: Browse available workflow recipes."""
    r = client.get("/api/workflow-recipes", params={"limit": 10})
    assert r.status_code == 200


def test_daily_12_system_metrics(client):
    """Step 12: Quick system metrics check."""
    r = client.get("/api/system/metrics")
    assert r.status_code == 200


def test_daily_cleanup(client, daily_state):
    """Cleanup: remove test chat."""
    if daily_state["chat_id"]:
        client.delete(f"/api/chat/{daily_state['chat_id']}")
