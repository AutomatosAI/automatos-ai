"""Journey 03: Agent management — CRUD + detail endpoints."""

import pytest


AGENT_PAYLOAD = {
    "name": "nightly-test-agent",
    "agent_type": "custom",
    "description": "Ephemeral agent created by the nightly test suite",
    "configuration": {
        "model": "openai/gpt-4o-mini",
        "system_prompt": "You are a test agent.",
    },
}


# ── CRUD ─────────────────────────────────────────────────────────────


def test_list_agents(client):
    r = client.get("/api/agents/")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_create_agent(client, created_agent_ids):
    r = client.post("/api/agents/", json=AGENT_PAYLOAD)
    assert r.status_code in (200, 201)
    data = r.json()
    assert "id" in data
    created_agent_ids.append(data["id"])


def test_get_agent(client, created_agent_ids):
    if not created_agent_ids:
        pytest.skip("No agent was created in this session")
    agent_id = created_agent_ids[-1]
    r = client.get(f"/api/agents/{agent_id}")
    assert r.status_code == 200
    assert r.json()["id"] == agent_id


def test_update_agent(client, created_agent_ids):
    if not created_agent_ids:
        pytest.skip("No agent was created in this session")
    agent_id = created_agent_ids[-1]
    r = client.put(f"/api/agents/{agent_id}", json={"description": "updated by nightly"})
    assert r.status_code == 200


def test_delete_agent(client, created_agent_ids):
    if not created_agent_ids:
        pytest.skip("No agent was created in this session")
    agent_id = created_agent_ids.pop()
    r = client.delete(f"/api/agents/{agent_id}")
    assert r.status_code in (200, 204)


# ── Detail endpoints (Journey 03) ───────────────────────────────────


def test_agent_statistics(client):
    r = client.get("/api/system/agent-statistics")
    assert r.status_code == 200
    assert "total_agents" in r.json()


def test_agent_status(client, first_agent_id):
    if not first_agent_id:
        pytest.skip("No agent available")
    r = client.get(f"/api/agents/{first_agent_id}/status")
    assert r.status_code == 200


def test_agent_performance(client, first_agent_id):
    if not first_agent_id:
        pytest.skip("No agent available")
    r = client.get(f"/api/agents/{first_agent_id}/performance", params={"period": "all"})
    assert r.status_code == 200


def test_agent_model_config(client, first_agent_id):
    if not first_agent_id:
        pytest.skip("No agent available")
    r = client.get(f"/api/agents/{first_agent_id}/model-config")
    assert r.status_code == 200


def test_agent_logs(client, first_agent_id):
    if not first_agent_id:
        pytest.skip("No agent available")
    r = client.get(f"/api/agents/{first_agent_id}/logs", params={"limit": 10})
    assert r.status_code == 200


def test_agent_types(client):
    r = client.get("/api/system/agent-types")
    assert r.status_code == 200
