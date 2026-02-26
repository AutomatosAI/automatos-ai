"""Journey 10: Heartbeat monitoring — status, orchestrator, agents."""

import pytest


def test_heartbeat_status(client):
    r = client.get("/api/heartbeat/status")
    assert r.status_code == 200


def test_heartbeat_orchestrator_run(client):
    r = client.post("/api/heartbeat/orchestrator/run")
    assert r.status_code == 200


def test_heartbeat_orchestrator_history(client):
    r = client.get("/api/heartbeat/orchestrator/history", params={"limit": 5})
    assert r.status_code == 200


def test_heartbeat_analytics(client):
    r = client.get("/api/heartbeat/analytics")
    assert r.status_code == 200


def test_heartbeat_agent_run(client, first_agent_id):
    if not first_agent_id:
        pytest.skip("No agent available")
    r = client.post(f"/api/heartbeat/agents/{first_agent_id}/run")
    assert r.status_code == 200
