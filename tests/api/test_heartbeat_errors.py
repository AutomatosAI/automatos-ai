"""Heartbeat error handling — validates graceful failures for edge cases."""


def test_heartbeat_agent_run_nonexistent(client):
    """POST /api/heartbeat/agents/{id}/run for non-existent agent."""
    r = client.post("/api/heartbeat/agents/999999/run")
    assert r.status_code != 500, (
        f"Heartbeat for non-existent agent returned 500: {r.text[:300]}"
    )


def test_heartbeat_agent_run_invalid_id(client):
    """POST /api/heartbeat/agents/{id}/run with non-numeric ID."""
    r = client.post("/api/heartbeat/agents/not-a-number/run")
    assert r.status_code in (400, 404, 422), (
        f"Heartbeat with invalid agent ID returned {r.status_code}"
    )


def test_heartbeat_history_invalid_limit(client):
    """GET /api/heartbeat/orchestrator/history with negative limit."""
    r = client.get("/api/heartbeat/orchestrator/history", params={"limit": -1})
    assert r.status_code != 500, (
        f"Heartbeat history with negative limit returned 500: {r.text[:300]}"
    )


def test_heartbeat_history_huge_limit(client):
    """GET /api/heartbeat/orchestrator/history with absurd limit should not crash."""
    r = client.get("/api/heartbeat/orchestrator/history", params={"limit": 999999})
    assert r.status_code != 500, (
        f"Heartbeat history with huge limit returned 500: {r.text[:300]}"
    )
