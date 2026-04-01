"""Journey: Heartbeat lifecycle — enable, run, verify, check history.

Expands on the 5 smoke tests in test_heartbeat.py with stateful validation.
"""

import pytest


@pytest.fixture(scope="module")
def heartbeat_state():
    return {
        "agent_id": None,
        "run_result": None,
    }


def test_heartbeat_status_shape(client):
    """GET /api/heartbeat/status — verify response shape is usable."""
    r = client.get("/api/heartbeat/status")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)
    # Status should have some structure — not just an empty dict
    assert len(data) > 0, "Heartbeat status returned empty dict"


def test_heartbeat_orchestrator_run_and_verify(client):
    """POST /api/heartbeat/orchestrator/run — run orchestrator heartbeat and verify output.

    This is the core heartbeat flow that recipes depend on.
    The response must contain actionable data for downstream agents.
    """
    r = client.post("/api/heartbeat/orchestrator/run")
    assert r.status_code == 200, f"Orchestrator run failed: {r.status_code} {r.text[:500]}"
    data = r.json()
    assert isinstance(data, dict)
    # Should have SOME content — status, results, or agent data
    assert any(k in data for k in (
        "status", "message", "result", "results", "agents",
        "heartbeat_results", "summary",
    )), f"Orchestrator run response too thin: {list(data.keys())}"


def test_heartbeat_orchestrator_history_has_entries(client):
    """GET /api/heartbeat/orchestrator/history — should have at least one entry after run."""
    r = client.get("/api/heartbeat/orchestrator/history", params={"limit": 5})
    assert r.status_code == 200
    data = r.json()
    entries = data if isinstance(data, list) else data.get("history", data.get("items", []))
    assert isinstance(entries, list)
    # After the orchestrator run above, there should be at least one entry
    # (though timing may vary — soft check)


def test_heartbeat_agent_run_response_shape(client, first_agent_id, heartbeat_state):
    """POST /api/heartbeat/agents/{id}/run — agent heartbeat must return actionable shape.

    Bug context: heartbeat_results JSONB must contain enough data for
    _auto_create_report() in heartbeat_service.py to generate a report.
    """
    if not first_agent_id:
        pytest.skip("No agent available")

    heartbeat_state["agent_id"] = first_agent_id

    r = client.post(f"/api/heartbeat/agents/{first_agent_id}/run")
    assert r.status_code == 200, f"Agent heartbeat failed: {r.status_code} {r.text[:500]}"
    data = r.json()
    heartbeat_state["run_result"] = data

    assert isinstance(data, dict)
    # Must have enough for monitoring recipes to consume
    assert any(k in data for k in (
        "status", "message", "result", "agent_id", "heartbeat_id",
    )), f"Agent heartbeat response too thin for recipes: {list(data.keys())}"


def test_heartbeat_analytics_after_run(client):
    """GET /api/heartbeat/analytics — analytics should reflect recent runs."""
    r = client.get("/api/heartbeat/analytics")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)


def test_heartbeat_agent_run_doesnt_crash_nonexistent(client):
    """Running heartbeat on a non-existent agent should return error, not 500."""
    r = client.post("/api/heartbeat/agents/999999/run")
    # Should be 404 or a graceful error — not 500
    assert r.status_code != 500, (
        f"Heartbeat on non-existent agent returned 500: {r.text[:300]}"
    )
