"""Agent error handling — validates graceful failures for edge cases.

Every CRUD endpoint must return proper error codes, never 500.
"""

import pytest

from .helpers import uid


# ── GET nonexistent ──────────────────────────────────────────────────

def test_get_nonexistent_agent(client):
    """GET /api/agents/{id} for non-existent agent should return 404."""
    r = client.get("/api/agents/999999")
    assert r.status_code in (404, 400), (
        f"Non-existent agent returned {r.status_code}, expected 404"
    )


def test_get_agent_invalid_id_type(client):
    """GET /api/agents/{id} with non-numeric ID should return 400/404/422."""
    r = client.get("/api/agents/not-a-number")
    assert r.status_code in (400, 404, 422), (
        f"Invalid agent ID type returned {r.status_code}"
    )


# ── CREATE validation ────────────────────────────────────────────────

def test_create_agent_empty_body(client):
    """POST /api/agents/ with empty body should return 400/422, not 500."""
    r = client.post("/api/agents/", json={})
    assert r.status_code in (400, 422), (
        f"Empty agent body returned {r.status_code}, expected 400/422"
    )


def test_create_agent_missing_name(client):
    """POST /api/agents/ without name should fail validation."""
    r = client.post("/api/agents/", json={
        "agent_type": "custom",
        "description": "Missing name field",
    })
    assert r.status_code != 500, (
        f"Agent creation without name returned 500: {r.text[:300]}"
    )


def test_create_agent_invalid_type(client):
    """POST /api/agents/ with invalid agent_type should fail gracefully."""
    r = client.post("/api/agents/", json={
        "name": uid("err-agent"),
        "agent_type": "INVALID_TYPE_THAT_DOES_NOT_EXIST",
    })
    # Could be 400 (validation) or 200 (platform accepts custom types) — just not 500
    assert r.status_code != 500, (
        f"Invalid agent_type returned 500: {r.text[:300]}"
    )


# ── UPDATE nonexistent ───────────────────────────────────────────────

def test_update_nonexistent_agent(client):
    """PUT /api/agents/{id} for non-existent agent should return 404."""
    r = client.put("/api/agents/999999", json={"description": "ghost"})
    assert r.status_code in (404, 400), (
        f"Update non-existent agent returned {r.status_code}"
    )


# ── DELETE nonexistent ───────────────────────────────────────────────

def test_delete_nonexistent_agent(client):
    """DELETE /api/agents/{id} for non-existent agent should return 404 or 204."""
    r = client.delete("/api/agents/999999")
    assert r.status_code in (200, 204, 404), (
        f"Delete non-existent agent returned {r.status_code}"
    )


# ── Sub-resource errors ──────────────────────────────────────────────

def test_agent_status_nonexistent(client):
    """GET /api/agents/{id}/status for non-existent agent."""
    r = client.get("/api/agents/999999/status")
    assert r.status_code != 500, (
        f"Status for non-existent agent returned 500: {r.text[:300]}"
    )


def test_agent_performance_nonexistent(client):
    """GET /api/agents/{id}/performance for non-existent agent."""
    r = client.get("/api/agents/999999/performance", params={"period": "all"})
    assert r.status_code != 500, (
        f"Performance for non-existent agent returned 500: {r.text[:300]}"
    )


def test_agent_logs_nonexistent(client):
    """GET /api/agents/{id}/logs for non-existent agent."""
    r = client.get("/api/agents/999999/logs", params={"limit": 10})
    assert r.status_code != 500, (
        f"Logs for non-existent agent returned 500: {r.text[:300]}"
    )


def test_agent_model_config_nonexistent(client):
    """GET /api/agents/{id}/model-config for non-existent agent."""
    r = client.get("/api/agents/999999/model-config")
    assert r.status_code != 500, (
        f"Model config for non-existent agent returned 500: {r.text[:300]}"
    )


def test_assign_persona_to_nonexistent_agent(client):
    """PUT /api/agents/{id}/persona for non-existent agent."""
    r = client.put("/api/agents/999999/persona", json={"persona_id": 1})
    assert r.status_code != 500, (
        f"Assign persona to non-existent agent returned 500: {r.text[:300]}"
    )
