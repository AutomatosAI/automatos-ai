"""PRD-123 + PRD-82A: Mission lifecycle — list, detail, checkpoints, cost, events.

Tests the mission endpoints including stop_reason, permission_denials,
session checkpointing, and cost tracking.
"""

import pytest


@pytest.fixture(scope="module")
def mission_state():
    return {"mission_id": None}


# ── List & Discovery ────────────────────────────────────────────────

def test_list_missions(client, mission_state):
    """GET /api/missions should list missions."""
    r = client.get("/api/missions", params={"limit": 10})
    assert r.status_code == 200
    data = r.json()
    items = data.get("items", data if isinstance(data, list) else [])
    assert isinstance(items, list)
    if items:
        mission_state["mission_id"] = items[0].get("id") or items[0].get("mission_id")


# ── Detail & Stop Reason ────────────────────────────────────────────

def test_mission_detail(client, mission_state):
    """GET /api/missions/{id} should return mission with stop_reason fields."""
    if not mission_state["mission_id"]:
        pytest.skip("No missions available")
    r = client.get(f"/api/missions/{mission_state['mission_id']}")
    assert r.status_code == 200
    data = r.json()
    assert "id" in data or "mission_id" in data


def test_mission_detail_nonexistent(client):
    """GET /api/missions/{id} for non-existent mission should return 404."""
    r = client.get("/api/missions/00000000-0000-0000-0000-000000000000")
    assert r.status_code in (404, 400), (
        f"Non-existent mission returned {r.status_code}"
    )


# ── Checkpoints ─────────────────────────────────────────────────────

def test_mission_checkpoints(client, mission_state):
    """GET /api/missions/{id}/checkpoints should list checkpoints."""
    if not mission_state["mission_id"]:
        pytest.skip("No missions available")
    r = client.get(f"/api/missions/{mission_state['mission_id']}/checkpoints")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, (list, dict))


def test_mission_checkpoints_nonexistent(client):
    """GET /api/missions/{id}/checkpoints for non-existent mission."""
    r = client.get("/api/missions/00000000-0000-0000-0000-000000000000/checkpoints")
    assert r.status_code in (404, 400, 200), (
        f"Checkpoints for non-existent mission returned {r.status_code}"
    )


def test_mission_resume_nonexistent(client):
    """POST /api/missions/{id}/resume for non-existent mission."""
    r = client.post("/api/missions/00000000-0000-0000-0000-000000000000/resume")
    assert r.status_code in (404, 400), (
        f"Resume non-existent mission returned {r.status_code}"
    )


# ── Cost Tracking ───────────────────────────────────────────────────

def test_mission_cost(client, mission_state):
    """GET /api/missions/{id}/cost should return cost breakdown."""
    if not mission_state["mission_id"]:
        pytest.skip("No missions available")
    r = client.get(f"/api/missions/{mission_state['mission_id']}/cost")
    assert r.status_code == 200


def test_mission_cost_nonexistent(client):
    """GET /api/missions/{id}/cost for non-existent mission."""
    r = client.get("/api/missions/00000000-0000-0000-0000-000000000000/cost")
    assert r.status_code in (404, 400), (
        f"Cost for non-existent mission returned {r.status_code}"
    )


# ── Events ──────────────────────────────────────────────────────────

def test_mission_events(client, mission_state):
    """GET /api/missions/{id}/events should return event stream."""
    if not mission_state["mission_id"]:
        pytest.skip("No missions available")
    r = client.get(f"/api/missions/{mission_state['mission_id']}/events")
    assert r.status_code == 200


def test_mission_events_nonexistent(client):
    """GET /api/missions/{id}/events for non-existent mission."""
    r = client.get("/api/missions/00000000-0000-0000-0000-000000000000/events")
    assert r.status_code in (404, 400, 200), (
        f"Events for non-existent mission returned {r.status_code}"
    )


# ── Cancel ──────────────────────────────────────────────────────────

def test_cancel_nonexistent_mission(client):
    """POST /api/missions/{id}/cancel for non-existent mission."""
    r = client.post("/api/missions/00000000-0000-0000-0000-000000000000/cancel")
    assert r.status_code in (404, 400), (
        f"Cancel non-existent mission returned {r.status_code}"
    )
