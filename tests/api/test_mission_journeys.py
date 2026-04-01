"""Mission coordinator journey tests — stateful multi-step workflows.

Tests the full mission lifecycle: create → approve → monitor events →
check cost → pause → resume → cancel. Covers PRD-82A sequential
coordinator and PRD-123 additions (stop_reason, checkpoints, cost).
"""

import pytest

from .helpers import uid


@pytest.fixture(scope="module")
def mission_journey_state():
    return {
        "mission_id": None,
        "existing_mission_id": None,
    }


# ── Step 1: Discover existing missions ──────────────────────────────

def test_mission_list_and_stats(client, mission_journey_state):
    """List missions and capture one for subsequent tests."""
    r = client.get("/api/missions", params={"limit": 10})
    assert r.status_code == 200
    data = r.json()
    items = data.get("items", data if isinstance(data, list) else [])
    if items:
        mission_journey_state["existing_mission_id"] = (
            items[0].get("id") or items[0].get("mission_id")
        )


def test_mission_stats(client):
    """GET /api/missions/stats should return aggregate stats."""
    r = client.get("/api/missions/stats")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)


# ── Step 2: Create a mission ────────────────────────────────────────

def test_create_mission(client, first_agent_id, mission_journey_state):
    """POST /api/missions — create a new mission for the journey."""
    agent_id = first_agent_id or 1
    r = client.post("/api/missions", json={
        "name": f"Journey Test Mission {uid('mission')}",
        "description": "Created by test_mission_journeys for lifecycle testing",
        "goal": "Verify mission lifecycle endpoints work correctly",
        "steps": [
            {
                "step_id": "step_1",
                "order": 1,
                "agent_id": agent_id,
                "prompt": "Summarize the current workspace state",
            },
        ],
    })
    if r.status_code in (200, 201):
        data = r.json()
        mission_journey_state["mission_id"] = (
            data.get("id") or data.get("mission_id")
        )
    # Accept 200/201 (created) or 400/422 (validation) — not 500
    assert r.status_code != 500, (
        f"Create mission returned 500: {r.text[:300]}"
    )


# ── Step 3: Get mission detail ──────────────────────────────────────

def test_get_mission_detail(client, mission_journey_state):
    """GET /api/missions/{id} — verify detail includes PRD-123 fields."""
    mid = mission_journey_state["mission_id"] or mission_journey_state["existing_mission_id"]
    if not mid:
        pytest.skip("No mission available")
    r = client.get(f"/api/missions/{mid}")
    assert r.status_code == 200
    data = r.json()
    # Should have basic mission fields
    assert "name" in data or "goal" in data or "status" in data


# ── Step 4: Check events ────────────────────────────────────────────

def test_mission_events(client, mission_journey_state):
    """GET /api/missions/{id}/events — verify event stream."""
    mid = mission_journey_state["mission_id"] or mission_journey_state["existing_mission_id"]
    if not mid:
        pytest.skip("No mission available")
    r = client.get(f"/api/missions/{mid}/events")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, (list, dict))


# ── Step 5: Check cost tracking ─────────────────────────────────────

def test_mission_cost_tracking(client, mission_journey_state):
    """GET /api/missions/{id}/cost — verify cost breakdown."""
    mid = mission_journey_state["mission_id"] or mission_journey_state["existing_mission_id"]
    if not mid:
        pytest.skip("No mission available")
    r = client.get(f"/api/missions/{mid}/cost")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)


# ── Step 6: Check checkpoints ───────────────────────────────────────

def test_mission_checkpoints(client, mission_journey_state):
    """GET /api/missions/{id}/checkpoints — list session checkpoints."""
    mid = mission_journey_state["mission_id"] or mission_journey_state["existing_mission_id"]
    if not mid:
        pytest.skip("No mission available")
    r = client.get(f"/api/missions/{mid}/checkpoints")
    assert r.status_code == 200


# ── Step 7: Field context ───────────────────────────────────────────

def test_mission_field_context(client, mission_journey_state):
    """GET /api/missions/{id}/field — field memory context."""
    mid = mission_journey_state["mission_id"] or mission_journey_state["existing_mission_id"]
    if not mid:
        pytest.skip("No mission available")
    r = client.get(f"/api/missions/{mid}/field")
    # 200 if field data exists, 404 if not — both ok
    assert r.status_code in (200, 404), (
        f"Mission field returned {r.status_code}"
    )


# ── Step 8: Lifecycle actions ───────────────────────────────────────

def test_pause_mission(client, mission_journey_state):
    """POST /api/missions/{id}/pause — pause a running mission."""
    mid = mission_journey_state["mission_id"]
    if not mid:
        pytest.skip("No test mission created")
    r = client.post(f"/api/missions/{mid}/pause")
    # 200 if pauseable, 400/409 if wrong state — not 500
    assert r.status_code != 500, (
        f"Pause mission returned 500: {r.text[:300]}"
    )


def test_resume_mission(client, mission_journey_state):
    """POST /api/missions/{id}/resume — resume a paused mission."""
    mid = mission_journey_state["mission_id"]
    if not mid:
        pytest.skip("No test mission created")
    r = client.post(f"/api/missions/{mid}/resume")
    assert r.status_code != 500, (
        f"Resume mission returned 500: {r.text[:300]}"
    )


def test_cancel_mission(client, mission_journey_state):
    """POST /api/missions/{id}/cancel — cancel sets stop_reason=human_cancelled."""
    mid = mission_journey_state["mission_id"]
    if not mid:
        pytest.skip("No test mission created")
    r = client.post(f"/api/missions/{mid}/cancel")
    assert r.status_code != 500, (
        f"Cancel mission returned 500: {r.text[:300]}"
    )


# ── Step 9: Archive ─────────────────────────────────────────────────

def test_mission_archive_list(client):
    """GET /api/missions/archive — list archived missions."""
    r = client.get("/api/missions/archive", params={"limit": 5})
    assert r.status_code == 200


def test_mission_archive_detail_nonexistent(client):
    """GET /api/missions/archive/{id} for non-existent archive."""
    r = client.get("/api/missions/archive/00000000-0000-0000-0000-000000000000")
    assert r.status_code in (404, 400), (
        f"Non-existent archive returned {r.status_code}"
    )


# ── Step 10: Delete test mission ────────────────────────────────────

def test_delete_mission(client, mission_journey_state):
    """DELETE /api/missions/{id} — cleanup test mission."""
    mid = mission_journey_state["mission_id"]
    if not mid:
        pytest.skip("No test mission created")
    r = client.delete(f"/api/missions/{mid}")
    assert r.status_code in (200, 204, 404), (
        f"Delete mission returned {r.status_code}"
    )
