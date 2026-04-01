"""User Journey: Mission-driven research workflow.

Simulates a user who creates a research mission, monitors progress,
checks costs, reviews results, and archives completed work.

Flow:
  1. List agents → pick a research-capable agent
  2. Create mission → define research goal with steps
  3. Check mission detail → verify structure
  4. Monitor events → track progress
  5. Check cost → budget awareness
  6. Review checkpoints → session state
  7. Check mission field context → knowledge accumulation
  8. Cancel/complete → lifecycle end
  9. Check mission stats → aggregate view
  10. Archive check → verify archival
"""

import pytest

from .helpers import uid


@pytest.fixture(scope="module")
def research_state():
    return {
        "agent_id": None,
        "mission_id": None,
    }


def test_research_01_pick_agent(client, first_agent_id, research_state):
    """Step 1: Identify an agent for research."""
    research_state["agent_id"] = first_agent_id
    if not first_agent_id:
        # Try listing
        r = client.get("/api/agents/")
        assert r.status_code == 200
        agents = r.json()
        if agents:
            research_state["agent_id"] = agents[0]["id"]


def test_research_02_create_mission(client, research_state):
    """Step 2: Create a research mission with clear goal and steps."""
    agent_id = research_state["agent_id"]
    if not agent_id:
        pytest.skip("No agent available")

    r = client.post("/api/missions", json={
        "name": f"Research Mission {uid('research')}",
        "description": "Automated research journey test",
        "goal": "Investigate and summarize the current state of the workspace",
        "steps": [
            {
                "step_id": "step_1",
                "order": 1,
                "agent_id": agent_id,
                "prompt": "List all available agents and their capabilities",
            },
            {
                "step_id": "step_2",
                "order": 2,
                "agent_id": agent_id,
                "prompt": "Summarize findings from step 1",
            },
        ],
    })
    if r.status_code in (200, 201):
        data = r.json()
        research_state["mission_id"] = data.get("id") or data.get("mission_id")
    assert r.status_code != 500, (
        f"Mission creation returned 500: {r.text[:300]}"
    )


def test_research_03_verify_mission_detail(client, research_state):
    """Step 3: Check mission detail has correct structure."""
    if not research_state["mission_id"]:
        pytest.skip("No mission created")
    r = client.get(f"/api/missions/{research_state['mission_id']}")
    assert r.status_code == 200
    data = r.json()
    assert "name" in data or "goal" in data or "status" in data


def test_research_04_monitor_events(client, research_state):
    """Step 4: Check mission events for progress tracking."""
    if not research_state["mission_id"]:
        pytest.skip("No mission created")
    r = client.get(f"/api/missions/{research_state['mission_id']}/events")
    assert r.status_code == 200


def test_research_05_check_cost(client, research_state):
    """Step 5: Review mission cost for budget awareness."""
    if not research_state["mission_id"]:
        pytest.skip("No mission created")
    r = client.get(f"/api/missions/{research_state['mission_id']}/cost")
    assert r.status_code == 200


def test_research_06_check_checkpoints(client, research_state):
    """Step 6: Review session checkpoints."""
    if not research_state["mission_id"]:
        pytest.skip("No mission created")
    r = client.get(f"/api/missions/{research_state['mission_id']}/checkpoints")
    assert r.status_code == 200


def test_research_07_field_context(client, research_state):
    """Step 7: Check field context — knowledge accumulated during mission."""
    if not research_state["mission_id"]:
        pytest.skip("No mission created")
    r = client.get(f"/api/missions/{research_state['mission_id']}/field")
    assert r.status_code in (200, 404)


def test_research_08_cancel_mission(client, research_state):
    """Step 8: Cancel the test mission."""
    if not research_state["mission_id"]:
        pytest.skip("No mission created")
    r = client.post(f"/api/missions/{research_state['mission_id']}/cancel")
    assert r.status_code != 500


def test_research_09_check_stats(client):
    """Step 9: Review aggregate mission stats."""
    r = client.get("/api/missions/stats")
    assert r.status_code == 200


def test_research_10_check_archive(client):
    """Step 10: Browse archived missions."""
    r = client.get("/api/missions/archive", params={"limit": 5})
    assert r.status_code == 200


def test_research_11_cleanup(client, research_state):
    """Step 11: Delete test mission if it still exists."""
    if not research_state["mission_id"]:
        return
    r = client.delete(f"/api/missions/{research_state['mission_id']}")
    assert r.status_code in (200, 204, 404)
