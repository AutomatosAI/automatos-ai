"""Journey 04: Persona lifecycle — CRUD + agent assignment."""

import pytest
from .helpers import pick, uid, PERSONA_NAMES, PERSONA_PROMPTS


@pytest.fixture(scope="module")
def persona_state():
    return {"persona_id": None}


def test_list_personas(client):
    r = client.get("/api/personas")
    assert r.status_code == 200
    data = r.json()
    assert "items" in data or isinstance(data, list)


def test_create_persona(client, workspace_id, persona_state, created_persona_ids):
    name = f"{pick(PERSONA_NAMES)} {uid('persona')}"
    r = client.post(
        f"/api/workspaces/{workspace_id}/personas",
        json={
            "name": name,
            "description": "Created by nightly test suite",
            "system_prompt": pick(PERSONA_PROMPTS),
            "category": "assistant",
        },
    )
    assert r.status_code in (200, 201)
    data = r.json()
    assert "id" in data
    persona_state["persona_id"] = data["id"]
    created_persona_ids.append(data["id"])


def test_get_persona(client, persona_state):
    if not persona_state["persona_id"]:
        pytest.skip("No persona created")
    r = client.get(f"/api/personas/{persona_state['persona_id']}")
    assert r.status_code == 200
    assert "system_prompt" in r.json()


def test_update_persona(client, workspace_id, persona_state):
    if not persona_state["persona_id"]:
        pytest.skip("No persona created")
    r = client.put(
        f"/api/workspaces/{workspace_id}/personas/{persona_state['persona_id']}",
        json={"description": "Updated by nightly test"},
    )
    assert r.status_code == 200


def test_assign_persona_to_agent(client, first_agent_id, persona_state):
    if not first_agent_id or not persona_state["persona_id"]:
        pytest.skip("Need agent + persona")
    r = client.put(
        f"/api/agents/{first_agent_id}/persona",
        json={"persona_id": persona_state["persona_id"]},
    )
    assert r.status_code == 200


def test_get_agent_persona(client, first_agent_id):
    if not first_agent_id:
        pytest.skip("No agent available")
    r = client.get(f"/api/agents/{first_agent_id}/persona")
    assert r.status_code == 200


def test_delete_persona(client, workspace_id, persona_state, created_persona_ids):
    if not persona_state["persona_id"]:
        pytest.skip("No persona created")
    pid = persona_state["persona_id"]
    r = client.delete(f"/api/workspaces/{workspace_id}/personas/{pid}")
    assert r.status_code == 200
    # Remove from cleanup list since we already deleted
    if pid in created_persona_ids:
        created_persona_ids.remove(pid)
