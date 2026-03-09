"""Expanded user journeys — stateful flows across agents, chat, workflows, and heartbeats."""

import pytest

from .helpers import pick, uid, CHAT_MESSAGES


@pytest.fixture(scope="module")
def journey_state():
    return {
        "agent_id": None,
        "chat_id": None,
        "workflow_id": None,
        "execution_id": None,
    }


def test_journey_agent_model_config_round_trip(client, created_agent_ids, journey_state):
    """Journey: create agent -> inspect model config -> update config -> verify persisted."""
    payload = {
        "name": uid("journey-agent"),
        "agent_type": "custom",
        "description": "Stateful journey test agent",
        "configuration": {
            "model": "openai/gpt-4o-mini",
            "system_prompt": "You are a journey test agent.",
        },
    }
    create = client.post("/api/agents/", json=payload)
    assert create.status_code in (200, 201)
    created = create.json()
    agent_id = created["id"]
    created_agent_ids.append(agent_id)
    journey_state["agent_id"] = agent_id

    current = client.get(f"/api/agents/{agent_id}/model-config")
    assert current.status_code == 200
    current_data = current.json()
    current_cfg = current_data.get("model_config", {}) or {}

    update_payload = {
        "provider": current_cfg.get("provider") or "openai",
        "model_id": current_cfg.get("model_id") or "openai/gpt-4o-mini",
        "temperature": 0.2,
        "max_tokens": current_cfg.get("max_tokens") or 2000,
        "top_p": current_cfg.get("top_p", 1.0),
        "frequency_penalty": current_cfg.get("frequency_penalty", 0.0),
        "presence_penalty": current_cfg.get("presence_penalty", 0.0),
        "fallback_model_id": current_cfg.get("fallback_model_id"),
    }
    updated = client.put(f"/api/agents/{agent_id}/model-config", json=update_payload)
    assert updated.status_code == 200, updated.text[:500]
    updated_data = updated.json()
    assert updated_data["status"] == "success"
    assert float(updated_data["model_config"]["temperature"]) == 0.2

    reloaded = client.get(f"/api/agents/{agent_id}/model-config")
    assert reloaded.status_code == 200
    reloaded_cfg = reloaded.json().get("model_config", {})
    assert float(reloaded_cfg["temperature"]) == 0.2


def test_journey_agent_execute_returns_handle(client, first_agent_id, journey_state):
    """Journey: execute an active agent and verify execution handle metadata."""
    agent_id = journey_state["agent_id"] or first_agent_id
    if not agent_id:
        pytest.skip("No agent available for execute journey")

    r = client.post(
        f"/api/agents/{agent_id}/execute",
        json={"task": "Summarize latest workspace state", "mode": "test"},
    )
    assert r.status_code == 200, r.text[:500]
    data = r.json()
    assert data["agent_id"] == agent_id
    assert data["status"] == "started"
    assert "execution_id" in data
    assert data["execution_id"]


def test_journey_chat_title_round_trip(client, journey_state):
    """Journey: create chat -> rename chat -> fetch chat and confirm title persists."""
    first = client.post(
        "/api/chat",
        json={
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": pick(CHAT_MESSAGES)}],
            }
        },
        timeout=60.0,
    )
    assert first.status_code == 200

    from .helpers import parse_sse_response

    parsed = parse_sse_response(first)
    if not parsed["chat_id"]:
        pytest.skip("Chat stream did not emit chat_id")

    chat_id = parsed["chat_id"]
    journey_state["chat_id"] = chat_id
    new_title = uid("journey-chat")

    updated = client.patch(f"/api/chat/{chat_id}", json={"title": new_title})
    assert updated.status_code == 200

    fetched = client.get(f"/api/chat/{chat_id}")
    assert fetched.status_code == 200
    chat = fetched.json()
    assert chat.get("title") == new_title

    deleted = client.delete(f"/api/chat/{chat_id}")
    assert deleted.status_code == 200


def test_journey_workflow_execute_and_status(client, journey_state):
    """Journey: pick an existing workflow -> execute it -> verify status endpoint works."""
    listed = client.get("/api/workflows", params={"limit": 5})
    assert listed.status_code == 200
    data = listed.json()
    items = data.get("items", data if isinstance(data, list) else [])
    if not items:
        pytest.skip("No workflows available in workspace")

    workflow_id = items[0]["id"]
    journey_state["workflow_id"] = workflow_id

    started = client.post(
        f"/api/workflows/{workflow_id}/execute",
        json={"input_data": {"source": "nightly-journey", "mode": "smoke"}},
    )
    assert started.status_code == 200, started.text[:500]
    execution = started.json()
    execution_id = execution.get("execution_id") or execution.get("id")
    assert execution_id
    journey_state["execution_id"] = execution_id

    status = client.get(f"/api/workflows/executions/{execution_id}")
    assert status.status_code == 200, status.text[:500]
    status_data = status.json()
    assert status_data.get("id") == execution_id or status_data.get("execution_id") == execution_id
    assert "status" in status_data


def test_journey_heartbeat_agent_response_shape(client, first_agent_id, journey_state):
    """Journey: run agent heartbeat and verify response shape is actionable for monitoring recipes."""
    agent_id = journey_state["agent_id"] or first_agent_id
    if not agent_id:
        pytest.skip("No agent available for heartbeat journey")

    r = client.post(f"/api/heartbeat/agents/{agent_id}/run")
    assert r.status_code == 200, r.text[:500]
    data = r.json()

    assert isinstance(data, dict)
    assert any(key in data for key in ("status", "message", "result", "agent_id")), (
        f"Heartbeat response shape too thin for monitoring workflow: {data}"
    )
