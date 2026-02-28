"""Journey 02: Chatbot conversation — SSE streaming, history, CRUD."""

import pytest
from .helpers import pick, uid, CHAT_MESSAGES, CHAT_FOLLOWUPS, parse_sse_response


@pytest.fixture(scope="module")
def chat_state():
    """Shared state across chat tests in this module."""
    return {"chat_id": None, "message_id": None}


def test_chat_stream_first_message(client, chat_state):
    """POST /api/chat — SSE stream, capture chatId."""
    body = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": pick(CHAT_MESSAGES)}],
        }
    }
    r = client.post("/api/chat", json=body, timeout=60.0)
    assert r.status_code == 200
    parsed = parse_sse_response(r)
    if parsed["chat_id"]:
        chat_state["chat_id"] = parsed["chat_id"]


def test_chat_stream_followup(client, chat_state):
    """POST /api/chat — second message with followup."""
    body = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": pick(CHAT_FOLLOWUPS)}],
        }
    }
    if chat_state["chat_id"]:
        body["chatId"] = chat_state["chat_id"]
    r = client.post("/api/chat", json=body, timeout=60.0)
    assert r.status_code == 200


def test_chat_history(client):
    """GET /api/chat/history — returns list."""
    r = client.get("/api/chat/history", params={"limit": 10})
    assert r.status_code == 200


def test_get_chat(client, chat_state):
    """GET /api/chat/{id} — chat details."""
    if not chat_state["chat_id"]:
        pytest.skip("No chatId captured")
    r = client.get(f"/api/chat/{chat_state['chat_id']}")
    assert r.status_code == 200
    assert "id" in r.json()


def test_chat_messages(client, chat_state):
    """GET /api/chat/{id}/messages — message array."""
    if not chat_state["chat_id"]:
        pytest.skip("No chatId captured")
    r = client.get(f"/api/chat/{chat_state['chat_id']}/messages")
    assert r.status_code == 200
    data = r.json()
    if isinstance(data, list) and data:
        chat_state["message_id"] = data[0].get("id")


def test_update_chat_title(client, chat_state):
    """PATCH /api/chat/{id} — update title."""
    if not chat_state["chat_id"]:
        pytest.skip("No chatId captured")
    r = client.patch(
        f"/api/chat/{chat_state['chat_id']}",
        json={"title": f"API Test {uid('chat')}"},
    )
    assert r.status_code == 200


def test_agent_statistics(client):
    """GET /api/system/agent-statistics — active_agents count."""
    r = client.get("/api/system/agent-statistics")
    assert r.status_code == 200
    assert "active_agents" in r.json()


def test_delete_chat(client, chat_state):
    """DELETE /api/chat/{id} — cleanup."""
    if not chat_state["chat_id"]:
        pytest.skip("No chatId captured")
    r = client.delete(f"/api/chat/{chat_state['chat_id']}")
    assert r.status_code == 200
