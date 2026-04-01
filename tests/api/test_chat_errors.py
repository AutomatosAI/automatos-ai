"""Chat error handling — validates graceful failures for edge cases.

Pilot users will hit these scenarios. The platform must not 500.
"""

import pytest

from .helpers import uid


def test_chat_with_nonexistent_agent(client):
    """Chat targeting a non-existent agent should fail gracefully."""
    body = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": "Hello"}],
        },
        "agentId": 999999,
    }
    r = client.post("/api/chat", json=body, timeout=30.0)
    # Should be 4xx or a graceful 200 with error in stream — not 500
    assert r.status_code != 500, f"Chat with bad agent returned 500: {r.text[:300]}"


def test_chat_empty_message(client):
    """Chat with empty message text should be handled gracefully."""
    body = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": ""}],
        },
    }
    r = client.post("/api/chat", json=body, timeout=30.0)
    # Either 400 validation error or 200 with some response — not 500
    assert r.status_code != 500, f"Empty message returned 500: {r.text[:300]}"


def test_chat_malformed_body(client):
    """Chat with malformed body should return 422, not 500."""
    r = client.post("/api/chat", json={"bad": "payload"}, timeout=15.0)
    assert r.status_code in (400, 422), (
        f"Malformed chat body returned {r.status_code}, expected 400/422"
    )


def test_chat_get_nonexistent(client):
    """GET /api/chat/{id} for non-existent chat should return 404."""
    r = client.get("/api/chat/00000000-0000-0000-0000-000000000000")
    assert r.status_code in (404, 400), (
        f"Non-existent chat returned {r.status_code}, expected 404"
    )


def test_chat_delete_nonexistent(client):
    """DELETE /api/chat/{id} for non-existent chat should return 404, not 500."""
    r = client.delete("/api/chat/00000000-0000-0000-0000-000000000000")
    assert r.status_code in (200, 204, 404), (
        f"Delete non-existent chat returned {r.status_code}"
    )


def test_chat_update_title_nonexistent(client):
    """PATCH /api/chat/{id} for non-existent chat should return 404."""
    r = client.patch(
        "/api/chat/00000000-0000-0000-0000-000000000000",
        json={"title": "should-not-work"},
    )
    assert r.status_code in (404, 400), (
        f"Patch non-existent chat returned {r.status_code}"
    )


def test_chat_history_with_pagination(client):
    """GET /api/chat/history with pagination params should work."""
    r = client.get("/api/chat/history", params={"limit": 5, "offset": 0})
    assert r.status_code == 200
    data = r.json()
    chats = data if isinstance(data, list) else data.get("chats", data.get("items", []))
    assert isinstance(chats, list)
    assert len(chats) <= 5
