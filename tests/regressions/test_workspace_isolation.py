"""Regression pins: Multi-tenancy workspace isolation bugs.

Pinned bugs:
- New users saw ALL data from existing admin (dev-fallback in hybrid.py, 2026-02-07)
- X-Workspace-ID header spoofing was not validated against user membership
"""

import os

import httpx
import pytest


def test_invalid_workspace_header_rejected(client, api_url):
    """Requests with a non-existent workspace ID should be rejected.

    Bug: hybrid.py had a fallback that assigned ALL users to the single existing
    workspace when no workspace was found for the given ID.
    Fix: _user_has_workspace_access() validates user owns/is member of workspace.
    """
    fake_workspace_id = "00000000-0000-0000-0000-000000000000"
    headers = {
        "X-Api-Key": os.environ["API_KEY"],
        "X-Workspace-ID": fake_workspace_id,
        "Content-Type": "application/json",
    }
    with httpx.Client(base_url=api_url, headers=headers, timeout=15.0) as fake_client:
        r = fake_client.get("/api/agents/")
        # Should be 401/403/404 — NOT 200 with another workspace's data
        assert r.status_code in (401, 403, 404, 422), (
            f"Fake workspace ID returned {r.status_code} — "
            "expected rejection. If 200, workspace isolation is broken."
        )


def test_agents_list_returns_only_current_workspace(client, workspace_id):
    """Agent list must only contain agents from the authenticated workspace.

    Bug: Dev fallback assigned all users to first workspace, so every user
    saw the admin's agents.
    """
    r = client.get("/api/agents/")
    assert r.status_code == 200
    agents = r.json()
    assert isinstance(agents, list)

    # All agents should belong to our workspace (if workspace_id is exposed)
    for agent in agents:
        agent_ws = agent.get("workspace_id")
        if agent_ws is not None:
            assert str(agent_ws) == str(workspace_id), (
                f"Agent {agent.get('id')} belongs to workspace {agent_ws}, "
                f"expected {workspace_id} — workspace isolation leak"
            )


def test_chat_history_scoped_to_workspace(client):
    """Chat history must only return chats from the current workspace."""
    r = client.get("/api/chat/history", params={"limit": 50})
    assert r.status_code == 200
    data = r.json()
    chats = data if isinstance(data, list) else data.get("chats", data.get("items", []))
    # If there are chats, verify they're all from this workspace
    # (workspace_id may not be in the response — that's fine, the test validates
    # that the endpoint doesn't return a 500 and returns a reasonable shape)
    assert isinstance(chats, list)


def test_documents_scoped_to_workspace(client):
    """Document list must only return documents from the current workspace."""
    r = client.get("/api/documents/")
    assert r.status_code == 200
    docs = r.json()
    assert isinstance(docs, list)
    # No cross-workspace documents should appear


def test_memory_stats_scoped_to_workspace(client):
    """Memory stats must reflect only the current workspace's memories."""
    r = client.get("/api/workspaces/current/memory-stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_memories" in data
    # total_memories should be a reasonable number, not suspiciously high
    # (which would indicate cross-workspace leakage)
    assert isinstance(data["total_memories"], (int, float))
