"""PRD-123: Tool tier stratification & permission management.

Tests the permission matrix, tool assignments, and audit endpoints
added by the harness pattern adoption work.
"""

import pytest

from .helpers import uid


def test_permission_matrix(client):
    """GET /api/permissions/matrix should return full permission state."""
    r = client.get("/api/permissions/matrix")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, (dict, list))


def test_permission_assignments(client):
    """GET /api/permissions/assignments should list tool assignments."""
    r = client.get("/api/permissions/assignments")
    assert r.status_code == 200


def test_permission_health(client):
    """GET /api/permissions/health should return health check."""
    r = client.get("/api/permissions/health")
    assert r.status_code == 200


def test_permission_audit(client):
    """GET /api/permissions/audit should return audit log."""
    r = client.get("/api/permissions/audit")
    assert r.status_code == 200


def test_assign_tool_to_agent(client, first_agent_id):
    """POST /api/permissions/assign should assign a tool to an agent."""
    if not first_agent_id:
        pytest.skip("No agent available")
    r = client.post("/api/permissions/assign", json={
        "agent_id": first_agent_id,
        "tool_name": "platform_list_agents",
        "tier": "platform",
    })
    # 200 if it works, 400/404 if tool doesn't exist — not 500
    assert r.status_code != 500, (
        f"Permission assign returned 500: {r.text[:300]}"
    )


def test_assign_tool_empty_body(client):
    """POST /api/permissions/assign with empty body should not 500."""
    r = client.post("/api/permissions/assign", json={})
    assert r.status_code in (400, 422), (
        f"Empty permission assign returned {r.status_code}"
    )


def test_revoke_nonexistent_permission(client):
    """DELETE /api/permissions/revoke for non-existent permission."""
    r = client.request("DELETE", "/api/permissions/revoke", json={
        "agent_id": 999999,
        "tool_name": "fake_tool",
    })
    assert r.status_code != 500, (
        f"Revoke nonexistent permission returned 500: {r.text[:300]}"
    )


def test_bulk_assign_empty_list(client):
    """POST /api/permissions/bulk-assign with empty assignments."""
    r = client.post("/api/permissions/bulk-assign", json={
        "assignments": [],
    })
    assert r.status_code != 500, (
        f"Empty bulk assign returned 500: {r.text[:300]}"
    )
