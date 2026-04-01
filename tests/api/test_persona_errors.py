"""Persona error handling — validates graceful failures for edge cases."""


def test_get_nonexistent_persona(client):
    """GET /api/personas/{id} for non-existent persona."""
    r = client.get("/api/personas/999999")
    assert r.status_code in (404, 400), (
        f"Non-existent persona returned {r.status_code}"
    )


def test_create_persona_empty_body(client, workspace_id):
    """POST persona with empty body should not 500."""
    r = client.post(f"/api/workspaces/{workspace_id}/personas", json={})
    assert r.status_code in (400, 422), (
        f"Empty persona body returned {r.status_code}, expected 400/422"
    )


def test_create_persona_missing_name(client, workspace_id):
    """POST persona without name should fail gracefully."""
    r = client.post(f"/api/workspaces/{workspace_id}/personas", json={
        "description": "Missing name",
        "system_prompt": "You are a test.",
    })
    assert r.status_code != 500, (
        f"Persona without name returned 500: {r.text[:300]}"
    )


def test_delete_nonexistent_persona(client, workspace_id):
    """DELETE persona that doesn't exist."""
    r = client.delete(f"/api/workspaces/{workspace_id}/personas/999999")
    assert r.status_code in (200, 204, 404), (
        f"Delete non-existent persona returned {r.status_code}"
    )


def test_update_nonexistent_persona(client, workspace_id):
    """PUT persona that doesn't exist."""
    r = client.put(f"/api/workspaces/{workspace_id}/personas/999999", json={
        "description": "ghost",
    })
    assert r.status_code in (404, 400), (
        f"Update non-existent persona returned {r.status_code}"
    )
