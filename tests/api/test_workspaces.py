"""Journey 01: Workspace setup + file/exec endpoints."""

import pytest


def test_current_workspace(client):
    r = client.get("/api/workspaces/current")
    assert r.status_code == 200
    data = r.json()
    assert "id" in data or "workspace" in data


def test_workspace_integrations(client):
    r = client.get("/api/workspaces/current/integrations")
    assert r.status_code == 200


def test_workspace_exec(client, workspace_id):
    """POST exec — 200 if worker is up, 404/503 if offline or unprovisioned."""
    r = client.post(
        f"/api/workspaces/{workspace_id}/exec",
        json={"command": "echo nightly-test-ok", "timeout": 10},
    )
    assert r.status_code in (200, 404, 503)
    if r.status_code == 200:
        assert "nightly-test-ok" in r.text


def test_workspace_files(client, workspace_id):
    """GET files — 200 if worker is up, 404/503 if offline or unprovisioned."""
    r = client.get(f"/api/workspaces/{workspace_id}/files")
    assert r.status_code in (200, 404, 503)


def test_workspace_file_content(client, workspace_id):
    """GET file content — 200 or 503 depending on worker."""
    r = client.get(
        f"/api/workspaces/{workspace_id}/files/content",
        params={"path": "requirements.txt"},
    )
    assert r.status_code in (200, 404, 503)
