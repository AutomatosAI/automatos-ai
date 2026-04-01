"""Workspace error handling — validates graceful failures for edge cases."""


def test_workspace_exec_empty_command(client, workspace_id):
    """POST exec with empty command should not 500."""
    r = client.post(
        f"/api/workspaces/{workspace_id}/exec",
        json={"command": "", "timeout": 10},
    )
    # 400 or 503 (worker offline) — not 500
    assert r.status_code != 500, (
        f"Empty exec command returned 500: {r.text[:300]}"
    )


def test_workspace_exec_invalid_workspace(client):
    """POST exec to invalid workspace ID should fail gracefully."""
    r = client.post(
        "/api/workspaces/00000000-0000-0000-0000-000000000000/exec",
        json={"command": "echo test", "timeout": 10},
    )
    assert r.status_code in (400, 403, 404, 503), (
        f"Exec on invalid workspace returned {r.status_code}"
    )


def test_workspace_files_invalid_workspace(client):
    """GET files for invalid workspace should fail gracefully."""
    r = client.get("/api/workspaces/00000000-0000-0000-0000-000000000000/files")
    assert r.status_code in (400, 403, 404, 503), (
        f"Files on invalid workspace returned {r.status_code}"
    )


def test_workspace_file_content_missing_path(client, workspace_id):
    """GET file content without path param should not 500."""
    r = client.get(f"/api/workspaces/{workspace_id}/files/content")
    # 400 (missing param) or 503 (worker offline)
    assert r.status_code != 500, (
        f"File content without path returned 500: {r.text[:300]}"
    )


def test_workspace_file_content_nonexistent_path(client, workspace_id):
    """GET file content for non-existent path should return 404."""
    r = client.get(
        f"/api/workspaces/{workspace_id}/files/content",
        params={"path": "/does/not/exist/file.txt"},
    )
    assert r.status_code in (404, 400, 503), (
        f"Non-existent file path returned {r.status_code}"
    )
