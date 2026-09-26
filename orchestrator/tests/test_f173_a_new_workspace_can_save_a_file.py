"""F173 (night 6) — a brand-new workspace can save a file.

A workspace made by the signup wizard has no directory on the workspace
worker's volume (the backend's PRD-130 note: no worker container is provisioned
for it). Only the file listing and the task runner provisioned one, so the
worker's write, exec, grep, git, html-to-png and download routes answered 404
"Workspace not found": night 6's report writes failed with it (02:09:45 to
02:16:01Z; task #1093's auto-report was lost). Every route now opens the
workspace through one helper that provisions it on first use, and only a
canonical UUID may name a workspace directory.

The worker's own HTTP server runs on a free port over a temporary volume, and
its task runner over a recorded Redis. No database.
"""
import uuid

import pytest

from tests.helpers_workspace_worker import answer, task_runner, worker_server


@pytest.mark.asyncio
async def test_a_new_workspaces_first_file_is_saved(monkeypatch, tmp_path):
    ws = str(uuid.uuid4())
    async with worker_server(monkeypatch, tmp_path) as (base, http):
        status, body = await answer(await http.post(
            f"{base}/workspaces/{ws}/files/write", json={"path": "reports/first-report.md", "content": "# Q3"}))

    assert status == 200, body
    assert (tmp_path / ws / "reports" / "first-report.md").read_text() == "# Q3"
    assert (tmp_path / ws / ".workspace_meta.json").is_file()  # provisioned like a listed one


@pytest.mark.asyncio
async def test_a_new_workspace_can_run_search_and_fetch(monkeypatch, tmp_path):
    ws = str(uuid.uuid4())
    async with worker_server(monkeypatch, tmp_path) as (base, http):
        ran = await answer(await http.post(f"{base}/workspaces/{ws}/exec", json={"command": "echo saved"}))
        searched = await answer(await http.get(f"{base}/workspaces/{ws}/files/grep", params={"pattern": "Q3"}))
        fetched = await answer(await http.get(f"{base}/workspaces/{ws}/files/download", params={"path": "none.md"}))

    assert ran[0] == 200 and "saved" in ran[1].get("stdout", ""), ran
    assert searched[0] == 200 and searched[1]["matches"] == [], searched
    assert fetched == (404, {"error": "File not found"})  # the file, not the workspace, is missing


@pytest.mark.parametrize("not_a_workspace", ["not-a-uuid", str(uuid.uuid4()).upper(), uuid.uuid4().hex])
@pytest.mark.asyncio
async def test_only_a_canonical_uuid_names_a_workspace_directory(monkeypatch, tmp_path, not_a_workspace):
    async with worker_server(monkeypatch, tmp_path) as (base, http):
        written = await answer(await http.post(
            f"{base}/workspaces/{not_a_workspace}/files/write", json={"path": "x.md", "content": "x"}))
        listed = await answer(await http.get(f"{base}/workspaces/{not_a_workspace}/files"))

    assert written == (400, {"error": "Invalid workspace id"})
    assert listed == (400, {"error": "Invalid workspace id"})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_a_queued_task_for_a_non_uuid_workspace_makes_nothing(monkeypatch, tmp_path):
    """Security review: the task runner also names a directory from its id."""
    volume = tmp_path / "volume"
    volume.mkdir()
    worker = task_runner(monkeypatch, volume)

    await worker._execute_task({"task_id": "task-f173-dotdot", "workspace_id": ".."})

    assert [p.name for p in tmp_path.iterdir()] == ["volume"]  # nothing beside the volume
    assert list(volume.iterdir()) == []
    last = worker._redis.statuses[-1]
    assert (last["status"], last["error"]) == ("failed", "Invalid workspace id")


@pytest.mark.asyncio
async def test_a_download_never_leaves_its_workspace(monkeypatch, tmp_path):
    ws = str(uuid.uuid4())
    (tmp_path / ws).mkdir()
    (tmp_path / "outside.txt").write_text("another workspace's")
    async with worker_server(monkeypatch, tmp_path) as (base, http):
        climbed = await answer(await http.get(
            f"{base}/workspaces/{ws}/files/download", params={"path": "../outside.txt"}))
        rooted = await answer(await http.get(
            f"{base}/workspaces/{ws}/files/download", params={"path": str(tmp_path / "outside.txt")}))

    assert climbed == (403, {"error": "Path traversal denied"})
    assert rooted == (403, {"error": "Path traversal denied"})
