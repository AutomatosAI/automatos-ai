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
import asyncio
import contextlib
import importlib.util
import json
import socket
import sys
import types
import uuid
from pathlib import Path

import aiohttp
import pytest

_WORKER_DIR = Path(__file__).resolve().parents[2] / "services" / "workspace-worker"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _load_worker_main(monkeypatch, volume):
    """The worker's main module over ``volume`` (its logging and metrics stubbed)."""
    monkeypatch.setattr(sys, "path", [str(_WORKER_DIR), *sys.path])  # main.py edits sys.path too
    monkeypatch.setitem(sys.modules, "automatos_logging", types.SimpleNamespace(setup_logging=lambda **_: None))
    # Prometheus metrics register once per process; a second server would collide.
    monkeypatch.setitem(sys.modules, "automatos_metrics",
                        types.SimpleNamespace(add_aiohttp_metrics=lambda app, **_: None))
    monkeypatch.setenv("WORKSPACE_VOLUME_PATH", str(volume))
    spec = importlib.util.spec_from_file_location("workspace_worker_main_f173", _WORKER_DIR / "main.py")
    worker_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker_main)
    return worker_main


@contextlib.asynccontextmanager
async def _worker(monkeypatch, volume):
    """The worker's HTTP server, listening, over ``volume``; its base URL."""
    port = _free_port()
    for name, value in {"WORKER_INTERNAL_TOKEN": "", "WORKER_BIND_HOST": "127.0.0.1",
                        "WORKER_HEALTH_PORT": str(port)}.items():
        monkeypatch.setenv(name, value)
    worker_main = _load_worker_main(monkeypatch, volume)

    worker = worker_main.WorkspaceWorker()
    server = asyncio.create_task(worker._health_server())
    base = f"http://127.0.0.1:{port}"
    try:
        async with aiohttp.ClientSession() as http:
            for _ in range(100):
                with contextlib.suppress(aiohttp.ClientError):
                    async with http.get(f"{base}/health") as health:
                        if health.status == 200:
                            break
                await asyncio.sleep(0.05)
            yield base, http
    finally:
        worker._running = False
        server.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server


async def _answer(response):
    return response.status, await response.json(content_type=None)


@pytest.mark.asyncio
async def test_a_new_workspaces_first_file_is_saved(monkeypatch, tmp_path):
    ws = str(uuid.uuid4())
    async with _worker(monkeypatch, tmp_path) as (base, http):
        status, body = await _answer(await http.post(
            f"{base}/workspaces/{ws}/files/write", json={"path": "reports/first-report.md", "content": "# Q3"}))

    assert status == 200, body
    assert (tmp_path / ws / "reports" / "first-report.md").read_text() == "# Q3"
    assert (tmp_path / ws / ".workspace_meta.json").is_file()  # provisioned like a listed one


@pytest.mark.asyncio
async def test_a_new_workspace_can_run_search_and_fetch(monkeypatch, tmp_path):
    ws = str(uuid.uuid4())
    async with _worker(monkeypatch, tmp_path) as (base, http):
        ran = await _answer(await http.post(f"{base}/workspaces/{ws}/exec", json={"command": "echo saved"}))
        searched = await _answer(await http.get(f"{base}/workspaces/{ws}/files/grep", params={"pattern": "Q3"}))
        fetched = await _answer(await http.get(f"{base}/workspaces/{ws}/files/download", params={"path": "none.md"}))

    assert ran[0] == 200 and "saved" in ran[1].get("stdout", ""), ran
    assert searched[0] == 200 and searched[1]["matches"] == [], searched
    assert fetched == (404, {"error": "File not found"})  # the file, not the workspace, is missing


@pytest.mark.parametrize("not_a_workspace", ["not-a-uuid", str(uuid.uuid4()).upper(), uuid.uuid4().hex])
@pytest.mark.asyncio
async def test_only_a_canonical_uuid_names_a_workspace_directory(monkeypatch, tmp_path, not_a_workspace):
    async with _worker(monkeypatch, tmp_path) as (base, http):
        written = await _answer(await http.post(
            f"{base}/workspaces/{not_a_workspace}/files/write", json={"path": "x.md", "content": "x"}))
        listed = await _answer(await http.get(f"{base}/workspaces/{not_a_workspace}/files"))

    assert written == (400, {"error": "Invalid workspace id"})
    assert listed == (400, {"error": "Invalid workspace id"})
    assert list(tmp_path.iterdir()) == []


class _Redis:
    """The task runner's Redis, recorded."""

    def __init__(self):
        self.statuses, self.results, self.events = [], {}, []

    async def hget(self, key, field):
        return None

    async def hset(self, key, mapping):
        self.statuses.append(dict(mapping))

    async def expire(self, key, seconds):
        pass

    async def set(self, key, value, ex=None):
        self.results[key] = json.loads(value)

    async def publish(self, channel, message):
        self.events.append(json.loads(message))


@pytest.mark.asyncio
async def test_a_queued_task_for_a_non_uuid_workspace_makes_nothing(monkeypatch, tmp_path):
    """Security review: the task runner also names a directory from its id."""
    volume = tmp_path / "volume"
    volume.mkdir()
    worker = _load_worker_main(monkeypatch, volume).WorkspaceWorker()
    worker._redis = redis = _Redis()

    async def _no_database(*args, **kwargs):
        return None

    worker._update_db_execution = _no_database

    await worker._execute_task({"task_id": "task-f173-dotdot", "workspace_id": ".."})

    assert [p.name for p in tmp_path.iterdir()] == ["volume"]  # nothing beside the volume
    assert list(volume.iterdir()) == []
    assert (redis.statuses[-1]["status"], redis.statuses[-1]["error"]) == ("failed", "Invalid workspace id")


@pytest.mark.asyncio
async def test_a_download_never_leaves_its_workspace(monkeypatch, tmp_path):
    ws = str(uuid.uuid4())
    (tmp_path / ws).mkdir()
    (tmp_path / "outside.txt").write_text("another workspace's")
    async with _worker(monkeypatch, tmp_path) as (base, http):
        climbed = await _answer(await http.get(
            f"{base}/workspaces/{ws}/files/download", params={"path": "../outside.txt"}))
        rooted = await _answer(await http.get(
            f"{base}/workspaces/{ws}/files/download", params={"path": str(tmp_path / "outside.txt")}))

    assert climbed == (403, {"error": "Path traversal denied"})
    assert rooted == (403, {"error": "Path traversal denied"})
