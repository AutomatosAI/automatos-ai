"""The workspace worker (services/workspace-worker) run inside a test: its main
module over a temporary volume, its HTTP server on a free port, and its task
runner over a recorded Redis. Shared by F173 and F178.

The worker's logging and Prometheus metrics are stubbed: its logging setup
reconfigures the root logger, and metrics register once per process, so a
second server would collide. main.py edits sys.path at import, so the loader
gives it a copy that monkeypatch restores.
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import socket
import sys
import types
from pathlib import Path

import aiohttp

WORKER_DIR = Path(__file__).resolve().parents[2] / "services" / "workspace-worker"


def load_worker_main(monkeypatch, volume):
    """The worker's main module over ``volume``."""
    monkeypatch.setattr(sys, "path", [str(WORKER_DIR), *sys.path])
    monkeypatch.setitem(sys.modules, "automatos_logging", types.SimpleNamespace(setup_logging=lambda **_: None))
    monkeypatch.setitem(sys.modules, "automatos_metrics",
                        types.SimpleNamespace(add_aiohttp_metrics=lambda app, **_: None))
    monkeypatch.setenv("WORKSPACE_VOLUME_PATH", str(volume))
    spec = importlib.util.spec_from_file_location("workspace_worker_main_under_test", WORKER_DIR / "main.py")
    worker_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker_main)
    return worker_main


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextlib.asynccontextmanager
async def worker_server(monkeypatch, volume):
    """The worker's HTTP server, listening, over ``volume``: (base URL, client)."""
    port = _free_port()
    for name, value in {"WORKER_INTERNAL_TOKEN": "", "WORKER_BIND_HOST": "127.0.0.1",
                        "WORKER_HEALTH_PORT": str(port)}.items():
        monkeypatch.setenv(name, value)
    worker = load_worker_main(monkeypatch, volume).WorkspaceWorker()
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


async def answer(response):
    """(status, JSON body) of a worker response; a non-JSON body comes back as bytes."""
    body = await response.read()
    try:
        return response.status, json.loads(body)
    except ValueError:
        return response.status, body


class RecordedRedis:
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


def task_runner(monkeypatch, volume):
    """A worker whose task runner runs over ``volume``, a recorded Redis and no database."""
    worker = load_worker_main(monkeypatch, volume).WorkspaceWorker()
    worker._redis = RecordedRedis()

    async def _no_database(*args, **kwargs):
        return None

    worker._update_db_execution = _no_database
    return worker
