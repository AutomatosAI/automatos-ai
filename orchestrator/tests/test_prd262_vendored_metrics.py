"""PRD-262 US-006: the services' Prometheus middleware labels by route template.

services/shared/automatos_metrics.py is the source; each service vendors a copy
because Railway builds every service from its own directory. Before, the label
was the raw request path, so every /optimize/{job_id} poll, every scanner path
and (under starlette 0.41.3) every forged Host header added a counter series and
a histogram family that lived until the next restart.

- every vendored copy is byte-identical to services/shared;
- the FastAPI middleware (agent-opt-worker) and the aiohttp middleware
  (workspace-worker) label by route template, with one "unmatched" label for
  paths no route matches, and an "OTHER" method for non-standard methods.
"""
from __future__ import annotations

import asyncio
import importlib.util
import uuid
from pathlib import Path

from prometheus_client import CollectorRegistry

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "services" / "shared" / "automatos_metrics.py"
SCANNER_PATHS = ("/wp-login.php", "/.env", "/optimize/a/b", "/x/y/z")


def _vendored_copies() -> list[Path]:
    return sorted(p for p in (ROOT / "services").glob("*/automatos_metrics.py") if p.parent.name != "shared")


def _load_shared():
    """A private copy of the shared module, with its own registry."""
    name = f"prd262_metrics_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SHARED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REGISTRY = CollectorRegistry()
    return module


def _labels(module, service: str, label: str) -> set[str]:
    sample_name = service.replace("-", "_") + "_requests_total"
    return {
        sample.labels[label]
        for metric in module.REGISTRY.collect()
        for sample in metric.samples
        if sample.name == sample_name
    }


def test_every_vendored_copy_is_byte_identical_to_services_shared():
    copies = _vendored_copies()
    assert {p.parent.name for p in copies} >= {"agent-opt-worker", "workspace-worker"}
    source = SHARED.read_bytes()
    differ = [str(p.relative_to(ROOT)) for p in copies if p.read_bytes() != source]
    assert not differ, (
        f"{differ} differ from services/shared/automatos_metrics.py. Change services/shared "
        "and copy it over every vendored copy."
    )


def test_fastapi_requests_are_labelled_by_route_template():
    from fastapi import APIRouter, FastAPI
    from fastapi.testclient import TestClient

    metrics = _load_shared()
    app = FastAPI()
    router = APIRouter()

    @router.get("/optimize/{job_id}")
    def poll(job_id: str):
        return {"job_id": job_id}

    @app.get("/health")
    def health():
        return {"ok": True}

    app.include_router(router)
    metrics.add_fastapi_metrics(app, service="prd262-fastapi")
    client = TestClient(app)
    for job_id in ("job-a", "job-b", "job-c"):
        assert client.get(f"/optimize/{job_id}").status_code == 200
    for path in SCANNER_PATHS:
        assert client.get(path).status_code == 404
    assert client.get("/health", headers={"Host": "evil.example/forged"}).status_code == 200
    assert client.request("PROPFIND", "/health").status_code == 405
    assert client.get("/metrics").status_code == 200

    assert _labels(metrics, "prd262-fastapi", "endpoint") == {"/optimize/{job_id}", "/health", "unmatched"}
    assert _labels(metrics, "prd262-fastapi", "method") == {"GET", "OTHER"}


def test_aiohttp_requests_are_labelled_by_route_template():
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    metrics = _load_shared()

    async def ok(request):
        return web.json_response({"ok": True})

    async def scenario() -> list[int]:
        app = web.Application()
        app.router.add_get("/workspaces/{workspace_id}/files", ok)
        app.router.add_get("/health", ok)
        metrics.add_aiohttp_metrics(app, service="prd262-aiohttp")
        statuses = []
        async with TestClient(TestServer(app)) as client:
            for workspace in ("ws-a", "ws-b", "ws-c"):
                statuses.append((await client.get(f"/workspaces/{workspace}/files")).status)
            for path in SCANNER_PATHS:
                statuses.append((await client.get(path)).status)
            statuses.append((await client.post("/health")).status)
            statuses.append((await client.request("PROPFIND", "/health")).status)
        return statuses

    statuses = asyncio.run(scenario())
    assert statuses == [200, 200, 200] + [404] * len(SCANNER_PATHS) + [405, 405]
    assert _labels(metrics, "prd262-aiohttp", "endpoint") == {"/workspaces/{workspace_id}/files", "unmatched"}
    assert _labels(metrics, "prd262-aiohttp", "method") == {"GET", "POST", "OTHER"}
