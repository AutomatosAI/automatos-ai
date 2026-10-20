"""PRD-262 US-006: the services' Prometheus middleware labels by route template.

services/shared/automatos_metrics.py is the source; each service vendors a copy
because Railway builds every service from its own directory. Before, the label
was the raw request path, so every /optimize/{job_id} poll, every scanner path
and (under starlette 0.41.3) every forged Host header added a counter series and
a histogram family that lived until the next restart.

- every vendored copy is byte-identical to services/shared;
- the FastAPI middleware (agent-opt-worker) and the Starlette middleware
  (workspace-worker) label by route template, with one "unmatched" label for
  paths no route matches, and an "OTHER" method for non-standard methods;
- the worker's /metrics answers 200 with prometheus_client's own content type
  (PRD-262 US-045: aiohttp raised on the charset in it, so the worker's /metrics was
  500). The worker moved from aiohttp to Starlette (PRD-262, row 36 of the 19 Oct
  size breakdown); these are the aiohttp tests, ported, with the status label now
  checked too.
"""
from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families

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


def _starlette_app(metrics, service: str):
    """A Starlette app with a workspace route and /health, metered as the worker is."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def ok(request):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/workspaces/{workspace_id}/files", ok), Route("/health", ok)])
    metrics.add_starlette_metrics(app, service=service)
    return app


def test_starlette_requests_are_labelled_by_route_template():
    from starlette.testclient import TestClient

    metrics = _load_shared()
    client = TestClient(_starlette_app(metrics, "prd262-starlette"))
    statuses = [client.get(f"/workspaces/{workspace}/files").status_code for workspace in ("ws-a", "ws-b", "ws-c")]
    statuses += [client.get(path).status_code for path in SCANNER_PATHS]
    statuses += [client.post("/health").status_code, client.request("PROPFIND", "/health").status_code]

    assert statuses == [200, 200, 200] + [404] * len(SCANNER_PATHS) + [405, 405]
    assert _labels(metrics, "prd262-starlette", "endpoint") == {"/workspaces/{workspace_id}/files", "unmatched"}
    assert _labels(metrics, "prd262-starlette", "method") == {"GET", "POST", "OTHER"}
    assert _labels(metrics, "prd262-starlette", "status") == {"200", "404", "405"}


def test_starlette_metrics_answers_200_with_the_client_librarys_content_type():
    """PRD-262 US-045: the workspace worker's /metrics answered 500 under aiohttp.
    prometheus_client's content type carries a charset; it goes out as it is."""
    from starlette.testclient import TestClient

    metrics = _load_shared()
    assert "charset=" in metrics.CONTENT_TYPE_LATEST, "the case this test guards no longer arises"
    client = TestClient(_starlette_app(metrics, "prd262-starlette-scrape"))
    assert client.get("/health").status_code == 200
    scrape = client.get("/metrics")

    assert scrape.status_code == 200, scrape.text
    assert scrape.headers.get("content-type") == metrics.CONTENT_TYPE_LATEST
    scraped = {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(scrape.text)
        for sample in family.samples
    }
    health = ("prd262_starlette_scrape_requests_total", (("endpoint", "/health"), ("method", "GET"), ("status", "200")))
    assert scraped.get(health) == 1.0, sorted(scraped)


def test_a_request_that_raises_is_counted_as_a_500():
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    metrics = _load_shared()

    async def boom(request):
        raise RuntimeError("boom")

    app = Starlette(routes=[Route("/boom", boom)])
    metrics.add_starlette_metrics(app, service="prd262-starlette-boom")
    assert TestClient(app, raise_server_exceptions=False).get("/boom").status_code == 500
    assert _labels(metrics, "prd262-starlette-boom", "status") == {"500"}
    assert _labels(metrics, "prd262-starlette-boom", "endpoint") == {"/boom"}


def test_no_copy_keeps_the_aiohttp_integration():
    """PRD-262, row 36: the worker, its only user, serves through Starlette now."""
    for path in [SHARED, *_vendored_copies()]:
        assert "add_aiohttp_metrics" not in path.read_text(), path
