"""PRD-262 US-008: the web stack on supported versions, and the API still serves every route.

starlette 0.41.3 carried GHSA-86qp-5c8j-p5mr (on CISA's KEV list), and moving it means
moving FastAPI, which F092 says is not a version edit: an unpinned fresh install once
served a route manifest of about a dozen routes. Since FastAPI 0.137 an included
router stays one node of ``app.routes`` with no path, so every walk over the routes
flattens them with ``iter_route_contexts``.

- requirements.in (requirements.txt until PRD-262 US-022; requirements.lock, which the
  image and CI install, is held to it by test_prd262_api_locks.py) pins fastapi,
  starlette (>= 1.0.1), uvicorn[standard] 0.50.1 (the newest that accepts websockets
  12.0) and websockets 12.0 exactly, and this suite runs on those versions;
- in a FRESH process (tests/web_stack_probe.py, DB unreachable), every router of
  router_manifest.py is mounted: the app serves each of its endpoints, and the app's
  routing answers each of its HTTP paths (405 to a method no route takes, never 404);
- no route of the committed reports/route-manifest.json is lost: scripts/dump_routes.py
  still collects each one, and the routing answers each HTTP one;
- FastAPI's strict_content_type (on by default since 0.132) stays on: a JSON body sent
  with no Content-Type is not parsed.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

ORCH_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ORCH_ROOT / "requirements.in"
PROBE = ORCH_ROOT / "tests" / "web_stack_probe.py"
# Read at collection, before any test runs: test_route_manifest.py rewrites this file
# with a fresh dump, and this module checks the COMMITTED one.
COMMITTED_MANIFEST = (ORCH_ROOT / "reports" / "route-manifest.json").read_text(encoding="utf-8")

SERVED = 405  # the probe's method on a served path
PIN = re.compile(r"^([A-Za-z0-9_.-]+)(\[[a-z,]+\])?==([0-9][0-9.]*)\s*(#.*)?$")
WEB_STACK = {"fastapi", "starlette", "uvicorn", "websockets"}
# GHSA-86qp-5c8j-p5mr is fixed in starlette 1.0.1.
STARLETTE_FIXED = (1, 0, 1)
# The newest uvicorn whose [standard] extra accepts websockets 12.0 (0.50.2 needs >= 13).
UVICORN_PIN = "0.50.1"
WEBSOCKETS_PIN = "12.0"


def _pins() -> dict:
    pins = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        match = PIN.match(line.strip())
        if match:
            pins[match.group(1).lower()] = (match.group(2) or "", match.group(3))
    return pins


def _numbers(text: str) -> tuple:
    return tuple(int(part) for part in text.split("."))


def test_the_web_stack_is_pinned_exactly():
    pins = _pins()
    assert WEB_STACK <= set(pins), f"pinned exactly (==): {sorted(WEB_STACK & set(pins))}"
    assert _numbers(pins["starlette"][1]) >= STARLETTE_FIXED, "starlette below 1.0.1 has GHSA-86qp-5c8j-p5mr"
    assert pins["uvicorn"] == ("[standard]", UVICORN_PIN)
    assert pins["websockets"] == ("", WEBSOCKETS_PIN)
    # The rate limiter's subclass keeps slowapi's dispatch line for line.
    assert pins["slowapi"] == ("", "0.1.10")


def test_the_suite_runs_on_the_pinned_versions():
    """A cached environment on the old stack would test nothing this story changed."""
    pins = _pins()
    for name in sorted(WEB_STACK | {"slowapi"}):
        assert version(name) == pins[name][1], f"{name} {version(name)} installed, {pins[name][1]} pinned"


@pytest.fixture(scope="module")
def fresh_process() -> dict:
    env = dict(os.environ)
    env.update(
        {
            # Unreachable Postgres (test_route_manifest.py's preamble): the probe is DB-free.
            "POSTGRES_USER": "test",
            "POSTGRES_PASSWORD": "test",
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": "59432",
            "POSTGRES_DB": "test",
            "DATABASE_URL": "postgresql://test:test@127.0.0.1:59432/test",
        }
    )
    proc = subprocess.run(
        [sys.executable, str(PROBE)],
        cwd=str(ORCH_ROOT),
        env=env,
        input=COMMITTED_MANIFEST,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert proc.returncode == 0, f"the probe failed (rc={proc.returncode})\nSTDERR:\n{proc.stderr[-3000:]}"
    line = next(line for line in proc.stdout.splitlines() if line.startswith("WEB_STACK_PROBE="))
    return json.loads(line.split("=", 1)[1])


def test_every_router_of_the_manifest_answers_on_a_fresh_process(fresh_process):
    answers = fresh_process["answers"]
    routers = fresh_process["routers"]
    # An optional router whose integration is missing is skipped, as the app skips it.
    missing = [r for r in routers if not r["imported"] and not r["optional"]]
    assert not missing, f"required routers that did not import: {missing}"
    mounted = [r for r in routers if r["imported"]]
    for router in mounted:
        assert not router["unserved"], f"{router['spec']}: routes the app does not serve: {router['unserved']}"
        silent = {path: answers[path] for path in router["http_paths"] if answers[path] != SERVED}
        assert not silent, f"{router['spec']}: paths the app's routing does not answer: {silent}"
    assert sum(len(r["http_paths"]) for r in mounted) > 100, "the probe asked about too few paths to mean anything"


def test_no_route_of_the_committed_manifest_is_lost(fresh_process):
    committed = {(r["method"], r["path"]) for r in json.loads(COMMITTED_MANIFEST)["routes"]}
    collected = {tuple(pair) for pair in fresh_process["collected"]}
    lost = sorted(committed - collected)
    assert not lost, f"reports/route-manifest.json routes a fresh process no longer collects: {lost}"
    answers = fresh_process["answers"]
    silent = {path: answers[path] for method, path in committed if method and answers[path] != SERVED}
    assert not silent, f"committed routes the app's routing does not answer on a fresh process: {silent}"


class _Thing(BaseModel):
    name: str


def test_a_json_body_with_no_content_type_is_not_parsed():
    """A browser sends a Blob body with no Content-Type without a CORS preflight, so a
    JSON endpoint that parsed such a body could be called from any web page the owner of
    a local edition (no login) opened. FastAPI 0.132+ refuses it by default, and the API
    keeps that default: its clients send application/json, and its webhooks read the
    raw request, which this setting does not touch."""
    app = FastAPI()

    @app.post("/things")
    def create(thing: _Thing):
        return {"name": thing.name}

    client = TestClient(app)
    assert client.post("/things", content=b'{"name": "a"}').status_code == 422
    assert client.post("/things", content=b'{"name": "a"}', headers={"Content-Type": "application/json"}).json() == {"name": "a"}
    assert "strict_content_type" not in (ORCH_ROOT / "main.py").read_text(encoding="utf-8")
