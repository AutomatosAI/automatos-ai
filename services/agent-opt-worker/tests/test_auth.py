"""PRD-262 US-006: every route but /health and /metrics answers only the platform.

The check is the router's dependency, so a forged Host header (which starlette
0.41.3 let rewrite request.url.path, GHSA-86qp-5c8j-p5mr) cannot talk past it.
"""
from __future__ import annotations

import pytest
from fastapi.routing import APIRoute

from helpers import TOKEN

PUBLIC = {"/health", "/metrics"}
# What orchestrator/core/services/futureagi_service.py calls, and nothing else.
PROTECTED = {
    ("POST", "/assess"),
    ("POST", "/safety"),
    ("POST", "/score"),
    ("POST", "/optimize"),
    ("GET", "/optimize/{job_id}"),
}
FORGED_HOSTS = ("evil.example", "testserver/health?", "testserver/metrics#", "testserver/health/../assess")


def _protected(worker_main) -> set[tuple[str, str]]:
    return {(method, route.path) for route in worker_main.router.routes for method in route.methods}


def _requests(worker_main):
    """(method, concrete path) for every route behind the token."""
    return sorted((method, path.replace("{job_id}", "some-job")) for method, path in _protected(worker_main))


def test_the_token_router_holds_exactly_the_routes_the_platform_calls(worker_main):
    assert _protected(worker_main) == PROTECTED


def test_only_health_and_metrics_sit_outside_the_token_router(worker_main):
    protected_paths = {path for _, path in _protected(worker_main)}
    public = {route.path for route in worker_main.app.routes if isinstance(route, APIRoute)} - protected_paths
    assert public == PUBLIC


@pytest.mark.parametrize("headers", [
    {},
    {"X-Internal-Token": ""},
    {"X-Internal-Token": "wrong"},
    {"X-Internal-Token": TOKEN + "x"},
    {"X-Internal-Token": TOKEN[:-1]},
], ids=["no-header", "empty", "wrong", "longer", "shorter"])
def test_every_protected_route_refuses_without_the_right_token(client, worker_main, headers):
    for method, path in _requests(worker_main):
        response = client.request(method, path, headers=headers)
        assert response.status_code == 401, (method, path, response.status_code, response.text)


@pytest.mark.parametrize("host", FORGED_HOSTS)
def test_a_forged_host_header_gets_no_further(client, worker_main, host):
    for method, path in _requests(worker_main):
        for headers in ({"Host": host}, {"Host": host, "X-Internal-Token": "wrong"}):
            response = client.request(method, path, headers=headers)
            assert response.status_code == 401, (method, path, host, response.status_code)


def test_a_json_body_does_not_get_past_the_token(client):
    response = client.post("/score", json={"input_text": "q", "output_text": "a"})
    assert response.status_code == 401


def test_the_right_token_reaches_the_route(client, authed, fake_scoring):
    response = client.post("/score", headers=authed, json={"input_text": "q", "output_text": "a", "metrics": ["is_helpful"]})
    assert response.status_code == 200, response.text
    assert response.json()["scores"]["is_helpful"]["score"] == 1.0
    assert fake_scoring == ["is_helpful"]


def test_an_unknown_job_with_the_token_is_a_404_not_a_401(client, authed):
    assert client.get("/optimize/no-such-job", headers=authed).status_code == 404


def test_health_and_metrics_answer_without_a_token(client):
    assert client.get("/health").status_code == 200
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "agent_opt_worker_requests_total" in metrics.text


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"])
def test_no_api_docs_are_served(client, authed, path):
    assert client.get(path).status_code == 404
    assert client.get(path, headers=authed).status_code == 404


def test_the_paid_sdk_smoke_route_is_gone(client, authed, worker_main):
    assert client.get("/test", headers=authed).status_code == 404
    assert not hasattr(worker_main, "test_sdk")
