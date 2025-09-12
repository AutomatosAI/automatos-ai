from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    payload = r.json()
    assert "status" in payload


def test_api_system_health(client: TestClient):
    r = client.get("/api/system/health")
    assert r.status_code == 200
    payload = r.json()
    assert payload.get("status") in {"healthy", "degraded", "unhealthy"}


def test_api_system_metrics(client: TestClient):
    r = client.get("/api/system/metrics")
    assert r.status_code == 200
    payload = r.json()
    assert "cpu" in payload and "memory" in payload and "disk" in payload


def test_request_id_header_echo(client: TestClient):
    r = client.get("/health", headers={"X-Request-ID": "test-req-123"})
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") == "test-req-123"


