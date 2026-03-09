"""Journey 01 + 16: Health & system endpoints."""

import pytest


# ─── Journey 01: Basic health ─────────────────────────────────────────────────


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_has_status_key(client):
    data = client.get("/health").json()
    assert "status" in data


def test_health_status_value(client):
    data = client.get("/health").json()
    assert data["status"] in ("healthy", "degraded", "ok")


# ─── Journey 16: System health ────────────────────────────────────────────────


def test_system_health(client):
    r = client.get("/api/system/health")
    assert r.status_code == 200


def test_system_state_summary(client):
    r = client.get("/api/system/state/summary")
    assert r.status_code == 200