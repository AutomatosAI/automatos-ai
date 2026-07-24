"""
PRD-008-A Phase 5 — Callback endpoint integration-ish tests
==============================================================

Uses FastAPI's TestClient with dependency overrides for ``widget_auth``
and ``get_db``. Exercises validation, feature-gate, idempotency,
rate-limit, and happy-path branches without a real DB or auth stack.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

import config  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_site(*, enabled=True, destinations=None, rate_limit=100):
    return SimpleNamespace(
        id=uuid4(),
        workspace_id=uuid4(),
        type="shopify",
        external_id="test.myshopify.com",
        display_name="Test",
        status="active",
        settings={
            "callback": {
                "enabled": enabled,
                "destinations": destinations or [{"type": "email", "address": "x@y"}],
                "sla_hours": 4,
                "team_capacity": "limited",
                "working_hours_only": False,
                "rate_limit_per_hour": rate_limit,
            }
        },
        capabilities={},
        secrets=None,
        created_at=datetime(2026, 5, 14, 12, 0, 0),
        updated_at=datetime(2026, 5, 14, 12, 0, 0),
    )


def _make_client(site, *, recent_dup_event=None, rate_session_count=0, rate_site_count=0):
    """Build a TestClient with widget_auth + get_db overrides."""
    from fastapi import FastAPI

    from api.widgets.callback import router as cb_router
    from api.widgets.auth import WidgetAuthContext, widget_auth
    from core.database.database import get_db
    from services.sites import get_default_site as _real_default
    import services.sites as sites_mod
    import services.callback as cb_mod

    app = FastAPI()
    app.include_router(cb_router, prefix="/api/widgets")

    fake_db = MagicMock()
    workspace_id = site.workspace_id if site else uuid4()

    # widget_auth override
    def _override_auth():
        return WidgetAuthContext(workspace_id=workspace_id, api_key_id=uuid4())

    # get_db override
    def _override_db():
        yield fake_db

    app.dependency_overrides[widget_auth] = _override_auth
    app.dependency_overrides[get_db] = _override_db

    # Patch service-layer functions that the endpoint calls. Avoids
    # constructing a fake SQLAlchemy query chain for each one.
    monkeys = []

    monkeys.append(("services.sites.get_default_site", lambda db, ws: site))
    monkeys.append((
        "services.callback.find_recent_duplicate",
        lambda db, **kw: recent_dup_event,
    ))
    monkeys.append((
        "services.callback.check_rate_limits",
        lambda db, **kw: cb_mod.RateLimitDecision(
            allowed=(rate_session_count == 0 and rate_site_count < 100),
            reason=(
                "per_session_cooldown" if rate_session_count > 0
                else ("per_site_hourly_cap" if rate_site_count >= 100 else None)
            ),
            retry_after_seconds=60 if rate_session_count > 0 else (3600 if rate_site_count >= 100 else None),
        ),
    ))
    # log_widget_event is async; make it a no-op coroutine
    async def _noop_log(*a, **kw): pass
    monkeys.append(("modules.widgets.telemetry.log_widget_event", _noop_log))

    import api.widgets.callback as cb_endpoint
    saved = []
    for path, replacement in monkeys:
        modname, attr = path.rsplit(".", 1)
        mod = __import__(modname, fromlist=[attr])
        # Patch on the endpoint module's namespace too if it imported the symbol
        saved.append((mod, attr, getattr(mod, attr, None)))
        setattr(mod, attr, replacement)
        if hasattr(cb_endpoint, attr):
            saved.append((cb_endpoint, attr, getattr(cb_endpoint, attr)))
            setattr(cb_endpoint, attr, replacement)

    from fastapi.testclient import TestClient
    client = TestClient(app)

    def teardown():
        for mod, attr, orig in saved:
            if orig is not None:
                setattr(mod, attr, orig)

    return client, teardown


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_callback_returns_202_and_request_id():
    site = _make_site()
    client, teardown = _make_client(site)
    try:
        resp = client.post(
            "/api/widgets/callback",
            json={
                "session_id": "sess_abc",
                "phone": "+447700900123",
                "name": "James",
                "product_context": "EN 12101-9 panel",
            },
        )
    finally:
        teardown()

    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["accepted"] is True
    assert body["request_id"].startswith("cb_")
    assert "aim to call" in body["eta_phrase"]
    assert "EN 12101-9 panel" in body["eta_phrase"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_callback_400_on_invalid_phone():
    site = _make_site()
    client, teardown = _make_client(site)
    try:
        resp = client.post(
            "/api/widgets/callback",
            json={"session_id": "sess", "phone": "not a phone", "name": "James"},
        )
    finally:
        teardown()

    assert resp.status_code == 400
    assert "E.164" in resp.json()["detail"]


def test_callback_422_on_missing_required_fields():
    site = _make_site()
    client, teardown = _make_client(site)
    try:
        resp = client.post(
            "/api/widgets/callback",
            json={"session_id": "sess"},
        )
    finally:
        teardown()

    # Pydantic validation → 422
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Feature-enabled gate
# ---------------------------------------------------------------------------

def test_callback_403_when_feature_disabled():
    site = _make_site(enabled=False)
    client, teardown = _make_client(site)
    try:
        resp = client.post(
            "/api/widgets/callback",
            json={
                "session_id": "sess", "phone": "+447700900123", "name": "James",
            },
        )
    finally:
        teardown()

    assert resp.status_code == 403
    assert "not enabled" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_callback_returns_existing_request_id_on_duplicate():
    site = _make_site()
    client, teardown = _make_client(site, recent_dup_event="cb_existing_123")
    try:
        resp = client.post(
            "/api/widgets/callback",
            json={
                "session_id": "sess_abc", "phone": "+447700900123",
                "name": "James", "product_context": "panel",
            },
        )
    finally:
        teardown()

    assert resp.status_code == 202
    assert resp.json()["request_id"] == "cb_existing_123"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_callback_429_on_per_session_cooldown():
    site = _make_site()
    client, teardown = _make_client(site, rate_session_count=1)
    try:
        resp = client.post(
            "/api/widgets/callback",
            json={"session_id": "x", "phone": "+447700900123", "name": "X"},
        )
    finally:
        teardown()

    assert resp.status_code == 429
    assert "per_session_cooldown" in resp.json()["detail"]
    assert resp.headers.get("retry-after") == "60"


def test_callback_429_on_per_site_hourly_cap():
    site = _make_site(rate_limit=100)
    client, teardown = _make_client(site, rate_site_count=100)
    try:
        resp = client.post(
            "/api/widgets/callback",
            json={"session_id": "x", "phone": "+447700900123", "name": "X"},
        )
    finally:
        teardown()

    assert resp.status_code == 429
    assert "per_site_hourly_cap" in resp.json()["detail"]
    assert resp.headers.get("retry-after") == "3600"


# ---------------------------------------------------------------------------
# Site-not-provisioned
# ---------------------------------------------------------------------------

def test_callback_503_when_no_site_for_workspace():
    """Transition window: workspace exists, migration hasn't run."""
    client, teardown = _make_client(site=None)
    try:
        resp = client.post(
            "/api/widgets/callback",
            json={"session_id": "x", "phone": "+447700900123", "name": "X"},
        )
    finally:
        teardown()

    assert resp.status_code == 503
    assert "contact support" in resp.json()["detail"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
