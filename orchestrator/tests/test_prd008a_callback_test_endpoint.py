"""PRD-008-A — POST /api/sites/{id}/callback/test endpoint.

The "Send test" button in the dashboard's CallbackPanel hits this route
to fire a synthetic callback through every configured destination so a
merchant can prove their Telegram/Slack/WhatsApp wiring works without
needing a real shopper submission.

These tests cover the contract:
- 404 when the Site doesn't belong to the requesting workspace
- 400 when no destinations are configured
- 200 + per-destination results on the happy path (dispatcher mocked)
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


def _make_site(*, workspace_id=None, destinations=None):
    return SimpleNamespace(
        id=uuid4(),
        workspace_id=workspace_id or uuid4(),
        type="shopify",
        external_id="test.myshopify.com",
        display_name="Test Store",
        status="active",
        settings={"callback": {"enabled": True, "destinations": destinations or []}},
        capabilities={},
        secrets=None,
        created_at=datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc),
    )


def _make_client(*, site, dispatch_results=None, monkeypatch=None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.sites import router as sites_router
    from core.auth.dependencies import RequestContext, UserContext
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db
    from services.destinations.base import DispatchResult

    app = FastAPI()
    app.include_router(sites_router)

    fake_db = MagicMock()
    workspace_id = site.workspace_id if site else uuid4()

    def _override_ctx():
        # PRD-195 S6 gated this route (workspace:manage); the anonymous lane
        # (trusted local single-user posture) owns its workspace, which keeps
        # this dispatch-logic test focused on dispatch, not membership rows.
        return RequestContext(
            workspace_id=workspace_id,
            user=UserContext(id="test-user", email="test@example.com"),
            auth_type="anonymous",
        )

    def _override_db():
        yield fake_db

    app.dependency_overrides[get_request_context_hybrid] = _override_ctx
    app.dependency_overrides[get_db] = _override_db

    import api.sites as sites_mod

    monkeypatch.setattr(sites_mod, "get_site", lambda db, ws, sid: site if site and ws == workspace_id else None)

    async def _fake_dispatch(*, site_id, workspace_id, session_id, request_id, payload, destinations):
        return dispatch_results or []

    monkeypatch.setattr(sites_mod, "dispatch_callback_for_site", _fake_dispatch)

    return TestClient(app), DispatchResult


class TestCallbackTestEndpoint:
    def test_404_when_site_not_in_workspace(self, monkeypatch):
        client, _ = _make_client(site=None, monkeypatch=monkeypatch)
        resp = client.post(f"/api/sites/{uuid4()}/callback/test")
        assert resp.status_code == 404

    def test_400_when_no_destinations(self, monkeypatch):
        site = _make_site(destinations=[])
        client, _ = _make_client(site=site, monkeypatch=monkeypatch)
        resp = client.post(f"/api/sites/{site.id}/callback/test")
        assert resp.status_code == 400
        assert "destinations" in resp.json()["detail"].lower()

    def test_200_returns_per_destination_results(self, monkeypatch):
        site = _make_site(destinations=[{"platform": "slack", "channel_id": "C123ABC"}])
        from services.destinations.base import DispatchResult

        results = [
            DispatchResult(
                success=True,
                destination_type="slack",
                latency_ms=142,
                extra={"platform": "slack", "target": "C123ABC"},
            )
        ]
        client, _ = _make_client(site=site, dispatch_results=results, monkeypatch=monkeypatch)
        resp = client.post(f"/api/sites/{site.id}/callback/test")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["destinations_attempted"] == 1
        assert body["results"][0]["success"] is True
        assert body["results"][0]["platform"] == "slack"
        assert body["results"][0]["target"] == "C123ABC"
        assert body["results"][0]["latency_ms"] == 142
        assert body["request_id"].startswith("cb_")

    def test_failure_result_surfaces_error(self, monkeypatch):
        site = _make_site(destinations=[{"platform": "telegram"}])
        from services.destinations.base import DispatchResult

        results = [
            DispatchResult(
                success=False,
                destination_type="telegram",
                latency_ms=7,
                error="telegram delivery returned False — send /start to the bot",
                retryable=False,
            )
        ]
        client, _ = _make_client(site=site, dispatch_results=results, monkeypatch=monkeypatch)
        resp = client.post(f"/api/sites/{site.id}/callback/test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"][0]["success"] is False
        assert "/start" in body["results"][0]["error"]
