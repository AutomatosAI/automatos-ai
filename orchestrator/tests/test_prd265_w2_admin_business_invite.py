"""PRD-265 W2 v2 — the Business offer (``POST /api/admin/workspaces/{id}/business-invite``).

No link, no portal: a platform admin offers Business to one workspace, and its
owner buys it on their own in-page Billing tab for ``BUSINESS_OFFER_DAYS``.
Platform admin only (403 otherwise); 404 for a missing or deleted workspace;
409 with a live subscription; 503 when Stripe isn't configured for a Business
price; 200 with the offer's expiry, and the owner is told by a notice.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.admin_credits import router
from config import config
from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db


class FakeProvider:
    name = "fake"

    def ensure_customer(self, **kwargs):
        return "cus_new"

    def create_checkout(self, request):  # pragma: no cover - the offer never checks out itself
        raise AssertionError("offering Business never opens a checkout")


@pytest.fixture(autouse=True)
def hosted(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")


def test_the_local_edition_has_no_business_offer(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    workspace = _workspace()
    assert _post(_app(system_role="super_admin", workspace=workspace), workspace.id).status_code == 404


def _workspace():
    return SimpleNamespace(id=uuid4(), name="Barber & Co", deleted_at=None)


def _app(*, system_role="user", workspace=None):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = workspace
    app = FastAPI()
    app.include_router(router)
    ctx = RequestContext(workspace_id=uuid4(),
                         user=UserContext(id="admin-1", email="admin@automatos.ai", system_role=system_role),
                         auth_type="clerk")
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: (yield db)
    return app, db


def _post(app, workspace_id):
    return TestClient(app).post(f"/api/admin/workspaces/{workspace_id}/business-invite")


def test_a_non_platform_admin_gets_403(monkeypatch):
    ws = _workspace()
    app, _ = _app(system_role="user", workspace=ws)
    assert _post(app, ws.id).status_code == 403


def test_a_missing_workspace_is_404(monkeypatch):
    app, _ = _app(system_role="super_admin", workspace=None)
    assert _post(app, uuid4()).status_code == 404


def test_a_soft_deleted_workspace_is_404(monkeypatch):
    ws = SimpleNamespace(id=uuid4(), name="Gone", deleted_at="2026-01-01T00:00:00Z")
    app, _ = _app(system_role="super_admin", workspace=ws)
    assert _post(app, ws.id).status_code == 404


def test_a_platform_admin_offers_business_and_returns_its_expiry(monkeypatch):
    from config import config

    ws = _workspace()
    monkeypatch.setattr(config, "BUSINESS_OFFER_DAYS", 14)
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: FakeProvider())
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, workspace_id: None)
    monkeypatch.setattr("services.billing.checkout.price_for_plan", lambda plan: "price_business")
    app, _ = _app(system_role="super_admin", workspace=ws)
    before = datetime.now(timezone.utc) + timedelta(days=14)
    res = _post(app, ws.id)
    after = datetime.now(timezone.utc) + timedelta(days=14)
    assert res.status_code == 200
    body = res.json()
    assert body["workspace_id"] == str(ws.id)
    offered_until = datetime.fromisoformat(body["business_offer_until"])
    assert before - timedelta(seconds=5) <= offered_until <= after + timedelta(seconds=5)


def test_a_workspace_with_a_live_subscription_gets_409(monkeypatch):
    ws = _workspace()
    live_account = SimpleNamespace(subscription_id="sub_1", subscription_status="active")
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: FakeProvider())
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, workspace_id: live_account)
    app, _ = _app(system_role="super_admin", workspace=ws)
    assert _post(app, ws.id).status_code == 409


def test_503_when_stripe_isnt_configured_for_a_business_price(monkeypatch):
    from config import config

    ws = _workspace()
    monkeypatch.setattr(config, "STRIPE_PRICE_BUSINESS", "")
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: FakeProvider())
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, workspace_id: None)
    app, _ = _app(system_role="admin", workspace=ws)
    assert _post(app, ws.id).status_code == 503


def test_the_owner_is_sent_a_business_offered_notice(monkeypatch):
    from services.billing import notices

    ws = _workspace()
    sent = []
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: FakeProvider())
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, workspace_id: None)
    monkeypatch.setattr("services.billing.checkout.price_for_plan", lambda plan: "price_business")
    monkeypatch.setattr(notices, "send_notice", lambda db, workspace_id, notice: sent.append((workspace_id, notice)))
    app, _ = _app(system_role="super_admin", workspace=ws)
    res = _post(app, ws.id)
    assert res.status_code == 200
    (sent_workspace_id, notice) = sent[0]
    assert str(sent_workspace_id) == str(ws.id)
    assert notice.event_type == notices.NOTICE_BUSINESS_OFFERED
    assert notice.status == "info"
