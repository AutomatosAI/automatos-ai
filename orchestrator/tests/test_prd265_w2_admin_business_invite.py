"""PRD-265 W2 — the Business invite link (``POST /api/admin/workspaces/{id}/business-invite``).

Platform admin only (403 otherwise); 404 for a missing workspace; 200 with a
Business checkout link and its expiry for a platform admin.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.admin_credits import router
from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from services.billing.provider import CheckoutLink


class FakeProvider:
    name = "fake"

    def ensure_customer(self, **kwargs):
        return "cus_new"

    def create_checkout(self, request):
        return CheckoutLink(url="https://checkout.stripe.com/pay/biz-1", session_id="cs_biz")

    def portal_link(self, **kwargs):  # pragma: no cover - not used by this route
        raise AssertionError("business-invite never opens the portal")


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


def test_a_platform_admin_gets_a_business_checkout_link(monkeypatch):
    ws = _workspace()
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: FakeProvider())
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, workspace_id: None)
    monkeypatch.setattr("services.billing.checkout.price_for_plan", lambda plan: "price_business")
    app, _ = _app(system_role="super_admin", workspace=ws)
    res = _post(app, ws.id)
    assert res.status_code == 200
    body = res.json()
    assert body["url"] == "https://checkout.stripe.com/pay/biz-1"
    assert body["workspace_id"] == str(ws.id)
    assert "expires_at" in body


def test_a_platform_admin_with_an_existing_subscription_gets_409(monkeypatch):
    ws = _workspace()
    live_account = SimpleNamespace(subscription_id="sub_1", subscription_status="active")
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: FakeProvider())
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, workspace_id: live_account)
    app, _ = _app(system_role="super_admin", workspace=ws)
    assert _post(app, ws.id).status_code == 409


def test_503_when_stripe_isnt_configured(monkeypatch):
    from config import config

    ws = _workspace()
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "")
    app, _ = _app(system_role="admin", workspace=ws)
    assert _post(app, ws.id).status_code == 503
