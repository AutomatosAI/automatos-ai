"""PRD-265 W2 — the Billing tab's routes (``api/billing.py``): plans, checkout, portal.

Hosted edition only (404 in local). ``/checkout`` and ``/portal`` are owner/admin
only (``require_workspace_admin``); while ``BILLING_OPEN`` is false, buying a
plan is further refused to everyone but a platform admin (``checkout_open_for``).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.billing import router
from config import config
from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db
from services.billing.provider import CheckoutLink
from services.billing.stripe_api import StripeError


class FakeProvider:
    name = "fake"

    def __init__(self, *, ensure_customer=None, create_checkout=None, portal_link=None):
        self._ensure_customer = ensure_customer or (lambda **k: "cus_new")
        self._create_checkout = create_checkout or (lambda request: CheckoutLink(url="https://checkout/1",
                                                                                 session_id="cs_1"))
        self._portal_link = portal_link or (lambda **k: "https://billing/portal-1")

    def ensure_customer(self, **kwargs):
        return self._ensure_customer(**kwargs)

    def create_checkout(self, request):
        return self._create_checkout(request)

    def portal_link(self, **kwargs):
        return self._portal_link(**kwargs)


def _workspace():
    return SimpleNamespace(id=uuid4(), name="Barber & Co", plan="basic")


def _ctx(system_role="user"):
    return RequestContext(workspace_id=uuid4(), user=UserContext(id="u-1", email="owner@example.com",
                                                                 system_role=system_role), auth_type="clerk")


def _app(*, ctx=None, bypass_admin_gate=False):
    db = MagicMock()
    app = FastAPI()
    app.include_router(router)
    ctx = ctx or _ctx()
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    if bypass_admin_gate:
        app.dependency_overrides[require_workspace_admin] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: (yield db)
    return app, db


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "BILLING_OPEN", False)


# --------------------------------------------------------------------------- #
# GET /plans
# --------------------------------------------------------------------------- #


def test_plans_is_hosted_only(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    app, db = _app()
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    assert TestClient(app).get("/api/v1/billing/plans").status_code == 404


def test_plans_has_the_expected_shape_for_a_member(hosted, monkeypatch):
    monkeypatch.setattr("core.auth.workspace_admin.may_see_own_workspace_health", lambda db, ctx: False)
    monkeypatch.setattr("services.billing.plans.account_for_workspace", lambda db, ws: None)
    app, db = _app()
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    body = TestClient(app).get("/api/v1/billing/plans").json()
    assert body["can_manage"] is False
    assert {"billing_open", "checkout_available", "current_plan", "subscription", "plans", "topup"} <= body.keys()
    assert [p["key"] for p in body["plans"]] == ["basic", "pro", "business", "enterprise"]


# --------------------------------------------------------------------------- #
# POST /checkout — who may call it at all
# --------------------------------------------------------------------------- #


def _checkout(app, body):
    return TestClient(app).post("/api/v1/billing/checkout", json=body)


def test_checkout_is_403_for_a_non_admin_member(hosted, monkeypatch):
    monkeypatch.setattr("core.auth.workspace_admin.may_see_own_workspace_health", lambda db, ctx: False)
    app, db = _app(ctx=_ctx("user"))
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    assert _checkout(app, {"kind": "plan", "plan": "basic"}).status_code == 403


def test_checkout_is_409_for_a_workspace_admin_who_isnt_a_platform_admin_while_closed(hosted):
    app, db = _app(ctx=_ctx("user"), bypass_admin_gate=True)
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    res = _checkout(app, {"kind": "plan", "plan": "basic"})
    assert res.status_code == 409
    assert res.json()["detail"] == "Paid plans aren't on sale yet."


def test_checkout_is_422_for_kind_plan_without_naming_a_plan(hosted):
    app, db = _app(ctx=_ctx("super_admin"), bypass_admin_gate=True)
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    assert _checkout(app, {"kind": "plan"}).status_code == 422


def test_checkout_200_for_a_platform_admin_before_billing_opens(hosted, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: provider)
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    monkeypatch.setattr("services.billing.checkout.price_for_plan", lambda plan: "price_basic")
    app, db = _app(ctx=_ctx("super_admin"), bypass_admin_gate=True)
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    res = _checkout(app, {"kind": "plan", "plan": "basic"})
    assert res.status_code == 200
    assert res.json() == {"url": "https://checkout/1", "flow": "checkout"}


def test_checkout_503_when_stripe_isnt_configured(hosted, monkeypatch):
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "")
    app, db = _app(ctx=_ctx("super_admin"), bypass_admin_gate=True)
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    assert _checkout(app, {"kind": "plan", "plan": "basic"}).status_code == 503


def test_checkout_502_when_stripe_refuses(hosted, monkeypatch):
    def _blow_up(**kwargs):
        raise StripeError(402, "Your card was declined.", "card_declined")

    provider = FakeProvider(ensure_customer=_blow_up)
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: provider)
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    monkeypatch.setattr("services.billing.checkout.price_for_plan", lambda plan: "price_basic")
    app, db = _app(ctx=_ctx("super_admin"), bypass_admin_gate=True)
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    assert _checkout(app, {"kind": "plan", "plan": "basic"}).status_code == 502


def test_checkout_accepts_the_display_name_own_keys_as_basic(hosted, monkeypatch):
    seen = {}

    def _start_plan_checkout(db, provider, workspace, *, plan, email):
        seen["plan"] = plan
        return {"url": "https://checkout/1", "flow": "checkout"}

    monkeypatch.setattr("services.billing.checkout.start_plan_checkout", _start_plan_checkout)
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: FakeProvider())
    app, db = _app(ctx=_ctx("super_admin"), bypass_admin_gate=True)
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    res = _checkout(app, {"kind": "plan", "plan": "Own keys"})
    assert res.status_code == 200
    assert seen["plan"] == "basic"


def test_checkout_409_for_a_topup_without_a_live_subscription(hosted, monkeypatch):
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: FakeProvider())
    app, db = _app(ctx=_ctx("super_admin"), bypass_admin_gate=True)
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    res = _checkout(app, {"kind": "topup"})
    assert res.status_code == 409


# --------------------------------------------------------------------------- #
# POST /portal
# --------------------------------------------------------------------------- #


def test_portal_404_with_no_billing_account(hosted, monkeypatch):
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: FakeProvider())
    monkeypatch.setattr("services.billing.checkout.account_for_workspace", lambda db, ws: None)
    app, db = _app(ctx=_ctx("super_admin"), bypass_admin_gate=True)
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    res = TestClient(app).post("/api/v1/billing/portal")
    assert res.status_code == 404
