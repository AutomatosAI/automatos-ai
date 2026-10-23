"""PRD-265 W2 v2 — managing a subscription inside Studio (``api/billing_manage.py``).

Every route is ``require_workspace_admin`` and hosted-only (404 in the local
edition): the card, invoices (PDF served by us), cancel/resume and plan changes.
The router is imported inside each test so one failure doesn't block the rest.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import config
from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_admin import require_workspace_admin
from core.database.database import get_db
from services.billing.provider import CardOnFile, InvoiceLine, PlanChangePreview, PlanChangeResult


def test_the_manage_router_imports_with_the_billing_helpers():
    import api.billing
    import api.billing_manage

    assert callable(api.billing.hosted_workspace) and callable(api.billing.billing_call)
    assert api.billing_manage.router.prefix == "/api/v1/billing"


class FakeProvider:
    name = "fake"

    def __init__(self):
        self.card = CardOnFile(brand="visa", last4="4242", exp_month=12, exp_year=2028)
        self.invoices_list = [InvoiceLine(id="in_1", number="ABCD-0001", status="paid", total_cents=2900,
                                          currency="usd", created=None)]
        self.pdf = (b"%PDF-1.4 bytes", "ABCD0001.pdf")
        self.preview = PlanChangePreview(amount_due_cents=1350, tax_cents=0, currency="usd")
        self.upgrade_result = PlanChangeResult(status="done")

    def card_on_file(self, **kwargs):
        return self.card

    def start_card_change(self, **kwargs):
        return "seti_1_secret_abc"

    def finish_card_change(self, **kwargs):
        return self.card

    def invoices(self, **kwargs):
        return self.invoices_list

    def invoice_pdf(self, **kwargs):
        return self.pdf

    def set_cancel_at_period_end(self, **kwargs):
        return None

    def preview_plan_change(self, **kwargs):
        return self.preview

    def upgrade_now(self, **kwargs):
        return self.upgrade_result

    def downgrade_at_period_end(self, **kwargs):
        return PlanChangeResult(status="scheduled", schedule_id="sub_sched_1", effective_at=None)

    def cancel_scheduled_change(self, **kwargs):
        return None


def _router():
    from api.billing_manage import router

    return router


def _workspace():
    return SimpleNamespace(id=uuid4(), name="Barber & Co", plan="pro")


def _ctx(system_role="user"):
    return RequestContext(workspace_id=uuid4(), user=UserContext(id="u-1", email="owner@example.com",
                                                                 system_role=system_role), auth_type="clerk")


def _app(*, ctx=None, bypass_admin_gate=False):
    db = MagicMock()
    app = FastAPI()
    app.include_router(_router())
    ctx = ctx or _ctx()
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    if bypass_admin_gate:
        app.dependency_overrides[require_workspace_admin] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: (yield db)
    return app, db


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")


ROUTES = [
    ("GET", "/api/v1/billing/payment-method", None),
    ("POST", "/api/v1/billing/payment-method/setup", None),
    ("POST", "/api/v1/billing/payment-method", {"setup_intent_id": "seti_1"}),
    ("GET", "/api/v1/billing/invoices", None),
    ("POST", "/api/v1/billing/subscription/cancel", None),
    ("POST", "/api/v1/billing/subscription/resume", None),
    ("POST", "/api/v1/billing/plan-change/preview", {"plan": "pro"}),
    ("POST", "/api/v1/billing/plan-change", {"plan": "pro"}),
    ("POST", "/api/v1/billing/plan-change/cancel", None),
]


def _call(client, method, path, body):
    return client.post(path, json=body) if method == "POST" else client.get(path)


# --------------------------------------------------------------------------- #
# 403 for a non-admin member — require_workspace_admin stays real
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method, path, body", ROUTES)
def test_every_route_is_403_for_a_non_admin_member(hosted, monkeypatch, method, path, body):
    monkeypatch.setattr("core.auth.workspace_admin.may_see_own_workspace_health", lambda db, ctx: False)
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: FakeProvider())
    app, db = _app(ctx=_ctx("user"))
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    assert _call(TestClient(app), method, path, body).status_code == 403


# --------------------------------------------------------------------------- #
# 404 in the local edition
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method, path, body", ROUTES)
def test_every_route_is_404_in_the_local_edition(monkeypatch, method, path, body):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: FakeProvider())
    app, db = _app(ctx=_ctx("super_admin"), bypass_admin_gate=True)
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    assert _call(TestClient(app), method, path, body).status_code == 404


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #


def test_get_payment_method_happy_path(hosted, monkeypatch):
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: FakeProvider())
    app, db = _app(ctx=_ctx("super_admin"), bypass_admin_gate=True)
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    res = TestClient(app).get("/api/v1/billing/payment-method")
    assert res.status_code == 200
    assert res.json()["payment_method"]["brand"] == "visa"


def test_cancel_subscription_happy_path(hosted, monkeypatch):
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: FakeProvider())
    app, db = _app(ctx=_ctx("super_admin"), bypass_admin_gate=True)
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    res = TestClient(app).post("/api/v1/billing/subscription/cancel")
    assert res.status_code == 200
    assert res.json()["cancel_at_period_end"] is True


# --------------------------------------------------------------------------- #
# The PDF route
# --------------------------------------------------------------------------- #


def test_download_invoice_answers_a_pdf_with_a_quoted_filename(hosted, monkeypatch):
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: FakeProvider())
    app, db = _app(ctx=_ctx("super_admin"), bypass_admin_gate=True)
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    res = TestClient(app).get("/api/v1/billing/invoices/in_1/pdf")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert res.headers["content-disposition"] == 'attachment; filename="ABCD0001.pdf"'
    assert res.content == b"%PDF-1.4 bytes"


# --------------------------------------------------------------------------- #
# setup_intent_id validation
# --------------------------------------------------------------------------- #


def test_save_payment_method_422_for_a_setup_intent_id_that_isnt_seti(hosted, monkeypatch):
    monkeypatch.setattr("services.billing.stripe_provider.stripe_provider", lambda: FakeProvider())
    app, db = _app(ctx=_ctx("super_admin"), bypass_admin_gate=True)
    db.query.return_value.filter.return_value.first.return_value = _workspace()
    res = TestClient(app).post("/api/v1/billing/payment-method", json={"setup_intent_id": "pm_not_a_setup_intent"})
    assert res.status_code == 422
