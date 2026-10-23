"""PRD-265 US-007 / US-008 — the billing summary, the ledger, and admin credit grants.

- ``GET /api/v1/billing/summary`` gives any member the plan, status and credits by
  bucket, and says when platform-funded work is paused; our own cost never appears.
- ``GET /api/v1/billing/ledger`` is for the workspace's owner or admins (403 otherwise).
- Both are hosted-only: 404 in the local edition.
- ``POST /api/admin/workspaces/{id}/credits`` is platform-admin only (403 otherwise),
  refuses a zero or reasonless adjustment, and writes one ``adjust`` ledger row with
  the admin and the reason.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import config
from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from services.credits.ledger import Balance


def _workspace(plan="basic", byok=None):
    return SimpleNamespace(id=uuid4(), name="Barber & Co", plan=plan, deleted_at=None,
                           settings={"byok_overrides": byok or {}})


def _app(router, workspace, system_role="user"):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = workspace
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(
        workspace_id=workspace.id,
        user=UserContext(id="u-1", email="owner@example.com", role="owner", system_role=system_role),
        auth_type="clerk",
    )
    app.dependency_overrides[get_db] = lambda: (yield db)
    return app, db


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "BILLING_OPEN", False)


def _summary(monkeypatch, workspace, balance):
    from api.billing import router

    monkeypatch.setattr("services.credits.summary.read_balance", lambda db, ws: balance)
    app, _ = _app(router, workspace)
    return TestClient(app).get("/api/v1/billing/summary")


def test_the_summary_shows_the_trial_credits(hosted, monkeypatch):
    res = _summary(monkeypatch, _workspace(), Balance(trial=420))
    assert res.status_code == 200
    body = res.json()
    assert body["plan"] == "basic" and body["plan_name"] == "Own keys" and body["price_label"] == "beta"
    assert body["balance"] == {"trial": 420, "monthly": 0, "topup": 0, "spendable": 420}
    assert body["status"] == "active" and body["paused"] is False and body["metered"] is True
    assert "actual_cost_usd" not in str(body)


def test_the_summary_says_when_work_is_paused(hosted, monkeypatch):
    body = _summary(monkeypatch, _workspace(byok={"openrouter": True}), Balance(trial=0, topup=300)).json()
    # Own keys: top-up credits can't pay for platform AI, so it is paused — own keys still work.
    assert body["paused"] is True and body["status"] == "paused"
    assert body["balance"]["spendable"] == 0 and body["own_keys"] is True


def test_a_low_balance_turns_amber(hosted, monkeypatch):
    # While billing is closed the allowance is the trial (500): low at 20% = 100 left.
    monkeypatch.setattr(config, "CREDITS_ALERT_LOW_SHARE", 0.20)
    monkeypatch.setattr(config, "TRIAL_CREDITS", 500)
    low = _summary(monkeypatch, _workspace(plan="pro"), Balance(trial=0, monthly=100)).json()
    assert low["low"] is True and low["paused"] is False
    fine = _summary(monkeypatch, _workspace(plan="pro"), Balance(trial=101)).json()
    assert fine["low"] is False


def test_an_allowance_plan_shows_no_meter(hosted, monkeypatch):
    body = _summary(monkeypatch, _workspace(plan="enterprise"), Balance()).json()
    assert body["metered"] is False and body["paused"] is False and body["balance"]["spendable"] is None


def test_billing_is_hosted_only(monkeypatch):
    from api.billing import router

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    app, _ = _app(router, _workspace())
    assert TestClient(app).get("/api/v1/billing/summary").status_code == 404


def test_the_ledger_is_for_owners_and_admins(hosted, monkeypatch):
    from api.billing import router

    monkeypatch.setattr("core.auth.workspace_admin.may_see_own_workspace_health", lambda db, ctx: False)
    app, _ = _app(router, _workspace())
    assert TestClient(app).get("/api/v1/billing/ledger").status_code == 403


def test_the_ledger_pages_for_an_owner(hosted, monkeypatch):
    from api.billing import router

    monkeypatch.setattr("core.auth.workspace_admin.may_see_own_workspace_health", lambda db, ctx: True)
    seen = {}

    def _page(db, workspace_id, *, page, page_size):
        seen.update(page=page, page_size=page_size)
        return {"items": [], "total": 0, "page": page, "page_size": page_size, "usage": {}}

    monkeypatch.setattr("services.credits.summary.ledger_page", _page)
    app, _ = _app(router, _workspace())
    res = TestClient(app).get("/api/v1/billing/ledger?page=2&page_size=10")
    assert res.status_code == 200 and seen == {"page": 2, "page_size": 10}
    assert TestClient(app).get("/api/v1/billing/ledger?page_size=1000").status_code == 422


# --------------------------------------------------------------------------- #
# Admin credit grants
# --------------------------------------------------------------------------- #


def _admin(monkeypatch, workspace, system_role):
    from api.admin_credits import router

    written = []
    monkeypatch.setattr("services.credits.ledger.record", lambda db, ws, entry, **k: written.append((ws, entry)) or True)
    monkeypatch.setattr("services.credits.ledger.read_balance", lambda db, ws: Balance(trial=1500))
    monkeypatch.setattr("core.workspaces.audit.AuditService.log", lambda self, **kwargs: None)
    monkeypatch.setattr("core.auth.actor.resolve_internal_user_id", lambda db, ctx: 1)
    app, db = _app(router, workspace, system_role=system_role)
    return TestClient(app), written, db


def test_an_admin_grants_credits_with_a_reason(hosted, monkeypatch):
    ws = _workspace()
    client, written, db = _admin(monkeypatch, ws, "super_admin")
    res = client.post(f"/api/admin/workspaces/{ws.id}/credits", json={"credits": 1000, "reason": "Web Summit demo"})
    assert res.status_code == 200 and res.json()["balance"]["trial"] == 1500
    (_, entry), = written
    assert entry.kind == "adjust" and entry.bucket == "trial" and entry.credits == 1000
    assert entry.reason == "Web Summit demo" and entry.actor == "owner@example.com"
    db.commit.assert_called()


def test_a_retried_request_with_its_id_changes_nothing_the_second_time(monkeypatch):
    ws = _workspace()
    client, written, _ = _admin(monkeypatch, ws, "super_admin")
    seen = set()

    def _record_once(db, workspace_id, entry, **_):
        first = entry.source_ref not in seen
        seen.add(entry.source_ref)
        written.append((workspace_id, entry))
        return first

    monkeypatch.setattr("services.credits.ledger.record", _record_once)
    body = {"credits": 500, "reason": "Web Summit demo", "request_id": "grant-0001"}
    first = client.post(f"/api/admin/workspaces/{ws.id}/credits", json=body).json()
    again = client.post(f"/api/admin/workspaces/{ws.id}/credits", json=body).json()
    assert first["applied"] is True and again["applied"] is False
    assert written[0][1].source_ref == written[1][1].source_ref == f"{ws.id}:grant-0001"


def test_a_non_admin_gets_403_and_nothing_is_written(hosted, monkeypatch):
    ws = _workspace()
    client, written, _ = _admin(monkeypatch, ws, "user")
    res = client.post(f"/api/admin/workspaces/{ws.id}/credits", json={"credits": 1000, "reason": "please"})
    assert res.status_code == 403 and written == []


@pytest.mark.parametrize("body", [
    {"credits": 0, "reason": "nothing"},
    {"credits": 10},
    {"credits": 10, "reason": "ok", "bucket": "overhead"},
    {"credits": 10, "reason": "ok", "request_id": "bad id with spaces"},
])
def test_a_zero_reasonless_or_odd_adjustment_is_refused(monkeypatch, body):
    ws = _workspace()
    client, written, _ = _admin(monkeypatch, ws, "super_admin")
    assert client.post(f"/api/admin/workspaces/{ws.id}/credits", json=body).status_code == 422
    assert written == []


def test_the_local_edition_has_no_credits_to_adjust(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    ws = _workspace()
    client, written, _ = _admin(monkeypatch, ws, "super_admin")
    res = client.post(f"/api/admin/workspaces/{ws.id}/credits", json={"credits": 1000, "reason": "a test"})
    assert res.status_code == 404
    assert written == []
