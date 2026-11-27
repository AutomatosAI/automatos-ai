"""PRD-274: a monthly spend cap per product workspace (the proposal in #1520).

A product's workspace spends its owner's credits up to its cap for the calendar month: its own (set by the
product's key) or its product's default. Each debit counts against it; the credit gate's budget is the smaller of
the owner's balance and the cap left, and work stops at the cap with ``workspace_cap_reached`` while the owner
still has credits. Background work and an Auto Live call don't start at the cap. The owner is told once at 80%
and once at the cap. A workspace with no cap is limited by the owner's balance only, as before.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from config import config
from core.models.products import Product, ProductWorkspaceSpend
from services.credits import asks, caps, gate, meter, routing
from services.credits.ledger import BUCKET_MONTHLY, KIND_GRANT, Entry, record
from services.products import mint_key


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    for table in (Product.__table__, ProductWorkspaceSpend.__table__):
        table.create(eng, checkfirst=True)  # a test database built before these
    with eng.begin() as conn:
        for ddl in ("ALTER TABLE products ADD COLUMN IF NOT EXISTS reuse_existing_key BOOLEAN NOT NULL DEFAULT true",
                    "ALTER TABLE products ADD COLUMN IF NOT EXISTS workspace_monthly_credit_cap INTEGER",
                    "ALTER TABLE credit_ledger ADD COLUMN IF NOT EXISTS sub_tenant_id UUID"):
            conn.execute(text(ddl))
    yield eng
    eng.dispose()


@pytest.fixture(autouse=True)
def hosted(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "PRODUCTS_ENABLED", True)
    monkeypatch.setattr(config, "BILLING_FREE_OWNER_EMAILS", frozenset())


@pytest.fixture
def world(new_session):
    """An owner on Pro with 1,000 credits, its product (cap 100 a workspace), the product's workspace, and a
    plain workspace with 1,000 credits of its own; all swept after."""
    owner, customer, plain = (str(uuid.uuid4()) for _ in range(3))
    key = mint_key()
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name, plan) VALUES (CAST(:o AS uuid), 'cap-owner', 'pro'), "
                   "(CAST(:p AS uuid), 'cap-plain', 'pro')"), {"o": owner, "p": plain})
    product = Product(slug=f"cap-{owner[:8]}", name="Support desk", owner_workspace_id=uuid.UUID(owner),
                      key_hash=key.key_hash, key_prefix=key.key_prefix, workspace_monthly_credit_cap=100)
    s.add(product)
    s.commit()
    s.execute(text("INSERT INTO workspaces (id, name, plan, settings) VALUES (CAST(:c AS uuid), 'cap-customer', "
                   "'basic', CAST(:s AS jsonb))"),
              {"c": customer, "s": f'{{"product_id": "{product.id}", "source_external_id": "acme-store"}}'})
    for ws in (owner, plain):
        record(s, ws, Entry(kind=KIND_GRANT, source_type="test_grant", source_ref=f"cap:{ws}", credits=1000,
                            bucket=BUCKET_MONTHLY, actor="test"))
    s.commit()
    s.close()
    yield SimpleNamespace(owner=owner, customer=customer, plain=plain, product=str(product.id), key=key.plaintext)
    s = new_session.sweep()
    for ws in (customer, plain, owner):
        for table in ("llm_usage", "credit_ledger", "credit_balances", "product_workspace_spend"):
            s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    s.execute(text("DELETE FROM workspaces WHERE id = ANY(CAST(:ids AS uuid[]))"), {"ids": [customer, plain]})
    s.execute(text("DELETE FROM products WHERE id = CAST(:p AS uuid)"), {"p": str(product.id)})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:o AS uuid)"), {"o": owner})
    s.commit()
    s.close()


def _spend(new_session, ws, credits):
    s = new_session()
    caps.count_spend(s, ws, credits)
    s.commit()
    s.close()


def _left(new_session, ws):
    s = new_session()
    try:
        return caps.cap_left(s, ws)
    finally:
        s.close()


def test_the_cap_is_the_workspaces_own_else_its_products_and_none_elsewhere(new_session, world):
    assert _left(new_session, world.customer) == 100 and _left(new_session, world.plain) is None
    s = new_session()
    s.execute(text("UPDATE workspaces SET settings = settings || '{\"product_credit_cap\": 40}'::jsonb "
                   "WHERE id = CAST(:c AS uuid)"), {"c": world.customer})
    s.commit()
    s.close()
    assert _left(new_session, world.customer) == 40


def test_each_metered_call_counts_against_the_cap(new_session, world, monkeypatch):
    monkeypatch.setattr("services.credits.alerts.check_alerts", lambda db, ws: None)
    monkeypatch.setattr(caps, "_notify_owner", lambda *a: None)
    s = new_session()
    s.execute(text(
        "INSERT INTO llm_usage (workspace_id, model_id, provider, tier, request_type, charge_class, input_tokens, "
        "output_tokens, total_tokens, input_cost, output_cost, total_cost, is_byok, status, created_at) VALUES "
        "(CAST(:ws AS uuid), 'claude-haiku-5-5', 'anthropic', 'direct', 'chat', 'customer', 100, 10, 110, 0.2, 0, "
        "0.2, false, 'success', (now() AT TIME ZONE 'utc'))"), {"ws": world.customer})
    s.commit()
    meter.meter_pending(s)
    spent = caps.spent_this_month(s, world.customer)
    s.close()
    assert spent == routing.credits_for(0.2) and _left(new_session, world.customer) == 100 - spent


def test_the_gate_budgets_the_cap_left_and_stops_at_it_while_the_owner_has_credits(new_session, world):
    state = asks.AskState(ask_id="cap-ask", lane="chat", workspace_id=world.customer)
    gate._check_balance(world.customer, state)
    assert state.budget_credits == 100 and state.capped
    state.cost_usd = 1.0  # well past 100 credits
    with pytest.raises(routing.WorkspaceCapReachedError):
        gate._within_budget(state)
    _spend(new_session, world.customer, 100)
    with pytest.raises(routing.WorkspaceCapReachedError) as stopped:
        gate._check_balance(world.customer, asks.AskState(ask_id="cap-ask-2", lane="chat", workspace_id=world.customer))
    assert stopped.value.error_code == "workspace_cap_reached" and routing.is_credit_stop(stopped.value)
    plain = asks.AskState(ask_id="plain-ask", lane="chat", workspace_id=world.plain)
    gate._check_balance(world.plain, plain)
    assert plain.budget_credits == 1000 and not plain.capped


def test_background_work_and_a_live_call_dont_start_at_the_cap(new_session, world):
    from modules.voice.live_billing import has_credits

    s = new_session()
    try:
        assert routing.background_allowed(s, world.customer) and has_credits(s, world.customer)
        caps.count_spend(s, world.customer, 100)
        s.commit()
        assert not routing.background_allowed(s, world.customer) and not has_credits(s, world.customer)
        assert routing.background_allowed(s, world.plain)
    finally:
        s.close()


def test_the_owner_is_told_once_at_80_percent_and_once_at_the_cap(new_session, world, monkeypatch):
    told = []
    monkeypatch.setattr(caps, "_notify_owner", lambda db, ws, level, spent, cap: told.append((level, spent, cap)))
    s = new_session()
    try:
        for credits in (50, 30, 5, 15, 10):
            caps.count_spend(s, world.customer, credits)
            s.commit()
            caps.check_cap_notices(s, world.customer)
    finally:
        s.close()
    assert told == [(caps.LEVEL_NEAR, 80, 100), (caps.LEVEL_REACHED, 100, 100)]


def test_the_products_key_sets_a_workspaces_own_cap(new_session, world):
    from api.products import router
    from core.database.database import get_db

    app = FastAPI()
    app.include_router(router)

    def db():
        s = new_session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = db
    client, auth = TestClient(app), {"Authorization": f"Bearer {world.key}"}
    url = f"/api/products/{world.product}/workspaces/{world.customer}"
    _spend(new_session, world.customer, 25)
    set_ = client.patch(url, json={"credit_cap": 500}, headers=auth).json()
    assert (set_["credit_cap"], set_["monthly_cap"], set_["spent_this_month"]) == (500, 500, 25)
    cleared = client.patch(url, json={"credit_cap": None}, headers=auth).json()
    assert (cleared["credit_cap"], cleared["monthly_cap"]) == (None, 100)
    assert client.patch(url, json={"credit_cap": 0}, headers=auth).status_code == 422
    assert client.patch(f"/api/products/{world.product}/workspaces/{world.plain}", json={"credit_cap": 5},
                        headers=auth).status_code == 404


def test_a_capped_turn_reports_its_own_code():
    from consumers.chatbot.turn_errors import describe_turn_error

    error = describe_turn_error(routing.WorkspaceCapReachedError())
    assert error.code == "workspace_cap_reached" and "monthly limit" in error.message


def test_of_two_runs_that_see_the_same_new_level_only_one_tells_the_owner(new_session, world, monkeypatch):
    told = []
    monkeypatch.setattr(caps, "_notify_owner", lambda db, ws, level, spent, cap: told.append(level))
    _spend(new_session, world.customer, 85)
    late = new_session()
    try:
        assert late.execute(caps._SPENT, {"ws": world.customer}).first()[1] == 0  # it read the row at level 0
        s = new_session()
        assert caps.check_cap_notices(s, world.customer) == caps.LEVEL_NEAR  # the other run tells the owner
        s.close()
        # The late run now makes the same move: the conditional update matches nothing, so it sends nothing.
        assert late.execute(caps._NOTIFIED, {"ws": world.customer, "level": caps.LEVEL_NEAR}).rowcount == 0
        late.rollback()
    finally:
        late.close()
    assert told == [caps.LEVEL_NEAR]


def test_a_changed_cap_starts_the_months_notices_again(new_session, world, monkeypatch):
    from api.products import router
    from core.database.database import get_db

    told = []
    monkeypatch.setattr(caps, "_notify_owner", lambda db, ws, level, spent, cap: told.append((level, cap)))
    _spend(new_session, world.customer, 100)
    s = new_session()
    caps.check_cap_notices(s, world.customer)
    s.close()
    app = FastAPI()
    app.include_router(router)

    def db():
        session = new_session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = db
    TestClient(app).patch(f"/api/products/{world.product}/workspaces/{world.customer}", json={"credit_cap": 120},
                          headers={"Authorization": f"Bearer {world.key}"})
    _spend(new_session, world.customer, 20)
    s = new_session()
    caps.check_cap_notices(s, world.customer)
    s.close()
    assert told == [(caps.LEVEL_REACHED, 100), (caps.LEVEL_REACHED, 120)]


def test_a_new_month_starts_with_nothing_spent_and_no_notice_sent(new_session, world):
    s = new_session()
    s.execute(text("INSERT INTO product_workspace_spend (workspace_id, month, credits, notified_level) VALUES "
                   "(CAST(:w AS uuid), CAST(date_trunc('month', now() AT TIME ZONE 'utc') - interval '1 month' AS date), "
                   "100, 2)"), {"w": world.customer})
    s.commit()
    s.close()
    assert _left(new_session, world.customer) == 100  # last month's spend and notices don't count this month
