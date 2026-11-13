"""PRD-274 Wave 5d (decisions 24 and 25): a product's owner workspace pays for the product's workspaces.

A product's workspace has no credits, plan or trial of its own. The credit gate reads its owner's plan and balance;
the meter debits its calls from the owner's balance and records the workspace each was for
(``credit_ledger.sub_tenant_id``), which outlives the workspace's purge; the alerts are the owner's; its own
summary says its product pays and shows none of the owner's figures; the owner's usage lists what each of the
product's workspaces spent. An admin grant to it is refused (409), and it is never given a trial.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from config import config
from core.models.credits import CreditLedgerEntry
from core.models.products import Product
from services.credits import alerts, gate, meter, payer, routing, summary, trial
from services.credits.ledger import BUCKET_MONTHLY, KIND_GRANT, Entry, read_balance, record
from services.products import mint_key


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    Product.__table__.create(eng, checkfirst=True)  # a test database built before these
    with eng.begin() as conn:
        conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS reuse_existing_key BOOLEAN NOT NULL "
                          "DEFAULT true"))
        conn.execute(text("ALTER TABLE credit_ledger ADD COLUMN IF NOT EXISTS sub_tenant_id UUID"))
    yield eng
    eng.dispose()


@pytest.fixture(autouse=True)
def hosted(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "BILLING_FREE_OWNER_EMAILS", frozenset())


def _sql(s, statement, **params):
    s.execute(text(statement), params)
    s.commit()


@pytest.fixture
def world(new_session):
    """An owner workspace on Pro with 1,000 monthly credits, its product, the product's workspace, and a
    plain workspace; all swept after."""
    owner, customer, plain = (str(uuid.uuid4()) for _ in range(3))
    key = mint_key()
    s = new_session()
    _sql(s, "INSERT INTO workspaces (id, name, plan) VALUES (CAST(:o AS uuid), 'w5d-owner', 'pro'), "
            "(CAST(:p AS uuid), 'w5d-plain', 'pro')", o=owner, p=plain)
    product = Product(slug=f"w5d-{owner[:8]}", name="Support desk", owner_workspace_id=uuid.UUID(owner),
                      key_hash=key.key_hash, key_prefix=key.key_prefix)
    s.add(product)
    s.commit()
    _sql(s, "INSERT INTO workspaces (id, name, plan, settings) VALUES (CAST(:c AS uuid), 'w5d-customer', 'basic', "
            "CAST(:settings AS jsonb))", c=customer, settings=f'{{"product_id": "{product.id}"}}')
    record(s, owner, Entry(kind=KIND_GRANT, source_type="test_grant", source_ref=f"w5d:{owner}", credits=1000,
                           bucket=BUCKET_MONTHLY, actor="test"))
    s.commit()
    s.close()
    yield SimpleNamespace(owner=owner, customer=customer, plain=plain, product=str(product.id))
    s = new_session.sweep()
    for ws in (customer, plain, owner):
        for table in ("llm_usage", "credit_ledger", "credit_balances", "audit_logs"):
            s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})  # noqa: S608
    s.execute(text("DELETE FROM workspaces WHERE id = ANY(CAST(:ids AS uuid[]))"), {"ids": [customer, plain]})
    s.execute(text("DELETE FROM products WHERE id = CAST(:p AS uuid)"), {"p": str(product.id)})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:o AS uuid)"), {"o": owner})
    s.commit()
    s.close()


def _usage(new_session, ws, cost=0.5) -> int:
    s = new_session()
    try:
        usage_id = s.execute(text(
            "INSERT INTO llm_usage (workspace_id, model_id, provider, tier, request_type, charge_class, input_tokens, "
            "output_tokens, total_tokens, input_cost, output_cost, total_cost, is_byok, status, created_at) VALUES "
            "(CAST(:ws AS uuid), 'claude-haiku-5-5', 'anthropic', 'direct', 'chat', 'customer', 100, 10, 110, :c, 0, "
            ":c, false, 'success', (now() AT TIME ZONE 'utc')) RETURNING id"), {"ws": ws, "c": cost}).scalar()
        s.commit()
        return usage_id
    finally:
        s.close()


def test_the_payer_is_the_products_owner_and_any_other_workspace_pays_for_itself(new_session, world):
    s = new_session()
    try:
        assert payer.paying_workspaces(s, [world.customer, world.plain, world.owner]) == {
            world.customer: world.owner, world.plain: world.plain, world.owner: world.owner}
        assert payer.is_paid_by_its_product(s, world.customer) and not payer.is_paid_by_its_product(s, world.owner)
    finally:
        s.close()


def test_the_gate_reads_the_owners_plan_and_balance(new_session, world):
    s = new_session()
    try:
        funding = routing.workspace_funding(s, world.customer)
        assert funding.funded_by == routing.FUNDED_BY_CREDITS and funding.spendable == 1000
        assert not funding.own_keys_plan  # the owner's Pro, not the workspace's own Basic
    finally:
        s.close()
    gate.before_platform_call(workspace_id=world.customer, is_byok=False, purpose="chat")  # the owner can pay
    s = new_session()
    _sql(s, "UPDATE credit_balances SET monthly = 0 WHERE workspace_id = CAST(:o AS uuid)", o=world.owner)
    s.close()
    with pytest.raises(routing.CreditsExhaustedError):
        gate.before_platform_call(workspace_id=world.customer, is_byok=False, purpose="chat")


def test_the_meter_debits_the_owner_and_records_the_workspace_it_was_for(new_session, world, monkeypatch):
    sent = []
    monkeypatch.setattr(alerts, "check_alerts", lambda db, ws: sent.append(ws))
    usage_id = _usage(new_session, world.customer)
    s = new_session()
    try:
        assert meter.meter_pending(s) >= 1
        rows = s.query(CreditLedgerEntry).filter(CreditLedgerEntry.source_ref == str(usage_id)).all()
        assert [(str(r.workspace_id), str(r.sub_tenant_id)) for r in rows] == [(world.owner, world.customer)]
        assert read_balance(s, world.owner).monthly == 1000 + rows[0].credits
        assert read_balance(s, world.customer).spendable() == 0  # it never had or spent credits of its own
    finally:
        s.close()
    assert world.owner in sent and world.customer not in sent


def test_the_owners_usage_lists_each_product_workspace_and_outlives_its_purge(new_session, world, monkeypatch):
    monkeypatch.setattr(alerts, "check_alerts", lambda db, ws: None)
    _usage(new_session, world.customer, cost=0.3)
    s = new_session()
    meter.meter_pending(s)
    _sql(s, "DELETE FROM llm_usage WHERE workspace_id = CAST(:c AS uuid)", c=world.customer)
    _sql(s, "DELETE FROM workspaces WHERE id = CAST(:c AS uuid)", c=world.customer)  # its purge's last step
    page = summary.ledger_page(s, world.owner, page=1, page_size=10)
    s.close()
    [spent] = page["usage"]["by_product_workspace"]
    assert spent["workspace_id"] == world.customer and spent["calls"] == 1 and spent["credits"] > 0


def test_the_alerts_are_the_owners(new_session, world, monkeypatch):
    told = []
    monkeypatch.setattr(alerts, "_notify", lambda db, ws, level, spendable: told.append((ws, level)))
    s = new_session()
    try:
        _sql(s, "UPDATE credit_balances SET monthly = 0 WHERE workspace_id = CAST(:o AS uuid)", o=world.owner)
        assert alerts.check_alerts(s, world.customer) == alerts.LEVEL_OUT
    finally:
        s.close()
    assert told == [(world.owner, alerts.LEVEL_OUT)]


def test_its_summary_says_its_product_pays_and_shows_none_of_the_owners_figures(new_session, world):
    from core.models.workspaces import Workspace

    s = new_session()
    try:
        theirs = summary.billing_summary(s, s.get(Workspace, uuid.UUID(world.customer)))
        owners = summary.billing_summary(s, s.get(Workspace, uuid.UUID(world.owner)))
    finally:
        s.close()
    assert theirs["paid_by_product"] is True and theirs["balance"] is None and theirs["plan"] is None
    assert theirs["paused"] is False and "1000" not in str(theirs)
    assert owners["paid_by_product"] is False and owners["balance"]["monthly"] == 1000


def test_it_gets_no_trial_and_no_admin_grant(new_session, world, monkeypatch):
    from api.admin_credits import _refuse_a_products_workspace
    from core.models.workspaces import Workspace

    monkeypatch.setattr(config, "TRIAL_ENABLED", True)
    s = new_session()
    try:
        assert trial.grant_trial(s, world.customer, owner_id=None) is None
        with pytest.raises(HTTPException) as refused:
            _refuse_a_products_workspace(s, s.get(Workspace, uuid.UUID(world.customer)))
        assert refused.value.status_code == 409 and world.owner in refused.value.detail
        _refuse_a_products_workspace(s, s.get(Workspace, uuid.UUID(world.plain)))  # any other: no refusal
    finally:
        s.close()
