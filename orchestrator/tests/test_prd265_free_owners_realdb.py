"""PRD-265 — a free owner's workspace against real Postgres (``@integration``).

The owner's email is in ``BILLING_FREE_OWNER_EMAILS``: with no credits at all the
workspace can still use Automatos's AI keys, the meter records its calls without a
debit, and the Billing tab shows it as unmetered and never paused. A workspace of
anyone else, with the same zero balance, is paused.
"""
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core.database.database import get_database_url  # noqa: E402
from services.credits.meter import PendingCall, meter_call  # noqa: E402
from services.credits.routing import (  # noqa: E402
    FUNDED_BY_ALLOWANCE,
    FUNDED_BY_CREDITS,
    can_pay,
    is_free_workspace,
    workspace_funding,
)
from services.credits.summary import billing_summary  # noqa: E402

pytestmark = pytest.mark.integration

FREE_EMAIL = "Founder-{tag}@Example.test"


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for tbl in ("workspaces", "users", "credit_ledger", "credit_balances"):
                c.execute(text(f"SELECT 1 FROM {tbl} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"free-owner suite needs Postgres with the PRD-265 schema: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def owners(engine, new_session, monkeypatch):
    """Two workspaces with no credits: one owned by a free owner, one by anyone else."""
    tag = uuid.uuid4().hex[:8]
    free_email, other_email = FREE_EMAIL.format(tag=tag), f"someone-{tag}@example.test"
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "BILLING_FREE_OWNER_EMAILS", frozenset({free_email.lower()}))
    s = new_session()
    made = {}
    for name, email in (("free", free_email), ("other", other_email)):
        uid = s.execute(text("INSERT INTO users (username, email, clerk_user_id, is_active) "
                             "VALUES (:e, :e, :c, true) RETURNING id"),
                        {"e": email, "c": f"user_free_{tag}_{name}"}).scalar()
        ws = str(uuid.uuid4())
        s.execute(text("INSERT INTO workspaces (id, name, slug, plan, owner_id, onboarding) "
                       "VALUES (CAST(:id AS uuid), :n, :n, 'pro', :o, CAST('{}' AS jsonb))"),
                  {"id": ws, "n": f"free-{tag}-{name}", "o": uid})
        made[name] = (ws, uid)
    s.commit()
    yield {name: ws for name, (ws, _) in made.items()}
    s = new_session.sweep()
    for ws, uid in made.values():
        s.execute(text("DELETE FROM credit_ledger WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": ws})
        s.execute(text("DELETE FROM credit_balances WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": ws})
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:ws AS uuid)"), {"ws": ws})
        s.execute(text("DELETE FROM users WHERE id = :id"), {"id": uid})
    s.commit()


def test_a_free_owner_uses_platform_ai_with_no_credits(owners, new_session):
    s = new_session()
    assert is_free_workspace(s, owners["free"]) and not is_free_workspace(s, owners["other"])
    free = workspace_funding(s, owners["free"])
    assert free.funded_by == FUNDED_BY_ALLOWANCE and can_pay(free, 10_000)
    other = workspace_funding(s, owners["other"])
    assert other.funded_by == FUNDED_BY_CREDITS and not can_pay(other, 1)


def test_the_meter_records_a_free_owners_call_without_a_debit(owners, new_session):
    s = new_session()
    call = PendingCall(usage_id=-1, workspace_id=owners["free"], model_id="claude-haiku-5-5",
                       provider="anthropic", tier=None, request_type="chat", input_tokens=1000,
                       output_tokens=500, total_cost=0.25, is_byok=False)
    assert meter_call(s, call, "pro", free=True) is False
    debits = s.execute(text("SELECT count(*) FROM credit_ledger WHERE workspace_id = CAST(:ws AS uuid)"),
                       {"ws": owners["free"]}).scalar()
    assert debits == 0


def test_the_billing_tab_shows_a_free_owner_unmetered_and_never_paused(owners, new_session):
    from core.models.workspaces import Workspace

    s = new_session()
    free = billing_summary(s, s.query(Workspace).get(owners["free"]))
    assert free["metered"] is False and free["paused"] is False
    other = billing_summary(s, s.query(Workspace).get(owners["other"]))
    assert other["metered"] is True and other["paused"] is True
