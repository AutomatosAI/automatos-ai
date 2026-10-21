"""PRD-265 against real Postgres (``@integration``): the trial, and the meter.

- The trial is granted once per person: a second workspace of the same owner gets
  none; the daily cap pauses new grants; a dev reset brings the trial back to full.
- The meter debits each customer call once, at its real cost × the markup, and
  marks it charged; own-key and overhead calls cost nothing; background work never
  spends the trial; an unpriced paid route is charged at the estimate.
- Balance alerts fire once per crossing and reset when credits are added.
- The nightly job records the day's absorbed overhead as a zero-credit row.

Skips cleanly when no Postgres with the ``prd265_credit_ledger`` schema is reachable
(CI runs it).
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core.database.database import get_database_url  # noqa: E402
from services.credits import alerts, ledger, meter, trial  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1 FROM credit_ledger LIMIT 1"))
            c.execute(text("SELECT ask_id FROM llm_usage LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"asks suite needs Postgres with prd265_credit_ledger applied: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def made(engine, new_session):
    rows = {"users": [], "workspaces": []}
    yield rows
    s = new_session.sweep()
    for ws in rows["workspaces"]:
        for table in ("llm_usage", "credit_ledger", "credit_balances"):
            s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    for uid in rows["users"]:
        s.execute(text("DELETE FROM users WHERE id = :id"), {"id": uid})
    s.commit()


def _user(s, made) -> int:
    tag = uuid.uuid4().hex[:10]
    uid = s.execute(text("INSERT INTO users (username, email, clerk_user_id, is_active) "
                         "VALUES (:e, :e, :c, true) RETURNING id"),
                    {"e": f"asks-{tag}@example.test", "c": f"user_asks_{tag}"}).scalar()
    s.commit()
    made["users"].append(uid)
    return uid


def _workspace(s, made, owner=None, plan="basic") -> str:
    ws = str(uuid.uuid4())
    s.execute(text("INSERT INTO workspaces (id, name, slug, owner_id, plan) VALUES (CAST(:id AS uuid), :n, :n, :o, :p)"),
              {"id": ws, "n": f"asks-{ws[:8]}", "o": owner, "p": plan})
    s.commit()
    made["workspaces"].append(ws)
    return ws


def _usage(s, ws, *, model="claude-haiku-5-5", provider="anthropic", status="success", cost=0.01, byok=False,
           request_type="chat", charge_class="customer", tokens=(100, 10)) -> int:
    usage_id = s.execute(text(
        "INSERT INTO llm_usage (workspace_id, model_id, provider, tier, request_type, charge_class, "
        "input_tokens, output_tokens, total_tokens, input_cost, output_cost, total_cost, is_byok, status, created_at) "
        "VALUES (CAST(:ws AS uuid), :m, :p, 'direct', :rt, :cc, :i, :o, :t, :c, 0, :c, :b, :st, "
        "(now() AT TIME ZONE 'utc')) RETURNING id"
    ), {"ws": ws, "m": model, "p": provider, "rt": request_type, "cc": charge_class, "i": tokens[0],
        "o": tokens[1], "t": sum(tokens), "c": cost, "b": byok, "st": status}).scalar()
    s.commit()
    return usage_id


def _charge_class(s, usage_id):
    return s.execute(text("SELECT charge_class FROM llm_usage WHERE id = :id"), {"id": usage_id}).scalar()


# --------------------------------------------------------------------------- #
# The trial
# --------------------------------------------------------------------------- #


def test_the_trial_is_granted_once_per_person(made, new_session, monkeypatch):
    monkeypatch.setattr(config, "TRIAL_ENABLED", True)
    monkeypatch.setattr(config, "TRIAL_GLOBAL_DAILY_CREDITS", 10_000_000)
    s = new_session()
    owner = _user(s, made)
    first, second = _workspace(s, made, owner), _workspace(s, made, owner)
    assert trial.grant_trial(s, first, owner_id=owner) == config.TRIAL_CREDITS
    s.commit()
    assert trial.grant_trial(s, second, owner_id=owner) is None
    s.commit()
    assert ledger.read_balance(s, first).trial == config.TRIAL_CREDITS
    assert ledger.read_balance(s, second).trial == 0


def test_the_daily_cap_pauses_new_grants(made, new_session, monkeypatch):
    monkeypatch.setattr(config, "TRIAL_ENABLED", True)
    monkeypatch.setattr(config, "TRIAL_GLOBAL_DAILY_CREDITS", 0)
    s = new_session()
    owner = _user(s, made)
    assert trial.grant_trial(s, _workspace(s, made, owner), owner_id=owner) is None


def test_a_dev_reset_brings_the_trial_back_to_full(made, new_session, monkeypatch):
    monkeypatch.setattr(config, "TRIAL_ENABLED", True)
    monkeypatch.setattr(config, "TRIAL_GLOBAL_DAILY_CREDITS", 10_000_000)
    s = new_session()
    owner = _user(s, made)
    ws = _workspace(s, made, owner)
    trial.grant_trial(s, ws, owner_id=owner)
    ledger.record(s, ws, ledger.Entry(kind=ledger.KIND_DEBIT, source_type="usage", source_ref=uuid.uuid4().hex,
                                      credits=-120))  # some of the trial spent
    assert trial.reset_trial(s, ws, owner_id=owner) == config.TRIAL_CREDITS
    s.commit()
    assert ledger.read_balance(s, ws).trial == config.TRIAL_CREDITS


# --------------------------------------------------------------------------- #
# The meter
# --------------------------------------------------------------------------- #


def _funded(s, made, credits=500, plan="basic", bucket=ledger.BUCKET_TRIAL):
    ws = _workspace(s, made, plan=plan)
    ledger.record(s, ws, ledger.Entry(kind=ledger.KIND_GRANT, source_type="test_grant", source_ref=f"{ws}:{bucket}",
                                      credits=credits, bucket=bucket))
    s.commit()
    return ws


@pytest.fixture
def metered(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "CREDITS_MARKUP", 1.2)
    monkeypatch.setattr(alerts, "_notify", lambda db, ws, level, spendable: None)


def test_a_call_is_debited_once_at_its_cost_times_the_markup(made, new_session, metered):
    s = new_session()
    ws = _funded(s, made)
    usage_id = _usage(s, ws, cost=0.10)  # 10 cents × 1.2 = 12 credits
    meter.meter_pending(s)
    meter.meter_pending(s)
    assert ledger.read_balance(s, ws).trial == 500 - 12
    assert _charge_class(s, usage_id) == "charged"
    rows = s.execute(text("SELECT credits, band, charged_credits, actual_cost_usd FROM credit_ledger "
                          "WHERE source_type = 'usage' AND source_ref = :r"), {"r": str(usage_id)}).fetchall()
    assert len(rows) == 1 and rows[0][0] == -12 and rows[0][1] == "metered" and rows[0][2] == 12
    assert float(rows[0][3]) == pytest.approx(0.10)


def test_own_key_and_overhead_calls_cost_nothing(made, new_session, metered):
    s = new_session()
    ws = _funded(s, made)
    own = _usage(s, ws, cost=0.50, byok=True)
    overhead = _usage(s, ws, cost=0.07, request_type="memory_distill", charge_class="overhead")
    meter.meter_pending(s)
    assert ledger.read_balance(s, ws).trial == 500
    assert _charge_class(s, own) == "charged"       # looked at, nothing to charge
    assert _charge_class(s, overhead) == "overhead"  # never the meter's
    meter.record_overhead(s, datetime.now(timezone.utc).date())
    row = s.execute(text("SELECT credits, actual_cost_usd FROM credit_ledger WHERE workspace_id = CAST(:ws AS uuid) "
                         "AND source_type = 'overhead_day'"), {"ws": ws}).fetchall()
    assert len(row) == 1 and row[0][0] == 0 and float(row[0][1]) == pytest.approx(0.07)


def test_nvidia_is_free_and_an_unpriced_paid_route_is_not(made, new_session, metered):
    s = new_session()
    ws = _funded(s, made)
    _usage(s, ws, provider="nvidia", model="nvidia/nemotron", cost=0.0)
    meter.meter_pending(s)
    assert ledger.read_balance(s, ws).trial == 500
    _usage(s, ws, provider="openrouter", model="nobody/priced-this-model", cost=0.0, tokens=(10_000, 2_000))
    meter.meter_pending(s)
    assert ledger.read_balance(s, ws).trial < 500


def test_background_work_never_spends_the_trial(made, new_session, metered):
    s = new_session()
    ws = _funded(s, made, plan="pro")
    ledger.record(s, ws, ledger.Entry(kind=ledger.KIND_GRANT, source_type="test_grant", source_ref=f"{ws}:m",
                                      credits=100, bucket=ledger.BUCKET_MONTHLY))
    s.commit()
    _usage(s, ws, cost=0.10, request_type="heartbeat")
    meter.meter_pending(s)
    balance = ledger.read_balance(s, ws)
    assert (balance.trial, balance.monthly) == (500, 100 - 12)


def test_an_enterprise_allowance_is_never_debited(made, new_session, metered):
    s = new_session()
    ws = _funded(s, made, plan="enterprise")
    _usage(s, ws, cost=3.0)
    meter.meter_pending(s)
    assert ledger.read_balance(s, ws).trial == 500


def test_alerts_fire_once_per_crossing_and_reset_on_a_top_up(made, new_session, monkeypatch, metered):
    sent = []
    monkeypatch.setattr(alerts, "_notify", lambda db, ws, level, spendable: sent.append(level))
    monkeypatch.setattr(config, "TRIAL_CREDITS", 500)
    monkeypatch.setattr(config, "CREDITS_ALERT_LOW_SHARE", 0.20)
    s = new_session()
    ws = _funded(s, made, credits=150)
    _usage(s, ws, cost=0.50)                      # 60 credits: 90 left, under 100 → low
    meter.meter_pending(s)
    _usage(s, ws, cost=0.05)                      # still low: no second alert
    meter.meter_pending(s)
    _usage(s, ws, cost=1.00)                      # 120 credits: run out
    meter.meter_pending(s)
    assert sent == [alerts.LEVEL_LOW, alerts.LEVEL_OUT]
    ledger.record(s, ws, ledger.Entry(kind=ledger.KIND_ADJUST, source_type="test_grant", source_ref=f"{ws}:more",
                                      credits=1000, bucket=ledger.BUCKET_TRIAL))
    s.commit()
    assert alerts.check_alerts(s, ws) is None     # credits added: the level drops back, silently
    assert s.execute(text("SELECT alert_level FROM credit_balances WHERE workspace_id = CAST(:ws AS uuid)"),
                     {"ws": ws}).scalar() == alerts.LEVEL_OK
