"""PRD-265 US-004 — the gate in front of the platform's keys, and the ask's budget.

Credits are metered (every platform-funded call costs its real cost × markup,
``services/credits/meter.py``); the gate's job is to stop work that can't be paid
for. ``services.credits.gate.before_platform_call`` runs before every
``LLMManager`` call. Pinned here with the database parts replaced:

- own-key calls, calls with no workspace, the local edition and platform overhead
  pass without a look (overhead is absorbed, decision D-3);
- a workspace with nothing spendable stops with ``CreditsExhaustedError``;
- inside an ask, the balance read at its first call is the ask's budget, read once;
  the ask stops at the next call once its running cost (in credits) reaches it —
  a long mission ends when the credits do;
- an allowance plan (enterprise) is never stopped;
- own-key cost never counts against the budget;
- an ask scope opened inside another joins it; ``new=True`` starts its own;
- every usage row carries the ask and the charge class in scope.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from config import config
from services.credits import asks, gate
from services.credits.routing import FUNDED_BY_ALLOWANCE, FUNDED_BY_CREDITS, CreditsExhaustedError, Funding


class _Session:
    def close(self):
        pass


@pytest.fixture
def balance(monkeypatch):
    """Replace the gate's database read: what the workspace can spend."""
    state = {"funding": Funding(FUNDED_BY_CREDITS, ("trial",), 500), "reads": 0}
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "CREDITS_MARKUP", 1.0)
    monkeypatch.setattr("core.database.database.SessionLocal", lambda: _Session())

    def _funding(db, ws, background=False):
        state["reads"] += 1
        return state["funding"]

    monkeypatch.setattr(gate, "workspace_funding", _funding)
    return state


def _call(purpose="orchestrator", is_byok=False, workspace_id="ws-1"):
    gate.before_platform_call(workspace_id=workspace_id, is_byok=is_byok, purpose=purpose)


def _spent(cost_usd, is_byok=False):
    gate.after_platform_call(workspace_id="ws-1", is_byok=is_byok, purpose="orchestrator", cost_usd=cost_usd)


@pytest.mark.parametrize("kwargs", [
    {"is_byok": True}, {"workspace_id": None}, {"purpose": "memory_distill"}, {"purpose": "complexity_assessor"},
])
def test_own_keys_no_workspace_and_overhead_pass_untouched(balance, kwargs):
    balance["funding"] = Funding(FUNDED_BY_CREDITS, ("trial",), 0)  # would pause anything gated
    with asks.ask_scope("chat", workspace_id="ws-1"):
        _call(**kwargs)
    assert balance["reads"] == 0


def test_the_local_edition_passes(balance, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    balance["funding"] = Funding(FUNDED_BY_CREDITS, ("trial",), 0)
    with asks.ask_scope("chat"):
        _call()
    assert balance["reads"] == 0


def test_nothing_spendable_pauses_the_work(balance):
    balance["funding"] = Funding(FUNDED_BY_CREDITS, ("trial",), 0, own_keys_plan=True)
    with pytest.raises(CreditsExhaustedError) as stop:
        with asks.ask_scope("chat", workspace_id="ws-1"):
            _call()
    assert stop.value.own_keys_plan is True
    with pytest.raises(CreditsExhaustedError):
        _call()  # outside any ask too


def test_an_ask_reads_the_balance_once_and_keeps_it_as_its_budget(balance):
    with asks.ask_scope("chat", workspace_id="ws-1") as ask:
        _call()
        _call()
        _call(purpose="planner")
    assert balance["reads"] == 1
    assert ask.checked and ask.funded_by == FUNDED_BY_CREDITS and ask.budget_credits == 500


def test_a_long_run_stops_when_its_cost_reaches_the_balance(balance):
    balance["funding"] = Funding(FUNDED_BY_CREDITS, ("trial",), 100)  # $1.00 of credits at markup 1.0
    with asks.ask_scope("mission", workspace_id="ws-1"):
        _call()
        _spent(0.60)
        _call()  # 60 of 100 credits used — still going
        _spent(0.45)
        with pytest.raises(CreditsExhaustedError):
            _call()  # 105 ≥ 100 — the next call stops, instead of overdrawing further


def test_the_markup_counts_against_the_budget(balance, monkeypatch):
    monkeypatch.setattr(config, "CREDITS_MARKUP", 2.0)
    balance["funding"] = Funding(FUNDED_BY_CREDITS, ("trial",), 100)
    with asks.ask_scope("chat", workspace_id="ws-1"):
        _call()
        _spent(0.50)  # 50 cents × 2 = 100 credits
        with pytest.raises(CreditsExhaustedError):
            _call()


def test_an_allowance_plan_is_never_stopped(balance):
    balance["funding"] = Funding(FUNDED_BY_ALLOWANCE)
    with asks.ask_scope("chat", workspace_id="ws-1"):
        _call()
        _spent(500.0)
        _call()


def test_own_key_cost_never_counts_against_the_budget(balance):
    balance["funding"] = Funding(FUNDED_BY_CREDITS, ("trial",), 10)
    with asks.ask_scope("chat", workspace_id="ws-1") as ask:
        _call()
        _spent(50.0, is_byok=True)
        _call()
    assert ask.cost_usd == 0.0


# --------------------------------------------------------------------------- #
# The ask scope
# --------------------------------------------------------------------------- #


def test_a_scope_inside_an_ask_joins_it_unless_new():
    with asks.ask_scope("chat", workspace_id="ws-1") as outer:
        with asks.ask_scope("mission") as joined:
            assert joined is outer
            assert asks.current_ask_id() == outer.ask_id
        with asks.ask_scope("recipe", new=True) as own:
            assert own is not outer and asks.current_ask_id() == own.ask_id
        assert asks.current_ask_id() == outer.ask_id
    assert asks.current_ask() is None


def test_background_lanes_are_marked():
    assert asks.is_background_lane("heartbeat") and asks.is_background_lane("scheduled_task")
    assert not asks.is_background_lane("chat") and not asks.is_background_lane(None)


def test_an_escaping_error_leaves_nothing_behind():
    with pytest.raises(RuntimeError):
        with asks.ask_scope("mission", workspace_id="ws-1"):
            raise RuntimeError("a tool failed")
    assert asks.current_ask() is None  # the registry is cleaned up either way


def test_runs_as_an_ask_names_the_workspace_from_the_arguments():
    seen = {}

    class _Ticker:
        @asks.runs_as_an_ask("heartbeat", background=True)
        async def tick(self, workspace_id, hb_config, result):
            ask = asks.current_ask()
            seen.update(lane=ask.lane, workspace_id=ask.workspace_id, background=ask.background)

    asyncio.run(_Ticker().tick("ws-9", {}, {}))
    assert seen == {"lane": "heartbeat", "workspace_id": "ws-9", "background": True}


def test_the_usage_row_is_stamped_with_the_ask_and_the_charge_class():
    from core.llm.usage_context import usage_scope
    from core.models.core import _stamp_the_ask

    row = SimpleNamespace(ask_id=None, charge_class=None)
    with asks.ask_scope("chat", workspace_id="ws-1") as ask, usage_scope(charge_class="customer"):
        _stamp_the_ask(None, None, row)
    assert (row.ask_id, row.charge_class) == (ask.ask_id, "customer")
    outside = SimpleNamespace(ask_id=None, charge_class=None)
    _stamp_the_ask(None, None, outside)
    assert (outside.ask_id, outside.charge_class) == (None, None)


def test_the_manager_records_each_call_under_its_charge_class():
    """``LLMManager._record_usage`` writes the row inside ``usage_scope(charge_class=…)``,
    named by ``classification.call_class`` of the manager's own purpose."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "core" / "llm" / "manager.py").read_text()
    start = src.index("def _record_usage(")
    body = src[start:start + 2500]
    assert "usage_scope(charge_class=call_class(self._credit_purpose()))" in body
