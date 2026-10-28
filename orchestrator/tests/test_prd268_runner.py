"""PRD-268 US-C03d (§6.2.1, §6.9, FR-2, FR-3, FR-13, K8, K9, K14) — the check-in runner.

Against Postgres, inside a rolled-back transaction (the suite's ``db_session``, joined by
SAVEPOINT so the runner's commits stay inside it). The model is a scripted fake, the
context build is recorded, and Auto's signals are given; everything else is real: the
switches, ``background_allowed``, the spend guard, the caps, the run rows, and every tool
call through ``PlatformActionExecutor.execute`` in the lane, so a suggestion is recorded
by ``platform_suggest_next_step`` itself. Proven here:

* no signal: a ``silent`` run, no model built or called, no ``llm_usage`` row; the window
  is the last day, or starts where the last successful run read up to;
* signals: one model call whose prompt lists the signals, the open and recently hidden
  titles and "most check-ins should record nothing", writing one suggestion through the
  tool; every call carries the lane, the run and the run's successful reads, so a ticket a
  read returned can be cited and one nobody read cannot (the model may correct it); a tool
  that fails, arguments that are not JSON and calls past eight a turn are answered, and the
  run goes on; the web reads are neither offered nor run in a check-in;
* an agent's check-in: its heartbeat prompt and identity through ``ContextService`` in
  ``ContextMode.CHECK_IN``, on the heartbeat model, never the agent's own;
* the ceilings: four model calls at most; past $0.05 the run stops as an ``error``;
* refusals, each a run row and no model call: switched off (platform, workspace, an
  unreadable setting), the owner's pause, a trial workspace, a set spend ceiling, the
  model-runs cap; ``check_in`` is a background lane the trial never pays for.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from services.proactive import runner
from services.proactive.signals import Signal

ORCH = Path(__file__).resolve().parents[1]
SUGGEST = "platform_suggest_next_step"
LIST_TASKS = "platform_list_tasks"
MODEL = "claude-haiku-5-5"
SIGNAL_SUMMARY = "Ticket #0846 failed: Publish the October SEO report. The provider refused the key"
OPEN_TITLE = "Ticket #0123 waits for the owner's answer"
HIDDEN_TITLE = "VECTOR's report is late"
READ_TICKET = {"id": 9, "number": "#0912", "title": "Refresh the sitemap", "status": "failed"}


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here):
    the runner and the suggestion tool commit, and neither may take the seed."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _user(db, tag: str) -> int:
    name = f"prd268rn_{tag}_{uuid4().hex[:10]}"
    return int(db.execute(text(
        "INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
        {"u": name, "e": f"{name}@t.test", "c": name}).scalar())


def _workspace(db, owner: int, enabled: bool = True) -> UUID:
    ws = uuid4()
    db.execute(text(
        "INSERT INTO workspaces (id, name, slug, plan, owner_id, settings) VALUES (CAST(:id AS uuid), "
        "'prd268 runner', :slug, 'pro', :o, CAST(:settings AS jsonb))"),
        {"id": str(ws), "slug": f"prd268-runner-{ws.hex[:10]}", "o": owner,
         "settings": json.dumps({"proactive": {"enabled": enabled}})})
    return ws


def _agent(db, ws: UUID) -> int:
    return int(db.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
        "VALUES ('VECTOR', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(ws)}).scalar())


@pytest.fixture
def shop(db_session):
    """A workspace with check-ins switched on, its owner and an agent."""
    owner = _user(db_session, "owner")
    ws = _workspace(db_session, owner)
    shop = NS(db=db_session, ws=ws, owner=owner, agent=_agent(db_session, ws))
    db_session.commit()
    return shop


def _suggestion(shop, title: str, status: str, decided_at=None) -> None:
    now = datetime.now(timezone.utc)
    shop.db.execute(text(
        "INSERT INTO proactive_suggestions (id, workspace_id, user_id, source_type, title, proposed_action, "
        "subject_key, status, decided_at, created_at, last_seen_at) VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), "
        ":user, 'check_in', :title, CAST(:action AS jsonb), :key, :status, :decided, :now, :now)"),
        {"id": str(uuid4()), "ws": str(shop.ws), "user": shop.owner, "title": title,
         "action": json.dumps({"kind": "chat", "payload": {}}), "key": f"title:{title.lower()}",
         "status": status, "decided": decided_at, "now": now})
    shop.db.commit()


def _run_row(shop, *, outcome: str, model=None, window_until=None, agent=None, kind="workspace_check_in",
             created_at=None) -> None:
    shop.db.execute(text(
        "INSERT INTO proactive_runs (id, workspace_id, agent_id, kind, outcome, model, window_until, created_at) "
        "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), :agent, :kind, :outcome, :model, :until, "
        "COALESCE(CAST(:at AS timestamptz), NOW()))"),
        {"id": str(uuid4()), "ws": str(shop.ws), "agent": agent, "kind": kind, "outcome": outcome,
         "model": model, "until": window_until, "at": created_at})
    shop.db.commit()


def _runs(shop):
    from core.models.proactive import ProactiveRun

    shop.db.expire_all()
    return shop.db.query(ProactiveRun).filter(ProactiveRun.workspace_id == shop.ws).order_by(
        ProactiveRun.created_at).all()


def _the_run(shop, result):
    [row] = [run for run in _runs(shop) if run.id == result.run_id]
    return row


def _suggestions(shop):
    from core.models.proactive import ProactiveSuggestion

    shop.db.expire_all()
    return shop.db.query(ProactiveSuggestion).filter(ProactiveSuggestion.workspace_id == shop.ws,
                                                     ProactiveSuggestion.run_id.isnot(None)).all()


def _usage_rows(shop) -> int:
    return int(shop.db.execute(text("SELECT count(*) FROM llm_usage WHERE workspace_id = CAST(:w AS uuid)"),
                               {"w": str(shop.ws)}).scalar())


# ── The fakes ───────────────────────────────────────────────────────────────

def _call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _reply(*calls: dict, content: str = "") -> NS:
    return NS(content=content, tool_calls=list(calls), usage={"input_tokens": 1000, "output_tokens": 100})


def _suggest(ticket: str, title: str = "Ticket #0846 failed: the provider refused the key") -> dict:
    return _call(SUGGEST, {"title": title, "detail": "The October SEO report did not publish.",
                           "evidence": [{"type": "ticket", "id": ticket}], "proposed_action": {"kind": "chat"}})


class FakeLLM:
    """The heartbeat model, scripted: each call answers the next reply (the last repeats)."""

    def __init__(self, replies, cost: float = 0.001, fails: bool = False):
        self.config = NS(model=MODEL)
        self.replies, self.cost, self.fails, self.calls = list(replies), cost, fails, []

    async def generate_response(self, messages, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if self.fails:
            raise RuntimeError("provider refused the key (HTTP 401)")
        return self.replies[min(len(self.calls), len(self.replies)) - 1]

    def call_cost(self, input_tokens, output_tokens, reported_cost=None):
        return self.cost, "estimate"


def _signal(at: datetime) -> Signal:
    return Signal(kind="ticket_failed", subject_key="ticket:0846", ref=("ticket", "0846"), at=at,
                  summary=SIGNAL_SUMMARY)


@pytest.fixture
def wired(shop, monkeypatch):
    """The runner on the test's session; proactive mode shadow; the model, the context build
    and the signals scripted; the executor real, with its rate limiter and autonomy read stubbed."""
    from core.llm import manager
    from modules.context.result import ContextResult
    from modules.context.service import ContextService
    from modules.tools.discovery import platform_executor
    from modules.tools.discovery.action_registry import get_action_registry

    rig = NS(mode="shadow", signals=(), signal_reads=[], built=[], llm=FakeLLM([_reply()]), made_llm=[],
             executed=[], listed=0)

    def setting(category, key):
        if rig.mode is RuntimeError:
            raise RuntimeError("system_settings unreadable")
        return rig.mode if (category, key) == ("proactive", "mode") else None

    async def build_context(self, **kwargs):
        rig.built.append(kwargs)
        tools = [get_action_registry().get(name).to_openai_schema() for name in (LIST_TASKS, SUGGEST)]
        return ContextResult(system_prompt=f"SYSTEM\n{kwargs['task_description']}", tools=tools)

    def signals(db, workspace_id, since, until):
        rig.signal_reads.append((workspace_id, since, until))
        return rig.signals

    def heartbeat_llm(workspace_id, agent_id):
        rig.made_llm.append((workspace_id, agent_id))
        return rig.llm

    async def list_tasks(db, workspace_id, params):
        rig.listed += 1
        return {"success": True, "tasks": [dict(READ_TICKET)], "total": 1}

    original = platform_executor.PlatformActionExecutor.execute

    async def execute(self, action_name, params, caller_context=None):
        rig.executed.append((action_name, json.loads(json.dumps(caller_context))))
        return await original(self, action_name, params, caller_context)

    monkeypatch.setattr(manager, "read_system_setting", setting)
    monkeypatch.setattr(ContextService, "build_context", build_context)
    monkeypatch.setattr(runner, "_session", lambda: nullcontext(shop.db))
    monkeypatch.setattr(runner, "read_signals", signals)
    monkeypatch.setattr(runner, "heartbeat_llm", heartbeat_llm)
    monkeypatch.setattr(platform_executor, "list_board_tasks", list_tasks)
    monkeypatch.setattr(platform_executor.PlatformActionExecutor, "_full_autonomy", lambda self: False)
    monkeypatch.setattr(platform_executor.PlatformActionExecutor, "execute", execute)
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        yield rig


def _workspace_check_in(shop):
    return asyncio.run(runner.run_workspace_check_in(shop.ws))


def _agent_check_in(shop, hb_config=None):
    return asyncio.run(runner.run_agent_check_in(shop.agent, shop.ws, hb_config or {}))


# ── Auto's workspace check-in ───────────────────────────────────────────────

def test_no_signal_is_a_silent_run_with_no_model_call_and_no_llm_usage(shop, wired):
    result = _workspace_check_in(shop)

    assert (result.outcome, result.suggestions) == ("silent", 0)
    assert wired.made_llm == [] and wired.llm.calls == [] and wired.built == []
    assert _usage_rows(shop) == 0
    run = _the_run(shop, result)
    assert (run.kind, run.agent_id, run.outcome, run.model, run.tokens, float(run.cost_usd)) == (
        "workspace_check_in", None, "silent", None, 0, 0.0)
    assert run.signals == [] and run.window_until - run.window_since == timedelta(hours=24)
    [(read_ws, since, until)] = wired.signal_reads
    assert (read_ws, since, until) == (shop.ws, run.window_since, run.window_until)


def test_a_later_check_in_reads_from_where_the_last_successful_run_read_up_to(shop, wired):
    read_up_to = datetime.now(timezone.utc) - timedelta(hours=3)
    _run_row(shop, outcome="silent", window_until=read_up_to)
    _run_row(shop, outcome="error", window_until=read_up_to + timedelta(hours=2))

    _workspace_check_in(shop)

    [(_ws, since, _until)] = wired.signal_reads
    assert since == read_up_to


def test_signals_make_one_model_call_that_records_one_suggestion_through_the_tool(shop, wired):
    _suggestion(shop, OPEN_TITLE, "open")
    _suggestion(shop, HIDDEN_TITLE, "hidden", decided_at=datetime.now(timezone.utc) - timedelta(days=2))
    wired.signals = (_signal(datetime.now(timezone.utc) - timedelta(hours=1)),)
    wired.llm = FakeLLM([_reply(_suggest("0846"))])

    result = _workspace_check_in(shop)

    assert (result.outcome, result.suggestions) == ("suggested", 1)
    assert len(wired.llm.calls) == 1 and wired.made_llm == [(shop.ws, None)]
    [built] = wired.built
    assert built["mode"].value == "check_in" and built["agent"].name == "Auto"
    task = built["task_description"]
    assert SIGNAL_SUMMARY in task and OPEN_TITLE in task and HIDDEN_TITLE in task
    assert "most check-ins should record nothing" in task.lower()
    assert '"ref": {"type": "ticket", "id": "0846"}' in task
    [suggestion] = _suggestions(shop)
    assert (suggestion.run_id, suggestion.user_id, suggestion.source_type) == (result.run_id, shop.owner, "check_in")
    run = _the_run(shop, result)
    assert (run.outcome, run.suggestions, run.model, run.tokens) == ("suggested", 1, MODEL, 1100)
    assert float(run.cost_usd) == pytest.approx(0.001)
    assert [signal["subject_key"] for signal in run.signals] == ["ticket:0846"]


def test_every_call_carries_the_lane_the_run_and_the_reads_that_succeeded(shop, wired):
    wired.signals = (_signal(datetime.now(timezone.utc)),)
    wired.llm = FakeLLM([_reply(_call(LIST_TASKS, {})), _reply(_suggest("0912", "Ticket #0912 failed"))])

    result = _workspace_check_in(shop)

    assert (result.outcome, result.suggestions) == ("suggested", 1)
    [(first, read_context), (second, suggest_context)] = wired.executed
    assert (first, second) == (LIST_TASKS, SUGGEST)
    for context in (read_context, suggest_context):
        assert context["lane"] == "check_in" and context["run_id"] == str(result.run_id)
        assert "check_in_agent_id" not in context
    assert read_context["check_in_reads"] == [{"type": "ticket", "id": "0846"}]
    assert {"type": "ticket", "id": "0912"} in suggest_context["check_in_reads"]


def test_a_suggestion_citing_a_ticket_this_run_never_read_is_refused_and_the_model_told(shop, wired):
    wired.signals = (_signal(datetime.now(timezone.utc)),)
    wired.llm = FakeLLM([_reply(_suggest("0999", "Ticket #0999 failed")), _reply(content="Nothing new.")])

    result = _workspace_check_in(shop)

    assert (result.outcome, result.suggestions) == ("silent", 0) and _suggestions(shop) == []
    assert len(wired.llm.calls) == 2 and _the_run(shop, result).model == MODEL
    refusal = wired.llm.calls[1]["messages"][-1]
    assert refusal["role"] == "tool" and "no_provenance" in refusal["content"]


def test_a_tool_that_fails_is_answered_and_its_read_never_logged(shop, wired, monkeypatch):
    from modules.tools.discovery import platform_executor

    async def broken(db, workspace_id, params):
        raise RuntimeError("board unreachable")

    monkeypatch.setattr(platform_executor, "list_board_tasks", broken)
    wired.signals = (_signal(datetime.now(timezone.utc)),)
    wired.llm = FakeLLM([_reply(_call(LIST_TASKS, {})), _reply(_suggest("0846"))])

    result = _workspace_check_in(shop)

    assert (result.outcome, result.suggestions) == ("suggested", 1)
    [(_read, _), (_suggest_call, suggest_context)] = wired.executed
    assert suggest_context["check_in_reads"] == [{"type": "ticket", "id": "0846"}]


def test_arguments_that_are_not_json_run_nothing_and_are_replayed_as_an_empty_object(shop, wired):
    wired.signals = (_signal(datetime.now(timezone.utc)),)
    broken = {"id": "call_1", "type": "function", "function": {"name": LIST_TASKS, "arguments": "{not json"}}
    wired.llm = FakeLLM([_reply(broken), _reply()])

    _workspace_check_in(shop)

    assert wired.listed == 0 and wired.executed == []
    assistant, answer = wired.llm.calls[1]["messages"][-2:]
    assert assistant["tool_calls"][0]["function"]["arguments"] == "{}"
    assert "not valid JSON" in answer["content"]


def test_a_turn_runs_at_most_eight_calls_and_answers_the_rest(shop, wired):
    from services.proactive.loop import MAX_CALLS_PER_TURN, NOT_RUN

    wired.signals = (_signal(datetime.now(timezone.utc)),)
    wired.llm = FakeLLM([_reply(*(_call(LIST_TASKS, {}, f"call_{n}") for n in range(10))), _reply()])

    _workspace_check_in(shop)

    assert MAX_CALLS_PER_TURN == 8 and wired.listed == 8
    answers = [m for m in wired.llm.calls[1]["messages"] if m["role"] == "tool"]
    assert len(answers) == 10 and [NOT_RUN in m["content"] for m in answers].count(True) == 2


# ── An agent's check-in ─────────────────────────────────────────────────────

def test_an_agents_check_in_is_its_heartbeat_in_the_check_in_context_on_the_heartbeat_model(shop, wired):
    wired.llm = FakeLLM([_reply(_call(LIST_TASKS, {})), _reply(_suggest("0912", "Ticket #0912 failed"))])

    result = _agent_check_in(shop, {"prompt": "Check the SEO dashboards for a drop."})

    assert (result.outcome, result.suggestions) == ("suggested", 1)
    assert wired.signal_reads == [] and wired.made_llm == [(shop.ws, shop.agent)]
    [built] = wired.built
    assert built["mode"].value == "check_in" and built["agent"].id == shop.agent
    assert "Check the SEO dashboards for a drop." in built["task_description"]
    assert "most check-ins should record nothing" in built["task_description"].lower()
    assert all(context["check_in_agent_id"] == shop.agent for _name, context in wired.executed)
    [suggestion] = _suggestions(shop)
    assert (suggestion.source_type, suggestion.source_agent_id) == ("heartbeat", shop.agent)
    run = _the_run(shop, result)
    assert (run.kind, run.agent_id, run.outcome) == ("agent_check_in", shop.agent, "suggested")


def test_an_agent_check_in_for_an_agent_not_in_the_workspace_is_an_error(shop, wired):
    result = asyncio.run(runner.run_agent_check_in(shop.agent + 10_000, shop.ws, {}))
    assert (result.outcome, result.error_class) == ("error", runner.AGENT_NOT_FOUND)
    assert wired.made_llm == []


# ── The ceilings ────────────────────────────────────────────────────────────

def test_the_iteration_ceiling_stops_the_run_at_four_model_calls(shop, wired):
    wired.signals = (_signal(datetime.now(timezone.utc)),)
    wired.llm = FakeLLM([_reply(_call(LIST_TASKS, {}))])

    result = _workspace_check_in(shop)

    assert len(wired.llm.calls) == 4 and wired.listed == 4
    run = _the_run(shop, result)
    assert (run.outcome, run.tokens, run.error_class) == ("silent", 4 * 1100, None)


def test_past_the_cost_ceiling_the_run_stops_and_records_an_error(shop, wired):
    wired.signals = (_signal(datetime.now(timezone.utc)),)
    wired.llm = FakeLLM([_reply(_call(LIST_TASKS, {}))], cost=0.03)

    result = _workspace_check_in(shop)

    assert (result.outcome, result.error_class) == ("error", "cost_ceiling")
    assert len(wired.llm.calls) == 2 and wired.listed == 1  # the second turn's call never ran
    assert float(_the_run(shop, result).cost_usd) == pytest.approx(0.06)


def test_a_model_call_that_fails_records_an_error_named_by_its_class(shop, wired):
    wired.signals = (_signal(datetime.now(timezone.utc)),)
    wired.llm = FakeLLM([], fails=True)

    result = _workspace_check_in(shop)

    assert (result.outcome, result.error_class) == ("error", "RuntimeError")
    assert _suggestions(shop) == []


# ── Refused at the start: a run row, no model call ──────────────────────────

def _refused_before_the_model(shop, wired, result, outcome: str, reason: str) -> None:
    assert (result.outcome, result.error_class, result.suggestions) == (outcome, reason, 0)
    run = _the_run(shop, result)
    assert (run.outcome, run.error_class, run.model) == (outcome, reason, None)
    assert wired.signal_reads == [] and wired.made_llm == [] and wired.built == []


@pytest.mark.parametrize("mode", ["off", None, "sideways", RuntimeError])
def test_with_the_platform_mode_off_or_unreadable_the_check_in_is_skipped(shop, wired, mode):
    wired.mode = mode
    _refused_before_the_model(shop, wired, _workspace_check_in(shop), "skipped_off", "switched_off")


def test_a_workspace_that_did_not_switch_check_ins_on_is_skipped(shop, wired):
    shop.db.execute(text("UPDATE workspaces SET settings = CAST('{}' AS jsonb) WHERE id = CAST(:w AS uuid)"),
                    {"w": str(shop.ws)})
    shop.db.commit()
    _refused_before_the_model(shop, wired, _agent_check_in(shop), "skipped_off", "switched_off")


def test_an_owner_who_paused_suggestions_is_skipped(shop, wired):
    shop.db.execute(text(
        "INSERT INTO proactive_person_settings (workspace_id, user_id, paused) VALUES (CAST(:w AS uuid), :u, true)"),
        {"w": str(shop.ws), "u": shop.owner})
    shop.db.commit()
    _refused_before_the_model(shop, wired, _workspace_check_in(shop), "skipped_paused", "paused")


def test_a_trial_workspace_is_skipped(shop, wired, monkeypatch):
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "BILLING_FREE_OWNER_EMAILS", frozenset())
    shop.db.execute(text("INSERT INTO credit_balances (workspace_id, trial, monthly, topup) "
                         "VALUES (CAST(:w AS uuid), 500, 0, 0)"), {"w": str(shop.ws)})
    shop.db.commit()
    _refused_before_the_model(shop, wired, _workspace_check_in(shop), "skipped_credits", "trial_only")


def test_a_set_spend_ceiling_refuses_the_check_in(shop, wired, monkeypatch):
    from config import config
    from services import daily_spend_guard

    monkeypatch.setattr(config, "SPEND_CEILING_USD", 1.0)
    monkeypatch.setattr(daily_spend_guard, "spent_today_usd", lambda db, workspace_id: 2.5)
    _refused_before_the_model(shop, wired, _agent_check_in(shop), "skipped_cap", "spend_ceiling")


def test_the_49th_check_in_that_would_reach_the_model_in_a_day_is_refused(shop, wired):
    from modules.proactive import Caps

    assert Caps().max_model_runs_per_day == 48
    for _ in range(Caps().max_model_runs_per_day):
        _run_row(shop, outcome="silent", model=MODEL)
    _run_row(shop, outcome="silent")  # no model call: not counted
    _run_row(shop, outcome="suggested", model=MODEL, created_at=datetime.now(timezone.utc) - timedelta(hours=25))
    _refused_before_the_model(shop, wired, _workspace_check_in(shop), "skipped_cap", "max_model_runs_per_day")


# ── The lane, the model and the money ──────────────────────────────────────

def test_check_in_is_a_background_lane_the_trial_never_pays_for():
    from services.credits.asks import BACKGROUND_LANES, is_background_lane
    from services.credits.ledger import Balance
    from services.credits.routing import can_pay, funding_for

    assert "check_in" in BACKGROUND_LANES and is_background_lane("check_in")
    funding = funding_for("pro", Balance(trial=500), edition="saas", background=is_background_lane("check_in"))
    assert not can_pay(funding, 1)


def test_the_check_in_model_is_the_heartbeat_services_booked_as_check_in(monkeypatch):
    from core.llm import manager, output_budget
    from modules.tools.discovery.check_in_lane import CHECK_IN_LANE
    from services.credits.classification import call_class

    made = []
    monkeypatch.setattr(manager, "LLMManager", lambda **kwargs: made.append(kwargs) or NS(**kwargs))
    ws = uuid4()
    runner.heartbeat_llm(ws, 7)
    assert made == [{"service_name": "heartbeat", "workspace_id": str(ws), "agent_id": 7, "request_type": "check_in"}]
    assert runner.CHECK_IN_PURPOSE == CHECK_IN_LANE
    assert output_budget.DEFAULT_BUDGETS["check_in"] == 800
    assert call_class("check_in") == "customer"


def test_the_runner_never_goes_through_the_agent_factory():
    paths = sorted((ORCH / "services" / "proactive").glob("*.py"))
    assert {path.name for path in paths} >= {"runner.py", "loop.py", "prompts.py"}
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "agent_factory" not in source and "AgentFactory(" not in source, path.name


def test_the_web_reads_are_neither_offered_nor_run_in_a_check_in():
    from modules.tools.discovery.action_registry import get_action_registry
    from modules.tools.discovery.check_in_lane import CHECK_IN_REFUSAL, may_clear_in_a_check_in, offered_in_a_check_in
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(None, uuid4())
    for name in ("platform_web_fetch", "platform_web_search"):
        action = get_action_registry().get(name)
        assert (action.permission_level, action.category) == ("read", "web")
        assert not offered_in_a_check_in(action) and not may_clear_in_a_check_in(name)
        refused = executor.clear(name, {"url": "https://example.test", "query": "x"}, {"lane": "check_in"})
        assert refused == {"success": False, "error": CHECK_IN_REFUSAL}
    assert may_clear_in_a_check_in(LIST_TASKS) and offered_in_a_check_in(get_action_registry().get(LIST_TASKS))


# ── The read log ────────────────────────────────────────────────────────────

def _refs(action, params, answer, shown=None):
    from services.proactive.reads import refs_of_read

    return refs_of_read(action, params, answer, json.dumps(answer) if shown is None else shown)


def test_a_read_logs_the_tickets_and_ids_its_answer_names():
    answer = {"success": True, "tasks": [dict(READ_TICKET)], "data": {"report_id": "r-17"},
              "watch": {"id": "w-1", "title": "Weekly SEO digest"}, "missions": [{"id": 51}],
              "workspace_id": "5b0c7a1e"}
    refs = _refs(LIST_TASKS, {}, answer)
    assert {"type": "ticket", "id": "0912"} in refs and {"type": "task", "id": "9"} in refs
    assert {"type": "report", "id": "r-17"} in refs and {"type": "watch", "id": "w-1"} in refs
    assert {"type": "mission", "id": "51"} in refs
    assert not any(ref["type"] == "workspace" for ref in refs)


def test_an_argument_counts_only_when_the_answer_names_it_and_the_action_declares_it():
    found = _refs("platform_get_mission", {"mission_id": "m-5"}, {"success": True, "id": "m-5", "title": "Q4"})
    assert {"type": "mission", "id": "m-5"} in found
    matched_nothing = _refs("platform_get_mission", {"mission_id": "m-404"}, {"success": True, "items": []})
    assert matched_nothing == ()
    undeclared = _refs("platform_get_mission", {"mission_id": "m-5", "agent_id": 3},
                       {"success": True, "id": "m-5", "owner": {"id": 3}})
    assert {"type": "agent", "id": "3"} not in undeclared  # not an argument platform_get_mission declares


def test_only_what_the_model_was_shown_is_logged():
    answer = {"success": True, "tasks": [dict(READ_TICKET), {"id": 10, "number": "#0913", "title": "Later"}]}
    shown = json.dumps(answer)[: json.dumps(answer).index("#0913")]
    refs = _refs(LIST_TASKS, {}, answer, shown)
    assert {"type": "ticket", "id": "0912"} in refs and {"type": "ticket", "id": "0913"} not in refs


def test_a_ticket_id_is_a_task_and_free_form_blobs_are_never_walked():
    answer = {"success": True, "item": {"id": 4, "ticket_id": 9,
                                        "metadata": {"task": {"number": "#0846"}, "report_id": "r-1"}}}
    refs = _refs(LIST_TASKS, {}, answer)
    assert {"type": "task", "id": "9"} in refs and {"type": "ticket", "id": "0009"} not in refs
    assert {"type": "ticket", "id": "0846"} not in refs and {"type": "report", "id": "r-1"} not in refs


def test_a_failed_read_logs_nothing_and_words_inside_a_value_are_never_refs():
    assert _refs("platform_get_mission", {"mission_id": "m-5"}, {"success": False, "error": "nope"}) == ()
    assert _refs(LIST_TASKS, {}, {"success": True, "summary": "see ticket #0777 and report_id r-9"}) == ()
