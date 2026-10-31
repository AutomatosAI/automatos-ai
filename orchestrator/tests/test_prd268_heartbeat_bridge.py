"""PRD-268 US-C04b (§6.2.1, FR-4, K7, K10) — the heartbeat bridge: check-ins beside or instead of today's ticks.

The scheduler runs ``services.proactive.heartbeat_bridge``'s jobs, not the ticks. Proven here:

* **the paths** (pure: the ticks, the check-ins and the bell recorded): off runs today's tick
  only; shadow runs it, then the check-in, which rings nothing; live runs the check-in only,
  for ``check_in`` agents (stated, or unset without ``auto_act``) and for the orchestrator at
  silent or notify; ``act`` agents, Claude Code agents and the orchestrator at act_notify or
  autonomous run today's tick. A mode that cannot be read is off; a check-in runs within the
  heartbeat's active hours; a failing check-in never reaches the scheduler or a shadow's tick;
* **the scheduler** passes the bridge's jobs, bound to the service, in place of the ticks;
* **the payload**: ``mode`` is ``check_in`` or ``act``; the GET echoes it, else its effective
  default; a PUT that sends it states it, and one that leaves it out keeps the block's own;
* **live, against Postgres** (the suite's rolled-back transaction, joined by SAVEPOINT; the
  model scripted, every tool call through the real executor and suggestion tool, the
  notification dispatcher recorded since the CI database has no notifications table): a
  ``check_in`` agent with ``report_to=auto`` creates zero board tickets and files no per-tick
  report; it writes a ``heartbeat_complete`` bell row only when its run made a suggestion or
  started an error streak. Auto's workspace check-in rings as Auto at notify and never at
  silent; a bell that fails never reaches the heartbeat.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import text

from services.proactive import heartbeat_bell, heartbeat_bridge
from services.proactive.heartbeat_bridge import HeartbeatJobs, agent_job, orchestrator_job

WS = "11111111-1111-1111-1111-111111111111"
AGENT = 578
# Active all day, so a real active-hours check never depends on the clock.
ALL_DAY = {"inherit_active_hours": False, "active_hours_start": "00:00", "active_hours_end": "23:59"}
CHECK_IN_HB = {**ALL_DAY, "enabled": True, "mode": "check_in", "report_to": "auto", "interval_minutes": 60,
               "prompt": "Check the SEO tickets."}


# ── The paths (pure) ────────────────────────────────────────────────────────

class FakeService:
    """The heartbeat service's side of the bridge: the ticks and the active-hours check."""

    def __init__(self, within_hours: bool = True):
        self.ticks, self.within_hours = [], within_hours

    async def _agent_tick(self, agent_id, workspace_id, hb_config):
        self.ticks.append(("agent", agent_id, workspace_id))
        return {"status": "success", "source_type": "agent"}

    async def _orchestrator_tick(self, workspace_id, hb_config):
        self.ticks.append(("orchestrator", workspace_id))
        return {"status": "success", "source_type": "orchestrator"}

    async def _is_within_active_hours(self, hb_config, workspace_id=""):
        return self.within_hours


@pytest.fixture
def rig(monkeypatch):
    """The mode, the CLI check, the orchestrator settings, the check-ins and the bell, recorded."""
    from consumers.chatbot import personality
    from services.proactive import runner
    from services.proactive.runner import CheckInResult

    rig = NS(mode="off", cli=False, level="notify", check_ins=[], rang=[], run_fails=False, mode_reads=0)

    def mode(db, workspace_id):
        rig.mode_reads += 1
        if rig.mode is RuntimeError:
            raise RuntimeError("system_settings unreadable")
        return rig.mode

    async def run_agent(agent_id, workspace_id, hb_config):
        rig.check_ins.append(("agent", agent_id, workspace_id))
        if rig.run_fails:
            raise RuntimeError("no database")
        return CheckInResult(run_id=uuid4(), outcome="suggested", suggestions=1)

    async def run_workspace(workspace_id):
        rig.check_ins.append(("workspace", workspace_id))
        return CheckInResult(run_id=uuid4(), outcome="suggested", suggestions=1)

    async def ring(run, workspace_id, agent_id):
        rig.rang.append((agent_id, run.outcome))
        return True

    monkeypatch.setattr(heartbeat_bridge, "effective_mode", mode)
    monkeypatch.setattr(heartbeat_bridge, "_session", lambda: nullcontext(NS()))
    monkeypatch.setattr(heartbeat_bridge, "_runs_on_the_cli", lambda db, agent_id, workspace_id: rig.cli)
    monkeypatch.setattr(personality, "load_orchestrator_settings", lambda ws: {"proactive_level": rig.level})
    monkeypatch.setattr(runner, "run_agent_check_in", run_agent)
    monkeypatch.setattr(runner, "run_workspace_check_in", run_workspace)
    monkeypatch.setattr(heartbeat_bell, "ring_if_new", ring)
    return rig


def _agent_fire(service, hb_config):
    return asyncio.run(agent_job(service)(AGENT, WS, dict(hb_config)))


def _orchestrator_fire(service, hb_config):
    return asyncio.run(orchestrator_job(service)(WS, dict(hb_config)))


@pytest.mark.parametrize("mode, hb_config, cli, tick, check_in, rings", [
    ("off", {"mode": "check_in"}, False, True, False, False),
    ("shadow", {"mode": "check_in"}, False, True, True, False),
    ("live", {"mode": "check_in"}, False, False, True, True),
    ("live", {}, False, False, True, True),                          # unset, no auto_act: check_in
    ("live", {"auto_act": True}, False, True, False, False),         # unset, auto_act: act
    ("live", {"mode": "check_in", "auto_act": True}, False, False, True, True),  # stated wins
    ("live", {"mode": "act"}, False, True, False, False),
    ("shadow", {"mode": "act"}, False, True, False, False),
    ("live", {"mode": "check_in"}, True, True, False, False),        # a Claude Code agent keeps its ticket
    ("shadow", {"mode": "check_in"}, True, True, False, False),
])
def test_an_agents_heartbeat_takes_the_path_its_mode_and_config_choose(rig, mode, hb_config, cli, tick, check_in,
                                                                        rings):
    rig.mode, rig.cli = mode, cli
    service = FakeService()

    result = _agent_fire(service, hb_config)

    assert service.ticks == ([("agent", AGENT, WS)] if tick else [])
    assert rig.check_ins == ([("agent", AGENT, WS)] if check_in else [])
    assert rig.rang == ([(AGENT, "suggested")] if rings else [])
    if tick and check_in:  # shadow: the tick's result, with the check-in's beside it
        assert result["status"] == "success" and result["check_in"]["outcome"] == "suggested"
        assert result["check_in"]["rang"] is False
    elif check_in:
        assert (result["source"], result["outcome"], result["rang"]) == ("check_in", "suggested", True)


@pytest.mark.parametrize("mode, level, tick, check_in, rings", [
    ("off", "notify", True, False, False),
    ("shadow", "notify", True, True, False),
    ("shadow", "silent", True, True, False),
    ("live", "notify", False, True, True),
    ("live", "silent", False, True, False),                          # silent never rings
    ("live", "act_notify", True, False, False),
    ("live", "autonomous", True, False, False),
])
def test_the_orchestrator_heartbeat_takes_the_path_its_mode_and_level_choose(rig, mode, level, tick, check_in,
                                                                             rings):
    rig.mode = mode
    service = FakeService()

    _orchestrator_fire(service, {"enabled": True, "proactive_level": level})

    assert service.ticks == ([("orchestrator", WS)] if tick else [])
    assert rig.check_ins == ([("workspace", WS)] if check_in else [])
    assert rig.rang == ([(None, "suggested")] if rings else [])


def test_the_orchestrators_level_comes_from_its_settings_when_the_heartbeat_states_none(rig):
    rig.mode, rig.level = "live", "autonomous"
    service = FakeService()

    _orchestrator_fire(service, {"enabled": True})
    assert service.ticks == [("orchestrator", WS)] and rig.check_ins == [] and rig.mode_reads == 0

    rig.level = "silent"
    _orchestrator_fire(service, {"enabled": True})
    assert rig.check_ins == [("workspace", WS)] and rig.rang == []


def test_an_act_heartbeat_never_reads_the_mode(rig):
    rig.mode = "live"

    _agent_fire(FakeService(), {"mode": "act"})

    assert rig.mode_reads == 0


def test_a_mode_that_cannot_be_read_is_off_and_todays_tick_runs(rig):
    rig.mode = RuntimeError
    service = FakeService()

    _agent_fire(service, {"mode": "check_in"})
    _orchestrator_fire(service, {"proactive_level": "notify"})

    assert service.ticks == [("agent", AGENT, WS), ("orchestrator", WS)] and rig.check_ins == []


def test_a_live_check_in_outside_active_hours_runs_nothing(rig):
    rig.mode = "live"
    service = FakeService(within_hours=False)

    result = _agent_fire(service, {"mode": "check_in"})

    assert result == {"status": "skipped", "reason": "outside_active_hours"}
    assert service.ticks == [] and rig.check_ins == []


def test_a_failing_check_in_is_the_jobs_result_and_never_the_ticks(rig):
    rig.mode, rig.run_fails = "live", True
    service = FakeService()

    assert _agent_fire(service, {"mode": "check_in"}) == {"status": "error", "reason": "check_in_failed"}
    assert service.ticks == [] and rig.rang == []

    rig.mode = "shadow"
    result = _agent_fire(service, {"mode": "check_in"})
    assert result["status"] == "success" and result["check_in"] == {"status": "error", "reason": "check_in_failed"}


def test_a_shadow_tick_that_returns_nothing_still_carries_the_check_in(rig):
    rig.mode = "shadow"
    service = FakeService()

    async def no_result(agent_id, workspace_id, hb_config):
        service.ticks.append(("agent", agent_id, workspace_id))

    service._agent_tick = no_result
    result = _agent_fire(service, {"mode": "check_in"})

    assert set(result) == {"check_in"} and result["check_in"]["outcome"] == "suggested"


# ── The scheduler passes the bridge's jobs ──────────────────────────────────

class _Scheduler:
    def __init__(self):
        self.jobs = {}

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def remove_job(self, job_id):
        self.jobs.pop(job_id, None)

    def add_job(self, func, trigger=None, id=None, **kwargs):  # noqa: A002 — APScheduler's name
        self.jobs[id] = NS(func=func, kwargs=kwargs)


def test_the_scheduler_runs_the_bridges_jobs_bound_to_the_service():
    from services.heartbeat_service import HeartbeatService

    service = HeartbeatService()
    service._scheduler = _Scheduler()
    service.schedule_agent_heartbeat(AGENT, WS, {"enabled": True, "interval_minutes": 60})
    service.schedule_orchestrator_heartbeat(WS, {"enabled": True, "interval_minutes": 30})

    agent, orchestrator = service._scheduler.jobs[f"agent_hb_{AGENT}"], service._scheduler.jobs[f"orch_hb_{WS}"]
    assert agent.func.__func__ is HeartbeatJobs.agent and agent.func.__self__._service is service
    assert orchestrator.func.__func__ is HeartbeatJobs.orchestrator and orchestrator.func.__self__._service is service
    assert agent.kwargs["args"] == [AGENT, WS, {"enabled": True, "interval_minutes": 60}]
    assert orchestrator.kwargs["args"] == [WS, {"enabled": True, "interval_minutes": 30}]


# ── The payload, the GET and the PUT ────────────────────────────────────────

def _route_db(block):
    """The SQL the config routes issue: the ownership check, the saved block, the update."""
    updates = []

    def execute(stmt, params=None):
        sql = str(stmt)
        if "SELECT workspace_id FROM agents" in sql:
            return NS(fetchone=lambda: NS(workspace_id=WS))
        if "SELECT configuration FROM agents" in sql:
            return NS(fetchone=lambda: NS(configuration={"heartbeat": dict(block)} if block is not None else {}))
        updates.append(json.loads(params["cfg"]))
        return NS(fetchone=lambda: None)

    return NS(execute=execute, commit=lambda: None, updates=updates)


@pytest.mark.parametrize("mode", ["check_in", "act", None])
def test_the_payload_accepts_check_in_and_act(mode):
    from api.heartbeat import HeartbeatConfigPayload

    assert HeartbeatConfigPayload(mode=mode).mode == mode


def test_the_payload_refuses_any_other_mode():
    from api.heartbeat import HeartbeatConfigPayload

    with pytest.raises(ValidationError, match="check_in, act"):
        HeartbeatConfigPayload(mode="sometimes")


@pytest.mark.parametrize("block, echoed", [
    ({}, "check_in"),
    ({"auto_act": False}, "check_in"),
    ({"auto_act": True}, "act"),
    ({"mode": "check_in", "auto_act": True}, "check_in"),
    ({"mode": "act"}, "act"),
    ({"mode": "sometimes", "auto_act": True}, "act"),
])
def test_the_get_echoes_the_mode_or_its_effective_default(block, echoed):
    import api.heartbeat as heartbeat_api

    got = asyncio.run(heartbeat_api.get_agent_heartbeat_config(AGENT, ctx=NS(workspace_id=WS), db=_route_db(block)))

    assert got["mode"] == echoed


def _put(monkeypatch, block, body):
    import api.heartbeat as heartbeat_api
    import services.heartbeat_service as hbs

    scheduled = []
    monkeypatch.setattr(hbs, "get_heartbeat_service", lambda: NS(
        schedule_agent_heartbeat=lambda agent_id, ws, hb: scheduled.append(hb), unschedule_heartbeat=lambda job: None))
    db = _route_db(block)
    payload = heartbeat_api.HeartbeatConfigPayload(**{"enabled": True, **body})
    asyncio.run(heartbeat_api.save_agent_heartbeat_config(AGENT, payload, ctx=NS(workspace_id=WS), db=db))
    [saved] = db.updates
    assert scheduled == [saved["heartbeat"]]
    return saved["heartbeat"]


@pytest.mark.parametrize("block, body, stored", [
    ({"mode": "act"}, {"prompt": "x"}, "act"),                       # left out: the stated mode stays
    ({"mode": " Act"}, {"prompt": "x"}, "act"),                      # read as the bridge reads it
    ({}, {"prompt": "x"}, None),                                     # nothing stated: still unset
    (None, {"prompt": "x"}, None),                                   # no block yet
    ({"mode": "act"}, {"mode": "check_in"}, "check_in"),             # sent: stated
    ({}, {"mode": "check_in", "auto_act": True}, "check_in"),        # an explicit check_in is never auto_act's
    ({"mode": "check_in"}, {"mode": None}, None),                    # null: unset
])
def test_a_put_states_the_mode_it_sends_and_keeps_the_blocks_own_when_it_sends_none(monkeypatch, block, body,
                                                                                   stored):
    assert _put(monkeypatch, block, body)["mode"] == stored


# ── The bell's copy ─────────────────────────────────────────────────────────

def test_the_bell_copy_is_built_from_structured_fields_in_both_tables():
    assert set(heartbeat_bell.STRINGS["en"]) == set(heartbeat_bell.STRINGS["pt-BR"])
    one = NS(outcome="suggested", suggestions=1, error_class=None)
    two = NS(outcome="suggested", suggestions=2, error_class=None)
    failed = NS(outcome="error", suggestions=0, error_class="RuntimeError")

    assert heartbeat_bell.bell_copy(one, "VECTOR") == (
        "VECTOR Heartbeat", "VECTOR's check-in suggested a next step.", "ok")
    assert heartbeat_bell.bell_copy(two, "VECTOR")[1] == "VECTOR's check-in suggested 2 next steps."
    assert heartbeat_bell.bell_copy(failed, "VECTOR")[1:] == ("VECTOR's check-in failed (RuntimeError).", "error")
    assert heartbeat_bell.bell_copy(one, "VECTOR", "pt-BR")[1] == "O check-in de VECTOR sugeriu um próximo passo."


# ── Live, against Postgres ──────────────────────────────────────────────────

SUGGEST = "platform_suggest_next_step"
LIST_TASKS = "platform_list_tasks"
MODEL = "claude-haiku-5-5"
READ_TICKET = {"id": 9, "number": "#0912", "title": "Refresh the sitemap", "status": "failed"}


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here):
    the runner, the suggestion tool and the bell commit, and none may take the seed."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _agent_row(db, ws: UUID, name: str, configuration: dict, system: bool = False) -> int:
    return int(db.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, is_system_agent, configuration) "
        "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', :s, CAST(:c AS json)) RETURNING id"),
        {"n": name, "w": str(ws), "s": system, "c": json.dumps(configuration)}).scalar())


@pytest.fixture
def shop(db_session):
    """A workspace with check-ins on, its owner, Auto (so ``report_to=auto`` has a ticket to
    file today) and VECTOR, a ``check_in`` agent reporting to Auto."""
    name = f"prd268br_{uuid4().hex[:10]}"
    owner = int(db_session.execute(text(
        "INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
        {"u": name, "e": f"{name}@t.test", "c": name}).scalar())
    ws = uuid4()
    db_session.execute(text(
        "INSERT INTO workspaces (id, name, slug, plan, owner_id, settings) VALUES (CAST(:id AS uuid), "
        "'prd268 bridge', :slug, 'pro', :o, CAST(:settings AS jsonb))"),
        {"id": str(ws), "slug": f"prd268-bridge-{ws.hex[:10]}", "o": owner,
         "settings": json.dumps({"proactive": {"enabled": True}})})
    _agent_row(db_session, ws, "Auto", {}, system=True)
    agent = _agent_row(db_session, ws, "VECTOR", {"heartbeat": CHECK_IN_HB})
    db_session.commit()
    return NS(db=db_session, ws=ws, owner=owner, agent=agent)


def _call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _reply(*calls: dict, content: str = "") -> NS:
    return NS(content=content, tool_calls=list(calls), usage={"input_tokens": 1000, "output_tokens": 100})


class FakeLLM:
    """The heartbeat model, scripted: each call answers the next reply (the last repeats)."""

    def __init__(self, replies=(), fails: bool = False):
        self.config = NS(model=MODEL)
        self.replies, self.fails, self.calls = list(replies) or [_reply()], fails, []

    async def generate_response(self, messages, tools=None):
        self.calls.append(len(messages))
        if self.fails:
            raise RuntimeError("provider refused the key (HTTP 401)")
        return self.replies[min(len(self.calls), len(self.replies)) - 1]

    def call_cost(self, input_tokens, output_tokens, reported_cost=None):
        return 0.001, "estimate"


async def no_delivery(db, run_id):
    """The runner's delivery, left out: these tests ring and count W1's heartbeat bell only."""
    from services.proactive.delivery import Delivery

    return Delivery()


@pytest.fixture
def live(shop, monkeypatch):
    """Platform mode live; every session the test's; the model and the context build
    scripted; the executor real; the tick, the report filer and the dispatcher recorded."""
    from core.llm import manager
    from core.services import notification_dispatcher
    from modules.context.result import ContextResult
    from modules.context.service import ContextService
    from modules.tools.discovery import platform_executor
    from modules.tools.discovery.action_registry import get_action_registry
    from services import heartbeat_report
    from services.heartbeat_service import HeartbeatService
    from services.proactive import runner

    rig = NS(mode="live", llm=FakeLLM(), bells=[], ticks=[], reports=[], service=HeartbeatService())

    async def build_context(self, **kwargs):
        tools = [get_action_registry().get(name).to_openai_schema() for name in (LIST_TASKS, SUGGEST)]
        return ContextResult(system_prompt=f"SYSTEM\n{kwargs['task_description']}", tools=tools)

    async def list_tasks(db, workspace_id, params):
        return {"success": True, "tasks": [dict(READ_TICKET)], "total": 1}

    async def tick(agent_id, workspace_id, hb_config):
        rig.ticks.append(agent_id)
        return {"status": "success"}

    async def orchestrator_tick(workspace_id, hb_config):
        rig.ticks.append(workspace_id)
        return {"status": "success"}

    async def report(*args, **kwargs):
        rig.reports.append(args)

    class Dispatcher:
        def __init__(self, db, workspace_id):
            self.workspace_id = str(workspace_id)

        async def dispatch(self, **kwargs):
            rig.bells.append({"workspace_id": self.workspace_id, **kwargs})
            return {"dispatched_to": ["in_app"]}

    for module in (runner, heartbeat_bridge, heartbeat_bell):
        monkeypatch.setattr(module, "_session", lambda: nullcontext(shop.db))
    monkeypatch.setattr(manager, "read_system_setting",
                        lambda category, key: rig.mode if (category, key) == ("proactive", "mode") else None)
    monkeypatch.setattr(ContextService, "build_context", build_context)
    monkeypatch.setattr(runner, "heartbeat_llm", lambda workspace_id, agent_id: rig.llm)
    monkeypatch.setattr(platform_executor, "list_board_tasks", list_tasks)
    # W2's delivery (the thread post and suggestion_ready) is proven in test_prd268w2_suggestion_delivery.py.
    monkeypatch.setattr(runner, "deliver_run", no_delivery)
    monkeypatch.setattr(platform_executor.PlatformActionExecutor, "_full_autonomy", lambda self: False)
    monkeypatch.setattr(notification_dispatcher, "NotificationDispatcher", Dispatcher)
    monkeypatch.setattr(heartbeat_report, "auto_create_heartbeat_report", report)
    monkeypatch.setattr(rig.service, "_agent_tick", tick)
    monkeypatch.setattr(rig.service, "_orchestrator_tick", orchestrator_tick)
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        yield rig


def _fire(shop, live):
    return asyncio.run(agent_job(live.service)(shop.agent, str(shop.ws), dict(CHECK_IN_HB)))


def _count(shop, table: str) -> int:
    return int(shop.db.execute(text(f"SELECT count(*) FROM {table} WHERE workspace_id = CAST(:w AS uuid)"),  # noqa: S608 — fixed names
                               {"w": str(shop.ws)}).scalar())


def _runs(shop):
    from core.models.proactive import ProactiveRun

    shop.db.expire_all()
    return shop.db.query(ProactiveRun).filter(ProactiveRun.workspace_id == shop.ws).all()


def _prior_run(shop, outcome: str, hours_ago: float, error_class=None) -> None:
    shop.db.execute(text(
        "INSERT INTO proactive_runs (id, workspace_id, agent_id, kind, outcome, error_class, created_at) "
        "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), :agent, 'agent_check_in', :outcome, :error, :at)"),
        {"id": str(uuid4()), "ws": str(shop.ws), "agent": shop.agent, "outcome": outcome, "error": error_class,
         "at": datetime.now(timezone.utc) - timedelta(hours=hours_ago)})
    shop.db.commit()


def _an_interval_passes(shop) -> None:
    """The scheduler's next fire comes one heartbeat interval later: every run the shop has
    moves back by it, so the next check-in is not refused as sooner than
    ``min_interval_minutes`` after the last (``services.proactive.gates``)."""
    shop.db.execute(text(
        "UPDATE proactive_runs SET created_at = created_at - :gap, window_since = window_since - :gap, "
        "window_until = window_until - :gap WHERE workspace_id = CAST(:ws AS uuid)"),
        {"gap": timedelta(minutes=CHECK_IN_HB["interval_minutes"]), "ws": str(shop.ws)})
    shop.db.commit()


def _assert_no_tick(shop, live) -> None:
    """Today's tick never ran: no board ticket, no heartbeat result, no per-tick report."""
    assert live.ticks == [] and live.reports == []
    assert _count(shop, "board_tasks") == 0 and _count(shop, "heartbeat_results") == 0


def test_live_a_silent_check_in_files_no_ticket_and_no_report_and_rings_nothing(shop, live):
    live.llm = FakeLLM([_reply(content="Nothing new.")])

    result = _fire(shop, live)

    assert (result["outcome"], result["suggestions"], result["rang"]) == ("silent", 0, False)
    _assert_no_tick(shop, live)
    assert live.bells == []
    [run] = _runs(shop)
    assert (run.kind, run.agent_id, run.outcome) == ("agent_check_in", shop.agent, "silent")


def test_live_a_suggestion_rings_one_heartbeat_bell_and_files_no_ticket(shop, live):
    live.llm = FakeLLM([_reply(_call(LIST_TASKS, {})), _reply(_call(SUGGEST, {
        "title": "Ticket #0912 failed", "detail": "The sitemap refresh failed.",
        "evidence": [{"type": "ticket", "id": "0912"}], "proposed_action": {"kind": "chat"}}))])

    result = _fire(shop, live)

    assert (result["outcome"], result["suggestions"], result["rang"]) == ("suggested", 1, True)
    _assert_no_tick(shop, live)
    [bell] = live.bells
    assert (bell["event_type"], bell["link_type"], bell["link_id"], bell["status"]) == (
        "heartbeat_complete", "heartbeat", result["run_id"], "ok")
    assert (bell["agent_id"], bell["agent_name"], bell["workspace_id"]) == (shop.agent, "VECTOR", str(shop.ws))
    assert bell["message"] == "VECTOR's check-in suggested a next step." and "#0912" not in bell["message"]


def test_live_the_first_failure_of_a_streak_rings_and_the_next_does_not(shop, live):
    live.llm = FakeLLM(fails=True)

    first = _fire(shop, live)
    _an_interval_passes(shop)
    second = _fire(shop, live)

    assert (first["outcome"], first["error_class"], first["rang"]) == ("error", "RuntimeError", True)
    assert (second["outcome"], second["rang"]) == ("error", False)
    _assert_no_tick(shop, live)
    [bell] = live.bells
    assert (bell["status"], bell["message"]) == ("error", "VECTOR's check-in failed (RuntimeError).")


@pytest.mark.parametrize("prior, rings", [
    ([("error", 2, "RuntimeError"), ("silent", 1, None)], True),      # a success ended the last streak
    ([("error", 1, "RuntimeError")], False),                          # the streak goes on
    ([("error", 2, "RuntimeError"), ("skipped_cap", 1, None)], False),  # a skip neither counts nor ends it
    ([("error", 1, "Timeout")], True),                                # another class: a new streak
])
def test_live_a_failure_rings_only_when_it_starts_a_streak(shop, live, prior, rings):
    for outcome, hours_ago, error_class in prior:
        _prior_run(shop, outcome, hours_ago, error_class)
    live.llm = FakeLLM(fails=True)

    result = _fire(shop, live)

    assert (result["outcome"], result["rang"]) == ("error", rings)
    assert len(live.bells) == (1 if rings else 0)
    _assert_no_tick(shop, live)


def test_a_bell_that_fails_never_reaches_the_heartbeat(shop, live, monkeypatch):
    from core.services import notification_dispatcher

    class Broken:
        def __init__(self, db, workspace_id):
            pass

        async def dispatch(self, **kwargs):
            raise RuntimeError("notifications is missing")

    monkeypatch.setattr(notification_dispatcher, "NotificationDispatcher", Broken)
    live.llm = FakeLLM(fails=True)

    result = _fire(shop, live)

    assert (result["outcome"], result["rang"]) == ("error", False)
    assert [run.outcome for run in _runs(shop)] == ["error"]
    _assert_no_tick(shop, live)


def _orchestrator_live(shop, live, monkeypatch, level: str):
    from services.proactive import runner
    from services.proactive.signals import Signal

    signal = Signal(kind="ticket_failed", subject_key="ticket:0846", ref=("ticket", "0846"),
                    at=datetime.now(timezone.utc), summary="Ticket #0846 failed")
    monkeypatch.setattr(runner, "read_signals", lambda db, workspace_id, since, until: (signal,))
    live.llm = FakeLLM(fails=True)
    hb_config = {**ALL_DAY, "enabled": True, "proactive_level": level}
    return asyncio.run(orchestrator_job(live.service)(str(shop.ws), hb_config))


def test_live_auto_rings_as_auto_at_notify_when_its_check_in_starts_a_streak(shop, live, monkeypatch):
    result = _orchestrator_live(shop, live, monkeypatch, "notify")

    assert (result["outcome"], result["rang"]) == ("error", True)
    [bell] = live.bells
    assert (bell["agent_id"], bell["agent_name"], bell["message"]) == (
        None, "Auto", "Auto's check-in failed (RuntimeError).")
    [run] = _runs(shop)
    assert (run.kind, run.agent_id) == ("workspace_check_in", None)
    _assert_no_tick(shop, live)


def test_live_auto_at_silent_rings_nothing(shop, live, monkeypatch):
    result = _orchestrator_live(shop, live, monkeypatch, "silent")

    assert (result["outcome"], result["rang"]) == ("error", False) and live.bells == []
    _assert_no_tick(shop, live)


def test_off_the_same_heartbeat_runs_todays_tick_and_no_check_in(shop, live):
    live.mode = "off"

    _fire(shop, live)

    assert live.ticks == [shop.agent] and _runs(shop) == [] and live.bells == []
