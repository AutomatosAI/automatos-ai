"""PRD-263 US-013: HARNESS's per-agent read model, against the real schema.

HARNESS judged four fields ``platform_list_agents`` never returns, so every agent was
STABLE. ``services.harness_metrics`` reads each agent's real weekly counts from SQL.
Each data trap in the design (section 4.1 #1) has its own case here, on two
workspaces: workspace B's rows, several of them carrying workspace A's agent id,
are seeded for every case, so each exact count in A also proves B never leaks in.

The judged week is fixed (2026-W40, Monday 28 Sep 2026) with four reference weeks
(W36-W39), so nothing depends on today's date.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

from core.models.composio_cache import ToolExecutionLog
from core.models.core import Agent, BoardTask, LLMUsage
from core.models.orchestration import OrchestrationRun, OrchestrationTask
from services import harness_metrics as hm

ORCHESTRATOR = Path(__file__).resolve().parent.parent
MIGRATIONS = ORCHESTRATOR / "alembic" / "versions"

WEEK = "2026-W40"
WEEKS = ("2026-W36", "2026-W37", "2026-W38", "2026-W39", "2026-W40")
MONDAY = datetime(2026, 9, 28)        # W40, naive UTC
IN_WEEK = MONDAY + timedelta(days=2, hours=10)
IN_REF_WEEK = IN_WEEK - timedelta(weeks=2)   # W38
BEFORE_WINDOW = IN_WEEK - timedelta(weeks=5)  # W35
AFTER_WINDOW = IN_WEEK + timedelta(weeks=1)   # W41
WINDOW_START = MONDAY - timedelta(weeks=4)    # Monday of W36, inside the window
WINDOW_END = MONDAY + timedelta(weeks=1)      # Monday of W41, outside it
IDENTITY = {"agent_id", "agent_name", "model_id", "heartbeat_enabled", "heartbeat_interval_minutes", "iso_week"}
NOTHING = (0, 0.0, ())   # what an agent-week with no rows holds: counts of 0, $0.0, no runs


def _aware(at: datetime) -> datetime:
    return at.replace(tzinfo=timezone.utc)


@pytest.fixture
def beat(db_session):
    """Insert a heartbeat_results row. The table has no model: init_test_db builds it
    from the migrations' DDL (the last test below holds the two together)."""

    def _beat(ws, source_type, source_id, at, *, objective_met=True, tokens=10, status="success"):
        db_session.execute(text(
            "INSERT INTO heartbeat_results (source_type, source_id, workspace_id, status, tokens_used, "
            "objective_met, created_at) VALUES (:st, :sid, CAST(:ws AS uuid), :status, :tokens, :met, :at)"),
            {"st": source_type, "sid": str(source_id), "ws": str(ws), "status": status, "tokens": tokens,
             "met": objective_met, "at": at})

    return _beat


def _agent(db, ws, name, **over):
    agent = Agent(name=name, agent_type=over.pop("agent_type", "custom"), workspace_id=UUID(ws),
                  status=over.pop("status", "active"), **over)
    db.add(agent)
    db.flush()
    return agent


def _ticket(db, ws, agent_id, status, at, **over):
    task = BoardTask(workspace_id=UUID(ws), title=f"A {status} ticket", status=status, priority="medium",
                     assigned_agent_id=agent_id, completed_at=_aware(at), updated_at=_aware(at), **over)
    db.add(task)
    db.flush()
    return task


def _usage(db, ws, agent_id, at, *, request_type="board_task", status="success", tier="direct",
           cost=0.0, tokens=0, execution_id=None):
    db.add(LLMUsage(workspace_id=UUID(ws), model_id="anthropic/claude-haiku-5.5", provider="anthropic",
                    tier=tier, agent_id=agent_id, execution_id=execution_id, request_type=request_type,
                    input_tokens=tokens, output_tokens=0, total_tokens=tokens, input_cost=cost,
                    output_cost=0.0, total_cost=cost, status=status, created_at=at))
    db.flush()


def _tool_run(db, ws, agent_id, at, *, app="GMAIL", status="success", error_code=None, source="production",
              error_message=None):
    run = ToolExecutionLog(workspace_id=UUID(ws), agent_id=agent_id, app_name=app,
                           action_name=f"{app}_SEND", status=status, error_code=error_code,
                           error_message=error_message, telemetry_source=source, executed_at=at)
    db.add(run)
    db.flush()
    return run


def _mission(db, ws):
    run = OrchestrationRun(workspace_id=UUID(ws), goal="Launch the autumn menu", created_by="user_test")
    db.add(run)
    db.flush()
    return run


def _mission_task(db, run, agent_id, state, at, seq):
    task = OrchestrationTask(run_id=run.id, title=f"Step {seq}", sequence_number=seq,
                             assigned_agent_id=agent_id, state=state, completed_at=_aware(at))
    db.add(task)
    db.flush()
    return task


def _seed_workspace_b(db, b, worker_id, intruder_id, beat):
    """Workspace B's own activity, and rows in B that carry A's agent id."""
    for agent_id in (intruder_id, worker_id):
        _ticket(db, b, agent_id, "done", IN_WEEK)
        _usage(db, b, agent_id, IN_WEEK, cost=5.0, tokens=5000, execution_id="board_task:1")
        _usage(db, b, agent_id, IN_WEEK, request_type="heartbeat", cost=5.0, tokens=5000)
        _tool_run(db, b, agent_id, IN_WEEK, status="error", error_code="401")
        beat(b, "agent", agent_id, IN_WEEK)
    _mission_task(db, _mission(db, b), worker_id, "verified", IN_WEEK, 1)


@pytest.fixture
def world(db_session, seed_workspace, beat):
    a, b = seed_workspace(name="harness-a"), seed_workspace(name="harness-b")
    worker = _agent(db_session, a, "Barista", model_config={"model_id": "anthropic/claude-haiku-5.5"},
                    configuration={"heartbeat": {"enabled": True, "interval_minutes": 30}})
    idle = _agent(db_session, a, "Baker")
    auto = _agent(db_session, a, "Auto", agent_type="system", slug=f"auto-{a}", is_system_agent=True)
    retired = _agent(db_session, a, "Old roaster", status="inactive")
    intruder = _agent(db_session, b, "Their barista")
    _seed_workspace_b(db_session, b, worker.id, intruder.id, beat)
    return NS(db=db_session, a=a, b=b, worker=worker.id, idle=idle.id, auto=auto.id, retired=retired.id,
              intruder=intruder.id, beat=beat)


def _read(world, workspace=None):
    return hm.read_sync(world.db, workspace or world.a, WEEK)


def _row(result, agent_id, week=WEEK):
    (row,) = [r for r in result.agent_rows if r["agent_id"] == agent_id and r["iso_week"] == week]
    return row


# ── roster and shape ─────────────────────────────────────────────────────────

def test_the_roster_is_active_agents_other_than_auto_with_their_model_and_heartbeat(world):
    result = _read(world)
    assert {r["agent_id"] for r in result.agent_rows} == {world.worker, world.idle}
    worker, idle = _row(result, world.worker), _row(result, world.idle)
    assert (worker["model_id"], worker["heartbeat_enabled"], worker["heartbeat_interval_minutes"]) == (
        "anthropic/claude-haiku-5.5", True, 30)
    # No heartbeat configured: off, at the heartbeat service's own default interval.
    assert (idle["model_id"], idle["heartbeat_enabled"], idle["heartbeat_interval_minutes"]) == (None, False, 60)


def test_a_heartbeat_setting_that_is_not_an_object_reads_as_off_and_the_read_goes_on(world):
    """The driver decodes the JSON, so "on" arrives as a str: never parsed again, never raised."""
    odd = _agent(world.db, world.a, "Odd config", configuration={"heartbeat": "on"})
    row = _row(_read(world), odd.id)
    assert (row["heartbeat_enabled"], row["heartbeat_interval_minutes"]) == (False, 60)


def test_every_agent_has_a_row_for_each_week_and_no_rows_reads_as_n_zero(world):
    result = _read(world)
    assert result.window.weeks == WEEKS and result.window.cur_week == WEEK
    assert sorted((r["agent_id"], r["iso_week"]) for r in result.agent_rows) == sorted(
        (agent, week) for agent in (world.worker, world.idle) for week in WEEKS)
    assert all(tuple(row) == hm.AGENT_ROW_FIELDS for row in result.agent_rows)
    idle = _row(result, world.idle)
    counts = {k: v for k, v in idle.items() if k not in IDENTITY}
    assert counts and all(v in NOTHING for v in counts.values()), counts   # n=0, never a rate or None


# ── board_tasks ──────────────────────────────────────────────────────────────

def test_tickets_count_done_and_failed_but_not_heartbeat_mission_or_harness_cards(world):
    db, a, worker = world.db, world.a, world.worker
    for status in ("done", "done", "done", "failed", "review", "in_progress", "cancelled"):
        _ticket(db, a, worker, status, IN_WEEK)
    run = _mission(db, a)
    step = _mission_task(db, run, worker, "running", IN_WEEK, 1)
    _ticket(db, a, worker, "done", IN_WEEK, source_type="heartbeat")
    _ticket(db, a, worker, "done", IN_WEEK, source_type="orchestration_task", orchestration_run_id=run.id)
    _ticket(db, a, worker, "done", IN_WEEK, source_type="orchestration", orchestration_run_id=run.id)
    # Each mission link on its own: a session-lane mission card, a run id, a step id.
    _ticket(db, a, worker, "done", IN_WEEK, source_type="mission")
    _ticket(db, a, worker, "done", IN_WEEK, orchestration_run_id=run.id)
    _ticket(db, a, worker, "done", IN_WEEK, orchestration_task_id=step.id)
    _ticket(db, a, worker, "failed", IN_WEEK, tags=["harness", "org-review"])
    _ticket(db, a, worker, "done", IN_REF_WEEK)
    _ticket(db, a, worker, "done", WINDOW_START)   # the window's start is in
    _ticket(db, a, worker, "done", WINDOW_END)     # its end is out
    _ticket(db, a, worker, "done", BEFORE_WINDOW)
    _ticket(db, a, worker, "done", AFTER_WINDOW)
    result = _read(world)
    cur = _row(result, world.worker)
    assert (cur["tasks_done"], cur["tasks_failed"]) == (3, 1)
    assert _row(result, world.worker, "2026-W38")["tasks_done"] == 1
    assert _row(result, world.worker, "2026-W36")["tasks_done"] == 1
    assert sum(r["tasks_done"] for r in result.agent_rows) == 5   # W35 and W41 are outside


def test_weeks_are_utc_whatever_the_session_zone(world):
    """date_trunc on a timestamptz follows the session zone; the read pins UTC."""
    db, a, w = world.db, world.a, world.worker
    db.execute(text("SET LOCAL TIME ZONE 'America/Los_Angeles'"))   # undone by db_session's rollback
    monday_early = MONDAY + timedelta(hours=3)        # W40 in UTC, Sunday evening in Los Angeles
    sunday_late = MONDAY - timedelta(minutes=30)      # W39 in UTC
    _ticket(db, a, w, "done", monday_early)
    _ticket(db, a, w, "failed", sunday_late)
    run = _mission(db, a)
    _mission_task(db, run, w, "verified", monday_early, 1)
    _mission_task(db, run, w, "failed", sunday_late, 2)
    result = _read(world)
    fields = ("tasks_done", "tasks_failed", "mission_tasks_done", "mission_tasks_failed")
    assert tuple(_row(result, w)[f] for f in fields) == (1, 0, 1, 0)
    assert tuple(_row(result, w, "2026-W39")[f] for f in fields) == (0, 1, 0, 1)


# ── llm_usage ────────────────────────────────────────────────────────────────

def test_spend_counts_whitelisted_lanes_and_statuses_with_plan_calls_apart(world):
    db, a, w = world.db, world.a, world.worker
    _usage(db, a, w, IN_WEEK, cost=0.10, tokens=1000, execution_id="board_task:901")
    _usage(db, a, w, IN_WEEK, status="error", cost=0.02, tokens=200, execution_id="board_task:901")
    _usage(db, a, w, IN_WEEK, request_type="recipe", cost=0.30, tokens=3000, execution_id="exec-45d8ac862a79")
    _usage(db, a, w, IN_WEEK, tier="subscription", tokens=5000, execution_id="board_task:902")
    _usage(db, a, w, IN_WEEK, cost=0.05, tokens=500)                       # no execution_id
    for request_type, status in (("memory_distill", "success"), ("complexity_assessor", "success"),
                                 ("media", "reserved"), ("board_task", "reserved"),
                                 ("board_task", "pending"), ("chat", "success"),
                                 ("heartbeat", "reserved"), ("heartbeat", "pending")):
        _usage(db, a, w, IN_WEEK, request_type=request_type, status=status, cost=9.0, tokens=9000)
    _usage(db, a, w, IN_WEEK, request_type="heartbeat", cost=0.01, tokens=100)
    for status in ("timeout", "rate_limited", "cancelled"):   # calls that happened, and count
        _usage(db, a, w, IN_WEEK, request_type="heartbeat", status=status, cost=0.02, tokens=200)
    cur = _row(_read(world), w)
    assert (cur["llm_calls"], cur["llm_plan_calls"], cur["llm_unattributed_calls"]) == (5, 1, 1)
    assert cur["llm_billed_cost_usd"] == pytest.approx(0.47)
    assert (cur["llm_billed_tokens"], cur["llm_plan_tokens"]) == (4700, 5000)
    # The plan-only run is a token sample, not a $0 cost sample.
    assert cur["llm_run_costs_usd"] == pytest.approx((0.12, 0.30))
    assert cur["llm_run_tokens"] == (1200, 3000, 5000)
    assert (cur["heartbeat_llm_calls"], cur["heartbeat_llm_billed_tokens"]) == (4, 700)
    assert cur["heartbeat_llm_billed_cost_usd"] == pytest.approx(0.07)
    assert (cur["heartbeat_llm_unattributed_calls"], cur["heartbeat_llm_run_costs_usd"]) == (4, ())


def test_a_session_tickets_spend_follows_its_ticket(world):
    """A Claude Code ticket books in the board_task lane whatever started it."""
    db, a, w = world.db, world.a, world.worker
    run = _mission(db, a)
    step = _mission_task(db, run, w, "running", IN_WEEK, 1)
    tickets = {
        "work": _ticket(db, a, w, "done", IN_WEEK),
        "heartbeat": _ticket(db, a, w, "done", IN_WEEK, source_type="heartbeat"),
        "mission_step": _ticket(db, a, w, "done", IN_WEEK, source_type="orchestration_task",
                                orchestration_run_id=run.id),
        "mission_card": _ticket(db, a, w, "done", IN_WEEK, source_type="mission"),
        "mission_run": _ticket(db, a, w, "done", IN_WEEK, orchestration_run_id=run.id),
        "mission_task": _ticket(db, a, w, "done", IN_WEEK, orchestration_task_id=step.id),
        "harness": _ticket(db, a, w, "done", IN_WEEK, tags=["harness"]),
    }
    for task in tickets.values():
        _usage(db, a, w, IN_WEEK, tier="subscription", tokens=1000, execution_id=f"board_task:{task.id}")
    # A's spend naming B's heartbeat ticket stays A's work: B's ticket never classifies it.
    theirs = _ticket(db, world.b, w, "done", IN_WEEK, source_type="heartbeat")
    _usage(db, a, w, IN_WEEK, tier="subscription", tokens=1000, execution_id=f"board_task:{theirs.id}")
    cur = _row(_read(world), w)
    assert (cur["llm_calls"], cur["llm_plan_tokens"]) == (2, 2000)
    assert (cur["llm_run_tokens"], cur["llm_run_costs_usd"]) == ((1000, 1000), ())
    assert (cur["heartbeat_llm_plan_calls"], cur["heartbeat_llm_run_tokens"]) == (1, (1000,))


# ── heartbeat_results ────────────────────────────────────────────────────────

def test_heartbeats_read_agent_rows_only_and_keep_unknown_outcomes_apart(world):
    a, w, beat = world.a, world.worker, world.beat
    for met in (True, True, False, None):
        beat(a, "agent", w, IN_WEEK, objective_met=met)
    beat(a, "agent", w, IN_REF_WEEK)
    # The orchestrator's own row keeps the workspace UUID in source_id; a malformed agent
    # row has no integer id at all. Neither may raise, and neither counts.
    beat(a, "orchestrator", a, IN_WEEK)
    beat(a, "agent", "not-a-number", IN_WEEK)
    result = _read(world)
    cur = _row(result, w)
    assert (cur["heartbeat_runs"], cur["heartbeat_objective_met"], cur["heartbeat_objective_missed"],
            cur["heartbeat_objective_unknown"], cur["heartbeat_tokens"]) == (4, 2, 1, 1, 40)
    assert _row(result, w, "2026-W39")["heartbeat_runs"] == 0
    assert _row(result, w, "2026-W38")["heartbeat_runs"] == 1


# ── orchestration_tasks ──────────────────────────────────────────────────────

def test_mission_tasks_are_scoped_through_their_run(world):
    db, a, w = world.db, world.a, world.worker
    run = _mission(db, a)
    for seq, state in enumerate(("verified", "verified", "failed", "running", "skipped", "completed"), 1):
        _mission_task(db, run, w, state, IN_WEEK, seq)
    # Workspace B's mission step carrying A's agent id is seeded by the fixture: not counted.
    cur = _row(_read(world), w)
    assert (cur["mission_tasks_done"], cur["mission_tasks_failed"]) == (2, 1)


# ── tool_execution_logs ──────────────────────────────────────────────────────

def test_tool_runs_are_production_only_grouped_by_app_and_error_class(world):
    db, a, w = world.db, world.a, world.worker
    for _ in range(2):
        _tool_run(db, a, w, IN_WEEK)
    _tool_run(db, a, w, IN_WEEK, status="error", error_code=" 401 ")
    _tool_run(db, a, w, IN_WEEK, status="error")
    # PRD-263 US-019: a row with no code (every row before US-018) classes from its message.
    _tool_run(db, a, w, IN_WEEK, status="error", error_message="httpx.ConnectError: [Errno 111] Connection refused")
    _tool_run(db, a, w, IN_WEEK, app="SLACK", status="timeout")
    _tool_run(db, a, w, IN_WEEK, status="asked")            # a wait for the owner's click
    _tool_run(db, a, w, IN_WEEK, status="error", source="eval")
    untagged = _tool_run(db, a, w, IN_WEEK, status="error")
    db.execute(text("UPDATE tool_execution_logs SET telemetry_source = NULL WHERE id = :id"), {"id": untagged.id})
    result = _read(world)
    got = {(r["app_name"], r["status"], r["error_class"]): r["calls"]
           for r in result.tool_rows if r["agent_id"] == w and r["iso_week"] == WEEK}
    assert got == {("GMAIL", "success", "none"): 2, ("GMAIL", "error", "auth"): 1,
                   ("GMAIL", "error", "connection"): 1, ("GMAIL", "error", "other"): 1,
                   ("SLACK", "timeout", "other"): 1}
    assert all(set(r) == set(hm.TOOL_ROW_FIELDS) for r in result.tool_rows)
    cur = _row(result, w)
    assert (cur["tool_calls"], cur["tool_failed_calls"]) == (6, 4)


# ── tenant isolation ─────────────────────────────────────────────────────────

def test_workspace_b_never_appears_in_workspace_as_result(world):
    mine = _read(world)
    assert world.intruder not in {r["agent_id"] for r in mine.agent_rows}
    assert all(r["agent_id"] != world.intruder for r in mine.tool_rows)
    # B's rows carrying A's agent id (seeded by the fixture) leave A's agent untouched.
    cur = _row(mine, world.worker)
    assert {k: v for k, v in cur.items() if k not in IDENTITY and v not in NOTHING} == {}
    theirs = _read(world, world.b)
    assert {r["agent_id"] for r in theirs.agent_rows} == {world.intruder}
    their_cur = _row(theirs, world.intruder)
    assert (their_cur["tasks_done"], their_cur["llm_calls"], their_cur["heartbeat_runs"]) == (1, 1, 1)


# ── the async entry point and the window ─────────────────────────────────────

def test_the_async_read_runs_the_same_read_on_a_worker_thread(world, monkeypatch):
    threaded = []
    real_to_thread = asyncio.to_thread

    async def _to_thread(func, *args, **kwargs):
        threaded.append(func)
        return await real_to_thread(func, *args, **kwargs)

    # Only the module's own binding: the stdlib's asyncio stays untouched for every other caller.
    monkeypatch.setattr(hm, "asyncio", NS(to_thread=_to_thread))
    _ticket(world.db, world.a, world.worker, "done", IN_WEEK)
    result = asyncio.run(hm.read(world.db, UUID(world.a), WEEK))
    assert threaded == [hm.read_sync]
    assert result.agent_rows == _read(world).agent_rows


def test_the_window_is_iso_weeks_across_a_year_end_and_bad_input_fails_fast():
    window = hm.build_window("2026-W01", 4)
    assert window.ref_weeks == ("2025-W49", "2025-W50", "2025-W51", "2025-W52")
    assert (window.start, window.end) == (datetime(2025, 12, 1), datetime(2026, 1, 5))
    assert hm.build_window(WEEK, 0).weeks == (WEEK,)
    assert (hm.build_window(WEEK).start, hm.build_window(WEEK).end) == (WINDOW_START, WINDOW_END)
    for bad in ("2026-42", "2026-W54", "", "W40"):
        with pytest.raises(ValueError):
            hm.build_window(bad)
    with pytest.raises(ValueError):
        hm.build_window(WEEK, -1)


def test_the_usage_vocabulary_matches_the_platforms():
    """Kept literal in the read model (it must not load the LLM client chain)."""
    from core.llm import usage_context as uc
    from core.llm import usage_tracker as ut
    from core.models.ticket_numbers import STEP_SOURCE
    from services.cli_ticket_lane import MISSION_SOURCE_TYPE

    assert hm.WORK_REQUEST_TYPES == (uc.LANE_BOARD_TASK, uc.LANE_SESSION, uc.LANE_RECIPE)
    assert hm.PLAN_TIER == ut.TIER_SUBSCRIPTION
    assert {ut.STATUS_SUCCESS, ut.STATUS_ERROR, ut.STATUS_CANCELLED} <= set(hm.WORK_STATUSES)
    assert not {"reserved", "pending"} & (set(hm.WORK_STATUSES) | set(hm.HEARTBEAT_STATUSES))
    assert {STEP_SOURCE, MISSION_SOURCE_TYPE} <= set(hm.MISSION_TICKET_SOURCES)


def test_the_test_database_builds_heartbeat_results_as_the_migrations_do():
    """heartbeat_results has no model, so init_test_db builds it for these tests: with
    the two migrations' columns and indexes, the read model meets prod's shape."""
    created = (MIGRATIONS / "20260215_add_heartbeat_and_channels.py").read_text(encoding="utf-8")
    added = (MIGRATIONS / "wave1b_heartbeat_completion.py").read_text(encoding="utf-8")
    fresh = (ORCHESTRATOR / "scripts" / "init_test_db.py").read_text(encoding="utf-8")
    (table,) = re.findall(r"op\.create_table\(\s*'heartbeat_results',(.*?)\n    \)", created, re.S)
    migrated = set(re.findall(r"sa\.Column\('(\w+)'", table))
    migrated |= set(re.findall(r"ADD COLUMN IF NOT EXISTS (\w+)", added))
    (ddl,) = re.findall(r"CREATE TABLE IF NOT EXISTS heartbeat_results \((.*?)\n\s*\)", fresh, re.S)
    assert {line.split()[0] for line in ddl.strip().splitlines()} == migrated
    indexes = re.findall(r"op\.create_index\('(ix_heartbeat_results_\w+)'", created)
    assert indexes and all(f"IF NOT EXISTS {name} ON heartbeat_results " in fresh for name in indexes)
