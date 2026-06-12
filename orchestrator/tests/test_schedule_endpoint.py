"""PRD-162 S1 — DB-first, stateless `get_schedule` + the shared schedule util.

Two layers:

  * Pure recurrence math (`schedule_util`) — no DB, no croniter needed for
    `interval_to_cron` (the legacy oracle); croniter-backed bits skip cleanly
    when the optional dep is absent (CI has it).
  * `ActivityService.get_schedule` over a tiny FAKE session (the repo's
    mock-db idiom) — proves the calendar is a STATELESS read: identical across
    instances/workers, one round-trip per source (no N+1), structured
    recurrence fields, and resilient (one bad source never blanks the rest).

No in-process APScheduler state is consulted, which is the whole point of the
PRD — so there is nothing per-worker to mock.
"""
from __future__ import annotations

import importlib.util as _ilu  # noqa: F401  (kept parallel to sibling test headers)
import os
from uuid import uuid4

import pytest

# Dummy POSTGRES_* satisfies the config import chain (blessed pattern — see
# tests/test_prd143_selection_at_scale.py). Nothing here touches a real DB.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from services import schedule_util  # noqa: E402
from services.activity_service import ActivityService  # noqa: E402

_HAS_CRONITER = _ilu.find_spec("croniter") is not None
_needs_croniter = pytest.mark.skipif(not _HAS_CRONITER, reason="croniter not installed (CI has it)")


# ── Pure recurrence math (no DB, no croniter) ──────────────────────────────

def test_interval_to_cron_matches_legacy_oracle():
    """interval_to_cron must reproduce the legacy heartbeat trigger exactly."""
    assert schedule_util.interval_to_cron(15) == "0,15,30,45 * * * *"
    assert schedule_util.interval_to_cron(30) == "0,30 * * * *"
    assert schedule_util.interval_to_cron(60) == "0 * * * *"
    assert schedule_util.interval_to_cron(120) == "0 */2 * * *"
    assert schedule_util.interval_to_cron(480) == "0 */8 * * *"
    assert schedule_util.interval_to_cron(1440) == "0 9 * * *"
    assert schedule_util.interval_to_cron(10080) == "0 9 * * 1"


def test_interval_to_cron_guards_nonsense():
    assert schedule_util.interval_to_cron(0) == "0 * * * *"
    assert schedule_util.interval_to_cron(-5) == "0 * * * *"


def test_parse_hhmm():
    assert schedule_util._parse_hhmm("09:30") == 570
    assert schedule_util._parse_hhmm("00:00") == 0
    assert schedule_util._parse_hhmm("nonsense") is None
    assert schedule_util._parse_hhmm(None) is None


# ── croniter-backed math ───────────────────────────────────────────────────

@_needs_croniter
def test_is_valid_cron():
    assert schedule_util.is_valid_cron("0 9 * * 1") is True
    assert schedule_util.is_valid_cron("*/15 * * * *") is True
    assert schedule_util.is_valid_cron("not a cron") is False
    assert schedule_util.is_valid_cron("") is False
    assert schedule_util.is_valid_cron(None) is False


@_needs_croniter
def test_next_run_is_future_and_utc():
    from datetime import datetime, timezone
    now = datetime(2026, 6, 12, 10, 0, tzinfo=timezone.utc)
    nxt = schedule_util.next_run("0 * * * *", now=now)
    assert nxt is not None
    assert nxt.tzinfo is not None
    assert nxt > now
    assert nxt.hour == 11 and nxt.minute == 0


@_needs_croniter
def test_next_run_active_hours_mask():
    """An occurrence outside the active window is skipped to the next inside it."""
    from datetime import datetime, timezone
    now = datetime(2026, 6, 12, 2, 0, tzinfo=timezone.utc)  # 02:00
    # hourly cron, but only 09:00–17:00 is active → next must land at 09:00
    nxt = schedule_util.next_run(
        "0 * * * *", now=now, tz="UTC", active_hours={"start": "09:00", "end": "17:00"}
    )
    assert nxt is not None and nxt.hour == 9


@_needs_croniter
def test_next_run_bad_expression_returns_none():
    assert schedule_util.next_run("totally invalid") is None


# ── get_schedule over a fake session — stateless DB read ───────────────────

class _Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def all(self):
        return self._rows


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeScheduleDB:
    """Minimal session: 1st query()→agents, 2nd→templates, execute()→tasks.

    Order is fixed because get_schedule calls heartbeat→playbook→task. Counts
    calls so a test can prove there is no per-row N+1.
    """

    def __init__(self, agents=None, templates=None, tasks=None, raise_on_query=None):
        self._agents = agents or []
        self._templates = templates or []
        self._tasks = tasks or []
        self._raise_on_query = raise_on_query
        self.query_calls = 0
        self.execute_calls = 0
        self.rollbacks = 0

    def query(self, *cols):
        self.query_calls += 1
        if self._raise_on_query == self.query_calls:
            raise RuntimeError("simulated source failure")
        return _FakeQuery(self._agents if self.query_calls == 1 else self._templates)

    def execute(self, stmt, params=None):
        self.execute_calls += 1
        return _FakeResult(self._tasks)

    def rollback(self):
        self.rollbacks += 1


def _svc(db):
    return ActivityService(db, uuid4())


def _heartbeat_agent(agent_id, minutes=30, **hb):
    cfg = {"heartbeat": {"interval_minutes": minutes, **hb}}
    return _Row(id=agent_id, name=f"Agent {agent_id}", configuration=cfg)


@_needs_croniter
def test_get_schedule_structured_recurrence_fields():
    db = _FakeScheduleDB(
        agents=[_heartbeat_agent(1, 30, timezone="UTC")],
        templates=[_Row(id=10, name="Weekly Report",
                        schedule_config={"type": "cron", "cron_expression": "0 9 * * 1"})],
        tasks=[],
    )
    out = _svc(db).get_schedule(range_days=30)
    assert out["scheduler_active"] is True
    assert len(out["scheduled"]) == 2
    for item in out["scheduled"]:
        rec = item["recurrence"]
        assert set(rec) == {"cron_expression", "interval_minutes", "timezone", "active_hours"}
        assert item["next_run_at"] is not None
    routine = next(i for i in out["scheduled"] if i["type"] == "routine")
    assert routine["recurrence"]["interval_minutes"] == 30
    assert routine["recurrence"]["cron_expression"] == "0,30 * * * *"


@_needs_croniter
def test_get_schedule_no_n_plus_one_at_scale():
    """200 heartbeat agents → still exactly 2 queries + 1 execute (no N+1)."""
    db = _FakeScheduleDB(
        agents=[_heartbeat_agent(i, 60) for i in range(200)],
        templates=[],
        tasks=[],
    )
    out = _svc(db).get_schedule(range_days=7)
    assert len(out["scheduled"]) == 200
    assert db.query_calls == 2      # agents + templates, regardless of row count
    assert db.execute_calls == 1    # scheduled tasks


@_needs_croniter
def test_get_schedule_identical_across_instances():
    """Two independent services over identical data → identical listing.

    This is the worker-invariance guarantee: no in-process scheduler state, so
    every worker computes the same answer from the same DB rows.
    """
    def fresh_db():
        return _FakeScheduleDB(
            agents=[_heartbeat_agent(1, 30), _heartbeat_agent(2, 1440)],
            templates=[_Row(id=10, name="P", schedule_config={"type": "cron", "cron_expression": "0 9 * * 1"})],
            tasks=[],
        )
    a = _svc(fresh_db()).get_schedule(range_days=30)
    b = _svc(fresh_db()).get_schedule(range_days=30)
    assert [i["id"] for i in a["scheduled"]] == [i["id"] for i in b["scheduled"]]
    assert [i["recurrence"]["cron_expression"] for i in a["scheduled"]] == \
           [i["recurrence"]["cron_expression"] for i in b["scheduled"]]


@_needs_croniter
def test_get_schedule_resilient_to_one_failing_source():
    """A failing source (Q49) must not blank the others; session is rolled back."""
    db = _FakeScheduleDB(
        agents=[_heartbeat_agent(1, 30)],
        templates=[],
        tasks=[],
        raise_on_query=2,  # the playbook/template query blows up
    )
    out = _svc(db).get_schedule(range_days=7)
    assert any(i["type"] == "routine" for i in out["scheduled"])  # heartbeat survived
    assert db.rollbacks >= 1


@_needs_croniter
def test_get_schedule_disabled_heartbeat_excluded():
    db = _FakeScheduleDB(
        agents=[_heartbeat_agent(1, 30, enabled=False), _heartbeat_agent(2, 30)],
        templates=[],
        tasks=[],
    )
    out = _svc(db).get_schedule(range_days=7)
    ids = {i["id"] for i in out["scheduled"]}
    assert "routine-2" in ids and "routine-1" not in ids


# ── S3: platform_get_schedule tool ─────────────────────────────────────────

@_needs_croniter
def test_platform_get_schedule_tool_returns_workspace_unified_feed():
    """The tool returns the SAME DB-first feed the calendar shows, scoped to ws."""
    import asyncio
    from modules.tools.discovery.handlers_scheduling import get_schedule as schedule_tool

    db = _FakeScheduleDB(
        agents=[_heartbeat_agent(1, 60)],
        templates=[_Row(id=10, name="Weekly", schedule_config={"type": "cron", "cron_expression": "0 9 * * 1"})],
        tasks=[],
    )
    out = asyncio.run(schedule_tool(db, uuid4(), {}))
    assert out["success"] is True
    assert out["count"] == 2
    assert {i["type"] for i in out["scheduled"]} == {"routine", "recipe"}
    assert all("recurrence" in i for i in out["scheduled"])
