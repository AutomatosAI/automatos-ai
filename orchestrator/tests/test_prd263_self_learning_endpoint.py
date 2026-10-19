"""PRD-263 US-021 — GET /api/harness/self-learning reads the run table and counts only this workspace.

The tile's endpoint was an ``async def`` over a synchronous session (the request
blocked the event loop on every database read), with no role check, and its
Signals figure was the tool-signal recorder's in-process counter: one number for
every workspace that worker served, reset on each restart. It now:

* is a plain ``def``, gated so any member of the workspace reads it (the US-002
  decision, maintainer issue #1148, at its proposed default);
* returns the latest harness_runs run as get_status reads it (status,
  iteration_count, coverage and the INSUFFICIENT_DATA, WATCH and REGRESSION
  counts) and the workspace's HARNESS mode, counts only;
* counts ``signals_7d`` from this workspace's production tool_execution_logs rows
  of the last 7 days.
"""
from __future__ import annotations

import inspect
import json
import uuid
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from sqlalchemy import create_engine, text

from core.auth.dependencies import RequestContext, UserContext

ROUTE_PATH = "/api/harness/self-learning"
JUDGED_WEEK = "2026-W41"
EARLIER_WEEK = "2026-W40"


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


def _seed_workspace(new_session, settings=None) -> UUID:
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name, is_active, settings) "
                   "VALUES (CAST(:id AS uuid), 'prd263-self-learning', true, CAST(:settings AS jsonb))"),
              {"id": ws, "settings": json.dumps(settings or {})})
    s.commit()
    s.close()
    return UUID(ws)


def _sweep(new_session, workspaces) -> None:
    s = new_session.sweep()
    for ws in workspaces:
        for sql in ("DELETE FROM tool_execution_logs WHERE workspace_id = CAST(:id AS uuid)",
                    "DELETE FROM harness_runs WHERE workspace_id = CAST(:id AS uuid)",
                    "DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"):
            s.execute(text(sql), {"id": str(ws)})
    s.commit()
    s.close()


@pytest.fixture
def workspaces(new_session):
    made = []

    def make(settings=None):
        ws = _seed_workspace(new_session, settings)
        made.append(ws)
        return ws

    yield make
    _sweep(new_session, made)


def _add_run(new_session, ws: UUID, *, window: str, iteration: int, days_ago: int, coverage, counts, verdicts):
    started = datetime.now(timezone.utc) - timedelta(days=days_ago)
    s = new_session()
    s.execute(text(
        "INSERT INTO harness_runs (workspace_id, cur_window, started_at, finished_at, status, iteration, "
        "coverage, counts, verdicts, prescriptions, report_md, report_status) "
        "VALUES (CAST(:ws AS uuid), :window, :started, :finished, 'completed', :iteration, :coverage, "
        "CAST(:counts AS jsonb), CAST(:verdicts AS jsonb), CAST('[]' AS jsonb), '# report', 'ok')"),
        {"ws": str(ws), "window": window, "started": started, "finished": started + timedelta(minutes=1),
         "iteration": iteration, "coverage": coverage, "counts": json.dumps(counts),
         "verdicts": json.dumps(verdicts)})
    s.commit()
    s.close()


def _add_tool_logs(new_session, ws: UUID, n: int, *, source: str = "production", days_ago: float = 1) -> None:
    executed = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_ago)
    s = new_session()
    for _ in range(n):
        s.execute(text(
            "INSERT INTO tool_execution_logs (app_name, action_name, workspace_id, status, telemetry_source, "
            "executed_at) VALUES ('gmail', 'GMAIL_SEND_EMAIL', CAST(:ws AS uuid), 'success', :source, :executed)"),
            {"ws": str(ws), "source": source, "executed": executed})
    s.commit()
    s.close()


def _ctx(ws: UUID) -> RequestContext:
    return RequestContext(workspace_id=ws, user=UserContext(id="clerk_member"), auth_type="clerk")


def _read(new_session, ws: UUID):
    from api.harness import self_learning_health

    s = new_session()
    try:
        return self_learning_health(ctx=_ctx(ws), db=s)
    finally:
        s.close()


def test_reads_harness_runs(workspaces, new_session):
    ws = workspaces({"orchestrator": {"harness": {"mode": "propose"}}})
    _add_run(new_session, ws, window=EARLIER_WEEK, iteration=1, days_ago=8, coverage=0.25,
             counts={"active_agents": 4, "judged_agents": 1, "insufficient_data": 3, "watch": 0, "regression": 0},
             verdicts={"convergence": "insufficient_data", "agents": {}})
    _add_run(new_session, ws, window=JUDGED_WEEK, iteration=2, days_ago=1, coverage=0.5,
             counts={"active_agents": 4, "judged_agents": 2, "insufficient_data": 2, "watch": 1,
                     "regression": 0, "prescriptions": 1},
             verdicts={"convergence": "watching", "agents": {}})

    out = _read(new_session, ws)

    assert out == {
        "workspace_id": str(ws), "status": "completed", "iteration_count": 2, "coverage": 0.5,
        "insufficient_data": 2, "watch": 1, "regression": 0, "mode": "propose", "signals_7d": 0,
        "unavailable": [],
    }


def test_a_workspace_with_no_run_reads_none_not_zero(workspaces, new_session):
    ws = workspaces()

    out = _read(new_session, ws)

    # Fewer than three active agents and no run: the dormancy status, and no fake figures
    # (get_status says iteration_count 0 here; the tile's endpoint has no run to read).
    assert out["status"] == "dormant_insufficient_agents" and out["mode"] == "report_only"
    assert [out[field] for field in ("iteration_count", "coverage", "insufficient_data", "watch", "regression")] == [
        None, None, None, None, None]


def test_signal_count_is_workspace_scoped(workspaces, new_session):
    ws, other = workspaces(), workspaces()
    _add_tool_logs(new_session, ws, 3)
    _add_tool_logs(new_session, ws, 2, source="eval")      # not production traffic
    _add_tool_logs(new_session, ws, 1, days_ago=8)          # older than the 7-day window
    _add_tool_logs(new_session, other, 4)

    assert _read(new_session, ws)["signals_7d"] == 3
    assert _read(new_session, other)["signals_7d"] == 4


def _route():
    from api.harness import router

    (route,) = [r for r in router.routes if getattr(r, "path", None) == ROUTE_PATH]
    return route


def test_route_is_not_a_coroutine():
    from api.harness import self_learning_health

    route = _route()
    assert route.methods == {"GET"}
    assert route.endpoint is self_learning_health
    assert not inspect.iscoroutinefunction(self_learning_health)


def test_every_workspace_member_may_read_it_and_only_members():
    from api.harness import SELF_LEARNING_PERMISSION
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR, workspace_permission_granted
    from modules.policy.roles import WorkspaceRole, workspace_has_permission

    gates = [getattr(d.dependency, PERMISSION_MARKER_ATTR, None) for d in _route().dependencies]
    assert SELF_LEARNING_PERMISSION in gates
    assert all(workspace_has_permission(role.value, SELF_LEARNING_PERMISSION) for role in WorkspaceRole)
    # No role (an API key or SDK key with no membership) reads nothing.
    assert not workspace_has_permission(None, SELF_LEARNING_PERMISSION)
    sdk = RequestContext(workspace_id=uuid.uuid4(), user=UserContext(id="sdk"), auth_type="sdk_key")
    assert not workspace_permission_granted(None, sdk, SELF_LEARNING_PERMISSION)
