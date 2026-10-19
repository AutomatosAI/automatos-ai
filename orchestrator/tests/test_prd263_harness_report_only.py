"""PRD-263 US-016 — HARNESS reports and changes nothing.

At 697ca7d6e every workspace defaulted to ``full_auto``: a low-risk prescription was
applied through PlatformActionExecutor with no owner's click, a risky one was filed
as a [HARNESS] board task (and a risk-4 one sent a channel notice), and done
[HARNESS] tasks were applied on the next tick. Through beta every workspace is
``report_only`` (D2): a run's suggestions are lines of its audit report and nothing
else, ``propose`` behaves the same until US-054, and ``disabled`` (or a stored
``enabled: false``) skips the workspace. The mode is read through
services.harness_mode, which maps the values stored before PRD-263.

Each case runs a full tick on committed rows (the tick opens its own sessions). The
seeded evidence is an agent with a week of tickets and a judged week in which every
GMAIL call failed on its expired connection, so PRESCRIBE has a "reconnect GMAIL" to
suggest (PRD-263 US-019 replaced the tool-routing graph's lifetime failure count, and
the tool removal it suggested, with that week's production tool logs). The platform
executor records every
action and runs the real platform_submit_report handler (ReportService) with the
workspace file and the knowledge ingest stubbed; every notification the run
dispatches and every channel notice it sends is recorded, and so is any call to the
decision engine's report triage or to an LLM client.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy import create_engine, text

from tests import helpers_harness_diagnose as hdx
from tests import helpers_harness_runs as hr

# agent_reports has no SQLAlchemy model, so a database built from the models has no
# such table; the audit report's insert needs these columns (ReportService.create_report).
AGENT_REPORTS = """
    CREATE TABLE agent_reports (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        workspace_id UUID NOT NULL, agent_id INTEGER, agent_name TEXT, heartbeat_result_id INTEGER,
        report_type TEXT, title TEXT, summary TEXT, status TEXT,
        file_path TEXT, file_type TEXT, file_size_bytes INTEGER,
        metrics JSONB, attachments JSONB, recommendations JSONB, action_items JSONB, linked_task_ids JSONB,
        requires_approval BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMPTZ DEFAULT now(), updated_at TIMESTAMPTZ DEFAULT now()
    )
"""
FAILING_APP = "GMAIL"
EXPIRED = "Connected account ca_scout_gmail has expired; reconnect it to continue"


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    with eng.begin() as conn:
        built = not conn.execute(text("SELECT to_regclass('agent_reports') IS NOT NULL")).scalar()
        if built:
            conn.execute(text(AGENT_REPORTS))
    yield eng
    if built:
        with eng.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS agent_reports"))
    eng.dispose()


@pytest.fixture
def workspace(new_session):
    ws = hr.seed_workspace(new_session)
    yield ws
    s = new_session.sweep()
    for sql in ("DELETE FROM tool_execution_logs WHERE workspace_id = CAST(:id AS uuid)",
                "DELETE FROM agent_reports WHERE workspace_id = CAST(:id AS uuid)"):
        s.execute(text(sql), {"id": str(ws)})
    s.commit()
    s.close()
    hr.sweep_workspace(new_session, ws)


@pytest.fixture
def record(monkeypatch, tmp_path):
    """What a run does: its platform actions, notifications, channel notices,
    report-triage calls and LLM calls."""
    seen = SimpleNamespace(actions=[], notifications=[], channel_notices=[], triage=[], llm_calls=[])
    hr.fake_platform(monkeypatch, tmp_path)
    monkeypatch.setattr("modules.tools.discovery.platform_executor.PlatformActionExecutor", _recording_executor(seen))
    _stub_report_side_effects(monkeypatch, seen)
    _record_llm(monkeypatch, seen)
    return seen


def _recording_executor(seen):
    class _Executor:
        def __init__(self, db=None, workspace_id=None, **_kwargs):
            self.db, self.workspace_id = db, workspace_id

        async def execute(self, action, params, caller_context=None):
            seen.actions.append(action)
            if action == hr.REPORT:
                from modules.tools.discovery.handlers_reports import submit_report

                return await submit_report(self.db, self.workspace_id, params)
            return {"success": True, "data": {"id": 1}}

    return _Executor


def _stub_report_side_effects(monkeypatch, seen):
    import core.services.notification_service as notification_service
    import services.knowledge_flywheel as flywheel
    import services.report_service as report_service
    from config import config
    from core.services.notification_dispatcher import NotificationDispatcher

    class _Files:
        def __init__(self, workspace_id):
            pass

        async def write_file(self, path, content):
            return {"success": True}

    async def _no_ingest(*_args, **_kwargs):
        return None

    async def _dispatch(self, event_type, title, *_args, **_kwargs):
        seen.notifications.append(event_type)
        return {}

    async def _channel_notice(workspace_id, message, channel=None):
        seen.channel_notices.append(message)
        return True

    monkeypatch.setattr(report_service, "WorkspaceClient", _Files)
    monkeypatch.setattr(flywheel, "ingest_agent_output", _no_ingest)
    monkeypatch.setattr(NotificationDispatcher, "dispatch", _dispatch)
    monkeypatch.setattr(notification_service, "send_workspace_notification", _channel_notice)
    monkeypatch.setattr(config, "DOCUMENT_THUMBNAILS_ENABLED", False)


def _record_llm(monkeypatch, seen):
    """Any report triage or LLM call recorded. The dials are the engine's real ones,
    reloaded from the test database, where nothing turns report_triage_mode on (FR-28)."""
    from core.llm import manager as llm_manager
    from core.llm.decisions import engine as decisions

    async def _generate(self, *_args, **_kwargs):
        seen.llm_calls.append("generate_response")

    decisions.get_decision_engine().reload()
    monkeypatch.setattr(decisions.DecisionEngine, "shadow",
                        lambda self, coro, purpose="shadow": seen.triage.append(purpose) or False)
    monkeypatch.setattr(llm_manager.LLMManager, "generate_response", _generate)
    monkeypatch.setattr(llm_manager.LLMManager, "generate_response_sync",
                        lambda self, *_args, **_kwargs: seen.llm_calls.append("generate_response_sync"))


def _seed_evidence(new_session, ws: UUID) -> int:
    """Scout: two weeks of tickets and, in the judged week, 20 GMAIL calls that all failed
    on an expired connection (a "reconnect GMAIL" to suggest)."""
    s = new_session()
    scout = hdx.agent(s, ws, "Scout")
    hdx.tickets(s, ws, scout, -1, 16, 4)
    hdx.tickets(s, ws, scout, 0, 14, 6)
    hdx.tool_runs(s, ws, scout, 0, 20, app=FAILING_APP, status="error", error_message=EXPIRED)
    s.commit()
    s.close()
    return scout


def _store_harness(new_session, ws: UUID, harness) -> None:
    s = new_session()
    s.execute(text("UPDATE workspaces SET settings = CAST(:settings AS jsonb) WHERE id = CAST(:id AS uuid)"),
              {"settings": json.dumps({"orchestrator": {"harness": harness}}), "id": str(ws)})
    s.commit()
    s.close()


def _count(new_session, sql: str, ws: UUID) -> int:
    s = new_session()
    try:
        return s.execute(text(sql), {"id": str(ws)}).scalar()
    finally:
        s.close()


def _board_tasks(new_session, ws: UUID) -> int:
    return _count(new_session, "SELECT count(*) FROM board_tasks WHERE workspace_id = CAST(:id AS uuid)", ws)


def _grants(new_session, ws: UUID) -> int:
    return _count(new_session, "SELECT count(*) FROM approval_grants WHERE workspace_id = CAST(:id AS uuid)", ws)


async def _swept(monkeypatch) -> list:
    """The workspace ids a sweep ticks. The tick only records them, so nothing runs
    for the other workspaces in the shared test database."""
    from services.harness_service import HarnessService

    ticked = []

    async def _tick(self, workspace_id, sweep=False):
        ticked.append(workspace_id)

    monkeypatch.setattr(HarnessService, "_harness_tick", _tick)
    await HarnessService()._harness_sweep()
    return ticked


def _assert_reported_only(record, new_session, ws: UUID, tickets_before: int) -> dict:
    """One completed run whose suggestion is a report line; nothing filed, granted or sent but the report."""
    from services.harness_service import RECONNECT_RISK, SUGGESTIONS_HEADING

    (run,) = hr.runs(new_session, ws)
    assert (run["status"], run["report_status"]) == ("completed", "ok")
    (rx,) = run["prescriptions"]
    assert (rx["prescription_id"], rx["change_type"], rx["risk_score"], rx["proposed_value"]) == (
        f"rx-{run['id']}-001", "reconnect_app", RECONNECT_RISK, {"app_name": FAILING_APP})
    assert SUGGESTIONS_HEADING in run["report_md"] and rx["prescription_id"] in run["report_md"]
    assert f"reconnect {FAILING_APP}" in run["report_md"]
    assert record.actions == [hr.REPORT]               # no platform write but the audit report
    assert record.notifications == ["report_submitted"]
    assert record.channel_notices == []                # no escalation
    assert _board_tasks(new_session, ws) == tickets_before and _grants(new_session, ws) == 0
    return run


@pytest.mark.asyncio
async def test_a_tick_on_seeded_evidence_reports_its_suggestion_and_changes_nothing(
        workspace, record, new_session, monkeypatch):
    from config import config
    from services.harness_service import HarnessService

    _seed_evidence(new_session, workspace)
    before = _board_tasks(new_session, workspace)
    # The legacy global flag on changes nothing: a report_only tick still applies nothing
    # (this replaces Half 2 of test_prd143_full_surface_positive.test_kill_switch_halts).
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)

    await HarnessService()._harness_tick(workspace)

    _assert_reported_only(record, new_session, workspace, before)
    # FR-28: with the report_triage_mode dial off by default, no triage and no LLM call.
    assert record.triage == [] and record.llm_calls == []


@pytest.mark.asyncio
async def test_trigger_now_reports_and_changes_nothing(workspace, record, new_session):
    from services.harness_service import HarnessService

    _seed_evidence(new_session, workspace)
    before = _board_tasks(new_session, workspace)
    running = asyncio.all_tasks()

    await HarnessService().trigger_now(workspace)
    await asyncio.gather(*(asyncio.all_tasks() - running))

    _assert_reported_only(record, new_session, workspace, before)


@pytest.mark.asyncio
@pytest.mark.parametrize("stored, reads", [
    ({"mode": "full_auto"}, "report_only"),
    ({"mode": "manual"}, "report_only"),
    ({"mode": "full_auto", "enabled": True}, "report_only"),
    ({"mode": "propose"}, "propose"),
])
async def test_a_legacy_or_propose_workspace_runs_report_only(workspace, record, new_session, monkeypatch,
                                                              stored, reads):
    """A stored full_auto or manual reads report_only; propose runs exactly like it until US-054.
    The sweep includes it."""
    from services.harness_mode import harness_mode
    from services.harness_service import HarnessService

    _seed_evidence(new_session, workspace)
    _store_harness(new_session, workspace, stored)
    before = _board_tasks(new_session, workspace)

    await HarnessService()._harness_tick(workspace)

    _assert_reported_only(record, new_session, workspace, before)
    assert harness_mode({"orchestrator": {"harness": stored}}) == reads
    assert workspace in await _swept(monkeypatch)


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", [
    {"enabled": False}, {"mode": "full_auto", "enabled": False}, {"disabled": True}, {"mode": "disabled"},
])
async def test_a_disabled_workspace_is_skipped_by_the_tick_trigger_and_sweep(workspace, record, new_session,
                                                                            monkeypatch, stored):
    from modules.tools.discovery.handlers_harness import harness_trigger
    from services.harness_service import DISABLED_MESSAGE, HarnessService

    _seed_evidence(new_session, workspace)
    _store_harness(new_session, workspace, stored)

    await HarnessService()._harness_tick(workspace)
    running = asyncio.all_tasks()
    await HarnessService().trigger_now(workspace)
    await asyncio.gather(*(asyncio.all_tasks() - running))

    assert hr.runs(new_session, workspace) == [] and record.actions == [] and record.notifications == []
    s = new_session()
    try:
        ws_row = s.execute(text("SELECT id, settings FROM workspaces WHERE id = CAST(:id AS uuid)"),
                           {"id": str(workspace)}).one()
        assert HarnessService._workspace_opted_in(ws_row) is False
        status = HarnessService().get_status(workspace, db=s)
        reply = await harness_trigger(s, workspace, {})  # Auto's platform_harness_trigger
    finally:
        s.close()
    assert (status["status"], status["message"]) == ("disabled", DISABLED_MESSAGE)
    assert "Report only" in DISABLED_MESSAGE and "Settings > Orchestrator" in DISABLED_MESSAGE
    # Auto is told nothing ran, not that a run was triggered.
    assert reply == {"success": True, "data": {"status": "disabled", "message": DISABLED_MESSAGE}}
    assert workspace not in await _swept(monkeypatch)
