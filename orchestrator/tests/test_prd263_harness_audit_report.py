"""PRD-263 US-048 — the weekly audit report goes straight to ReportService, attributed to Auto.

Until US-048 the tick submitted its audit report as the platform action
platform_submit_report, through the PlatformActionExecutor that HARNESS's direct
apply helper also used. With that helper gone, services.harness_service holds no
executor (the grep gate, FR-12): the report is one ReportService.create_report call,
and its agent_reports row names the workspace's Auto. The PRD-248 shadow triage still
wraps create_report: it starts with the decision engine's report_triage_mode dial on,
and not with it off (FR-28).

Each case runs a full tick on committed rows with the real ReportService (its workspace
file and knowledge ingest stubbed, thumbnails off). Any platform action is recorded and
answered without running its handler, so a report submitted that way leaves no row. The
triage is recorded when it is scheduled, and never run.
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy import text

from tests import helpers_harness_runs as hr

TITLE = "HARNESS Weekly Org Review — Run #1"


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    yield from hr.agent_reports_engine()


@pytest.fixture
def workspace(new_session):
    ws = hr.seed_workspace(new_session)
    yield ws
    s = new_session.sweep()
    s.execute(text("DELETE FROM agent_reports WHERE workspace_id = CAST(:id AS uuid)"), {"id": str(ws)})
    s.commit()
    s.close()
    hr.sweep_workspace(new_session, ws)


@pytest.fixture
def seen(monkeypatch, tmp_path):
    """What a run does: its platform actions, notifications, channel notices and report triage."""
    seen = SimpleNamespace(actions=[], notifications=[], channel_notices=[], triage=[])
    hr.pin_run(monkeypatch, tmp_path)
    hr.record_platform_actions(monkeypatch, seen.actions)
    hr.stub_report_side_effects(monkeypatch, seen)
    return seen


def _triage_dial(monkeypatch, seen, mode: str) -> None:
    """Set the report_triage_mode dial to ``mode``; record each triage scheduled, unrun."""
    from core.llm.decisions import engine as decisions

    def _shadow(self, coro, purpose="shadow"):
        coro.close()
        seen.triage.append(purpose)
        return True

    monkeypatch.setattr(decisions.DecisionEngine, "dials",
                        lambda self, force=False: decisions.Dials(report_triage_mode=mode))
    monkeypatch.setattr(decisions.DecisionEngine, "shadow", _shadow)


def _reports(new_session, ws: UUID) -> list:
    s = new_session()
    try:
        rows = s.execute(text("SELECT agent_id, agent_name, report_type, title FROM agent_reports "
                              "WHERE workspace_id = CAST(:id AS uuid)"), {"id": str(ws)}).mappings().all()
        return [dict(row) for row in rows]
    finally:
        s.close()


def _auto_id(new_session, ws: UUID) -> int:
    s = new_session()
    try:
        return s.execute(text("SELECT id FROM agents WHERE workspace_id = CAST(:id AS uuid) AND name = 'Auto'"),
                         {"id": str(ws)}).scalar_one()
    finally:
        s.close()


@pytest.mark.asyncio
async def test_the_audit_report_is_one_report_row_attributed_to_auto(workspace, seen, new_session, monkeypatch):
    from core.llm.decisions import MODE_OFF
    from services.harness_service import AUDIT_REPORT_TYPE, HarnessService

    _triage_dial(monkeypatch, seen, MODE_OFF)

    await HarnessService()._harness_tick(workspace)

    (run,) = hr.runs(new_session, workspace)
    assert (run["status"], run["report_status"]) == ("completed", "ok")
    assert _reports(new_session, workspace) == [{
        "agent_id": _auto_id(new_session, workspace), "agent_name": "Auto",
        "report_type": AUDIT_REPORT_TYPE, "title": TITLE,
    }]
    assert run["report_md"].startswith(f"# {TITLE}")
    assert seen.actions == []                        # no platform action
    assert seen.notifications == ["report_submitted"]
    assert seen.triage == []                         # FR-28: the dial is off


@pytest.mark.asyncio
async def test_the_shadow_triage_still_wraps_the_audit_report_when_its_dial_is_on(
        workspace, seen, new_session, monkeypatch):
    from core.llm.decisions import MODE_SHADOW, judgements
    from services.harness_service import HarnessService

    _triage_dial(monkeypatch, seen, MODE_SHADOW)

    await HarnessService()._harness_tick(workspace)

    assert [report["agent_name"] for report in _reports(new_session, workspace)] == ["Auto"]
    assert seen.actions == []
    assert seen.triage == [judgements.PURPOSE_REPORT]
