"""13 Nov 2026, Automatos HQ: a Claude Code agent's heartbeat reports when its session ends.

The News & Research agent (``runtime: cli``, hourly heartbeat, ``report_to: auto``) filed, every
hour, a report holding only "queued for your Claude Code session as ticket #0330" and a
"Review News & Research heartbeat — queued for …" ticket that ran Auto on nothing. Proven here:

* a tick that only queued a session files no report and no review ticket;
* any other tick files its review exactly as before;
* when the session's ticket ends, Auto's review carries what the session found, only for the
  heartbeat lane's tickets of an agent that reports to Auto, never for Auto's own review card;
* the ticket's report path calls it.

PURE tests: fake sessions and recorded calls; no database.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS

import pytest

WS = "11111111-1111-1111-1111-111111111111"
QUEUED = [{"action": "file_cli_ticket", "task_id": 330}]


@pytest.mark.parametrize("actions, queued_only", [
    (QUEUED, True),
    (json.dumps(QUEUED), True),                                      # as a json column may hand it back
    (QUEUED + [{"action": "file_cli_ticket", "task_id": 331}], True),
    (QUEUED + [{"action": "create_task", "task_id": 9}], False),
    ([{"action": "llm_analysis"}], False),
    ([], False), (None, False), ("[]", False), ("not json", False),
])
def test_a_tick_that_only_queued_a_session(actions, queued_only):
    from services.heartbeats.auto_review import only_queued_a_session

    assert only_queued_a_session(actions) is queued_only


def _route(monkeypatch, queued_only):
    from services.heartbeat_service import HeartbeatService
    from services.heartbeats import auto_review

    filed = []
    monkeypatch.setattr(auto_review, "tick_only_queued_a_session", lambda ws, link: queued_only)
    monkeypatch.setattr(auto_review, "file_auto_review", lambda **kw: filed.append(kw) or 7)
    asyncio.run(HeartbeatService()._route_heartbeat_to_auto(
        workspace_id=WS, source_agent_id=17, source_agent_name="News & Research", title="t",
        message="queued for your Claude Code session as ticket #0330", link_id=171, status="ok",
    ))
    return filed


class _TickDb:
    def __init__(self, actions=None, broken=False):
        self.actions, self.broken, self.params, self.closed = actions, broken, [], False

    def execute(self, _statement, params):
        if self.broken:
            raise RuntimeError("database gone")
        self.params.append(params)
        return NS(fetchone=lambda: (self.actions,) if self.actions is not None else None)

    def close(self):
        self.closed = True


@pytest.mark.parametrize("db, link_id, queued_only", [
    (_TickDb(QUEUED), 171, True),
    (_TickDb([{"action": "llm_analysis"}]), 171, False),
    (_TickDb(None), 171, False),                    # no such tick in this workspace: review as before
    (_TickDb(QUEUED), None, False),                 # the tick was never stored: review as before
    (_TickDb(broken=True), 171, False),             # unreadable: review as before, never lost
])
def test_the_stored_tick_decides_and_unknown_means_review(monkeypatch, db, link_id, queued_only):
    import core.database.database as database
    from services.heartbeats.auto_review import tick_only_queued_a_session

    monkeypatch.setattr(database, "SessionLocal", lambda: db)
    assert tick_only_queued_a_session(WS, link_id) is queued_only
    assert db.params in ([], [{"id": 171, "workspace_id": WS}])
    assert db.closed or link_id is None


def test_a_queued_tick_files_no_review_for_auto(monkeypatch):
    assert _route(monkeypatch, queued_only=True) == []


def test_any_other_tick_files_its_review_as_before(monkeypatch):
    (call,) = _route(monkeypatch, queued_only=False)
    assert call == {"workspace_id": WS, "source_agent_id": 17, "source_agent_name": "News & Research",
                    "message": "queued for your Claude Code session as ticket #0330", "link_id": 171,
                    "status": "ok"}


def test_a_queued_tick_files_no_report(monkeypatch):
    import core.database.database as database
    from services.heartbeat_report import auto_create_heartbeat_report

    def _never():
        raise AssertionError("a queued tick opened a session to file a report")

    monkeypatch.setattr(database, "SessionLocal", _never)
    asyncio.run(auto_create_heartbeat_report(17, WS, {"actions_taken": QUEUED, "findings": []}))


def test_the_review_says_what_the_heartbeat_found():
    from services.heartbeats.auto_review import _review_fields

    fields = _review_fields(source_agent_id=17, source_agent_name="News & Research",
                            message="Nothing is assigned to me.\nThe board has 13 cards in review.",
                            link_id=None, status="ok")
    assert fields["title"] == "Review News & Research heartbeat — Nothing is assigned to me."
    assert "Findings:\nNothing is assigned to me.\nThe board has 13 cards in review." in fields["description"]
    assert fields["priority"] == "medium" and fields["source_id"] is None and fields["created_by_id"] == "17"
    assert _review_fields(source_agent_id=None, source_agent_name="X", message="", link_id=171,
                          status="error")["priority"] == "high"


class _Db:
    """The one query ``review_finished_session`` makes: the agent and its heartbeat block."""

    def __init__(self, heartbeat):
        self.agent = NS(configuration={"heartbeat": heartbeat})

    def query(self, _model):
        return self

    def filter(self, *_args):
        return self

    def first(self):
        return self.agent


def _ticket(**over):
    base = dict(id=330, source_type="heartbeat", source_id="agent:17", assigned_agent_id=17,
                status="done", error_message=None)
    return NS(**{**base, **over})


def _finished(monkeypatch, task, heartbeat=None):
    from services.heartbeats import auto_review

    filed = []
    monkeypatch.setattr(auto_review, "file_auto_review", lambda **kw: filed.append(kw) or 9)
    db = _Db({"report_to": "auto"} if heartbeat is None else heartbeat)
    out = asyncio.run(auto_review.review_finished_session(db, WS, task, agent_name="News & Research",
                                                          answer="Heartbeat #330: nothing is assigned to me."))
    return out, filed


def test_a_finished_session_heartbeat_files_auto_s_review_with_its_findings(monkeypatch):
    out, (call,) = _finished(monkeypatch, _ticket())
    assert out == 9
    assert call == {"workspace_id": WS, "source_agent_id": 17, "source_agent_name": "News & Research",
                    "message": "Heartbeat #330: nothing is assigned to me.", "link_id": None, "status": "ok"}


def test_a_failed_session_heartbeat_reports_an_error(monkeypatch):
    _, (call,) = _finished(monkeypatch, _ticket(status="failed", error_message="usage limit"))
    assert call["status"] == "error"


@pytest.mark.parametrize("task, heartbeat", [
    (_ticket(source_id="171"), None),                         # Auto's own review card: never reviewed again
    (_ticket(source_type="scheduled_task", source_id="task:1:20261113T0700"), None),
    (_ticket(), {"report_to": "telegram"}),
    (_ticket(), {}),
])
def test_nothing_else_files_a_review(monkeypatch, task, heartbeat):
    out, filed = _finished(monkeypatch, task, heartbeat)
    assert out is None and filed == []


def test_the_ticket_report_path_asks_for_the_review(monkeypatch):
    import services.report_metrics as metrics
    from services import task_report
    from services.heartbeats import auto_review
    from services.report_service import ReportService

    asked = []

    async def _metrics(*_args, **_kwargs):
        return {}

    async def _create(self, **_kwargs):
        return {"success": True}

    async def _review(db, workspace_id, task, *, agent_name, answer):
        asked.append((workspace_id, task.id, agent_name, answer))

    monkeypatch.setattr(metrics, "settled_execution_metrics", _metrics)
    monkeypatch.setattr(ReportService, "create_report", _create)
    monkeypatch.setattr(task_report, "_agent_name", lambda db, agent_id: "News & Research")
    monkeypatch.setattr(task_report, "task_report_content", lambda *args: "report")
    monkeypatch.setattr(auto_review, "review_finished_session", _review)
    task = _ticket(title="Heartbeat: News & Research", result=None, started_at=None, completed_at=None)

    asyncio.run(task_report.auto_create_task_report(None, WS, task, {"result": "Nothing to pick up."}))

    assert asked == [(WS, 330, "News & Research", "Nothing to pick up.")]
