"""13 Nov 2026, Automatos HQ: a Claude Code agent's heartbeat reports when its session ends.

The News & Research agent (``runtime: cli``, hourly heartbeat, ``report_to: auto``) filed, every
hour, a report holding only "queued for your Claude Code session as ticket #0330" and a
"Review News & Research heartbeat — queued for …" ticket that ran Auto on nothing; an agent
reporting anywhere else got the same line as a notification. Proven here:

* a tick that only queued a session files no report and sends nothing, on any route;
* any other tick is routed exactly as before (auto, webhook, the workspace's channels), with the
  same message, and the triage shadow still sees it;
* the dispatcher's moved pieces and Auto's review ticket do what the inline code did;
* when the session's ticket ends, what it found goes through the same dispatcher, only for the
  heartbeat lane's tickets, never for Auto's own review card; the ticket's report path calls it.

PURE tests: fake sessions and recorded calls; no database.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS

import pytest

WS = "11111111-1111-1111-1111-111111111111"
QUEUED = [{"action": "file_cli_ticket", "task_id": 330}]
FOUND = "Nothing is assigned to me.\nThe board has 13 cards in review."


@pytest.mark.parametrize("actions, queued_only", [
    (QUEUED, True),
    (json.dumps(QUEUED), True),                                      # as a json column may hand it back
    (QUEUED + [{"action": "file_cli_ticket", "task_id": 331}], True),
    (QUEUED + [{"action": "create_task", "task_id": 9}], False),
    ([{"action": "llm_analysis"}], False),
    ([], False), (None, False), ("[]", False), ("not json", False),
])
def test_a_tick_that_only_queued_a_session(actions, queued_only):
    from services.heartbeats.session_ticks import only_queued_a_session

    assert only_queued_a_session(actions) is queued_only


# ── the dispatcher ──────────────────────────────────────────────────────────

def _dispatch(monkeypatch, result, hb_config):
    """Run the real dispatcher with its routes and the source lookup recorded."""
    from services import heartbeat_service
    from services.heartbeat_service import HeartbeatService
    from services.heartbeats import notification

    calls = []

    def _source(res):
        calls.append(("source", res.get("source_id")))
        return notification.HeartbeatSource(agent_id=17, agent_name="News & Research",
                                            title="News & Research Heartbeat", hb_config=hb_config)

    async def _auto(self, **kw):
        calls.append(("auto", kw))

    async def _webhook(self, **kw):
        calls.append(("webhook", kw))

    async def _workspace(**kw):
        calls.append(("workspace", kw))

    monkeypatch.setattr(notification, "source_of", _source)
    monkeypatch.setattr(notification, "dispatch_to_workspace", _workspace)
    monkeypatch.setattr(HeartbeatService, "_route_heartbeat_to_auto", _auto)
    monkeypatch.setattr(HeartbeatService, "_route_heartbeat_to_webhook", _webhook)
    monkeypatch.setattr(heartbeat_service, "_shadow_heartbeat_triage", lambda **kw: calls.append(("shadow", kw)))
    asyncio.run(HeartbeatService()._dispatch_heartbeat_notification(result))
    return calls


def _tick(actions, findings=None, **over):
    return {"workspace_id": WS, "source_type": "agent", "source_id": "17", "status": "success",
            "actions_taken": actions, "_heartbeat_result_id": 171,
            "findings": findings if findings is not None else [{"check": "llm_analysis", "detail": FOUND}], **over}


@pytest.mark.parametrize("hb_config", [{"report_to": "auto"}, {"report_to": "webhook", "webhook_url": "https://x"},
                                       {"report_to": "telegram"}, {}])
def test_a_queued_tick_sends_nothing_on_any_route(monkeypatch, hb_config):
    queued = _tick(QUEUED, [{"check": "cli_ticket", "detail": "queued for your Claude Code session as ticket #0330"}])
    assert _dispatch(monkeypatch, queued, hb_config) == []


def test_any_other_tick_goes_to_auto_as_before(monkeypatch):
    calls = _dispatch(monkeypatch, _tick([]), {"report_to": "auto"})
    assert [c[0] for c in calls] == ["source", "shadow", "auto"]
    assert calls[2][1] == {"workspace_id": WS, "source_agent_id": 17, "source_agent_name": "News & Research",
                           "title": "News & Research Heartbeat", "message": FOUND, "link_id": 171, "status": "ok"}
    assert calls[1][1]["platform_action"] == "report_to=auto" and calls[1][1]["message"] == FOUND


def test_a_webhook_and_the_workspace_get_it_as_before(monkeypatch):
    hook = _dispatch(monkeypatch, _tick([], status="error"), {"report_to": "webhook", "webhook_url": "https://x"})
    assert hook[-1] == ("webhook", {"webhook_url": "https://x", "title": "News & Research Heartbeat", "message": FOUND,
                                    "agent_id": 17, "agent_name": "News & Research", "link_id": 171, "status": "error"})
    channels = _dispatch(monkeypatch, _tick([]), {"report_to": "telegram"})
    assert channels[-1][0] == "workspace" and channels[-1][1]["message"] == FOUND
    assert channels[1][1]["platform_action"] == "report_to=telegram"


def test_a_dispatch_that_fails_never_reaches_the_tick(monkeypatch, caplog):
    from services.heartbeat_service import HeartbeatService
    from services.heartbeats import notification

    def _boom(_result):
        raise RuntimeError("database gone")

    monkeypatch.setattr(notification, "source_of", _boom)
    asyncio.run(HeartbeatService()._dispatch_heartbeat_notification(_tick([])))
    assert "Notification dispatch failed" in caplog.text


# ── the moved pieces ────────────────────────────────────────────────────────

def test_the_message_is_the_analysis_else_the_details_else_no_findings():
    from services.heartbeats.notification import DETAIL_MAX_CHARS, MESSAGE_MAX_CHARS, NO_FINDINGS, message_of

    long = "a" * (MESSAGE_MAX_CHARS + 5)
    assert message_of([{"check": "x", "detail": "d"}, {"check": "llm_analysis", "detail": long}]) == long[:MESSAGE_MAX_CHARS]
    assert message_of([{"check": "x", "detail": "b" * 400}, {"check": "y", "detail": "c"}]) == "b" * DETAIL_MAX_CHARS + "\nc"
    assert message_of([]) == message_of(None) == message_of([{"check": "x"}]) == NO_FINDINGS


def test_the_source_of_an_orchestrator_tick_or_an_unreadable_agent_needs_no_lookup():
    from services.heartbeats.notification import source_of

    orch = source_of({"source_type": "orchestrator"})
    assert (orch.agent_id, orch.agent_name, orch.title, orch.report_to) == (None, "Orchestrator", "Orchestrator Heartbeat", "")
    odd = source_of({"source_type": "agent", "source_id": "not-a-number"})
    assert (odd.agent_id, odd.agent_name, odd.title) == (None, None, "Agent Heartbeat")


def test_the_workspace_fan_out_dispatches_and_commits(monkeypatch):
    import core.database.database as database
    import core.services.notification_dispatcher as nd
    from services.heartbeats.notification import HeartbeatSource, dispatch_to_workspace

    sent, db = [], NS(committed=False, closed=False)
    db.commit = lambda: setattr(db, "committed", True)
    db.rollback = lambda: None
    db.close = lambda: setattr(db, "closed", True)

    class _Dispatcher:
        def __init__(self, session, workspace_id):
            assert session is db and workspace_id == WS

        async def dispatch(self, **kw):
            sent.append(kw)
            return {"dispatched_to": ["in_app"]}

    monkeypatch.setattr(database, "SessionLocal", lambda: db)
    monkeypatch.setattr(nd, "NotificationDispatcher", _Dispatcher)
    asyncio.run(dispatch_to_workspace(workspace_id=WS, source=HeartbeatSource(17, "News & Research", "N Heartbeat"),
                                      message="m", link_id=171, status="ok"))
    assert sent == [{"event_type": "heartbeat_complete", "title": "N Heartbeat", "message": "m", "link_type": "heartbeat",
                     "link_id": "171", "agent_id": 17, "agent_name": "News & Research", "status": "ok"}]
    assert db.committed and db.closed


def test_route_to_auto_files_the_review_ticket(monkeypatch):
    from services.heartbeat_service import HeartbeatService
    from services.heartbeats import auto_review

    filed = []
    monkeypatch.setattr(auto_review, "file_auto_review", lambda **kw: filed.append(kw) or 7)
    asyncio.run(HeartbeatService()._route_heartbeat_to_auto(
        workspace_id=WS, source_agent_id=17, source_agent_name="News & Research", title="t",
        message=FOUND, link_id=171, status="ok",
    ))
    assert filed == [{"workspace_id": WS, "source_agent_id": 17, "source_agent_name": "News & Research",
                      "message": FOUND, "link_id": 171, "status": "ok"}]


def test_the_review_says_what_the_heartbeat_found():
    from services.heartbeats.auto_review import _review_fields

    fields = _review_fields(source_agent_id=17, source_agent_name="News & Research", message=FOUND,
                            link_id=None, status="ok")
    assert fields["title"] == "Review News & Research heartbeat — Nothing is assigned to me."
    assert f"Findings:\n{FOUND}" in fields["description"]
    assert fields["priority"] == "medium" and fields["source_id"] is None and fields["created_by_id"] == "17"
    assert _review_fields(source_agent_id=None, source_agent_name="X", message="", link_id=171,
                          status="error")["priority"] == "high"


class _ReviewDb:
    def __init__(self, auto):
        self.auto, self.added, self.committed, self.closed = auto, [], False, False

    def query(self, _model):
        return self

    def filter(self, *_args):
        return self

    def first(self):
        return self.auto

    def add(self, row):
        self.added.append(row)

    def commit(self):
        self.committed = True
        for row in self.added:
            row.id = 400

    def rollback(self):
        pass

    def close(self):
        self.closed = True


@pytest.mark.parametrize("auto", [NS(id=1), None])
def test_auto_s_review_ticket_is_filed_for_the_workspace_s_auto(monkeypatch, auto):
    import core.database.database as database
    import services.board_consent as consent
    from services.heartbeats.auto_review import file_auto_review

    db, consented = _ReviewDb(auto), []
    monkeypatch.setattr(database, "SessionLocal", lambda: db)
    monkeypatch.setattr(consent, "consent_for_lane_ticket", lambda session, **kw: consented.append(kw))

    out = file_auto_review(workspace_id=WS, source_agent_id=17, source_agent_name="News & Research",
                           message=FOUND, link_id=171, status="ok")

    assert db.closed
    if auto is None:
        assert out is None and db.added == [] and consented == []
        return
    (task,) = db.added
    assert out == 400 and task.assigned_agent_id == 1 and task.status == "assigned"
    assert task.source_type == "heartbeat" and task.source_id == "171" and task.title.startswith("Review News")
    assert consented == [{"workspace_id": WS, "task": task, "source_type": "heartbeat"}]


def test_a_queued_tick_files_no_report(monkeypatch):
    import core.database.database as database
    from services.heartbeat_report import auto_create_heartbeat_report

    def _never():
        raise AssertionError("a queued tick opened a session to file a report")

    monkeypatch.setattr(database, "SessionLocal", _never)
    asyncio.run(auto_create_heartbeat_report(17, WS, {"actions_taken": QUEUED, "findings": []}))


# ── when the session ends ───────────────────────────────────────────────────

def _ticket(**over):
    base = dict(id=330, source_type="heartbeat", source_id="agent:17", assigned_agent_id=17,
                status="done", error_message=None)
    return NS(**{**base, **over})


def _finished(monkeypatch, task, answer="Heartbeat #330: nothing is assigned to me."):
    import services.heartbeat_service as hs
    from services.heartbeats.session_ticks import report_finished_session

    dispatched = []

    async def _dispatch(result):
        dispatched.append(result)

    monkeypatch.setattr(hs, "get_heartbeat_service", lambda: NS(_dispatch_heartbeat_notification=_dispatch))
    return asyncio.run(report_finished_session(WS, task, answer=answer)), dispatched


def test_a_finished_session_heartbeat_reports_what_it_found(monkeypatch):
    sent, (result,) = _finished(monkeypatch, _ticket())
    assert sent is True
    assert result == {"workspace_id": WS, "source_type": "agent", "source_id": "17", "status": "success",
                      "findings": [{"check": "llm_analysis", "detail": "Heartbeat #330: nothing is assigned to me."}],
                      "actions_taken": []}


def test_a_failed_session_heartbeat_reports_an_error(monkeypatch):
    _, (result,) = _finished(monkeypatch, _ticket(status="failed", error_message="usage limit"), answer="")
    assert result["status"] == "error" and result["findings"] == [{"check": "exec_error", "detail": "usage limit"}]


@pytest.mark.parametrize("task", [
    _ticket(source_id="171"),                                         # Auto's own review card: never again
    _ticket(source_id=None),
    _ticket(source_type="scheduled_task", source_id="task:1:20261113T0700"),
    _ticket(assigned_agent_id=None),
])
def test_nothing_else_reports(monkeypatch, task):
    sent, dispatched = _finished(monkeypatch, task)
    assert sent is False and dispatched == []


def test_the_ticket_report_path_hands_the_session_on(monkeypatch):
    import services.report_metrics as metrics
    from services import task_report
    from services.heartbeats import session_ticks
    from services.report_service import ReportService

    handed = []

    async def _metrics(*_args, **_kwargs):
        return {}

    async def _create(self, **_kwargs):
        return {"success": True}

    async def _report(workspace_id, task, *, answer):
        handed.append((workspace_id, task.id, answer))
        return True

    monkeypatch.setattr(metrics, "settled_execution_metrics", _metrics)
    monkeypatch.setattr(ReportService, "create_report", _create)
    monkeypatch.setattr(task_report, "_agent_name", lambda db, agent_id: "News & Research")
    monkeypatch.setattr(task_report, "task_report_content", lambda *args: "report")
    monkeypatch.setattr(session_ticks, "report_finished_session", _report)
    task = _ticket(title="Heartbeat: News & Research", result=None, started_at=None, completed_at=None)

    asyncio.run(task_report.auto_create_task_report(None, WS, task, {"result": "Nothing to pick up."}))

    assert handed == [(WS, 330, "Nothing to pick up.")]
