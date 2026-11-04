"""F105 (wave 2): a session host's event flush holds nothing on the event loop.

``POST /api/v1/cli-hosts/{id}/tasks/{task}/events`` (``record_events``) is posted
constantly (eight session agents on one install). Every batch ran the ticket lookup, the
lease renewal, its commits and the decisions hand-back synchronously on the event loop. A
batch carrying a held command or a plan also filed a question through
``handlers_asks.stage_question`` and then awaited the bell and the Telegram send while the
request's connection sat idle in transaction.

Now every step over the request's session runs in the threadpool, the request's
transaction ends before the sends, and the sends use a short-lived session of their own,
connected in the threadpool. The answers, the rows and the messages are unchanged.

DB-free: a fake session records the thread of every call and whether a transaction is open.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
from sqlalchemy.exc import TimeoutError as PoolTimeout

from channels.drivers.base import SendResult
from core.models.approval_grants import ApprovalGrant
from core.models.core import Agent, BoardTask
from modules.tools.discovery import handlers_asks
from services import cli_host_service as svc

WS = uuid4()
HOST = NS(id=uuid4(), workspace_id=WS)
TICKET = 105
TELEGRAM = {"telegram": {"chat_id": "chat-105", "message_id": "7105"}}


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *conds):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _Session:
    """A session that remembers the thread of each call and whether a transaction is open."""

    def __init__(self):
        self.rows, self.calls, self.open = [], [], False

    def use(self, call, *, ends=False):
        self.calls.append((call, threading.get_ident()))
        self.open = not ends

    def query(self, model):
        self.use("query")
        return _Query([row for row in self.rows if isinstance(row, model)])

    def add(self, row):
        self.use("add")
        row.id = row.id or 900 + len(self.rows)
        self.rows.append(row)

    def flush(self):
        self.use("flush")

    def refresh(self, row):
        self.use("refresh")

    def connection(self):
        self.use("connection")

    def commit(self):
        self.use("commit", ends=True)

    def rollback(self):
        self.use("rollback", ends=True)

    def close(self):
        self.use("close", ends=True)

    def names(self):
        return [call for call, _ in self.calls]

    def threads(self):
        return {thread for _, thread in self.calls}

    def questions(self):
        return [row for row in self.rows if isinstance(row, ApprovalGrant)]


@pytest.fixture
def seen(monkeypatch):
    """The request's session, the sends' own sessions, and what each send saw."""
    seen = NS(db=_Session(), sessions=[], sends=[], canvas=[])
    monkeypatch.setattr(svc, "renew_lease", lambda db, task_id, lease_seconds: db.use("lease") or True)
    monkeypatch.setattr(svc, "publish_canvas_events", lambda ws, events: seen.canvas.append(threading.get_ident()) or 0)
    monkeypatch.setattr("services.ask_cascade.count_downstream_blocked", lambda db, *a: db.use("cascade counted") or 0)
    monkeypatch.setattr("core.services.approval_grants._shadow_hold", lambda grant, workspace_id: None)

    def sends_session():
        seen.sessions.append(_Session())
        return seen.sessions[-1]

    async def bell(self, **kwargs):
        seen.sends.append(("bell", self.db, seen.db.open))
        return {"dispatched_to": ["in_app"]}

    async def telegram(**kwargs):
        seen.sends.append(("telegram", kwargs["db"], seen.db.open))
        return SendResult(ok=True, latency_ms=1, message_id="7105", target="chat-105")

    monkeypatch.setattr("core.database.database.SessionLocal", sends_session)
    monkeypatch.setattr("core.services.notification_dispatcher.NotificationDispatcher.dispatch", bell)
    monkeypatch.setattr("channels.sender.send_to_channel", telegram)
    return seen


def _ticket(db, status="in_progress", **ref):
    task = BoardTask(id=TICKET, workspace_id=WS, title="Ship the page", status=status, assigned_agent_id=57,
                     runtime_ref={"runtime": "cli", "host_id": str(HOST.id), "session_id": "s-105", "attempt": 1,
                                  **ref})
    db.rows.extend([task, Agent(id=57, name="CODER", agent_type="custom", configuration={"runtime": "cli"})])
    return task


def _hold(request_id):
    return {"event": "PermissionRequest", "at": 1.0, "request_id": request_id, "tool_name": "Bash",
            "subject": "pip --version", "reason": "outside this ticket's Bash allowlist", "session_id": "s-105"}


def _flush(db, events):
    async def run():
        return threading.get_ident(), await svc.record_events(db, HOST, TICKET, events)

    return asyncio.run(run())


def _sends_saw(seen):
    """(what was sent, whether the request's session held a transaction) for each send."""
    return [(what, request_open) for what, _, request_open in seen.sends]


def test_a_batch_without_a_question_touches_its_session_only_off_the_loop(seen):
    _ticket(seen.db, permission_decisions={"r9": {"approved": True, "delivered": False}})

    loop, out = _flush(seen.db, [{"event": "PreToolUse", "tool_name": "Bash", "subject": "ls"},
                                 {"event": "PostToolUse", "tool_name": "Bash"}])

    assert out == {"status": "in_progress", "lease_renewed": True, "control": [],
                   "decisions": [{"request_id": "r9", "approved": True}]}
    assert seen.db.names() == ["query", "lease", "commit", "commit"]  # the ticket, the lease, the batch, the answers
    assert loop not in seen.db.threads() and seen.canvas and loop not in seen.canvas
    assert seen.sends == [] and seen.sessions == []


def test_a_held_command_is_filed_off_the_loop_and_its_sends_hold_no_request_transaction(seen):
    task = _ticket(seen.db)

    loop, out = _flush(seen.db, [_hold("r1")])

    assert out == {"status": "in_progress", "lease_renewed": True, "control": [], "decisions": []}
    (question,) = seen.db.questions()
    assert question.options == ["allow", "deny"] and question.details["cli_permission"]["request_id"] == "r1"
    assert task.runtime_ref["pending_permissions"][0]["grant_id"] == question.id
    assert question.channel_refs == TELEGRAM  # a reply to the Telegram message still answers it
    assert loop not in seen.db.threads()
    assert _sends_saw(seen) == [("bell", False), ("telegram", False)]
    (sends,) = seen.sessions
    assert all(session is sends for _, session, _ in seen.sends)
    assert sends.names() == ["connection", "commit", "close"] and loop not in sends.threads()


def test_a_plan_card_is_filed_off_the_loop_and_its_sends_hold_no_request_transaction(seen):
    task = _ticket(seen.db)

    loop, out = _flush(seen.db, [{"event": "PlanReady", "text": "1. Read the brief.\n2. Write the page."}])

    assert out["status"] == "in_progress" and out["lease_renewed"] is True
    (card,) = seen.db.questions()
    assert card.options == ["Approve", "Reject"] and card.details["cli_plan"]["task_id"] == TICKET
    assert task.runtime_ref["session_plans"][0]["grant_id"] == card.id
    assert card.channel_refs == TELEGRAM
    assert loop not in seen.db.threads()
    assert _sends_saw(seen) == [("bell", False), ("telegram", False)]


def test_a_terminal_batch_runs_off_the_loop(seen, monkeypatch):
    monkeypatch.setattr(svc, "notify_board_event", lambda db, **kwargs: db.use("board event"))
    _ticket(seen.db, status="done", mode=svc.SESSION_MODE_TERMINAL)

    loop, out = _flush(seen.db, [{"hook_event_name": "TerminalOpened", "session_id": "sid-105"}])

    assert out == {"status": "in_progress", "lease_renewed": False, "control": {}, "decisions": []}
    assert seen.db.names() == ["query", "board event", "commit"] and loop not in seen.db.threads()


def test_the_shared_ask_internals_answer_as_before_with_the_transaction_ended_before_the_sends(seen):
    park = BoardTask(id=7, workspace_id=WS, title="Price the box", status="in_progress")

    async def run():
        staged = await handlers_asks.stage_question(seen.db, WS, subject_type="board_task", subject_id="7",
                                                    question="Which box size?", park=park)
        return threading.get_ident(), staged

    loop, staged = asyncio.run(run())

    (question,) = seen.db.questions()
    assert staged == {
        "success": True, "ask_id": question.id, "parked": True, "subject_type": "board_task", "subject_id": "7",
        "downstream_blocked": 0,
        "message": f"Parked board_task 7 and asked the human (ask #{question.id}). "
                   "Move on to other work — the answer will resume it.",
    }
    assert (park.status, park.blocked_reason) == ("blocked", f"Awaiting human answer (ask #{question.id})")
    # filed, parked and committed; the count's read ended; the Telegram reference kept
    assert seen.db.names() == ["add", "flush", "commit", "cascade counted", "rollback", "commit"]
    assert loop not in seen.db.threads()
    assert _sends_saw(seen) == [("bell", False), ("telegram", False)]
    assert question.channel_refs == TELEGRAM


def _pool_exhausted():
    raise PoolTimeout("QueuePool limit reached, connection timed out")


def test_with_no_connection_for_the_sends_the_question_still_stands(seen, monkeypatch):
    def exhausted():
        sends = _Session()
        sends.connection = _pool_exhausted
        seen.sessions.append(sends)
        return sends

    monkeypatch.setattr("core.database.database.SessionLocal", exhausted)
    task = _ticket(seen.db)

    _, out = _flush(seen.db, [_hold("r1")])

    assert out["lease_renewed"] is True
    (question,) = seen.db.questions()
    assert task.runtime_ref["pending_permissions"][0]["grant_id"] == question.id
    assert seen.sends == [] and question.channel_refs is None
    assert seen.sessions[0].names() == ["close"]
