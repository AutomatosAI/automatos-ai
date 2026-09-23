"""PRD-245 S0.4 — a held command reaches the Questions tab, the bell and Telegram.

Nineteen holds, zero answered (2026-09-17): a held command was a card on the
ticket's Canvas over live SSE and nowhere else. Now the host's
``PermissionRequest`` event becomes ONE PRD-225 question row through the shared
ask internals (the function ``platform_ask_human`` dispatches to) — unparked,
the ticket keeps running — with a marker in ``ApprovalGrant.details`` the answer
path reads. The Questions tab lists it with the ticket's cascade, the bell rings,
the Telegram bridge delivers and correlates it: none of them changed.

Pure tests over a fake session (the test_prd225_* pattern) driving the REAL
service functions end to end: event → row → answer (tab route, Telegram reply,
Canvas card) → the ticket's decision → the host's next event flush. Nothing is
re-queued; a result with the hold unanswered expires the row.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from channels.drivers.base import SendResult  # noqa: E402
from core.models.approval_grants import ApprovalGrant, GrantStatus, KIND_QUESTION  # noqa: E402
from core.models.core import Agent, BoardTask  # noqa: E402
from services import cli_host_service as svc  # noqa: E402

WS = uuid4()
HOST = SimpleNamespace(id=uuid4(), workspace_id=WS)
CTX = SimpleNamespace(workspace_id=WS, user_id=5, internal_user_id=5)
TICKET = 116


# ---------------------------------------------------------------------------
# Fake session — equality + ``in_`` filters off SQLAlchemy expressions, ``get``,
# and the filtered ``update`` the answer / expiry compare-and-swaps run.
# ---------------------------------------------------------------------------

class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *conds):
        rows = self._rows
        for cond in conds:
            key = cond.left.key
            op = getattr(getattr(cond, "operator", None), "__name__", "")
            value = getattr(cond.right, "value", None)
            if op == "in_op":
                allowed = {str(v) for v in (value or [])}
                rows = [r for r in rows if str(getattr(r, key, None)) in allowed]
            else:
                rows = [r for r in rows if str(getattr(r, key, None)) == str(value)]
        return _Query(rows)

    def order_by(self, *a):
        return _Query(list(reversed(self._rows)))

    def limit(self, *a):
        return self

    def get(self, pk):
        return next((r for r in self._rows if getattr(r, "id", None) == pk), None)

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)

    def update(self, values, synchronize_session=False):
        for r in self._rows:
            for col, val in values.items():
                setattr(r, getattr(col, "key", col), val)
        return len(self._rows)


class _ColumnQuery(_Query):
    """``db.query(Model.a, Model.b)`` — rows of the model as tuples (F091's
    grant_owners names each card's agent and ticket this way)."""

    def __init__(self, rows, keys):
        super().__init__(rows)
        self._keys = keys

    def filter(self, *conds):
        return _ColumnQuery(super().filter(*conds)._rows, self._keys)

    def all(self):
        return [tuple(getattr(r, k, None) for k in self._keys) for r in self._rows]


class _FakeSession:
    def __init__(self):
        self.rows = []
        self.commits = 0

    def add(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self.rows) + 1
        self.rows.append(obj)

    def flush(self):
        pass

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def refresh(self, obj):
        pass

    def query(self, *entities):
        if len(entities) == 1 and isinstance(entities[0], type):
            return _Query([r for r in self.rows if isinstance(r, entities[0])])
        model = entities[0].class_
        return _ColumnQuery([r for r in self.rows if isinstance(r, model)], [e.key for e in entities])


def _ticket(db):
    """A running session ticket claimed by HOST, one downstream task, its agent."""
    task = BoardTask(
        id=TICKET, workspace_id=WS, title="Find things out", status="in_progress", assigned_agent_id=57,
        review_mode="auto",
        runtime_ref={"runtime": "cli", "host_id": str(HOST.id), "session_id": "s-116", "attempt": 1},
    )
    db.add(task)
    db.add(BoardTask(id=TICKET + 1, workspace_id=WS, title="Write it up", status="assigned", parent_task_id=TICKET))
    db.add(Agent(id=57, name="RESEARCHER", agent_type="custom", configuration={"runtime": "cli"}))
    return task


def _hold(request_id, command="pip --version"):
    """The event ``session.py::_ask_operator`` puts on the bus for a held command."""
    return {"event": "PermissionRequest", "at": 1.0, "request_id": request_id, "tool_name": "Bash",
            "subject": command, "reason": f"'{command}' is outside this ticket's Bash allowlist", "session_id": "s-116"}


def _question_rows(db):
    return [r for r in db.rows if isinstance(r, ApprovalGrant)]


@pytest.fixture()
def quiet(monkeypatch):
    """Everything around the row, captured: lease, canvas, bell, Telegram, chat, dispatcher."""
    seen = {"bell": [], "telegram": [], "chat": [], "requeue": []}
    monkeypatch.setattr(svc, "renew_lease", lambda db, task_id, lease_seconds: True)
    monkeypatch.setattr(svc, "publish_canvas_events", lambda *a, **k: 0)

    async def _dispatch(self, **kw):
        seen["bell"].append(kw)
        return {"dispatched_to": ["in_app"]}

    monkeypatch.setattr("core.services.notification_dispatcher.NotificationDispatcher.dispatch", _dispatch)

    async def _send(**kw):
        seen["telegram"].append(kw)
        return SendResult(ok=True, latency_ms=1, message_id=str(700 + len(seen["telegram"])), target="chat-9")

    monkeypatch.setattr("channels.sender.send_to_channel", _send)
    monkeypatch.setattr("services.chat_messenger.deliver_background_message", lambda db, **kw: seen["chat"].append(kw))
    monkeypatch.setattr("services.board_dispatcher.notify_task_available", lambda *a, **k: seen["requeue"].append(k))
    return seen


# ===========================================================================
# 1. The event → one row, unparked, through the shared internals
# ===========================================================================

@pytest.mark.asyncio
async def test_a_hold_becomes_one_question_row_and_the_ticket_keeps_running(quiet):
    db = _FakeSession()
    task = _ticket(db)

    out = await svc.record_events(db, HOST, TICKET, [_hold("r1")])

    assert out["status"] == "in_progress" and out["decisions"] == []
    rows = _question_rows(db)
    assert len(rows) == 1
    q = rows[0]
    assert q.kind == KIND_QUESTION and q.status == GrantStatus.PENDING.value
    assert q.subject_type == "board_task" and q.subject_id == str(TICKET)
    assert q.options == ["allow", "deny"] and q.asked_by_agent_id == 57 and q.agent_id == 57
    assert q.details == {"cli_permission": {"request_id": "r1", "task_id": TICKET}}
    # night 1 (64fc8dc4f): what the agent wants in a sentence first, the exact
    # command folded away under it, the gate's reason last
    assert q.question_md.startswith(f"**Allow this command in ticket #{TICKET}?**\n\nThe agent wants to run **pip**.\n\n")
    assert "<summary>The exact command</summary>\n\n```sh\npip --version\n```" in q.question_md
    assert "outside this ticket's Bash allowlist" in q.question_md and "Answer `allow` or `deny`." in q.question_md
    # never parked — the host is waiting on the answer, the session is alive
    assert task.status == "in_progress" and task.blocked_reason is None
    # the pending entry remembers its row; the Canvas card still gets its request id
    assert task.runtime_ref["pending_permissions"][0]["grant_id"] == q.id
    assert task.runtime_ref["pending_permissions"][0]["request_id"] == "r1"
    # the bell and Telegram: the shared internals did it, nothing new here
    bell = quiet["bell"][0]
    assert bell["event_type"] == "question_pending" and bell["link_type"] == "question" and bell["link_id"] == str(q.id)
    assert bell["title"] == "Question from RESEARCHER" and bell["severity"] is None
    assert q.channel_refs["telegram"] == {"chat_id": "chat-9", "message_id": "701"}
    assert "/answer" in quiet["telegram"][0]["text"] and "```sh\npip --version\n```" in quiet["telegram"][0]["text"]


@pytest.mark.asyncio
async def test_a_reflushed_event_creates_nothing(quiet):
    """The host re-sends a batch whose POST failed (host.py::_flush_events)."""
    db = _FakeSession()
    _ticket(db)
    await svc.record_events(db, HOST, TICKET, [_hold("r1")])
    await svc.record_events(db, HOST, TICKET, [_hold("r1"), {"event": "PreToolUse", "tool_name": "Bash", "subject": "ls"}])

    rows = _question_rows(db)
    assert len(rows) == 1 and len(quiet["bell"]) == 1 and len(quiet["telegram"]) == 1
    task = db.query(BoardTask).get(TICKET)
    assert [p["grant_id"] for p in task.runtime_ref["pending_permissions"]] == [rows[0].id]


@pytest.mark.asyncio
async def test_two_holds_are_two_rows(quiet):
    db = _FakeSession()
    _ticket(db)
    await svc.record_events(db, HOST, TICKET, [_hold("r1"), _hold("r2", "python3 -m pip --version")])
    rows = _question_rows(db)
    assert [r.details["cli_permission"]["request_id"] for r in rows] == ["r1", "r2"]
    assert "```sh\npython3 -m pip --version\n```" in rows[1].question_md


@pytest.mark.asyncio
async def test_a_row_that_cannot_be_filed_leaves_the_canvas_card_and_the_flush_intact(quiet, monkeypatch):
    import modules.tools.discovery.handlers_asks as asks

    async def _boom(*a, **k):
        raise RuntimeError("grants table unavailable")

    monkeypatch.setattr(asks, "stage_question", _boom)
    db = _FakeSession()
    task = _ticket(db)
    out = await svc.record_events(db, HOST, TICKET, [_hold("r1")])
    assert out["status"] == "in_progress" and out["lease_renewed"] is True
    assert _question_rows(db) == []
    assert task.runtime_ref["pending_permissions"][0]["request_id"] == "r1"
    assert "grant_id" not in task.runtime_ref["pending_permissions"][0]


# ===========================================================================
# 2. The answer — Questions tab route → the ticket's decision → the host
# ===========================================================================

@pytest.mark.asyncio
async def test_answering_allow_in_the_questions_tab_reaches_the_host_and_requeues_nothing(quiet):
    from api.approval_grants import AnswerRequest, answer_question

    db = _FakeSession()
    task = _ticket(db)
    await svc.record_events(db, HOST, TICKET, [_hold("r1")])
    q = _question_rows(db)[0]

    res = await answer_question(q.id, AnswerRequest(option="allow"), CTX, db)

    assert res["grant"]["status"] == GrantStatus.GRANTED.value and q.answer_text == "allow"
    assert task.status == "in_progress" and quiet["requeue"] == []        # never re-queued: it is running
    assert task.runtime_ref["pending_permissions"] == []
    decision = task.runtime_ref["permission_decisions"]["r1"]
    assert decision["approved"] is True and decision["by"] == "user:5" and decision["delivered"] is False
    assert "resuming" in quiet["chat"][-1]["text"].lower()                # honest: the held command goes on
    # the next event flush carries it to the host, exactly once
    out = await svc.record_events(db, HOST, TICKET, [{"event": "PostToolUse", "tool_name": "Bash"}])
    assert out["decisions"] == [{"request_id": "r1", "approved": True}]
    assert (await svc.record_events(db, HOST, TICKET, []))["decisions"] == []


@pytest.mark.asyncio
async def test_answering_deny_records_a_denial_for_the_host(quiet):
    from api.approval_grants import AnswerRequest, answer_question

    db = _FakeSession()
    task = _ticket(db)
    await svc.record_events(db, HOST, TICKET, [_hold("r2", "python3 -m pip --version")])
    q = _question_rows(db)[0]

    await answer_question(q.id, AnswerRequest(answer_text="deny — not on this machine"), CTX, db)

    assert task.runtime_ref["permission_decisions"]["r2"]["approved"] is False
    out = await svc.record_events(db, HOST, TICKET, [])
    assert out["decisions"] == [{"request_id": "r2", "approved": False}]
    # what the host then reports for that command is a hold — the one kind that puts the ticket in review (S0.3)
    denied = svc._denial_summary({"tool": "Bash", "stage": "PreToolUse",
                                  "reason": "'python3 -m pip --version' is outside this ticket's Bash allowlist — denied by the operator"})
    assert denied["kind"] == "hold"
    # only 'allow' allows — anything else denies (fail closed)
    assert svc.is_allow_answer("Allow it") and svc.is_allow_answer("  allow")
    assert not svc.is_allow_answer("yes") and not svc.is_allow_answer("ok, allow") and not svc.is_allow_answer(None)


@pytest.mark.asyncio
async def test_an_answer_for_a_hold_already_decided_records_but_resumes_nothing(quiet):
    """The Canvas decided a hold whose row had not been linked yet: the tab's
    answer is recorded on the row, the ticket is untouched, and the confirmation
    does not claim anything resumed."""
    from api.approval_grants import apply_question_answer

    db = _FakeSession()
    task = _ticket(db)
    await svc.record_events(db, HOST, TICKET, [_hold("r1")])
    q = _question_rows(db)[0]
    ref = dict(task.runtime_ref)
    assert svc.record_permission_decision(ref, "r1", True, "user:2")
    task.runtime_ref = ref

    outcome = await apply_question_answer(db, q, answer_text="deny", answered_by="user:9")

    assert outcome.applied is True and outcome.resumed is False
    assert task.runtime_ref["permission_decisions"]["r1"]["approved"] is True   # the first decision stands
    assert "nothing to auto-resume" in quiet["chat"][-1]["text"]


# ===========================================================================
# 3. Telegram — the polling bridge's shared entry answers it, no new code
# ===========================================================================

@pytest.mark.asyncio
async def test_a_telegram_reply_answers_the_hold_through_the_same_path(quiet, monkeypatch):
    """``channels/telegram_adapter.py::_on_message`` hands every polled message to
    ``api.webhooks.maybe_answer_polled_telegram_message`` → ``_maybe_answer_question``
    (PRD-225 US-005). A reply to the correlated message answers the hold."""
    import api.webhooks as webhooks

    replies = []

    async def _reply(text, reply_ctx, integrations, *, workspace_id=None):
        replies.append(text)
        return True

    monkeypatch.setattr(webhooks, "_deliver_reply", _reply)
    db = _FakeSession()
    task = _ticket(db)
    await svc.record_events(db, HOST, TICKET, [_hold("r1")])
    q = _question_rows(db)[0]

    body = {"update_id": 1, "message": {
        "text": "allow", "chat": {"id": "chat-9"}, "from": {"id": 555, "first_name": "Ger"},
        "message_id": 1000, "reply_to_message": {"message_id": 701},
    }}
    reply_ctx = webhooks._extract_reply_context(body, "telegram")
    handled = await webhooks._maybe_answer_question(db, SimpleNamespace(id=WS), body, reply_ctx, {})

    assert handled["route_type"] == "question_answer" and handled["ask_id"] == q.id
    assert q.status == GrantStatus.GRANTED.value and q.answered_by == "telegram:555"
    decision = task.runtime_ref["permission_decisions"]["r1"]
    assert decision["approved"] is True and decision["by"] == "telegram:555"
    assert replies == [f"Answered #{q.id} — the agent is resuming."]
    assert task.status == "in_progress" and quiet["requeue"] == []


# ===========================================================================
# 4. The Canvas card — both surfaces resolve the same request id
# ===========================================================================

@pytest.mark.asyncio
async def test_the_canvas_card_answered_first_closes_the_row_and_a_second_answer_is_a_noop(quiet):
    from fastapi import HTTPException

    from api.approval_grants import AnswerRequest, answer_question

    db = _FakeSession()
    task = _ticket(db)
    await svc.record_events(db, HOST, TICKET, [_hold("r1")])
    q = _question_rows(db)[0]

    out = svc.decide_session_permission(db, task, "r1", False, "user:2")

    assert out == {"task_id": TICKET, "request_id": "r1", "approved": False, "pending": 0}
    assert q.status == GrantStatus.GRANTED.value and q.answer_text == "deny" and q.answered_by == "user:2"
    # the Questions tab's answer to the same hold: refused as already answered
    with pytest.raises(HTTPException) as ei:
        await answer_question(q.id, AnswerRequest(option="allow"), CTX, db)
    assert ei.value.status_code == 422 and "not open" in ei.value.detail
    # and the card's second answer: no pending question, said so
    with pytest.raises(LookupError, match="no pending permission question r1"):
        svc.decide_session_permission(db, task, "r1", True, "user:3")
    assert task.runtime_ref["permission_decisions"]["r1"]["approved"] is False


# ===========================================================================
# 5. The result — an unanswered hold's row expires; an answered one is left alone
# ===========================================================================

def _landing(monkeypatch):
    import api.board_tasks as board

    async def _finalize(db, **kw):
        return "review" if kw["force_review"] else "done"

    monkeypatch.setattr(board, "finalize_board_task_run", _finalize)
    monkeypatch.setattr(svc, "_register_session_deliverables", lambda *a, **k: [])


def _result_with_expired_hold(command="pip --version"):
    return {"attempt": 1, "status": "success", "result_text": "gave up on pip", "permission_denials": [{
        "tool": "Bash", "stage": "PreToolUse",
        "reason": f"'{command}' is outside this ticket's Bash allowlist — no answer from the operator within 120 s",
        "input": {"command": command},
    }]}


@pytest.mark.asyncio
async def test_the_result_expires_an_unanswered_hold_row(quiet, monkeypatch):
    import api.webhooks as webhooks

    _landing(monkeypatch)
    db = _FakeSession()
    task = _ticket(db)
    await svc.record_events(db, HOST, TICKET, [_hold("r1")])
    q = _question_rows(db)[0]

    out = await svc.apply_result(db, HOST, TICKET, _result_with_expired_hold())

    assert out == {"applied": True, "status": "review"}                    # the hold went unanswered → review
    assert q.status == GrantStatus.EXPIRED.value and q.revoked_by == f"cli-host:{HOST.id}" and q.revoked_at is not None
    assert task.runtime_ref["pending_permissions"] == []
    assert task.runtime_ref["expired_permissions"][0]["grant_id"] == q.id
    # gone from the Telegram bridge's view of open questions too
    assert webhooks._find_question_by_telegram_message(db, WS, "701", "chat-9") is None


@pytest.mark.asyncio
async def test_the_result_expires_a_hold_row_the_capped_pending_list_forgot(quiet, monkeypatch):
    """``pending_permissions`` keeps the last PENDING_PERMISSIONS_KEPT entries; a
    row whose entry was pushed out still belongs to this session and closes."""
    _landing(monkeypatch)
    db = _FakeSession()
    task = _ticket(db)
    await svc.record_events(db, HOST, TICKET, [_hold("r1")])
    q = _question_rows(db)[0]
    task.runtime_ref = {**task.runtime_ref, "pending_permissions": []}   # the entry is gone, the row is not

    await svc.apply_result(db, HOST, TICKET, {"attempt": 1, "status": "success", "result_text": "moved on"})

    assert q.status == GrantStatus.EXPIRED.value and q.revoked_by == f"cli-host:{HOST.id}"


@pytest.mark.asyncio
async def test_the_result_leaves_an_answered_row_alone(quiet, monkeypatch):
    from api.approval_grants import AnswerRequest, answer_question

    _landing(monkeypatch)
    db = _FakeSession()
    _ticket(db)
    await svc.record_events(db, HOST, TICKET, [_hold("r1")])
    q = _question_rows(db)[0]
    await answer_question(q.id, AnswerRequest(option="allow"), CTX, db)

    out = await svc.apply_result(db, HOST, TICKET, {"attempt": 1, "status": "success", "result_text": "pip works"})

    assert out == {"applied": True, "status": "done"}
    assert q.status == GrantStatus.GRANTED.value and q.answer_text == "allow"


# ===========================================================================
# 6. The Questions tab — the existing list route, with the ticket's cascade
# ===========================================================================

@pytest.mark.asyncio
async def test_the_questions_tab_lists_the_hold_with_the_tickets_cascade(quiet):
    from api.approval_grants import list_grants

    db = _FakeSession()
    _ticket(db)
    await svc.record_events(db, HOST, TICKET, [_hold("r1")])

    res = await list_grants(status="pending", kind="question", ctx=SimpleNamespace(workspace_id=WS, user_id=1), db=db)

    assert len(res["grants"]) == 1
    row = res["grants"][0]
    assert row["kind"] == KIND_QUESTION and row["subject_type"] == "board_task" and row["subject_id"] == str(TICKET)
    assert row["options"] == ["allow", "deny"]
    assert row["details"]["cli_permission"] == {"request_id": "r1", "task_id": TICKET}
    assert row["cascade"] == {"total": 1, "tasks": [{"id": TICKET + 1, "title": "Write it up", "status": "assigned"}]}


# ===========================================================================
# 7. The marker, and the ordinary PRD-225 board question still re-queues
# ===========================================================================

def test_the_marker_is_read_strictly():
    assert svc.session_hold_marker(SimpleNamespace(details={"cli_permission": {"request_id": "r", "task_id": 1}})) == {"request_id": "r", "task_id": 1}
    assert svc.session_hold_marker(SimpleNamespace(details={"cli_permission": {"request_id": "r"}})) is None
    assert svc.session_hold_marker(SimpleNamespace(details={"cli_permission": "r:1"})) is None
    assert svc.session_hold_marker(SimpleNamespace(details={"human_qa": []})) is None
    assert svc.session_hold_marker(SimpleNamespace(details=None)) is None
    # the row's expiry mirrors the host's --ask-timeout default
    # (services/cli-host config.DEFAULT_ASK_TIMEOUT_SECONDS), an hour since night 1 (64fc8dc4f)
    assert svc.SESSION_HOLD_OPTIONS == ("allow", "deny") and svc.SESSION_HOLD_TTL_SECONDS == 3600


@pytest.mark.asyncio
async def test_a_plain_board_question_still_requeues_the_parked_task(quiet):
    from api.approval_grants import apply_question_answer

    db = _FakeSession()
    parked = BoardTask(id=200, workspace_id=WS, title="Parked", status="blocked", blocked_reason="Awaiting human answer (ask #1)")
    db.add(parked)
    plain = ApprovalGrant(workspace_id=WS, subject_type="board_task", subject_id="200", kind=KIND_QUESTION,
                          question_md="Ship A or B?", status=GrantStatus.PENDING.value)
    db.add(plain)

    outcome = await apply_question_answer(db, plain, answer_text="A", answered_by="user:1")

    assert outcome.resumed is True and parked.status == "assigned" and len(quiet["requeue"]) == 1
