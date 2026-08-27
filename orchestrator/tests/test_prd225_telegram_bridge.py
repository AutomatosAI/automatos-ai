"""PRD-225 US-005 — the Telegram answer bridge.

Pure tests: outbound capture writes ``channel_refs.telegram`` (driver mocked) and
degrades fail-soft; inbound correlation answers a pending question from a reply
or ``/answer <id> <text>`` via the SHARED service (no HTTP self-call), leaves
unmatched messages to route as before, and safely no-ops a wrong-workspace /
already-answered target. Fixtures use obviously-fake ids (gitleaks-safe).
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from channels.drivers.base import SendResult
from core.models.approval_grants import ApprovalGrant, GrantStatus, KIND_QUESTION


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def get(self, pk):
        for r in self._rows:
            if getattr(r, "id", None) == pk:
                return r
        return None

    def filter(self, *conds):
        rows = self._rows
        for cond in conds:
            key = cond.left.key
            value = getattr(cond.right, "value", None)
            rows = [r for r in rows if str(getattr(r, key, None)) == str(value)]
        return _Query(rows)

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    def __init__(self):
        self.rows = []

    def add(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self.rows) + 1
        self.rows.append(obj)

    def flush(self):
        pass

    def commit(self):
        pass

    def query(self, model):
        return _Query([r for r in self.rows if isinstance(r, model)])


@pytest.fixture(autouse=True)
def _quiet_chat_confirm(monkeypatch):
    """The answer path confirms into chat; stub it (no DB/network)."""
    monkeypatch.setattr(
        "services.chat_messenger.deliver_background_message",
        lambda db, **kw: None,
    )


# ===========================================================================
# Outbound — capture message_id into channel_refs (driver mocked)
# ===========================================================================

@pytest.mark.asyncio
async def test_outbound_captures_channel_refs(monkeypatch):
    from modules.tools.discovery import handlers_asks

    async def fake_send(**kwargs):
        return SendResult(ok=True, latency_ms=1, message_id="777", target="chat-9")

    monkeypatch.setattr("channels.sender.send_to_channel", fake_send)
    grant = SimpleNamespace(id=5, channel_refs={"existing": 1})
    await handlers_asks._capture_question_telegram(
        _FakeSession(), uuid.uuid4(), grant, agent_name="Scout", question="Go?",
    )
    # Rebuild-don't-mutate: prior keys preserved, telegram ref added.
    assert grant.channel_refs == {
        "existing": 1,
        "telegram": {"chat_id": "chat-9", "message_id": "777"},
    }


@pytest.mark.asyncio
async def test_outbound_capture_failure_degrades(monkeypatch):
    from modules.tools.discovery import handlers_asks

    async def fake_send_fail(**kwargs):
        return SendResult(ok=False, latency_ms=1, error="no telegram connected")

    monkeypatch.setattr("channels.sender.send_to_channel", fake_send_fail)
    grant = SimpleNamespace(id=6, channel_refs=None)
    await handlers_asks._capture_question_telegram(
        _FakeSession(), uuid.uuid4(), grant, agent_name=None, question="Q",
    )
    assert grant.channel_refs is None  # in-app-only, no error


@pytest.mark.asyncio
async def test_outbound_capture_swallows_raise(monkeypatch):
    from modules.tools.discovery import handlers_asks

    async def fake_send_raise(**kwargs):
        raise RuntimeError("telegram down")

    monkeypatch.setattr("channels.sender.send_to_channel", fake_send_raise)
    grant = SimpleNamespace(id=7, channel_refs=None)
    # Must not raise.
    await handlers_asks._capture_question_telegram(
        _FakeSession(), uuid.uuid4(), grant, agent_name=None, question="Q",
    )
    assert grant.channel_refs is None


# ===========================================================================
# Inbound — reply / /answer correlation via the shared service
# ===========================================================================

def _tg_body(text, *, reply_to=None, chat="c1", first_name="Ger"):
    msg = {"text": text, "chat": {"id": chat}, "from": {"first_name": first_name}, "message_id": 1000}
    if reply_to is not None:
        msg["reply_to_message"] = {"message_id": reply_to}
    return {"update_id": 1, "message": msg}


def _question(db, ws, *, gid, telegram_message_id=None, subject_id="call-1"):
    refs = {"telegram": {"chat_id": "c1", "message_id": telegram_message_id}} if telegram_message_id else None
    g = ApprovalGrant(
        id=gid, workspace_id=ws, subject_type="tool_call", subject_id=subject_id,
        kind=KIND_QUESTION, question_md="Which vendor?", channel_refs=refs,
        status=GrantStatus.PENDING.value,
    )
    db.add(g)
    return g


@pytest.fixture()
def replies(monkeypatch):
    calls = []

    async def _record(text, reply_ctx, integrations, *, workspace_id=None):
        calls.append({"text": text, "workspace_id": workspace_id})
        return True

    monkeypatch.setattr("api.webhooks._deliver_reply", _record)
    return calls


@pytest.mark.asyncio
async def test_reply_to_correlated_message_answers(replies):
    from api.webhooks import _maybe_answer_question, _extract_reply_context

    db = _FakeSession()
    ws = uuid.uuid4()
    grant = _question(db, ws, gid=42, telegram_message_id="777")
    workspace = SimpleNamespace(id=ws)

    body = _tg_body("Use vendor X", reply_to=777)
    reply_ctx = _extract_reply_context(body, "telegram")
    res = await _maybe_answer_question(db, workspace, body, reply_ctx, {})

    assert res["route_type"] == "question_answer"
    assert res["ask_id"] == 42
    assert grant.status == GrantStatus.GRANTED.value
    assert grant.answer_text == "Use vendor X"
    assert grant.answered_by == "telegram:Ger"
    assert len(replies) == 1  # confirmation into the same thread


@pytest.mark.asyncio
async def test_slash_answer_command(replies):
    from api.webhooks import _maybe_answer_question, _extract_reply_context

    db = _FakeSession()
    ws = uuid.uuid4()
    grant = _question(db, ws, gid=8)
    workspace = SimpleNamespace(id=ws)

    body = _tg_body("/answer 8 ship it now")
    reply_ctx = _extract_reply_context(body, "telegram")
    res = await _maybe_answer_question(db, workspace, body, reply_ctx, {})

    assert res["route_type"] == "question_answer"
    assert grant.status == GrantStatus.GRANTED.value
    assert grant.answer_text == "ship it now"


@pytest.mark.asyncio
async def test_unmatched_message_falls_through(replies):
    from api.webhooks import _maybe_answer_question, _extract_reply_context

    db = _FakeSession()
    ws = uuid.uuid4()
    _question(db, ws, gid=8, telegram_message_id="777")
    workspace = SimpleNamespace(id=ws)

    # Plain chatter, no reply, no /answer → not correlated.
    body = _tg_body("hey what's the weather")
    reply_ctx = _extract_reply_context(body, "telegram")
    res = await _maybe_answer_question(db, workspace, body, reply_ctx, {})
    assert res is None
    assert replies == []


@pytest.mark.asyncio
async def test_reply_to_unknown_message_falls_through(replies):
    from api.webhooks import _maybe_answer_question, _extract_reply_context

    db = _FakeSession()
    ws = uuid.uuid4()
    _question(db, ws, gid=8, telegram_message_id="777")
    workspace = SimpleNamespace(id=ws)

    # A reply, but to a message we never correlated → fall through to routing.
    body = _tg_body("random reply", reply_to=999999)
    reply_ctx = _extract_reply_context(body, "telegram")
    res = await _maybe_answer_question(db, workspace, body, reply_ctx, {})
    assert res is None


@pytest.mark.asyncio
async def test_wrong_workspace_answer_is_safe(replies):
    from api.webhooks import _maybe_answer_question, _extract_reply_context

    db = _FakeSession()
    ws = uuid.uuid4()
    grant = _question(db, ws, gid=8)  # only #8 exists here
    workspace = SimpleNamespace(id=ws)

    body = _tg_body("/answer 999 sneaky")  # #999 is not in this workspace
    reply_ctx = _extract_reply_context(body, "telegram")
    res = await _maybe_answer_question(db, workspace, body, reply_ctx, {})

    assert res["reason"] == "answer_target_not_found"
    assert grant.status == GrantStatus.PENDING.value  # nothing changed
    assert len(replies) == 1
    assert "isn't open" in replies[0]["text"]


@pytest.mark.asyncio
async def test_already_answered_target_is_safe(replies):
    from api.webhooks import _maybe_answer_question, _extract_reply_context

    db = _FakeSession()
    ws = uuid.uuid4()
    grant = _question(db, ws, gid=8)
    grant.status = GrantStatus.GRANTED.value  # already answered
    workspace = SimpleNamespace(id=ws)

    body = _tg_body("/answer 8 too late")
    reply_ctx = _extract_reply_context(body, "telegram")
    res = await _maybe_answer_question(db, workspace, body, reply_ctx, {})
    assert res["reason"] == "answer_target_not_found"  # not pending → not found
    assert grant.answer_text is None
