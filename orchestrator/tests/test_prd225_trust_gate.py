"""PRD-225 US-006 — the per-channel ingress trust gate.

Pure tests: the conservative classifier (fixture table), the per-mode hold
decision, the gate holding a directive as a question-kind row vs routing chatter,
the strict default, correlated-answer bypass, and no message body in gate logs.
"""
from __future__ import annotations

import logging
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

from core.models.approval_grants import ApprovalGrant, GrantStatus, KIND_QUESTION
from core.models.channels import ChannelConnection


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *conds):
        rows = self._rows
        for cond in conds:
            key = cond.left.key
            value = getattr(cond.right, "value", None)
            rows = [r for r in rows if str(getattr(r, key, None)) == str(value)]
        return _Query(rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


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


# ===========================================================================
# 1. The classifier — conservative fixture table
# ===========================================================================

_DIRECTIVES = [
    "delete the user account",
    "send an email to the vendor",
    "please refund order 123",
    "create a new task for the analyst",
    "run the weekly report",
    "can you update the pricing page",
    "schedule a call for tomorrow",
    "deploy the app to production",
    "the numbers look off, dig into Q3",  # ambiguous ⇒ directive (conservative)
    "reset my password",
]

_CHATTER = [
    "hi",
    "hello there",
    "thanks!",
    "thank you",
    "good morning team",
    "ok",
    "yes",
    "great, cheers",
    "how are you doing today",
    "👍",
]


@pytest.mark.parametrize("text", _DIRECTIVES)
def test_directives_classify_as_directive(text):
    from services.ingress_gate import classify_inbound_message, VERDICT_DIRECTIVE

    assert classify_inbound_message(text) == VERDICT_DIRECTIVE


@pytest.mark.parametrize("text", _CHATTER)
def test_chatter_classifies_as_chatter(text):
    from services.ingress_gate import classify_inbound_message, VERDICT_CHATTER

    assert classify_inbound_message(text) == VERDICT_CHATTER


def test_should_hold_per_mode():
    from services.ingress_gate import should_hold

    # allow_all routes everything.
    assert should_hold("allow_all", "delete production") is False
    # communication_only: chatter routes, directives held.
    assert should_hold("communication_only", "hello") is False
    assert should_hold("communication_only", "delete production") is True
    # strict holds everything.
    assert should_hold("strict", "hello") is True
    assert should_hold("strict", "delete production") is True
    # unknown mode fails safe to strict (hold).
    assert should_hold("bogus", "hello") is True


# ===========================================================================
# 2. The gate — hold vs route, default strict, no channel = no gate
# ===========================================================================

def _channel(ws, *, config, platform="telegram"):
    return ChannelConnection(id=uuid.uuid4(), workspace_id=ws, platform=platform, config=config)


def _tg(text):
    return {"update_id": 1, "message": {"text": text, "chat": {"id": "c1"}}}


@pytest.fixture()
def acks(monkeypatch):
    calls = []

    async def _record(text, reply_ctx, integrations, *, workspace_id=None):
        calls.append(text)
        return True

    monkeypatch.setattr("api.webhooks._deliver_reply", _record)
    return calls


@pytest.mark.asyncio
async def test_strict_holds_directive_as_question(acks):
    from api.webhooks import _apply_trust_gate

    db = _FakeSession()
    ws = uuid.uuid4()
    db.add(_channel(ws, config={"trigger_mode": "strict"}))
    workspace = SimpleNamespace(id=ws)

    res = await _apply_trust_gate(db, workspace, "telegram", _tg("delete the vault"), {"platform": "telegram"}, {})

    assert res["reason"] == "trust_gate_hold"
    grant = next(r for r in db.rows if isinstance(r, ApprovalGrant))
    assert grant.kind == KIND_QUESTION
    assert grant.subject_type == "channel"
    assert grant.status == GrantStatus.PENDING.value
    assert "delete the vault" in grant.question_md
    assert len(acks) == 1  # sender acknowledged


@pytest.mark.asyncio
async def test_allow_all_routes_everything(acks):
    from api.webhooks import _apply_trust_gate

    db = _FakeSession()
    ws = uuid.uuid4()
    db.add(_channel(ws, config={"trigger_mode": "allow_all"}))
    workspace = SimpleNamespace(id=ws)

    res = await _apply_trust_gate(db, workspace, "telegram", _tg("delete the vault"), {"platform": "telegram"}, {})
    assert res is None  # routes as today
    assert not any(isinstance(r, ApprovalGrant) for r in db.rows)


@pytest.mark.asyncio
async def test_communication_only_routes_chatter_holds_directive(acks):
    from api.webhooks import _apply_trust_gate

    ws = uuid.uuid4()
    workspace = SimpleNamespace(id=ws)

    # chatter routes
    db1 = _FakeSession()
    db1.add(_channel(ws, config={"trigger_mode": "communication_only"}))
    assert await _apply_trust_gate(db1, workspace, "telegram", _tg("thanks so much"), {"platform": "telegram"}, {}) is None

    # directive holds
    db2 = _FakeSession()
    db2.add(_channel(ws, config={"trigger_mode": "communication_only"}))
    res = await _apply_trust_gate(db2, workspace, "telegram", _tg("send the invoice now"), {"platform": "telegram"}, {})
    assert res["reason"] == "trust_gate_hold"


@pytest.mark.asyncio
async def test_default_mode_is_strict(acks):
    """A channel row with no stored trigger_mode behaves as strict."""
    from api.webhooks import _apply_trust_gate

    db = _FakeSession()
    ws = uuid.uuid4()
    db.add(_channel(ws, config={}))  # no trigger_mode key
    workspace = SimpleNamespace(id=ws)

    res = await _apply_trust_gate(db, workspace, "telegram", _tg("hello"), {"platform": "telegram"}, {})
    assert res is not None and res["reason"] == "trust_gate_hold"  # strict holds even chatter


@pytest.mark.asyncio
async def test_no_channel_row_means_no_gate(acks):
    """Legacy-integration inbound (no channel connection) is unchanged."""
    from api.webhooks import _apply_trust_gate

    db = _FakeSession()
    ws = uuid.uuid4()  # no channel added
    workspace = SimpleNamespace(id=ws)

    res = await _apply_trust_gate(db, workspace, "telegram", _tg("delete everything"), {"platform": "telegram"}, {})
    assert res is None
    assert not any(isinstance(r, ApprovalGrant) for r in db.rows)


# ===========================================================================
# 2b. P225-RVW-2 — the gate scores the SAME content the router routes, so
# caption / edited_message / Meta-WhatsApp shapes cannot bypass strict.
# ===========================================================================

@pytest.mark.parametrize("body,expected", [
    # Telegram caption-only media (no `text`, only `caption`).
    ({"message": {"caption": "wire it", "chat": {"id": "c1"}}}, "wire it"),
    # Telegram edited_message — body carries `edited_message`, not `message`.
    ({"edited_message": {"text": "edit me", "chat": {"id": "c1"}}}, "edit me"),
    ({"edited_message": {"caption": "edit cap", "chat": {"id": "c1"}}}, "edit cap"),
    # Meta-WhatsApp text message — no text/event/Body key at all.
    ({"entry": [{"changes": [{"value": {"messages": [{"text": {"body": "wa hi"}}]}}]}]}, "wa hi"),
])
def test_inbound_text_matches_ingestor_shapes(body, expected):
    """The gate extracts the same real-message content the WebhookIngestor routes
    (caption / edited_message / WhatsApp body) — never scores it empty."""
    from api.webhooks import _inbound_text

    assert _inbound_text(body) == expected


@pytest.mark.asyncio
async def test_strict_holds_whatsapp_text_message(acks):
    """A Meta-WhatsApp text message is HELD under strict — it has no
    message.text / event.text / Body key; the gate reads messages[].text.body."""
    from api.webhooks import _apply_trust_gate

    db = _FakeSession()
    ws = uuid.uuid4()
    db.add(_channel(ws, config={"trigger_mode": "strict"}, platform="whatsapp"))
    workspace = SimpleNamespace(id=ws)

    wa = {"entry": [{"changes": [{"value": {"messages": [
        {"text": {"body": "delete the vault"}, "from": "15551230000"}
    ]}}]}]}
    res = await _apply_trust_gate(db, workspace, "whatsapp", wa, {"platform": "whatsapp"}, {})

    assert res["reason"] == "trust_gate_hold"
    grant = next(r for r in db.rows if isinstance(r, ApprovalGrant))
    assert grant.kind == KIND_QUESTION and grant.subject_type == "channel"
    assert "delete the vault" in grant.question_md


@pytest.mark.asyncio
async def test_strict_holds_telegram_caption(acks):
    """A Telegram caption-only media message is HELD under strict (the ingestor
    routes on message.caption; the gate must score it too)."""
    from api.webhooks import _apply_trust_gate

    db = _FakeSession()
    ws = uuid.uuid4()
    db.add(_channel(ws, config={"trigger_mode": "strict"}))
    workspace = SimpleNamespace(id=ws)

    body = {"update_id": 3, "message": {
        "caption": "wire the funds now", "chat": {"id": "c1"},
        "photo": [{"file_id": "AgACfake"}],
    }}
    res = await _apply_trust_gate(db, workspace, "telegram", body, {"platform": "telegram"}, {})

    assert res["reason"] == "trust_gate_hold"
    grant = next(r for r in db.rows if isinstance(r, ApprovalGrant))
    assert "wire the funds now" in grant.question_md


@pytest.mark.asyncio
async def test_strict_holds_telegram_edited_message(acks):
    """A Telegram edited_message is HELD under strict — its body carries
    'edited_message', not 'message', so the ingestor would json.dumps it and the
    directive would otherwise reach the router."""
    from api.webhooks import _apply_trust_gate

    db = _FakeSession()
    ws = uuid.uuid4()
    db.add(_channel(ws, config={"trigger_mode": "strict"}))
    workspace = SimpleNamespace(id=ws)

    body = {"update_id": 4, "edited_message": {
        "text": "deploy to production", "chat": {"id": "c1"},
    }}
    res = await _apply_trust_gate(db, workspace, "telegram", body, {"platform": "telegram"}, {})

    assert res["reason"] == "trust_gate_hold"
    grant = next(r for r in db.rows if isinstance(r, ApprovalGrant))
    assert "deploy to production" in grant.question_md


# ===========================================================================
# 3. Correlated answers bypass the gate (the handler runs correlation first)
# ===========================================================================

@pytest.mark.asyncio
async def test_correlated_answer_bypasses_gate_under_strict(monkeypatch):
    """A reply to a correlated question is ANSWERED, never held — even on a
    strict channel — because correlation (2c) runs before the gate (2d)."""
    from api.webhooks import _maybe_answer_question, _apply_trust_gate, _extract_reply_context

    monkeypatch.setattr("services.chat_messenger.deliver_background_message", lambda db, **k: None)

    async def _reply(*a, **k):
        return True

    monkeypatch.setattr("api.webhooks._deliver_reply", _reply)

    db = _FakeSession()
    ws = uuid.uuid4()
    workspace = SimpleNamespace(id=ws)
    # A strict channel...
    db.add(_channel(ws, config={"trigger_mode": "strict"}))
    # ...and a pending question correlated to Telegram message 777.
    db.add(ApprovalGrant(
        id=50, workspace_id=ws, subject_type="tool_call", subject_id="c1",
        kind=KIND_QUESTION, question_md="Which vendor?",
        channel_refs={"telegram": {"chat_id": "c1", "message_id": "777"}},
        status=GrantStatus.PENDING.value,
    ))

    body = {"update_id": 2, "message": {"text": "vendor X", "chat": {"id": "c1"},
                                        "from": {"first_name": "Ger"},
                                        "reply_to_message": {"message_id": 777}}}
    reply_ctx = _extract_reply_context(body, "telegram")

    # Handler sequence: correlation first…
    answered = await _maybe_answer_question(db, workspace, body, reply_ctx, {})
    assert answered is not None and answered["route_type"] == "question_answer"
    # …so the gate is never consulted. If it were, it would NOT have held this
    # (it was answered), and no new 'channel' question row exists.
    assert not any(
        isinstance(r, ApprovalGrant) and r.subject_type == "channel" for r in db.rows
    )


# ===========================================================================
# 4. Gate log lines carry no message body
# ===========================================================================

@pytest.mark.asyncio
async def test_gate_log_has_no_message_body(acks, caplog):
    from api.webhooks import _apply_trust_gate

    db = _FakeSession()
    ws = uuid.uuid4()
    db.add(_channel(ws, config={"trigger_mode": "strict"}))
    workspace = SimpleNamespace(id=ws)

    secret = "wire 9000 to account ABC-SECRET"
    with caplog.at_level(logging.INFO):
        await _apply_trust_gate(db, workspace, "telegram", _tg(secret), {"platform": "telegram"}, {})

    gate_logs = [r.getMessage() for r in caplog.records if "trust-gate" in r.getMessage()]
    assert gate_logs, "expected a gate decision log line"
    assert all("ABC-SECRET" not in line and "wire 9000" not in line for line in gate_logs)


# ===========================================================================
# 7. P225-RVW-5 — the gate fails CLOSED on any internal error
# ===========================================================================

def _boom(*_a, **_k):
    raise RuntimeError("classifier exploded")


@pytest.mark.asyncio
async def test_gate_error_fails_closed_under_strict(acks, caplog, monkeypatch):
    """A classify-path error under a non-allow_all channel HOLDS (never routes)
    and logs at error level — an internal gate error must not silently open
    strict. ``res is not None`` is exactly what makes the handler return without
    reaching UniversalRouter / the platform-tool interception."""
    from api.webhooks import _apply_trust_gate

    # _apply_trust_gate imports should_hold from the module at call time, so
    # patching the attribute lands on the runtime import.
    monkeypatch.setattr("services.ingress_gate.should_hold", _boom)

    db = _FakeSession()
    ws = uuid.uuid4()
    db.add(_channel(ws, config={"trigger_mode": "strict"}))
    workspace = SimpleNamespace(id=ws)

    with caplog.at_level(logging.ERROR):
        res = await _apply_trust_gate(
            db, workspace, "telegram", _tg("delete the vault"),
            {"platform": "telegram"}, {},
        )

    assert res is not None and res["routed"] is False  # held → not routed
    assert res["reason"] == "trust_gate_error"
    assert not any(isinstance(r, ApprovalGrant) for r in db.rows)  # nothing executed
    errs = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR and "trust-gate" in r.getMessage()
    ]
    assert errs, "expected an error-level gate log"
    assert all("delete the vault" not in r.getMessage() for r in errs)  # no body


@pytest.mark.asyncio
async def test_gate_error_under_allow_all_still_routes(acks, monkeypatch):
    """allow_all is resolved BEFORE the classifier, so a classifier error can't
    turn an allow_all channel into a hold — it routes as today (a gate error on
    the one route-everything mode is a safe no-op, not a bypass)."""
    from api.webhooks import _apply_trust_gate

    monkeypatch.setattr("services.ingress_gate.should_hold", _boom)

    db = _FakeSession()
    ws = uuid.uuid4()
    db.add(_channel(ws, config={"trigger_mode": "allow_all"}))
    workspace = SimpleNamespace(id=ws)

    res = await _apply_trust_gate(
        db, workspace, "telegram", _tg("delete the vault"),
        {"platform": "telegram"}, {},
    )
    assert res is None  # short-circuits before should_hold → routes


def test_channel_is_allow_all_fails_closed():
    """The handler's fallthrough decision: True ONLY for a provable allow_all;
    strict / no-channel / lookup-error all resolve to False (fail closed)."""
    from api.webhooks import _channel_is_allow_all

    ws = uuid.uuid4()
    db_allow = _FakeSession()
    db_allow.add(_channel(ws, config={"trigger_mode": "allow_all"}))
    assert _channel_is_allow_all(db_allow, ws, "telegram") is True

    db_strict = _FakeSession()
    db_strict.add(_channel(ws, config={"trigger_mode": "strict"}))
    assert _channel_is_allow_all(db_strict, ws, "telegram") is False

    assert _channel_is_allow_all(_FakeSession(), ws, "telegram") is False  # no channel
