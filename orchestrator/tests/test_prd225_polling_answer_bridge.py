"""PRD-225 US-005 on the Telegram POLLING path.

The webhook handler correlated replies to delivered questions; a bot in polling
mode (the only mode a local install without a public URL can run) never came
through it, so a reply reached the agent as chat. These tests drive the shared
entry point the polling adapter now consults, with the same fakes the webhook
bridge tests use, and check the adapter routes nothing once a message was
consumed as an answer.
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

import api.webhooks as webhooks
from core.models.approval_grants import GrantStatus
from tests.test_prd225_telegram_bridge import (  # noqa: F401 — fixtures (replies, the autouse chat stub)
    _FakeSession,
    _question,
    _quiet_chat_confirm,
    replies,
)


@pytest.fixture()
def fake_db(monkeypatch):
    db = _FakeSession()
    monkeypatch.setattr("core.database.database.SessionLocal", lambda: db)
    return db


@pytest.mark.asyncio
async def test_polled_reply_to_a_delivered_question_answers_it(fake_db, replies):
    ws = uuid4()
    grant = _question(fake_db, ws, gid=41, telegram_message_id=900, chat_id="c1")

    handled = await webhooks.maybe_answer_polled_telegram_message(
        str(ws), text="Vendor B", chat_id="c1", from_id=555, reply_to_message_id=900,
    )

    assert handled is True
    assert grant.status == GrantStatus.GRANTED.value
    assert grant.answer_text == "Vendor B"
    assert grant.answered_by == "telegram:555"
    assert replies and replies[0]["text"].startswith("Answer")


@pytest.mark.asyncio
async def test_polled_slash_answer_from_the_delivery_chat(fake_db, replies):
    ws = uuid4()
    grant = _question(fake_db, ws, gid=42, telegram_message_id=901, chat_id="c1")

    handled = await webhooks.maybe_answer_polled_telegram_message(
        str(ws), text="/answer 42 use the backup", chat_id="c1", from_id=555, reply_to_message_id=None,
    )

    assert handled is True
    assert grant.answer_text == "use the backup"


@pytest.mark.asyncio
async def test_polled_chat_that_answers_nothing_falls_through(fake_db, replies):
    ws = uuid4()
    _question(fake_db, ws, gid=43, telegram_message_id=902, chat_id="c1")

    assert await webhooks.maybe_answer_polled_telegram_message(
        str(ws), text="hello there", chat_id="c1", from_id=555, reply_to_message_id=None,
    ) is False
    assert await webhooks.maybe_answer_polled_telegram_message(
        str(ws), text="   ", chat_id="c1", from_id=555, reply_to_message_id=902,
    ) is False
    # a reply from a chat the question was never delivered to is not an answer
    assert await webhooks.maybe_answer_polled_telegram_message(
        str(ws), text="yes", chat_id="other-chat", from_id=1, reply_to_message_id=902,
    ) is False
    assert replies == []


@pytest.mark.asyncio
async def test_bridge_failure_never_eats_the_message(monkeypatch):
    async def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr("core.database.database.SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(webhooks, "_maybe_answer_question", _boom)
    assert await webhooks.maybe_answer_polled_telegram_message(
        str(uuid4()), text="yes", chat_id="c1", from_id=1, reply_to_message_id=1,
    ) is False


def _adapter():
    from channels.telegram_adapter import TelegramAdapter

    adapter = TelegramAdapter("conn-1", str(uuid4()), {"bot_token": "t"})
    adapter._persist_default_chat_id = lambda chat_id: None
    return adapter


def _update(text="Vendor B", reply_to=900):
    message = SimpleNamespace(
        text=text, caption=None, photo=None, document=None, message_id=7,
        reply_to_message=SimpleNamespace(message_id=reply_to) if reply_to is not None else None,
    )
    return SimpleNamespace(message=message, effective_chat=SimpleNamespace(id=1), effective_user=SimpleNamespace(id=555))


class _Bot:
    async def send_chat_action(self, **kwargs):
        return None

    async def send_message(self, **kwargs):
        return None


@pytest.mark.asyncio
async def test_adapter_routes_nothing_when_the_message_answered_a_question(monkeypatch):
    seen = {}

    async def _answered(workspace_id, **kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr("api.webhooks.maybe_answer_polled_telegram_message", _answered)
    adapter = _adapter()
    routed = []

    async def _handle(msg):
        routed.append(msg)

    adapter.handle_message = _handle
    await adapter._on_message(_update(), SimpleNamespace(bot=_Bot()))

    assert routed == []
    assert seen == {"text": "Vendor B", "chat_id": 1, "from_id": 555, "reply_to_message_id": 900}


@pytest.mark.asyncio
async def test_adapter_routes_ordinary_chat_as_before(monkeypatch):
    async def _not_an_answer(workspace_id, **kwargs):
        return False

    monkeypatch.setattr("api.webhooks.maybe_answer_polled_telegram_message", _not_an_answer)
    adapter = _adapter()
    routed = []

    async def _handle(msg):
        routed.append(msg)

    adapter.handle_message = _handle
    await adapter._on_message(_update(text="hello", reply_to=None), SimpleNamespace(bot=_Bot()))

    assert len(routed) == 1 and routed[0]["text"] == "hello"
