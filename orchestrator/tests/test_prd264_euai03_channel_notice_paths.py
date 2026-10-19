"""PRD-264 EUAI-03 — every channel adapter and driver puts the AI notice on the wire.

Article 50(1) and Guidelines ¶31: a person messaging a workspace's bot is told,
before or with the first reply, that it is an AI and for whom: "AI assistant
for {principal}." Both outbound paths add it:

* the adapter path (Telegram, Slack, Discord, WhatsApp adapters): the reply
  leaves through ``BaseChannelAdapter._send_reply``; each test reads what the
  adapter's own ``send_message`` handed to its platform client;
* the driver path (``channels.sender.send_to_channel``): webhook replies,
  notifications, agent asks and destination dispatches. The person-facing
  drivers get the notice as the first paragraph; the generic webhook driver
  gets ``ai_generated`` and ``ai_disclosure`` payload fields instead.

For each: the first message carries the notice once, a second within the gap
doesn't, one after the gap does, "are you human?" brings it back, and Redis
down adds it. The Telegram /start reply says "I'm an AI assistant for …" in
its first sentence. Both editions.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Optional
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from config import config as app_config  # noqa: E402

HOUR = 3600
T0 = 1_800_000_000.0
SHOP = "Harbour Coffee"
NOTICE = "AI assistant for Harbour Coffee."
REPLY = "We open at 8."
WS = str(uuid4())
CHAT_ID = "4242"


# ── fakes ──────────────────────────────────────────────────────────────────

class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self.values.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self.values = {**self.values, key: value}
        return True


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _Result:
    def __init__(self, row: Any) -> None:
        self._row = row

    def fetchone(self) -> Any:
        return self._row


class FakeDb:
    """The rows the sender and the notice read: the channel row and the principal."""

    def __init__(self, channel_config: Optional[dict] = None) -> None:
        self.channel_row = NS(id=uuid4(), config=channel_config or {}, metadata={})

    def execute(self, statement: Any, params: Any = None) -> _Result:
        sql = str(statement)
        if "FROM channel_connections" in sql:
            return _Result(self.channel_row)
        if "business_profiles" in sql:
            return _Result((SHOP,))
        return _Result(None)

    def begin_nested(self) -> "FakeDb":
        return self

    def __enter__(self) -> "FakeDb":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def close(self) -> None:
        return None


@pytest.fixture(params=("local", "saas"))
def edition(request, monkeypatch):
    monkeypatch.setattr(app_config, "AUTH_EDITION", request.param)
    return request.param


@pytest.fixture
def state(monkeypatch):
    """The real notice module on a fake Redis and a fake clock, 24 h gap."""
    import channels.ai_notice as ai_notice

    redis = FakeRedis()
    clock = Clock(T0)
    monkeypatch.setattr(ai_notice, "_get_redis", lambda: redis)
    monkeypatch.setattr(ai_notice, "_clock", clock)
    monkeypatch.setattr(app_config, "AI_DISCLOSURE_IDLE_HOURS", 24)
    return NS(module=ai_notice, redis=redis, clock=clock)


def _with_notice(text: str) -> str:
    return f"{NOTICE}\n\n{text}"


# ── the adapter path ───────────────────────────────────────────────────────

class _TelegramBot:
    def __init__(self, sent: list) -> None:
        self.sent = sent

    async def send_message(self, chat_id: Any, text: str) -> None:
        self.sent.append(text)


class _SlackClient:
    def __init__(self, sent: list) -> None:
        self.sent = sent

    async def chat_postMessage(self, channel: str, text: str, thread_ts: Any = None) -> dict:
        self.sent.append(text)
        return {"ok": True}


class _DiscordChannel:
    def __init__(self, sent: list) -> None:
        self.sent = sent

    async def send(self, chunk: str) -> None:
        self.sent.append(chunk)


class _HttpResp:
    status_code = 200
    text = "ok"

    def json(self) -> dict:
        return {}


class _HttpClient:
    """Stands in for ``httpx.AsyncClient``; records each JSON body posted."""

    def __init__(self, bodies: list) -> None:
        self.bodies = bodies

    async def __aenter__(self) -> "_HttpClient":
        return self

    async def __aexit__(self, *_exc: Any) -> bool:
        return False

    async def post(self, url: str, json: Any = None, **_kwargs: Any) -> _HttpResp:
        self.bodies.append(json)
        return _HttpResp()


def _telegram_adapter(monkeypatch):
    from channels.telegram_adapter import TelegramAdapter

    sent: list = []
    adapter = TelegramAdapter("conn-tg", WS, {"bot_token": "test-token"})
    adapter._app = NS(bot=_TelegramBot(sent))
    return adapter, lambda: list(sent)


def _slack_adapter(monkeypatch):
    from channels.slack_adapter import SlackAdapter

    sent: list = []
    adapter = SlackAdapter("conn-slack", WS, {"bot_token": "test-token"})
    adapter._app = NS(client=_SlackClient(sent))
    return adapter, lambda: list(sent)


def _discord_adapter(monkeypatch):
    from channels.discord_adapter import DiscordAdapter

    sent: list = []
    channel = _DiscordChannel(sent)
    adapter = DiscordAdapter("conn-discord", WS, {"bot_token": "test-token"})
    adapter._client = NS(get_channel=lambda _id: channel)
    return adapter, lambda: list(sent)


def _whatsapp_adapter(monkeypatch):
    import httpx

    from channels.whatsapp_adapter import WhatsAppAdapter

    bodies: list = []
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _HttpClient(bodies))
    adapter = WhatsAppAdapter("conn-wa", WS, {"access_token": "test-token", "phone_number_id": "100"})
    return adapter, lambda: [body["text"]["body"] for body in bodies]


ADAPTERS = {
    "telegram": _telegram_adapter,
    "slack": _slack_adapter,
    "discord": _discord_adapter,
    "whatsapp": _whatsapp_adapter,
}


def _adapter_and_outbox(name: str, monkeypatch):
    """An adapter whose platform client is fake, and a reader of the texts it sent."""
    return ADAPTERS[name](monkeypatch)


def _reply(adapter, text: str = REPLY, inbound: Optional[str] = "hi") -> None:
    message = {"channel_id": CHAT_ID, "reply_channel_id": CHAT_ID, "user_id": "person-1"}
    asyncio.run(adapter._send_reply(FakeDb(), message, text, inbound_text=inbound))


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_adapter_first_reply_carries_the_notice_once(name, edition, state, monkeypatch):
    adapter, outbox = _adapter_and_outbox(name, monkeypatch)

    _reply(adapter)
    state.clock.now = T0 + 60
    _reply(adapter, "Anything else?")

    assert outbox() == [_with_notice(REPLY), "Anything else?"]


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_adapter_notice_returns_after_the_gap(name, state, monkeypatch):
    adapter, outbox = _adapter_and_outbox(name, monkeypatch)

    _reply(adapter)
    state.clock.now = T0 + 24 * HOUR
    _reply(adapter)

    assert outbox() == [_with_notice(REPLY), _with_notice(REPLY)]


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_adapter_are_you_human_gets_the_notice(name, state, monkeypatch):
    adapter, outbox = _adapter_and_outbox(name, monkeypatch)

    _reply(adapter)
    state.clock.now = T0 + 60
    _reply(adapter, "Yes, I can help.", inbound="Are you a human?")

    assert outbox() == [_with_notice(REPLY), _with_notice("Yes, I can help.")]


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_adapter_redis_down_adds_the_notice(name, edition, state, monkeypatch):
    adapter, outbox = _adapter_and_outbox(name, monkeypatch)
    monkeypatch.setattr(state.module, "_get_redis", lambda: None)

    _reply(adapter)
    _reply(adapter)

    assert outbox() == [_with_notice(REPLY), _with_notice(REPLY)]


def test_a_failed_adapter_send_keeps_the_notice_for_the_next_reply(state):
    from channels.telegram_adapter import TelegramAdapter

    adapter = TelegramAdapter("conn-tg", WS, {"bot_token": "test-token"})  # no bot: send fails
    _reply(adapter)
    assert state.redis.values == {}

    sent: list = []
    adapter._app = NS(bot=_TelegramBot(sent))
    state.clock.now = T0 + 60
    _reply(adapter)
    assert sent == [_with_notice(REPLY)]


class _RefusedResp(_HttpResp):
    status_code = 400
    text = "refused"


class _RefusingHttpClient(_HttpClient):
    """The Graph API refusing the message."""

    async def post(self, url: str, json: Any = None, **_kwargs: Any) -> _HttpResp:
        self.bodies.append(json)
        return _RefusedResp()


def test_a_refused_whatsapp_send_keeps_the_notice_for_the_next_reply(state, monkeypatch):
    import httpx

    from channels.whatsapp_adapter import WhatsAppAdapter

    bodies: list = []
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _RefusingHttpClient(bodies))
    adapter = WhatsAppAdapter("conn-wa", WS, {"access_token": "test-token", "phone_number_id": "100"})

    assert asyncio.run(adapter.send_message(CHAT_ID, REPLY)) is False
    _reply(adapter)
    assert state.redis.values == {}

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _HttpClient(bodies))
    state.clock.now = T0 + 60
    _reply(adapter)
    assert bodies[-1]["text"]["body"] == _with_notice(REPLY)
    assert state.redis.values


def test_whatsapp_webhook_replies_to_the_sender_through_the_base_reply_point(monkeypatch):
    from channels.whatsapp_adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter("conn-wa", WS, {"access_token": "test-token", "phone_number_id": "100"})
    handled: list = []

    async def fake_handle(message):
        handled.append(message)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    monkeypatch.setattr(adapter, "_increment_message_count", lambda: None)
    payload = {"entry": [{"changes": [{"value": {
        "messages": [{"from": "351900000001", "type": "text", "text": {"body": "hi"}}],
        "contacts": [{"profile": {"name": "Ana"}}],
    }}]}]}

    assert asyncio.run(adapter.handle_webhook(payload)) == "ok"
    assert handled[0]["reply_channel_id"] == "351900000001"
    assert handled[0]["user_id"] == "351900000001"
    assert "reply_channel_id" not in payload["entry"][0]["changes"][0]["value"]["messages"][0]


# ── the Telegram /start reply ──────────────────────────────────────────────

def _start_reply(monkeypatch, db: Any) -> str:
    from channels.telegram_adapter import TelegramAdapter

    import core.database.database as database

    monkeypatch.setattr(database, "SessionLocal", lambda: db)
    adapter = TelegramAdapter("conn-tg", WS, {"bot_token": "test-token"})
    monkeypatch.setattr(adapter, "_persist_default_chat_id", lambda chat_id: None)
    sent: list = []
    update = NS(effective_chat=NS(id=int(CHAT_ID)))
    context = NS(bot=_TelegramBot(sent))
    asyncio.run(adapter._on_command(update, context))
    return sent[0]


def test_telegram_start_says_it_is_an_ai_for_the_principal_first(edition, monkeypatch):
    text = _start_reply(monkeypatch, FakeDb())

    assert text.startswith("I'm an AI assistant for Harbour Coffee.")
    assert "Connected" in text


def test_telegram_start_without_a_public_name_still_says_ai(monkeypatch):
    class NoName(FakeDb):
        def execute(self, statement, params=None):
            return _Result(None)

    text = _start_reply(monkeypatch, NoName())

    assert text.startswith("I'm an AI assistant. ")


# ── the driver path (channels.sender.send_to_channel) ──────────────────────

CHAT_DRIVERS = {
    "telegram": ("channels.drivers.telegram", "TelegramDriver"),
    "slack": ("channels.drivers.slack", "SlackDriver"),
    "discord": ("channels.drivers.discord", "DiscordDriver"),
    "whatsapp": ("channels.drivers.whatsapp", "WhatsAppDriver"),
}


def _patch_driver(monkeypatch, platform: str, ok: bool = True) -> list:
    """Replace the platform driver's ``send`` with a recorder; returns the texts."""
    import importlib

    from channels.drivers import SendResult

    module_name, class_name = CHAT_DRIVERS[platform]
    driver_cls = getattr(importlib.import_module(module_name), class_name)
    texts: list = []

    async def fake_send(self, *, workspace_id, config, target, text):
        texts.append(text)
        return SendResult(ok=ok, latency_ms=1, error=None if ok else "down")

    monkeypatch.setattr(driver_cls, "send", fake_send)
    return texts


def _send(platform: str, text: str = REPLY, in_reply_to: Optional[str] = None):
    from channels.sender import send_to_channel

    return asyncio.run(send_to_channel(
        db=FakeDb({"bot_token": "test-token"}), workspace_id=WS, platform=platform,
        text=text, target=CHAT_ID, in_reply_to=in_reply_to,
    ))


@pytest.mark.parametrize("platform", sorted(CHAT_DRIVERS))
def test_driver_first_message_carries_the_notice_once(platform, edition, state, monkeypatch):
    texts = _patch_driver(monkeypatch, platform)

    assert _send(platform).ok is True
    state.clock.now = T0 + 60
    _send(platform, "Anything else?")

    assert texts == [_with_notice(REPLY), "Anything else?"]


@pytest.mark.parametrize("platform", sorted(CHAT_DRIVERS))
def test_driver_notice_returns_after_the_gap(platform, state, monkeypatch):
    texts = _patch_driver(monkeypatch, platform)

    _send(platform)
    state.clock.now = T0 + 24 * HOUR
    _send(platform)

    assert texts == [_with_notice(REPLY), _with_notice(REPLY)]


@pytest.mark.parametrize("platform", sorted(CHAT_DRIVERS))
def test_driver_are_you_human_gets_the_notice(platform, state, monkeypatch):
    texts = _patch_driver(monkeypatch, platform)

    _send(platform)
    state.clock.now = T0 + 60
    _send(platform, "Yes, I can help.", in_reply_to="am I talking to a bot?")

    assert texts == [_with_notice(REPLY), _with_notice("Yes, I can help.")]


@pytest.mark.parametrize("platform", sorted(CHAT_DRIVERS))
def test_driver_redis_down_adds_the_notice(platform, edition, state, monkeypatch):
    texts = _patch_driver(monkeypatch, platform)
    monkeypatch.setattr(state.module, "_get_redis", lambda: None)

    _send(platform)
    _send(platform)

    assert texts == [_with_notice(REPLY), _with_notice(REPLY)]


def test_a_failed_driver_send_keeps_the_notice_for_the_next_message(state, monkeypatch):
    texts = _patch_driver(monkeypatch, "telegram", ok=False)

    _send("telegram")
    _send("telegram")

    assert texts == [_with_notice(REPLY), _with_notice(REPLY)]
    assert state.redis.values == {}


# ── the generic webhook driver: payload fields, never a text prefix ────────

def _webhook_bodies(monkeypatch) -> list:
    import httpx

    bodies: list = []
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _HttpClient(bodies))
    return bodies


def _send_webhook(text: str = REPLY, in_reply_to: Optional[str] = None):
    from channels.sender import send_to_channel

    return asyncio.run(send_to_channel(
        db=FakeDb({"webhook_url": "https://hooks.example.com/in"}), workspace_id=WS,
        platform="webhook", text=text, in_reply_to=in_reply_to,
    ))


def test_webhook_first_payload_carries_the_notice_fields_once(edition, state, monkeypatch):
    bodies = _webhook_bodies(monkeypatch)

    assert _send_webhook().ok is True
    state.clock.now = T0 + 60
    _send_webhook("Anything else?")

    assert bodies[0] == {"text": REPLY, "message": REPLY, "ai_generated": True, "ai_disclosure": NOTICE}
    assert bodies[1] == {"text": "Anything else?", "message": "Anything else?", "ai_generated": True}


def test_webhook_notice_returns_after_the_gap_and_on_are_you_human(state, monkeypatch):
    bodies = _webhook_bodies(monkeypatch)

    _send_webhook()
    state.clock.now = T0 + 60
    _send_webhook(in_reply_to="are you a bot?")
    state.clock.now = T0 + 25 * HOUR
    _send_webhook()

    assert [body.get("ai_disclosure") for body in bodies] == [NOTICE, NOTICE, NOTICE]
    assert all(body["text"] == REPLY for body in bodies)


def test_webhook_redis_down_adds_the_notice_fields(state, monkeypatch):
    bodies = _webhook_bodies(monkeypatch)
    monkeypatch.setattr(state.module, "_get_redis", lambda: None)

    _send_webhook()
    _send_webhook()

    assert [body.get("ai_disclosure") for body in bodies] == [NOTICE, NOTICE]


def test_webhook_driver_marks_every_payload_ai_generated(monkeypatch):
    from channels.drivers.webhook import WebhookDriver

    bodies = _webhook_bodies(monkeypatch)
    asyncio.run(WebhookDriver().send(
        workspace_id=WS, config={"webhook_url": "https://hooks.example.com/in"}, target=None, text="hi",
    ))

    assert bodies == [{"text": "hi", "message": "hi", "ai_generated": True}]


# ── webhook replies pass the message they answer ───────────────────────────

def test_deliver_reply_passes_the_inbound_text_to_the_sender(monkeypatch):
    import core.database.database as database
    from api import webhooks

    calls: list = []

    async def fake_send_to_channel(**kwargs):
        from channels.drivers import SendResult

        calls.append(kwargs)
        return SendResult(ok=True, latency_ms=1)

    monkeypatch.setattr("channels.sender.send_to_channel", fake_send_to_channel)
    monkeypatch.setattr(database, "SessionLocal", lambda: FakeDb())
    body = {"message": {"chat": {"id": 4242}, "from": {"id": 7}, "text": "are you human?", "message_id": 1}}
    reply_ctx = webhooks._extract_reply_context(body, "telegram")

    assert asyncio.run(webhooks._deliver_reply(REPLY, reply_ctx, {}, workspace_id=WS)) is True
    assert calls[0]["in_reply_to"] == "are you human?"
    assert calls[0]["target"] == "4242"


@pytest.mark.parametrize("platform, body, expected", [
    ("slack", {"event": {"channel": "C1", "text": "are you a bot?"}}, "are you a bot?"),
    ("whatsapp", {"entry": [{"changes": [{"value": {"messages": [
        {"from": "351900000001", "text": {"body": "és humano?"}}]}}]}]}, "és humano?"),
    ("twilio", {"Body": "is this a real person?", "From": "+1", "To": "+2"}, "is this a real person?"),
])
def test_reply_context_keeps_the_inbound_text(platform, body, expected):
    from api import webhooks

    assert webhooks._extract_reply_context(body, platform)["inbound_text"] == expected
