"""PRD-008-A.4 — per-platform channel driver tests.

Each driver is a thin wrapper around the platform's HTTP API; tests
mock ``httpx.AsyncClient`` at the boundary so we exercise the
driver's URL-building, error mapping, and result shaping without
hitting the network.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

import config  # noqa: E402,F401


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Test-only mock for httpx.AsyncClient. Each test patches the context
# manager onto the module under test.
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status_code: int, body: Any = None, headers=None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.headers = headers or {"content-type": "application/json"}

    def json(self):
        return self._body


class _FakeClient:
    """Records (method, url, kwargs) and returns a configured response."""

    def __init__(self, responses: dict[str, _FakeResp]):
        self._responses = responses  # keyed by URL path suffix
        self.calls: list[tuple[str, str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def get(self, url, **kwargs):
        return self._dispatch("GET", url, kwargs)

    async def post(self, url, **kwargs):
        return self._dispatch("POST", url, kwargs)

    def _dispatch(self, method, url, kwargs):
        self.calls.append((method, url, kwargs))
        for suffix, resp in self._responses.items():
            if suffix in url:
                return resp
        return _FakeResp(404, {"error": "no fake configured"})


def _patch_httpx(monkeypatch, mod, fake: _FakeClient) -> None:
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: fake)


# ---------------------------------------------------------------------------
# Telegram driver
# ---------------------------------------------------------------------------

class TestTelegramDriver:
    @pytest.fixture
    def driver(self):
        from channels.drivers.telegram import TelegramDriver
        return TelegramDriver()

    def test_verify_rejects_token_without_bot_id_prefix(self, driver):
        result = _run(driver.verify(workspace_id="ws", config={"bot_token": "AAF9onlysecret"}))
        assert result.ok is False
        assert "prefix" in (result.error or "").lower()

    def test_verify_happy_path(self, driver, monkeypatch):
        from channels.drivers import telegram as tg
        fake = _FakeClient({"getMe": _FakeResp(200, {"ok": True, "result": {"id": 1, "username": "automatos_bot"}})})
        _patch_httpx(monkeypatch, tg, fake)

        result = _run(driver.verify(workspace_id="ws", config={"bot_token": "1:AAF"}))
        assert result.ok is True
        assert result.identity == "automatos_bot"
        assert (result.metadata or {}).get("bot_id") == 1

    def test_verify_404_from_telegram(self, driver, monkeypatch):
        from channels.drivers import telegram as tg
        fake = _FakeClient({"getMe": _FakeResp(404, {})})
        _patch_httpx(monkeypatch, tg, fake)

        result = _run(driver.verify(workspace_id="ws", config={"bot_token": "1:AAF"}))
        assert result.ok is False
        assert "404" in (result.error or "")

    def test_send_requires_target(self, driver):
        result = _run(driver.send(
            workspace_id="ws", config={"bot_token": "1:AAF"},
            target=None, text="hi",
        ))
        assert result.ok is False
        assert result.retryable is False

    def test_send_happy_path(self, driver, monkeypatch):
        from channels.drivers import telegram as tg
        fake = _FakeClient({"sendMessage": _FakeResp(200, {"ok": True, "result": {}})})
        _patch_httpx(monkeypatch, tg, fake)

        result = _run(driver.send(
            workspace_id="ws", config={"bot_token": "1:AAF"},
            target="12345", text="hello",
        ))
        assert result.ok is True
        # Confirms the URL contains the full token + sendMessage path.
        assert any("bot1:AAF/sendMessage" in url for _, url, _ in fake.calls)

    def test_send_failure_with_description(self, driver, monkeypatch):
        from channels.drivers import telegram as tg
        fake = _FakeClient({
            "sendMessage": _FakeResp(
                400, {"ok": False, "description": "chat not found"},
            ),
        })
        _patch_httpx(monkeypatch, tg, fake)

        result = _run(driver.send(
            workspace_id="ws", config={"bot_token": "1:AAF"},
            target="x", text="hi",
        ))
        assert result.ok is False
        assert "chat not found" in (result.error or "")

    def test_install_webhook_writes_setwebhook(self, driver, monkeypatch):
        from channels.drivers import telegram as tg
        fake = _FakeClient({"setWebhook": _FakeResp(200, {"ok": True})})
        _patch_httpx(monkeypatch, tg, fake)

        result = _run(driver.install_webhook(
            workspace_id="ws", config={"bot_token": "1:AAF"},
            webhook_url="https://api.automatos.app/api/webhooks/ws/abc",
        ))
        assert result.ok is True
        assert any("setWebhook" in url for _, url, _ in fake.calls)

    def test_uninstall_webhook_writes_deletewebhook(self, driver, monkeypatch):
        from channels.drivers import telegram as tg
        fake = _FakeClient({"deleteWebhook": _FakeResp(200, {"ok": True})})
        _patch_httpx(monkeypatch, tg, fake)

        ok = _run(driver.uninstall_webhook(
            workspace_id="ws", config={"bot_token": "1:AAF"},
        ))
        assert ok is True
        assert any("deleteWebhook" in url for _, url, _ in fake.calls)


# ---------------------------------------------------------------------------
# Slack driver
# ---------------------------------------------------------------------------

class TestSlackDriver:
    @pytest.fixture
    def driver(self):
        from channels.drivers.slack import SlackDriver
        return SlackDriver()

    def test_verify_happy(self, driver, monkeypatch):
        from channels.drivers import slack as sl
        fake = _FakeClient({"auth.test": _FakeResp(200, {"ok": True, "team": "Acme", "user": "automatos"})})
        _patch_httpx(monkeypatch, sl, fake)

        result = _run(driver.verify(workspace_id="ws", config={"bot_token": "xoxb-1"}))
        assert result.ok is True
        assert result.identity == "Acme"

    def test_verify_invalid_auth(self, driver, monkeypatch):
        from channels.drivers import slack as sl
        fake = _FakeClient({"auth.test": _FakeResp(200, {"ok": False, "error": "invalid_auth"})})
        _patch_httpx(monkeypatch, sl, fake)

        result = _run(driver.verify(workspace_id="ws", config={"bot_token": "xoxb-1"}))
        assert result.ok is False
        assert "invalid_auth" in (result.error or "")

    def test_send_uses_default_channel(self, driver, monkeypatch):
        from channels.drivers import slack as sl
        fake = _FakeClient({"chat.postMessage": _FakeResp(200, {"ok": True})})
        _patch_httpx(monkeypatch, sl, fake)

        result = _run(driver.send(
            workspace_id="ws",
            config={"bot_token": "xoxb-1", "default_channel": "#sales"},
            target=None, text="hi",
        ))
        assert result.ok is True
        payload = fake.calls[0][2]["json"]
        assert payload["channel"] == "#sales"

    def test_send_no_target_no_default_returns_error(self, driver):
        result = _run(driver.send(
            workspace_id="ws", config={"bot_token": "xoxb-1"},
            target=None, text="hi",
        ))
        assert result.ok is False
        assert result.retryable is False

    def test_send_rate_limited_is_retryable(self, driver, monkeypatch):
        from channels.drivers import slack as sl
        fake = _FakeClient({"chat.postMessage": _FakeResp(200, {"ok": False, "error": "ratelimited"})})
        _patch_httpx(monkeypatch, sl, fake)

        result = _run(driver.send(
            workspace_id="ws", config={"bot_token": "xoxb-1"},
            target="C0123", text="hi",
        ))
        assert result.ok is False
        assert result.retryable is True


# ---------------------------------------------------------------------------
# WhatsApp driver
# ---------------------------------------------------------------------------

class TestWhatsAppDriver:
    @pytest.fixture
    def driver(self):
        from channels.drivers.whatsapp import WhatsAppDriver
        return WhatsAppDriver()

    def test_verify_requires_credentials(self, driver):
        result = _run(driver.verify(workspace_id="ws", config={}))
        assert result.ok is False

    def test_verify_happy(self, driver, monkeypatch):
        from channels.drivers import whatsapp as wa
        fake = _FakeClient({"/123": _FakeResp(200, {"verified_name": "Acme Co", "display_phone_number": "+447..."})})
        _patch_httpx(monkeypatch, wa, fake)

        result = _run(driver.verify(workspace_id="ws", config={"phone_number_id": "123", "access_token": "t"}))
        assert result.ok is True
        assert result.identity == "Acme Co"

    def test_send_requires_target(self, driver):
        result = _run(driver.send(
            workspace_id="ws",
            config={"phone_number_id": "123", "access_token": "t"},
            target=None, text="hi",
        ))
        assert result.ok is False
        assert result.retryable is False


# ---------------------------------------------------------------------------
# Discord driver
# ---------------------------------------------------------------------------

class TestDiscordDriver:
    @pytest.fixture
    def driver(self):
        from channels.drivers.discord import DiscordDriver
        return DiscordDriver()

    def test_verify_happy(self, driver, monkeypatch):
        from channels.drivers import discord as dc
        fake = _FakeClient({"users/@me": _FakeResp(200, {"id": "1", "username": "automatos"})})
        _patch_httpx(monkeypatch, dc, fake)

        result = _run(driver.verify(workspace_id="ws", config={"bot_token": "tok"}))
        assert result.ok is True
        assert result.identity == "automatos"


# ---------------------------------------------------------------------------
# Webhook (outbound) driver
# ---------------------------------------------------------------------------

class TestWebhookDriver:
    @pytest.fixture
    def driver(self):
        from channels.drivers.webhook import WebhookDriver
        return WebhookDriver()

    def test_verify_rejects_non_url(self, driver):
        result = _run(driver.verify(workspace_id="ws", config={"webhook_url": "not a url"}))
        assert result.ok is False

    def test_verify_accepts_https(self, driver):
        result = _run(driver.verify(workspace_id="ws", config={"webhook_url": "https://hooks.example.com/x"}))
        assert result.ok is True

    def test_send_5xx_is_retryable(self, driver, monkeypatch):
        from channels.drivers import webhook as wh
        fake = _FakeClient({"https://hooks": _FakeResp(503)})
        _patch_httpx(monkeypatch, wh, fake)

        result = _run(driver.send(
            workspace_id="ws", config={"webhook_url": "https://hooks.example.com/x"},
            target=None, text="hi",
        ))
        assert result.ok is False
        assert result.retryable is True


# ---------------------------------------------------------------------------
# Driver registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_concrete_drivers_registered(self):
        from channels.drivers import get_driver, list_platforms
        assert "telegram" in list_platforms()
        assert "slack" in list_platforms()
        assert "whatsapp" in list_platforms()
        assert "discord" in list_platforms()
        assert "webhook" in list_platforms()
        # Each registered name resolves to a usable class.
        for p in list_platforms():
            cls = get_driver(p)
            assert cls is not None

    def test_unknown_platform_raises(self):
        from channels.drivers import UnknownPlatform, get_driver
        with pytest.raises(UnknownPlatform):
            get_driver("nonexistent")
