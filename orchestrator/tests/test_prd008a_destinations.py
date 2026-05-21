"""
PRD-008-A.1 — callback dispatcher tests (platform-keyed shape)
==============================================================

These tests cover ``dispatch_via_channel`` after the rebuild that
replaced the bespoke ``channel_connection`` shape with the same
platform key heartbeat uses. Destinations are now of the form::

    {"platform": "telegram"}
    {"platform": "slack",   "channel_id":  "C01ABC..."}
    {"platform": "webhook", "webhook_url": "https://..."}

Telegram / Slack / WhatsApp go through ``send_workspace_notification``
(the heartbeat path). Webhook is dispatched directly via httpx.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

import config  # noqa: E402,F401


def _await(coro):
    return asyncio.run(coro)


def _payload(**overrides):
    from services.destinations.base import CallbackPayload
    defaults = dict(
        request_id="cb_test123",
        name="James Smith",
        phone="+447700900123",
        product_context="EN 12101-9 panel",
        urgency=None,
        preferred_time=None,
        site_display_name="INBUILD UK",
        site_external_id="inbuilduk.myshopify.com",
    )
    defaults.update(overrides)
    return CallbackPayload(**defaults)


# ---------------------------------------------------------------------------
# Static guards
# ---------------------------------------------------------------------------

def test_dispatch_rejects_missing_platform():
    from services.destinations.dispatcher import dispatch_via_channel

    result = _await(dispatch_via_channel(
        destination={},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is False
    assert result.retryable is False
    assert "platform" in result.error


def test_dispatch_rejects_unknown_platform():
    from services.destinations.dispatcher import dispatch_via_channel

    result = _await(dispatch_via_channel(
        destination={"platform": "pigeon_post"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is False
    assert result.retryable is False
    assert "unsupported platform" in result.error.lower()


# ---------------------------------------------------------------------------
# Telegram / Slack — go through send_workspace_notification (heartbeat path)
# ---------------------------------------------------------------------------

def test_dispatch_telegram_success(monkeypatch):
    from services.destinations import dispatcher as disp_mod

    captured = {}

    async def fake_send(*, workspace_id, message, channel):
        captured["workspace_id"] = workspace_id
        captured["message"] = message
        captured["channel"] = channel
        return True

    monkeypatch.setattr(disp_mod, "send_workspace_notification", fake_send)

    result = _await(disp_mod.dispatch_via_channel(
        destination={"platform": "telegram"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is True
    assert result.destination_type == "telegram"
    assert captured["channel"] == "telegram"
    assert "INBUILD UK" in captured["message"]
    assert "+447700900123" in captured["message"]


def test_dispatch_telegram_failure_returns_actionable_error(monkeypatch):
    from services.destinations import dispatcher as disp_mod

    async def fake_send(**_kw):
        return False

    monkeypatch.setattr(disp_mod, "send_workspace_notification", fake_send)

    result = _await(disp_mod.dispatch_via_channel(
        destination={"platform": "telegram"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is False
    assert result.destination_type == "telegram"
    assert "telegram" in result.error.lower()
    # Tells the merchant the next concrete action.
    assert "/start" in result.error or "chat_id" in result.error.lower()


def test_dispatch_slack_with_explicit_channel_id(monkeypatch):
    from services.destinations import dispatcher as disp_mod

    captured = {}

    async def fake_send(*, workspace_id, message, channel):
        captured["channel"] = channel
        return True

    stashed = {}

    async def fake_stash(*, workspace_id, channel_id):
        stashed["channel_id"] = channel_id

    monkeypatch.setattr(disp_mod, "send_workspace_notification", fake_send)
    monkeypatch.setattr(disp_mod, "_stash_slack_channel_override", fake_stash)

    result = _await(disp_mod.dispatch_via_channel(
        destination={"platform": "slack", "channel_id": "C01XYZ"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is True
    assert stashed["channel_id"] == "C01XYZ"
    assert captured["channel"] == "slack"


def test_dispatch_notification_exception_is_retryable(monkeypatch):
    from services.destinations import dispatcher as disp_mod

    async def boom(**_kw):
        raise RuntimeError("network blip")

    monkeypatch.setattr(disp_mod, "send_workspace_notification", boom)

    result = _await(disp_mod.dispatch_via_channel(
        destination={"platform": "telegram"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is False
    assert result.retryable is True
    assert "network blip" in result.error


# ---------------------------------------------------------------------------
# Webhook — direct httpx path
# ---------------------------------------------------------------------------

def test_dispatch_webhook_rejects_missing_url():
    from services.destinations.dispatcher import dispatch_via_channel

    result = _await(dispatch_via_channel(
        destination={"platform": "webhook"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is False
    assert result.retryable is False
    assert "webhook_url" in result.error


def test_dispatch_webhook_success(monkeypatch):
    from services.destinations import dispatcher as disp_mod

    class _Resp:
        status_code = 200

    class _Client:
        def __init__(self, *a, **kw): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json):
            captured["url"] = url
            captured["body"] = json
            return _Resp()

    captured = {}
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    result = _await(disp_mod.dispatch_via_channel(
        destination={"platform": "webhook", "webhook_url": "https://hooks.example.com/x"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is True
    assert result.destination_type == "webhook"
    assert captured["url"] == "https://hooks.example.com/x"
    assert "INBUILD UK" in captured["body"]["text"]
    assert captured["body"]["request_id"] == "cb_test123"


def test_dispatch_webhook_5xx_is_retryable(monkeypatch):
    from services.destinations import dispatcher as disp_mod

    class _Resp:
        status_code = 503

    class _Client:
        def __init__(self, *a, **kw): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw): return _Resp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    result = _await(disp_mod.dispatch_via_channel(
        destination={"platform": "webhook", "webhook_url": "https://hooks.example.com/x"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is False
    assert result.retryable is True
    assert "503" in result.error


def test_dispatch_webhook_4xx_is_permanent(monkeypatch):
    from services.destinations import dispatcher as disp_mod

    class _Resp:
        status_code = 401

    class _Client:
        def __init__(self, *a, **kw): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw): return _Resp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    result = _await(disp_mod.dispatch_via_channel(
        destination={"platform": "webhook", "webhook_url": "https://hooks.example.com/x"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is False
    assert result.retryable is False
    assert "401" in result.error


# ---------------------------------------------------------------------------
# Text rendering — unchanged behaviour
# ---------------------------------------------------------------------------

def test_render_callback_text_includes_optional_fields():
    from services.destinations.dispatcher import _render_callback_text

    txt = _render_callback_text(_payload(urgency="ASAP", preferred_time="this afternoon"))
    assert "INBUILD UK" in txt
    assert "James Smith" in txt
    assert "+447700900123" in txt
    assert "EN 12101-9 panel" in txt
    assert "ASAP" in txt
    assert "this afternoon" in txt


def test_render_callback_text_skips_unset_optional_fields():
    from services.destinations.dispatcher import _render_callback_text

    txt = _render_callback_text(_payload(product_context=None))
    assert "Topic:" not in txt
    assert "Urgency:" not in txt
    assert "Preferred time:" not in txt


def test_callback_platforms_constant():
    from services.destinations.base import CALLBACK_PLATFORMS

    assert "telegram" in CALLBACK_PLATFORMS
    assert "slack" in CALLBACK_PLATFORMS
    assert "whatsapp" in CALLBACK_PLATFORMS
    assert "webhook" in CALLBACK_PLATFORMS
