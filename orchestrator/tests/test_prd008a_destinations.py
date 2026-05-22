"""
PRD-008-A.1 / .4 — callback dispatcher tests
=============================================

The dispatcher is now a thin shape over ``channels.sender.send_to_channel``:
it picks a target from the destination dict, builds the text, calls the
sender, and maps the ``SendResult`` to a ``DispatchResult``. These tests
verify the shape mapping; per-platform behaviour is covered in
``test_prd008a4_channel_drivers.py`` and ``test_prd008a4_channel_sender.py``.
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


def _run(coro):
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

    result = _run(dispatch_via_channel(
        destination={},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is False
    assert result.retryable is False
    assert "platform" in result.error.lower()


def test_dispatch_rejects_unknown_platform():
    from services.destinations.dispatcher import dispatch_via_channel

    result = _run(dispatch_via_channel(
        destination={"platform": "pigeon_post"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is False
    assert result.retryable is False


def test_dispatch_webhook_requires_url():
    from services.destinations.dispatcher import dispatch_via_channel

    result = _run(dispatch_via_channel(
        destination={"platform": "webhook"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is False
    assert result.retryable is False
    assert "webhook_url" in result.error


# ---------------------------------------------------------------------------
# Sender delegation
# ---------------------------------------------------------------------------

def test_dispatch_telegram_delegates_to_sender(monkeypatch):
    """The dispatcher should call channels.sender.send_to_channel with
    the platform name and let the sender resolve creds + target."""
    from channels.drivers import SendResult
    from services.destinations import dispatcher as disp_mod

    captured = {}

    async def fake_sender(*, db, workspace_id, platform, text, target=None):
        captured.update(
            platform=platform, text=text, target=target,
            workspace_id=str(workspace_id),
        )
        return SendResult(ok=True, latency_ms=120)

    # Patch the import inside dispatcher.dispatch_via_channel
    import channels.sender as sender_mod
    monkeypatch.setattr(sender_mod, "send_to_channel", fake_sender)

    result = _run(disp_mod.dispatch_via_channel(
        destination={"platform": "telegram"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is True
    assert result.destination_type == "telegram"
    assert captured["platform"] == "telegram"
    # No explicit target: sender's resolution kicks in.
    assert captured["target"] is None
    # Text rendering carries the lead details.
    assert "James Smith" in captured["text"]
    assert "+447700900123" in captured["text"]


def test_dispatch_slack_passes_explicit_channel_id_as_target(monkeypatch):
    from channels.drivers import SendResult
    from services.destinations import dispatcher as disp_mod

    captured = {}

    async def fake_sender(*, db, workspace_id, platform, text, target=None):
        captured.update(target=target)
        return SendResult(ok=True, latency_ms=42)

    import channels.sender as sender_mod
    monkeypatch.setattr(sender_mod, "send_to_channel", fake_sender)

    result = _run(disp_mod.dispatch_via_channel(
        destination={"platform": "slack", "channel_id": "C01XYZ"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is True
    assert captured["target"] == "C01XYZ"


def test_dispatch_webhook_passes_url_as_target(monkeypatch):
    from channels.drivers import SendResult
    from services.destinations import dispatcher as disp_mod

    captured = {}

    async def fake_sender(*, db, workspace_id, platform, text, target=None):
        captured.update(platform=platform, target=target)
        return SendResult(ok=True, latency_ms=1)

    import channels.sender as sender_mod
    monkeypatch.setattr(sender_mod, "send_to_channel", fake_sender)

    result = _run(disp_mod.dispatch_via_channel(
        destination={"platform": "webhook", "webhook_url": "https://hooks.example.com/x"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is True
    assert captured["platform"] == "webhook"
    assert captured["target"] == "https://hooks.example.com/x"


def test_dispatch_failure_propagates_retryable_flag(monkeypatch):
    from channels.drivers import SendResult
    from services.destinations import dispatcher as disp_mod

    async def fake_sender(*, db, workspace_id, platform, text, target=None):
        return SendResult(
            ok=False, latency_ms=5,
            error="bot says no", retryable=True,
        )

    import channels.sender as sender_mod
    monkeypatch.setattr(sender_mod, "send_to_channel", fake_sender)

    result = _run(disp_mod.dispatch_via_channel(
        destination={"platform": "telegram"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is False
    assert result.retryable is True
    assert result.error == "bot says no"


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
