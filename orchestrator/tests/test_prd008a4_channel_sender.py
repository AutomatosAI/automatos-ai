"""PRD-008-A.4 — unified ``channels.sender.send_to_channel`` tests.

The sender is the single entry point every outbound platform message
now goes through. These tests cover:

- channel_connections is preferred over the legacy integrations bag
- legacy integrations fallback kicks in when no row exists
- target resolution precedence (explicit → metadata → legacy default)
- unknown platform / no connection paths return permanent SendResults
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

import config  # noqa: E402,F401


def _run(coro):
    return asyncio.run(coro)


def _db_with_responses(*, channel_row=None, workspace_settings=None):
    """Build a MagicMock SQLAlchemy session whose execute().fetchone()
    returns the right row depending on the query text."""
    db = MagicMock()

    def execute(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "FROM channel_connections" in sql:
            result.fetchone.return_value = channel_row
        elif "FROM workspaces" in sql:
            row = None
            if workspace_settings is not None:
                row = MagicMock()
                row.settings = workspace_settings
            result.fetchone.return_value = row
        else:
            result.fetchone.return_value = None
        return result

    db.execute.side_effect = execute
    return db


def _channel_row(*, config=None, metadata=None):
    """Build a SQLAlchemy-row-shaped MagicMock matching the columns the
    sender selects."""
    row = MagicMock()
    row.id = uuid4()
    row.config = config or {}
    row.metadata = metadata or {}
    return row


class TestSendToChannel:
    def test_unknown_platform_returns_permanent_error(self):
        from channels.sender import send_to_channel

        db = _db_with_responses(channel_row=None)
        result = _run(send_to_channel(
            db=db, workspace_id=uuid4(), platform="pigeon", text="hi",
        ))
        assert result.ok is False
        assert result.retryable is False

    def test_no_connection_and_no_legacy_returns_permanent_error(self):
        from channels.sender import send_to_channel

        db = _db_with_responses(channel_row=None, workspace_settings={})
        result = _run(send_to_channel(
            db=db, workspace_id=uuid4(), platform="telegram", text="hi",
        ))
        assert result.ok is False
        assert result.retryable is False
        assert "no telegram channel" in (result.error or "").lower()

    def test_calls_driver_with_channel_config_when_row_present(self, monkeypatch):
        from channels.sender import send_to_channel
        from channels.drivers import SendResult

        captured: dict = {}

        async def fake_send(self, *, workspace_id, config, target, text):
            captured.update(
                workspace_id=workspace_id,
                config=dict(config),
                target=target,
                text=text,
            )
            return SendResult(ok=True, latency_ms=42)

        # Patch the Telegram driver's send so we don't hit the network.
        from channels.drivers import telegram as tg
        monkeypatch.setattr(tg.TelegramDriver, "send", fake_send)

        row = _channel_row(
            config={"bot_token": "1:AAFfrom-row"},
            metadata={"default_target": "999"},
        )
        db = _db_with_responses(channel_row=row)
        result = _run(send_to_channel(
            db=db, workspace_id=uuid4(), platform="telegram", text="hello",
        ))
        assert result.ok is True
        assert captured["config"]["bot_token"] == "1:AAFfrom-row"
        # Target falls back to metadata.default_target when caller doesn't pass one.
        assert captured["target"] == "999"

    def test_falls_back_to_legacy_integrations_when_no_row(self, monkeypatch):
        from channels.sender import send_to_channel
        from channels.drivers import SendResult

        captured: dict = {}

        async def fake_send(self, *, workspace_id, config, target, text):
            captured.update(config=dict(config), target=target)
            return SendResult(ok=True, latency_ms=1)

        from channels.drivers import telegram as tg
        monkeypatch.setattr(tg.TelegramDriver, "send", fake_send)

        db = _db_with_responses(
            channel_row=None,
            workspace_settings={"integrations": {
                "telegram_bot_token": "1:AAFlegacy",
                "telegram_default_chat_id": "42",
            }},
        )
        result = _run(send_to_channel(
            db=db, workspace_id=uuid4(), platform="telegram", text="hi",
        ))
        assert result.ok is True
        assert captured["config"]["bot_token"] == "1:AAFlegacy"
        assert captured["target"] == "42"

    def test_explicit_target_overrides_metadata_default(self, monkeypatch):
        from channels.sender import send_to_channel
        from channels.drivers import SendResult

        captured: dict = {}

        async def fake_send(self, *, workspace_id, config, target, text):
            captured["target"] = target
            return SendResult(ok=True, latency_ms=1)

        from channels.drivers import telegram as tg
        monkeypatch.setattr(tg.TelegramDriver, "send", fake_send)

        row = _channel_row(
            config={"bot_token": "1:AAF"},
            metadata={"default_target": "999"},
        )
        db = _db_with_responses(channel_row=row)
        result = _run(send_to_channel(
            db=db, workspace_id=uuid4(), platform="telegram",
            text="hi", target="explicit-chat",
        ))
        assert result.ok is True
        assert captured["target"] == "explicit-chat"

    def test_driver_exception_returns_retryable_failure(self, monkeypatch):
        from channels.sender import send_to_channel

        async def boom(self, **_kw):
            raise RuntimeError("network blip")

        from channels.drivers import telegram as tg
        monkeypatch.setattr(tg.TelegramDriver, "send", boom)

        row = _channel_row(config={"bot_token": "1:AAF"})
        db = _db_with_responses(channel_row=row)
        result = _run(send_to_channel(
            db=db, workspace_id=uuid4(), platform="telegram",
            text="hi", target="x",
        ))
        assert result.ok is False
        assert result.retryable is True
        assert "network blip" in (result.error or "")
