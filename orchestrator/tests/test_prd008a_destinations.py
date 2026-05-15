"""
PRD-008-A.1 — channel_connection dispatcher tests
==================================================

Replaces the legacy email/slack_webhook/crm_webhook/shopify_note tests
with the single canonical destination type: dispatch_via_channel.

The dispatcher is a thin wire over an existing ChannelConnection +
running ChannelManager adapter (the same path heartbeats use).
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


def _stub_db_returning(conn_obj):
    """Build a SQLAlchemy-session-shaped MagicMock whose ``.first()``
    returns ``conn_obj`` (a ChannelConnection-shaped namespace, or None
    to simulate a missing row)."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = conn_obj
    return db


def _make_conn(*, status="active", platform="slack", workspace_id=None, conn_id=None):
    from core.models.channels import ChannelConnection
    conn = ChannelConnection()
    conn.id = conn_id or uuid4()
    conn.workspace_id = workspace_id or uuid4()
    conn.platform = platform
    conn.status = status
    return conn


# ---------------------------------------------------------------------------
# Static guards (reject before touching DB / adapter)
# ---------------------------------------------------------------------------

def test_dispatch_via_channel_rejects_missing_connection_id():
    from services.destinations.dispatcher import dispatch_via_channel

    result = _await(dispatch_via_channel(
        destination={"type": "channel_connection", "target": "C123"},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is False
    assert result.retryable is False
    assert "missing connection_id" in result.error


def test_dispatch_via_channel_rejects_missing_target():
    from services.destinations.dispatcher import dispatch_via_channel

    result = _await(dispatch_via_channel(
        destination={"type": "channel_connection", "connection_id": str(uuid4())},
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is False
    assert result.retryable is False
    assert "missing connection_id or target" in result.error


def test_dispatch_via_channel_rejects_invalid_uuid():
    from services.destinations.dispatcher import dispatch_via_channel

    result = _await(dispatch_via_channel(
        destination={
            "type": "channel_connection",
            "connection_id": "not-a-uuid",
            "target": "C123",
        },
        payload=_payload(),
        db=MagicMock(),
        workspace_id=uuid4(),
    ))
    assert result.success is False
    assert result.retryable is False
    assert "UUID" in result.error


# ---------------------------------------------------------------------------
# Workspace isolation
# ---------------------------------------------------------------------------

def test_dispatch_via_channel_workspace_isolation_rejects_foreign_connection():
    from services.destinations.dispatcher import dispatch_via_channel

    # The DB query filters by (id, workspace_id); when the connection
    # belongs to a different workspace, ``.first()`` returns None.
    db = _stub_db_returning(None)

    result = _await(dispatch_via_channel(
        destination={
            "type": "channel_connection",
            "connection_id": str(uuid4()),
            "target": "C123",
        },
        payload=_payload(),
        db=db,
        workspace_id=uuid4(),
    ))
    assert result.success is False
    assert result.retryable is False
    assert "not found in this workspace" in result.error


# ---------------------------------------------------------------------------
# Inactive connection rejected as permanent
# ---------------------------------------------------------------------------

def test_dispatch_via_channel_inactive_connection_is_permanent():
    from services.destinations.dispatcher import dispatch_via_channel

    conn = _make_conn(status="inactive")
    db = _stub_db_returning(conn)

    result = _await(dispatch_via_channel(
        destination={
            "type": "channel_connection",
            "connection_id": str(conn.id),
            "target": "C123",
        },
        payload=_payload(),
        db=db,
        workspace_id=conn.workspace_id,
    ))
    assert result.success is False
    assert result.retryable is False
    assert "inactive" in result.error


# ---------------------------------------------------------------------------
# Adapter not loaded -> retryable
# ---------------------------------------------------------------------------

def test_dispatch_via_channel_missing_adapter_is_retryable():
    from services.destinations import dispatcher as disp_mod

    conn = _make_conn(status="active")
    db = _stub_db_returning(conn)

    fake_manager = MagicMock()
    fake_manager._adapters = {}  # adapter not loaded

    with pytest.MonkeyPatch.context() as mp:
        # Patch get_channel_manager at the module path the dispatcher imports.
        import channels.manager as cm
        mp.setattr(cm, "get_channel_manager", lambda: fake_manager)

        result = _await(disp_mod.dispatch_via_channel(
            destination={
                "type": "channel_connection",
                "connection_id": str(conn.id),
                "target": "C123",
            },
            payload=_payload(),
            db=db,
            workspace_id=conn.workspace_id,
        ))

    assert result.success is False
    assert result.retryable is True
    assert "adapter not loaded" in result.error


# ---------------------------------------------------------------------------
# Happy path -> success with platform + target in extra
# ---------------------------------------------------------------------------

def test_dispatch_via_channel_happy_path():
    from services.destinations import dispatcher as disp_mod

    conn = _make_conn(status="active", platform="slack")
    db = _stub_db_returning(conn)

    sent = {}

    class FakeAdapter:
        async def send_message(self, channel_id, text, **kwargs):
            sent["channel_id"] = channel_id
            sent["text"] = text
            return True

    fake_manager = MagicMock()
    fake_manager._adapters = {str(conn.id): FakeAdapter()}

    with pytest.MonkeyPatch.context() as mp:
        import channels.manager as cm
        mp.setattr(cm, "get_channel_manager", lambda: fake_manager)

        result = _await(disp_mod.dispatch_via_channel(
            destination={
                "type": "channel_connection",
                "connection_id": str(conn.id),
                "target": "C0123456",
            },
            payload=_payload(),
            db=db,
            workspace_id=conn.workspace_id,
        ))

    assert result.success is True
    assert result.destination_type == "channel_connection"
    assert result.extra["platform"] == "slack"
    assert result.extra["target"] == "C0123456"
    assert sent["channel_id"] == "C0123456"
    assert "INBUILD UK" in sent["text"]
    assert "+447700900123" in sent["text"]


# ---------------------------------------------------------------------------
# Adapter returns False (bad target) -> permanent failure
# ---------------------------------------------------------------------------

def test_dispatch_via_channel_adapter_false_is_permanent():
    from services.destinations import dispatcher as disp_mod

    conn = _make_conn(status="active")
    db = _stub_db_returning(conn)

    class FakeAdapter:
        async def send_message(self, channel_id, text, **kwargs):
            return False

    fake_manager = MagicMock()
    fake_manager._adapters = {str(conn.id): FakeAdapter()}

    with pytest.MonkeyPatch.context() as mp:
        import channels.manager as cm
        mp.setattr(cm, "get_channel_manager", lambda: fake_manager)

        result = _await(disp_mod.dispatch_via_channel(
            destination={
                "type": "channel_connection",
                "connection_id": str(conn.id),
                "target": "C-bad",
            },
            payload=_payload(),
            db=db,
            workspace_id=conn.workspace_id,
        ))

    assert result.success is False
    assert result.retryable is False
    assert "rejected target" in result.error


# ---------------------------------------------------------------------------
# Adapter raises -> retryable
# ---------------------------------------------------------------------------

def test_dispatch_via_channel_adapter_exception_is_retryable():
    from services.destinations import dispatcher as disp_mod

    conn = _make_conn(status="active")
    db = _stub_db_returning(conn)

    class FakeAdapter:
        async def send_message(self, channel_id, text, **kwargs):
            raise RuntimeError("transient slack 5xx")

    fake_manager = MagicMock()
    fake_manager._adapters = {str(conn.id): FakeAdapter()}

    with pytest.MonkeyPatch.context() as mp:
        import channels.manager as cm
        mp.setattr(cm, "get_channel_manager", lambda: fake_manager)

        result = _await(disp_mod.dispatch_via_channel(
            destination={
                "type": "channel_connection",
                "connection_id": str(conn.id),
                "target": "C123",
            },
            payload=_payload(),
            db=db,
            workspace_id=conn.workspace_id,
        ))

    assert result.success is False
    assert result.retryable is True
    assert "adapter exception" in result.error


# ---------------------------------------------------------------------------
# Renderer carries optional fields
# ---------------------------------------------------------------------------

def test_render_callback_text_includes_optional_fields():
    from services.destinations.dispatcher import _render_callback_text

    text = _render_callback_text(_payload(urgency="urgent", preferred_time="3pm"))
    assert "Urgency: urgent" in text
    assert "Preferred time: 3pm" in text


def test_render_callback_text_skips_unset_optional_fields():
    from services.destinations.dispatcher import _render_callback_text

    text = _render_callback_text(_payload(urgency=None, preferred_time=None))
    assert "Urgency:" not in text
    assert "Preferred time:" not in text


# ---------------------------------------------------------------------------
# DESTINATION_TYPES whitelist
# ---------------------------------------------------------------------------

def test_destination_types_is_channel_connection_only():
    from services.destinations.base import DESTINATION_TYPES

    assert DESTINATION_TYPES == ("channel_connection",), (
        "PRD-008-A.1: legacy email/slack_webhook/crm_webhook/shopify_note "
        "destination types are gone. Adding a new destination type should be "
        "a deliberate change with a new dispatcher."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
