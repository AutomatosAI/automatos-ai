"""PRD-128 US-003: NotificationDispatcher unit tests.

These tests exercise the dispatcher against a ``MagicMock`` database session
so the full fan-out logic can be verified without a live Postgres. External
delivery (Telegram / Slack / webhook) is stubbed via
``send_workspace_notification`` patching.

Covered scenarios:

* Silent preference is skipped (no insert, no external call).
* in_app preference inserts a row and does NOT call ``db.commit``.
* Multi-destination fan-out: one ``in_app`` row + one ``telegram`` row
  triggers both an insert and an external send on a single event.
* User-specific rows shadow workspace defaults on the same destination.
* Empty preferences list defaults to a single in_app dispatch.
"""

from __future__ import annotations

import asyncio
import os

# notification_dispatcher imports notification_service which imports
# core.database.database (SessionLocal). That in turn pulls config.py which
# requires Postgres env vars. Seed harmless defaults — nothing here actually
# hits a real DB.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402
from uuid import uuid4  # noqa: E402

import pytest  # noqa: E402

from core.services.notification_dispatcher import NotificationDispatcher  # noqa: E402


# --------------------------------------------------------------- helpers


def _make_prefs_result(rows: list[tuple]) -> MagicMock:
    """Return a mock Result whose fetchall() yields the given row tuples.

    Column order must match the dispatcher's SELECT:
        (user_id, destination, enabled, channel_connection_id)
    """
    result = MagicMock()
    result.fetchall.return_value = rows
    return result


def _make_db(pref_rows: list[tuple]) -> MagicMock:
    """Build a mock session where the first execute() returns prefs and the
    rest (inserts, channel lookups) return a no-op result."""
    db = MagicMock()
    prefs_result = _make_prefs_result(pref_rows)
    insert_result = MagicMock()
    # First call: preferences SELECT. Subsequent calls: inserts / lookups.
    db.execute.side_effect = [prefs_result] + [insert_result] * 20
    return db


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _first_insert_params(db: MagicMock) -> dict:
    """Return the params dict from the first INSERT INTO notifications call."""
    for call in db.execute.call_args_list:
        sql_obj, params = call.args
        if "INSERT INTO notifications" in str(sql_obj):
            return params
    raise AssertionError("No INSERT INTO notifications call was made")


def _count_inserts(db: MagicMock) -> int:
    return sum(
        1
        for call in db.execute.call_args_list
        if "INSERT INTO notifications" in str(call.args[0])
    )


# -------------------------------------------------------------- tests


def test_silent_preference_skips_insert_and_external_send():
    db = _make_db([(None, "silent", True, None)])
    dispatcher = NotificationDispatcher(db, uuid4())

    with patch(
        "core.services.notification_dispatcher.send_workspace_notification",
        new=AsyncMock(return_value=True),
    ) as fake_send:
        result = _run(
            dispatcher.dispatch(
                event_type="mission_step_complete",
                title="Step 1 done",
            )
        )

    assert result == {"dispatched_to": []}
    assert _count_inserts(db) == 0
    fake_send.assert_not_awaited()
    db.commit.assert_not_called()


def test_in_app_preference_inserts_row_without_committing():
    ws_id = uuid4()
    db = _make_db([(None, "in_app", True, None)])
    dispatcher = NotificationDispatcher(db, ws_id)

    with patch(
        "core.services.notification_dispatcher.send_workspace_notification",
        new=AsyncMock(return_value=True),
    ) as fake_send:
        result = _run(
            dispatcher.dispatch(
                event_type="heartbeat_complete",
                title="Heartbeat finished",
                message="All 3 tasks passed",
                link_type="heartbeat",
                link_id="abc-123",
                agent_id=42,
                agent_name="Sentinel",
                status="ok",
            )
        )

    assert result == {"dispatched_to": ["in_app"]}
    assert _count_inserts(db) == 1

    params = _first_insert_params(db)
    assert params["ws_id"] == str(ws_id)
    assert params["event_type"] == "heartbeat_complete"
    assert params["title"] == "Heartbeat finished"
    assert params["message"] == "All 3 tasks passed"
    assert params["link_type"] == "heartbeat"
    assert params["link_id"] == "abc-123"
    assert params["agent_id"] == 42
    assert params["agent_name"] == "Sentinel"
    assert params["status"] == "ok"

    db.commit.assert_not_called()
    fake_send.assert_not_awaited()


def test_multi_destination_fanout_in_app_and_telegram():
    db = _make_db(
        [
            (None, "in_app", True, None),
            (None, "telegram", True, None),
        ]
    )
    dispatcher = NotificationDispatcher(db, uuid4())

    with patch(
        "core.services.notification_dispatcher.send_workspace_notification",
        new=AsyncMock(return_value=True),
    ) as fake_send:
        result = _run(
            dispatcher.dispatch(
                event_type="task_complete",
                title="Task done",
                message="Shipped",
                agent_name="Builder",
            )
        )

    assert set(result["dispatched_to"]) == {"in_app", "telegram"}
    assert _count_inserts(db) == 1
    fake_send.assert_awaited_once()
    # Telegram payload should contain the formatted header
    _, kwargs = fake_send.call_args
    sent_message = fake_send.call_args.args[1]
    assert "Task done" in sent_message
    assert "Agent: Builder" in sent_message
    assert fake_send.call_args.kwargs["channel"] == "telegram"
    db.commit.assert_not_called()


def test_user_specific_row_overrides_workspace_default():
    # Workspace default says in_app; user prefers silent for same destination.
    # Additional workspace default destination (telegram) should survive.
    db = _make_db(
        [
            (None, "in_app", True, None),      # workspace default
            (None, "telegram", True, None),    # workspace default (no user override)
            (7, "in_app", True, None),         # user override on in_app slot
        ]
    )
    # But the user chose a DIFFERENT destination for in_app slot? We actually
    # need a case where user row replaces default row on same destination.
    # Let's make user row disable in_app.
    db = _make_db(
        [
            (None, "in_app", True, None),
            (None, "telegram", True, None),
            (7, "in_app", False, None),  # user disables in_app
        ]
    )
    dispatcher = NotificationDispatcher(db, uuid4())

    with patch(
        "core.services.notification_dispatcher.send_workspace_notification",
        new=AsyncMock(return_value=True),
    ) as fake_send:
        result = _run(
            dispatcher.dispatch(
                event_type="report_submitted",
                title="Report ready",
                user_id=7,
            )
        )

    # User row (enabled=False) wins on in_app → no insert.
    # Workspace telegram row has no user override → still fires.
    assert result["dispatched_to"] == ["telegram"]
    assert _count_inserts(db) == 0
    fake_send.assert_awaited_once()
    assert fake_send.call_args.kwargs["channel"] == "telegram"


def test_no_preferences_defaults_to_single_in_app_dispatch():
    db = _make_db([])  # zero rows returned from SELECT
    dispatcher = NotificationDispatcher(db, uuid4())

    with patch(
        "core.services.notification_dispatcher.send_workspace_notification",
        new=AsyncMock(return_value=True),
    ) as fake_send:
        result = _run(
            dispatcher.dispatch(
                event_type="agent_error",
                title="Agent crashed",
                status="error",
            )
        )

    assert result == {"dispatched_to": ["in_app"]}
    assert _count_inserts(db) == 1
    fake_send.assert_not_awaited()
    db.commit.assert_not_called()


def test_format_external_message_truncates_and_uses_status_icon():
    long = "x" * 400
    out = NotificationDispatcher._format_external_message(
        title="Big job",
        status="error",
        agent_name="Runner",
        message=long,
    )
    lines = out.splitlines()
    assert lines[0].startswith("❌")
    assert "Big job" in lines[0]
    assert lines[1] == "Agent: Runner"
    # Truncated line should be <= 200 chars (including ellipsis)
    assert len(lines[2]) <= 200
    assert lines[2].endswith("…")
