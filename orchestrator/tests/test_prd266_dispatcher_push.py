"""PRD-266 US-010d (§4.4, FR-6, FR-7, K3, K6, K7) — push as the dispatcher's destination.

``NotificationDispatcher.dispatch`` against a ``MagicMock`` session (the prd128 style: the
first ``db.execute`` is the preferences SELECT), with ``services.push.fanout.hand_over``
patched. Proven here:

* ``dispatch`` is a short function (≤ 50 code lines, counted from the AST as CI's shape gate
  counts them) over helpers.
* A question with no preferences rings the bell once and hands its push over once, carrying
  that in-app row's id; quiet hours hold the push for normal severity and keep it for urgent
  and security; no event outside ``PUSH_EVENT_TYPES`` ever reaches ``hand_over`` (FR-7), even
  with an auto-reporting route to ``push``.
* K7: a person's ``push`` row is never a delivery destination, so it never costs the bell
  row; a route to ``push`` delivers the bell row (and, for a push type, the phone).
* ``_insert_in_app`` ends with ``RETURNING id`` and returns that id as a string (None when the
  result has none); a ``hand_over`` that raises leaves ``push`` out of ``dispatched_to`` and
  never raises out of ``dispatch``.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain when the suite's
# own environment does not set it (CI does, and these defaults never override it).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")

from api import notifications as notifications_api  # noqa: E402
from core.services import auto_reporting  # noqa: E402
from core.services import notification_dispatcher as nd  # noqa: E402
from core.services.notification_dispatcher import VALID_EVENT_TYPES, NotificationDispatcher  # noqa: E402
from services.push.payload import PUSH_EVENT_TYPES  # noqa: E402

DISPATCHER_FILE = Path(nd.__file__)
CODE_LINE_LIMIT = 50
ROW_ID = UUID("6f1c2d3e-4a5b-4c6d-8e9f-0a1b2c3d4e5f")
PERSON = 7
HAND_OVER = "services.push.fanout.hand_over"
SEND = "core.services.notification_dispatcher.send_workspace_notification"
NO_ROUTES: dict[str, Any] = {"enabled": True, "routes": {}, "quiet_hours": {"enabled": False}}
QUIET: dict[str, Any] = {"enabled": True, "routes": {}, "quiet_hours": {"enabled": True}}


# --------------------------------------------------------------- helpers


def _make_db(pref_rows: list[tuple], row_id: Optional[UUID] = ROW_ID) -> MagicMock:
    """First execute(): the preferences SELECT; every later one: an insert returning ``row_id``."""
    db = MagicMock()
    prefs_result = MagicMock()
    prefs_result.fetchall.return_value = pref_rows
    insert_result = MagicMock()
    insert_result.scalar.return_value = row_id
    db.execute.side_effect = [prefs_result] + [insert_result] * 20
    return db


def _inserts(db: MagicMock) -> list[dict]:
    return [call.args[1] for call in db.execute.call_args_list
            if "INSERT INTO notifications" in str(call.args[0])]


def _dispatch(db: MagicMock, settings: dict[str, Any], *, quiet: bool = False,
              hand_over: Optional[MagicMock] = None, **event: Any) -> tuple[dict, MagicMock, AsyncMock]:
    """Run one dispatch with auto_reporting ``settings`` and ``hand_over`` / the sender patched."""
    handed = hand_over if hand_over is not None else MagicMock(return_value=True)
    send = AsyncMock(return_value=True)
    with patch.object(NotificationDispatcher, "_load_auto_reporting", lambda self: settings), \
            patch.object(NotificationDispatcher, "_is_quiet_hours", staticmethod(lambda s: quiet)), \
            patch(HAND_OVER, new=handed), patch(SEND, new=send):
        result = asyncio.run(NotificationDispatcher(db, uuid4()).dispatch(**event))
    return result, handed, send


def _code_lines(node: ast.AST, lines: list[str]) -> int:
    """Lines that hold code: not blank, not comment-only, not the docstring (CI's shape gate)."""
    first = node.body[0] if node.body else None
    docstring: set[int] = set()
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        docstring = set(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    numbers = range(node.lineno, (node.end_lineno or node.lineno) + 1)
    return sum(1 for n in numbers if n not in docstring and lines[n - 1].strip()
               and not lines[n - 1].strip().startswith("#"))


# ------------------------------------------------------------ the shape


def test_dispatch_is_at_most_fifty_code_lines():
    source = DISPATCHER_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = [node for node in ast.walk(tree)
             if isinstance(node, ast.AsyncFunctionDef) and node.name == "dispatch"]
    assert len(found) == 1
    assert _code_lines(found[0], source.splitlines()) <= CODE_LINE_LIMIT


def test_push_is_a_valid_destination_but_never_a_channel_or_an_external_destination():
    assert "push" in notifications_api._VALID_DESTINATIONS
    assert "push" in auto_reporting._VALID_DESTINATIONS
    assert "push" not in auto_reporting._VALID_CHANNELS
    assert "push" not in nd._EXTERNAL_DESTINATIONS


# ------------------------------------------------------------ the push leg


def test_a_question_with_no_preferences_rings_the_bell_once_and_hands_its_id_over():
    db = _make_db([])
    result, hand_over, _ = _dispatch(
        db, NO_ROUTES, event_type="question_pending", title="Auto asks", link_type="question",
        link_id=41, agent_name="Scout", user_id=PERSON, severity="normal",
    )

    assert result == {"dispatched_to": ["in_app", "push"]}
    assert len(_inserts(db)) == 1
    hand_over.assert_called_once()
    session, event = hand_over.call_args.args
    assert session is db
    assert event.event_type == "question_pending"
    assert event.notification_id == str(ROW_ID)
    assert (event.link_type, event.link_id, event.user_id) == ("question", "41", PERSON)
    assert (event.agent_name, event.severity) == ("Scout", "normal")


def test_quiet_hours_hold_the_push_for_normal_severity():
    db = _make_db([])
    result, hand_over, _ = _dispatch(db, QUIET, quiet=True, event_type="approval_pending",
                                     title="Approve?", severity="normal")

    assert result == {"dispatched_to": ["in_app"]}
    assert len(_inserts(db)) == 1
    hand_over.assert_not_called()


@pytest.mark.parametrize("severity", ["urgent", "security"])
def test_quiet_hours_keep_the_push_for_urgent_and_security(severity):
    db = _make_db([])
    result, hand_over, _ = _dispatch(db, QUIET, quiet=True, event_type="question_pending",
                                     title="Auto asks", severity=severity)

    assert result == {"dispatched_to": ["in_app", "push"]}
    hand_over.assert_called_once()


@pytest.mark.parametrize("event_type", sorted(VALID_EVENT_TYPES - PUSH_EVENT_TYPES))
def test_an_event_that_is_not_a_push_type_never_reaches_hand_over(event_type):
    db = _make_db([])
    result, hand_over, _ = _dispatch(db, NO_ROUTES, event_type=event_type, title="Something")

    assert result == {"dispatched_to": ["in_app"]}
    hand_over.assert_not_called()


def test_a_route_to_push_for_another_event_is_the_bell_row_only():
    # The workspace's telegram row is overridden by the route: proof the route is read.
    db = _make_db([(None, "telegram", True, None)])
    routed = {**NO_ROUTES, "routes": {"task_complete": "push"}}
    result, hand_over, send = _dispatch(db, routed, event_type="task_complete", title="Task done")

    assert result == {"dispatched_to": ["in_app"]}
    assert len(_inserts(db)) == 1
    send.assert_not_awaited()
    hand_over.assert_not_called()


def test_a_route_to_push_for_a_push_type_gives_the_bell_row_and_the_push():
    db = _make_db([(None, "telegram", True, None)])
    routed = {**NO_ROUTES, "routes": {"mission_plan_ready": "push"}}
    result, hand_over, send = _dispatch(db, routed, event_type="mission_plan_ready", title="Plan ready")

    assert result == {"dispatched_to": ["in_app", "push"]}
    assert len(_inserts(db)) == 1
    send.assert_not_awaited()
    assert hand_over.call_args.args[1].notification_id == str(ROW_ID)


# --------------------------------------------------------- K7: push rows


@pytest.mark.parametrize("enabled", [True, False])
def test_a_persons_push_row_alone_still_gets_the_in_app_default(enabled):
    db = _make_db([(PERSON, "push", enabled, None)])
    result, hand_over, send = _dispatch(db, NO_ROUTES, event_type="task_complete", title="Task done",
                                        user_id=PERSON)

    assert result == {"dispatched_to": ["in_app"]}
    assert len(_inserts(db)) == 1
    assert _inserts(db)[0]["user_id"] == PERSON
    send.assert_not_awaited()
    hand_over.assert_not_called()


def test_a_push_row_beside_a_telegram_row_never_adds_the_bell_row():
    # The push row is dropped; the telegram row is still the person's only destination.
    db = _make_db([(None, "telegram", True, None), (PERSON, "push", True, None)])
    result, hand_over, send = _dispatch(db, NO_ROUTES, event_type="question_pending",
                                        title="Auto asks", user_id=PERSON, hand_over=MagicMock(return_value=False))

    assert result == {"dispatched_to": ["telegram"]}
    assert _inserts(db) == []
    send.assert_awaited_once()
    hand_over.assert_called_once()
    assert hand_over.call_args.args[1].notification_id is None


def test_an_in_app_row_beside_a_push_row_is_one_bell_row_and_the_push():
    db = _make_db([(None, "in_app", True, None), (PERSON, "push", True, None)])
    result, hand_over, _ = _dispatch(db, NO_ROUTES, event_type="question_pending", title="Auto asks",
                                     user_id=PERSON)

    assert result == {"dispatched_to": ["in_app", "push"]}
    assert len(_inserts(db)) == 1
    assert hand_over.call_args.args[1].notification_id == str(ROW_ID)


def test_a_silent_route_for_a_push_type_still_hands_the_push_over():
    # K7: the person's push row is the push switch (read at fan-out); 'silent' holds the bell row only.
    db = _make_db([])
    routed = {**NO_ROUTES, "routes": {"question_pending": "silent"}}
    result, hand_over, _ = _dispatch(db, routed, event_type="question_pending", title="Auto asks")

    assert result == {"dispatched_to": ["push"]}
    assert _inserts(db) == []
    assert hand_over.call_args.args[1].notification_id is None


# --------------------------------------------------------- the in-app row


def test_insert_in_app_returns_the_returning_id_as_a_string():
    db = MagicMock()
    db.execute.return_value.scalar.return_value = ROW_ID
    row_id = NotificationDispatcher(db, uuid4())._insert_in_app(
        user_id=None, event_type="task_complete", title="Task done", message=None,
        link_type=None, link_id=None, agent_id=None, agent_name=None, status="ok",
    )

    assert row_id == str(ROW_ID)
    sql = str(db.execute.call_args.args[0])
    assert sql.rstrip().endswith("RETURNING id")


@pytest.mark.parametrize("returned", [None, "no result"])
def test_insert_in_app_returns_none_when_the_result_has_no_id(returned):
    db = MagicMock()
    if returned is None:
        db.execute.return_value.scalar.return_value = None
    else:
        db.execute.return_value = None
    row_id = NotificationDispatcher(db, uuid4())._insert_in_app(
        user_id=None, event_type="task_complete", title="Task done", message=None,
        link_type=None, link_id=None, agent_id=None, agent_name=None, status="ok",
    )

    assert row_id is None


# ------------------------------------------------------- never raises


def test_a_hand_over_that_raises_leaves_push_out_without_raising(caplog):
    db = _make_db([])
    boom = MagicMock(side_effect=RuntimeError("fan-out down"))
    with caplog.at_level(logging.ERROR, logger=nd.__name__):
        result, hand_over, _ = _dispatch(db, NO_ROUTES, hand_over=boom, event_type="question_pending",
                                         title="Auto asks")

    assert result == {"dispatched_to": ["in_app"]}
    assert len(_inserts(db)) == 1
    hand_over.assert_called_once()
    assert any("Push hand-over failed" in record.getMessage() for record in caplog.records)


def test_a_hand_over_that_finds_no_phone_leaves_push_out():
    db = _make_db([])
    result, hand_over, _ = _dispatch(db, NO_ROUTES, hand_over=MagicMock(return_value=False),
                                     event_type="review_pending", title="Ready for review")

    assert result == {"dispatched_to": ["in_app"]}
    hand_over.assert_called_once()
