"""F460 (night 18): "Close the done cards from last week" never reached the bulk update: platform_list_tasks
had no completed-date filter, so Auto could not list last week's done cards. ``completed_after`` (on or
after that day) and ``completed_before`` (before that day), ISO dates in UTC, keep the cards finished in
the window, so the existing bulk update can take them.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID

LAST_WEEK = {"status": "done", "completed_after": "2026-10-12", "completed_before": "2026-10-19"}


def _at(day, hour=12):
    return datetime(2026, 10, day, hour, 0, tzinfo=timezone.utc)


def _card(db, ws, title, status="done", finished=None):
    from core.models.core import BoardTask

    card = BoardTask(workspace_id=ws, title=title, status=status, priority="medium", completed_at=finished)
    db.add(card)
    db.flush()
    return card


def _list(db, ws, **params):
    from modules.tools.discovery.board_task_list import list_board_tasks

    return asyncio.run(list_board_tasks(db, ws, params))


def test_last_weeks_done_cards_and_only_them(db_session, seed_workspace):
    ws, other = UUID(seed_workspace()), UUID(seed_workspace())
    monday = _card(db_session, ws, "Monday's roast plan", finished=_at(12, 0))           # the first moment counts
    sunday = _card(db_session, ws, "Sunday's stock count", finished=_at(18, 23))
    _card(db_session, ws, "The week before", finished=_at(11, 23))
    _card(db_session, ws, "This Monday", finished=_at(19, 0))                             # the end day is out
    _card(db_session, ws, "Never finished")
    _card(db_session, ws, "In review last week", status="review", finished=_at(14))
    _card(db_session, other, "Another shop's card", finished=_at(14))

    page = _list(db_session, ws, **LAST_WEEK)

    assert page["success"] is True and page["total"] == 2
    assert {t["title"] for t in page["tasks"]} == {monday.title, sunday.title}


def test_a_date_that_is_no_date_is_refused_in_words(db_session, seed_workspace):
    answer = _list(db_session, UUID(seed_workspace()), status="done", completed_after="last week")

    assert answer["success"] is False and "completed_after must be a date as YYYY-MM-DD" in answer["error"]


def test_the_action_takes_the_window():
    from modules.tools.discovery.action_registry import ActionRegistry
    from modules.tools.discovery.actions_board_tasks import _register_list_tasks

    registry = ActionRegistry()
    _register_list_tasks(registry)
    props = registry.get("platform_list_tasks").parameters["properties"]

    assert props["completed_after"]["type"] == props["completed_before"]["type"] == "string"
    assert "YYYY-MM-DD" in props["completed_after"]["description"]
