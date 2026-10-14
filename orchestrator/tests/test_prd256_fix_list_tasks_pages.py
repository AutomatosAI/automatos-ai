"""P256-FIX-N15-3 (F424): platform_list_tasks pages, and its total is the filter's.

Night 15: the action took no offset, so a full list of a big column was out of
reach, and each page's own length went out as ``total``. Six identical asks in
one second counted the review column as 66, 66, 66, 31, 66 and 13. Every card
here is made in ONE transaction, so Postgres gives them all the same
``created_at``: only a total order (created_at, then id) keeps consecutive
pages from repeating or skipping a card.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

REVIEW_CARDS = 70
PAGE = 30


def _cards(db, ws, status, count):
    from core.models.core import BoardTask

    cards = [BoardTask(workspace_id=ws, title=f"{status} card {i}", status=status, priority="medium")
             for i in range(count)]
    db.add_all(cards)
    db.flush()
    return {c.id for c in cards}


def _list(db, ws, **params):
    from modules.tools.discovery.board_task_list import list_board_tasks

    return asyncio.run(list_board_tasks(db, ws, {"status": "review", "limit": PAGE, **params}))


def test_seventy_review_cards_in_three_pages_each_once_and_seventy_on_every_page(db_session, seed_workspace):
    ws, other = UUID(seed_workspace()), UUID(seed_workspace())
    review = _cards(db_session, ws, "review", REVIEW_CARDS)
    _cards(db_session, ws, "inbox", 5)            # another column: not in the filter's total
    _cards(db_session, other, "review", 4)        # another workspace: never seen

    pages, offset = [], 0
    while offset is not None:
        page = _list(db_session, ws, offset=offset)
        pages.append(page)
        offset = page["next_offset"]

    assert len(pages) == 3
    assert [p["total"] for p in pages] == [REVIEW_CARDS] * 3
    assert [p["returned"] for p in pages] == [30, 30, 10]
    assert [p["offset"] for p in pages] == [0, 30, 60]
    seen = [t["id"] for p in pages for t in p["tasks"]]
    assert len(seen) == len(set(seen)) == REVIEW_CARDS        # every card exactly once
    assert set(seen) == review


def test_the_same_ask_twice_gives_the_same_page(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    _cards(db_session, ws, "review", REVIEW_CARDS)

    first, again = _list(db_session, ws, offset=PAGE), _list(db_session, ws, offset=PAGE)

    assert [t["id"] for t in first["tasks"]] == [t["id"] for t in again["tasks"]]
    assert first["total"] == again["total"] == REVIEW_CARDS


def test_an_offset_past_the_end_is_an_empty_last_page_with_the_same_total(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    _cards(db_session, ws, "review", 3)

    page = _list(db_session, ws, offset=10)

    assert page["tasks"] == [] and page["total"] == 3 and page["next_offset"] is None


def test_the_action_documents_paging():
    from modules.tools.discovery.action_registry import ActionRegistry
    from modules.tools.discovery.actions_board_tasks import _register_list_tasks

    registry = ActionRegistry()
    _register_list_tasks(registry)
    action = registry.get("platform_list_tasks")

    assert {"offset", "limit"} <= set(action.parameters["properties"])
    assert "next_offset" in action.description and "'total'" in action.description
