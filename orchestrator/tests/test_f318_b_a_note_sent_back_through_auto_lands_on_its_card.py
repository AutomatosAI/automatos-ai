"""F318 (night 9b): a card sent back through Auto carries the owner's note, as the board's Reject does.

Chat 6586c8bf: "Send card #0081 back to the Business Analyst with this: I only wanted this
September. Drop the 2025 figure and give me the one number." Auto's first call was
platform_update_task {description: …}, the board's Re-brief: #1938 re-ran with Auto's
rewording as its brief and showed times_sent_back 0 and no review feedback. Its last
calls moved the card to In progress with the note, which re-runs the old brief and drops
the note. Now a new brief for a card the owner said to send back is refused with the
send-back that carries their words, and a move of an answered card to In progress with a
note is the board's Reject: review_feedback, owner_corrections and times_sent_back, as
the board's own.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

from tests import test_1094_a_ticket_with_no_agent_is_never_in_progress as f1094

shop = f1094.board      # #1094's Postgres: a workspace, its Content Creator, Auto's real tool
OWNER = "owner@cafe.test"
SAID = ("Send card #0081 back to the Business Analyst with this: I only wanted this September. Drop the 2025 "
        "figure and give me the one number.")
NOTE = "I only wanted this September. Drop the 2025 figure and give me the one number."
CHAT = {"conversation_id": str(uuid4())}


def _answered(shop):
    return f1094._ticket(shop, status="review", assigned_agent_id=shop.agent, result="£1,455.00, and £1,210 in 2025.",
                         completed_at=datetime.now(timezone.utc))


def test_in_progress_with_the_owners_note_is_the_boards_reject(shop):
    from modules.tools.discovery.handlers_board_tasks import update_board_task_status
    from services.ticket_redo import times_sent_back

    task = _answered(shop)

    out = asyncio.run(update_board_task_status(shop.db, shop.ws, {"task_id": task.id, "status": "in_progress",
                                                                  "note": NOTE, "_user_id": OWNER}))

    shop.db.refresh(task)
    assert out["success"] is True and out["sent_back"] is True
    assert task.status == "assigned" and task.review_feedback == NOTE           # night 9b: null, the note dropped
    assert task.planning_data["owner_corrections"][-1]["note"] == NOTE
    assert times_sent_back(task) == 1 and shop.launched == []                   # not the old brief run again


def test_in_progress_with_no_note_still_runs_it_again(shop):
    from modules.tools.discovery.handlers_board_tasks import update_board_task_status

    task = _answered(shop)

    out = asyncio.run(update_board_task_status(shop.db, shop.ws, {"task_id": task.id, "status": "in_progress"}))

    assert out["success"] is True and "sent_back" not in out and shop.launched == [task.id]


# ── Auto's guard: a new brief is not a send-back ───────────────────────────────────────

@pytest.fixture
def turn(monkeypatch):
    """The owner's words this turn and the card they name, as owner_turn would read them."""
    import modules.tools.discovery.follows_the_owner as guard
    from modules.tools.discovery.owner_turn import NamedCard, OwnerTurn

    def _set(latest, status="review"):
        task = NS(id=1938, title="September retail takings", status=status, source_type="user", description="")
        card = NamedCard(ref="#0081", seq=81, step=None, task=task)
        monkeypatch.setattr(guard, "owner_turn", lambda db, ws, ctx: OwnerTurn(latest=latest, earlier="",
                                                                               cards=(card,)))

    monkeypatch.setattr(guard, "_the_cards_words", lambda db, ws, params: ())
    monkeypatch.setattr(guard, "owners_recent_words", lambda db, ws, turn: ())
    monkeypatch.setattr(guard, "autos_last_reply", lambda db, ws, turn: "")
    monkeypatch.setattr(guard, "autos_proposal", lambda db, ws, turn: "")
    return _set


def _refused(action, **params):
    from modules.tools.discovery.follows_the_owner import refusal_for

    return refusal_for(None, uuid4(), action, params, CHAT)


def test_a_new_brief_for_a_card_the_owner_sent_back_is_refused_with_the_send_back(turn):
    turn(SAID)

    refusal = _refused("platform_update_task", task_id="0081", description=NOTE)

    assert refusal is not None and 'status: "assigned"' in refusal and NOTE in refusal     # night 9b: a Re-brief


def test_a_new_brief_the_owner_asked_for_still_goes_on(turn):
    turn("Put this brief on #0081 and send it back: September 2026 retail takings, one number.")

    assert _refused("platform_update_task", task_id="0081",
                    description="September 2026 retail takings, one number.") is None


def test_in_progress_with_their_words_as_the_note_is_let_through(turn):
    turn("The Shopify Business Analyst, it's the one already on the card. I just want it sent back with my note: "
         f"{NOTE}")

    assert _refused("platform_update_task_status", task_id=81, status="in_progress", note=NOTE) is None
    assert 'status: "assigned"' in _refused("platform_update_task_status", task_id=81, status="in_progress")
