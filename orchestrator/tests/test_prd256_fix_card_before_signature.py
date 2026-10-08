"""PRD-256 FX-003 (night 12, B1, F386): the card comes before the signature rule.

Night 12: "Approve #0953 — looks good", "cancel #0953" and "delete every done card" from
chat got "A note on the card is signed as the owner's, so it is their own words…" and NO
approval card, 17 times: ``follows_the_owner`` wraps ``PlatformActionExecutor.execute``
and refused before ``asks_the_owner_first`` (on ``_run_cleared``) could raise the card.
An owner-only call from a person's chat now skips the signature rule: it raises the card,
which shows the call and its note, and the owner's click signs the move (``CLICKED_BY``).
A note on a card the call does not close is still judged by the rule. No regex reads the
owner's words for a click (Decision D1).

Each call runs through the executor's whole stack (the question guard, this guard, the
session ticket, the role gates, the hierarchy check, the rate limit and the ask).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from modules.tools.discovery.owner_only import MAX_CARDS_NAMED

MOVE = "platform_update_task_status"
SIGNATURE_REFUSAL = "signed as the owner's"
AUTOS_NOTE = "Approved — the owner is happy with the draft and it can go out."   # Auto's paraphrase
BULK_CARDS = MAX_CARDS_NAMED + 1


@pytest.fixture
def board(db_session, seed_workspace):
    """A workspace with Auto, a card in Review and one Done card more than an ask names; the move's
    handler is recorded, so a call that moves a card shows up on it."""
    from core.models.core import Agent, BoardTask
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    ws = UUID(seed_workspace())
    auto = Agent(name="Auto", agent_type="system", description="", status="active", configuration={},
                 workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws),
                 is_system_agent=True)
    card = BoardTask(workspace_id=ws, title="Wholesale reply to Larder & Loaf", status="review", source_type="user")
    done = [BoardTask(workspace_id=ws, title=f"Done card {n}", status="done", source_type="user")
            for n in range(BULK_CARDS)]
    db_session.add_all([auto, card, *done])
    db_session.flush()
    executor = PlatformActionExecutor(db_session, ws)
    executor._full_autonomy = lambda: True   # the dial on: an owner-only call still asks
    handler = AsyncMock(return_value={"success": True, "task_id": card.id})
    executor._handlers[MOVE] = handler
    return NS(db=db_session, ws=ws, card=card, done=done, number=f"#{card.workspace_seq:04d}",
              executor=executor, handler=handler, auto=auto.id)


@pytest.fixture
def says(monkeypatch):
    """The owner's latest message in the chat, and the cards it names, as owner_turn reads them."""
    import modules.tools.discovery.follows_the_owner as guard
    from modules.tools.discovery.owner_turn import NamedCard, OwnerTurn

    def _set(latest, *tasks):
        cards = tuple(NamedCard(ref=f"#{task.workspace_seq:04d}", seq=task.workspace_seq, step=None, task=task)
                      for task in tasks)
        turn = OwnerTurn(latest=latest, earlier="", cards=cards)
        monkeypatch.setattr(guard, "owner_turn", lambda db, ws, ctx: turn if ctx else None)
        monkeypatch.setattr(guard, "owners_recent_words", lambda db, ws, turn: (latest,))
    return _set


def _owners_chat():
    return {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}


def _call(board, params, caller_context):
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(board.executor.execute(MOVE, {**params, "_agent_id": board.auto}, caller_context))


def _asked(reply):
    return (reply["success"] is False and reply.get("requires_confirmation") is True
            and reply.get("owner_only") is True and isinstance(reply.get("grant_id"), int))


# ── The three night-12 shapes: the card, not the refusal ────────────────────────────

def test_approve_with_autos_note_raises_the_card_and_moves_nothing(board, says):
    """N5-a: 'Approve #0953 — looks good' → done with Auto's own note was refused; the card shows the note."""
    says(f"Approve {board.number} — looks good", board.card)

    reply = _call(board, {"task_id": board.number, "status": "done", "note": AUTOS_NOTE}, _owners_chat())

    assert _asked(reply), reply
    assert SIGNATURE_REFUSAL not in str(reply.get("error", ""))
    assert "approve" in reply["message"] and board.card.title in reply["message"]
    assert reply["params"]["note"] == AUTOS_NOTE        # the card shows what the click signs
    board.handler.assert_not_called()


def test_cancel_with_autos_note_raises_the_card_and_moves_nothing(board, says):
    """F263-a: 'cancel #0953' → cancelled, with a note the owner never wrote."""
    says(f"cancel {board.number}", board.card)

    reply = _call(board, {"task_id": board.number, "status": "cancelled",
                          "note": "Cancelled at the owner's request."}, _owners_chat())

    assert _asked(reply), reply
    assert "cancel" in reply["message"] and board.card.title in reply["message"]
    board.handler.assert_not_called()


def test_delete_every_done_card_asks_naming_up_to_the_cards_named(board, says):
    """S-policy-a: 'delete every done card' → one bulk cancel over task_ids."""
    says("delete every done card")
    refs = [f"#{task.workspace_seq:04d}" for task in board.done]

    reply = _call(board, {"task_ids": refs, "status": "cancelled", "note": "Clearing out finished work."},
                  _owners_chat())

    assert _asked(reply), reply
    named = [task.title for task in board.done if task.title in reply["message"]]
    assert named == [task.title for task in board.done[:MAX_CARDS_NAMED]]
    assert f"and {BULK_CARDS - MAX_CARDS_NAMED} more" in reply["message"]   # the card says it holds more
    board.handler.assert_not_called()


# ── The rule still holds where the card does not show the words ─────────────────────

def test_a_note_longer_than_the_card_shows_is_still_judged_by_the_rule(board, says):
    """The card cuts a long value (card_digest): the click signs only what the owner read."""
    from modules.tools.formatting.card_digest import DIGEST_CHARS

    long_note = ("Approved. The owner read the draft and is happy with the tone, the length and the price list; "
                 "it can go to Larder & Loaf today, with the delivery day kept as Tuesday.")
    assert len(long_note) > DIGEST_CHARS
    says(f"Approve {board.number} — looks good", board.card)

    reply = _call(board, {"task_id": board.number, "status": "done", "note": long_note}, _owners_chat())

    assert reply["success"] is False and SIGNATURE_REFUSAL in reply["error"]
    assert "requires_confirmation" not in reply
    board.handler.assert_not_called()


def test_the_card_digest_cuts_as_it_did():
    from modules.tools.formatting.card_digest import DIGEST_CHARS, fits_on_the_card, shown_on_the_card
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    whole, long = "x" * DIGEST_CHARS, "y" * (DIGEST_CHARS + 1)
    assert fits_on_the_card(whole) and not fits_on_the_card(long)
    assert shown_on_the_card(whole) == whole and shown_on_the_card(long) == "y" * 117 + "…"
    assert ToolResultFormatter._tool_params_digest({"note": long, "_agent_id": 3}) == {"note": "y" * 117 + "…"}


# ── The rule still holds where no card is raised ────────────────────────────────────

def test_a_paraphrased_note_on_a_move_that_does_not_close_the_card_is_still_refused(board, says):
    says(f"Move {board.number} to in progress with this note: lead with the Guji's price.", board.card)

    reply = _call(board, {"task_id": board.number, "status": "in_progress",
                          "note": "Please restructure the opening paragraph."}, _owners_chat())

    assert reply["success"] is False and SIGNATURE_REFUSAL in reply["error"]
    assert "requires_confirmation" not in reply
    board.handler.assert_not_called()


def test_the_owners_own_note_on_that_move_goes_through(board, says):
    says(f"Move {board.number} to in progress with this note: lead with the Guji's price.", board.card)

    reply = _call(board, {"task_id": board.number, "status": "in_progress",
                          "note": "lead with the Guji's price."}, _owners_chat())

    assert reply["success"] is True
    board.handler.assert_called_once()


def test_a_chat_no_person_drives_raises_no_card_so_the_rule_still_judges_it(board, says):
    """No driving user (no click to ask for): the signature rule is the only check, as before."""
    says(f"Approve {board.number} — looks good", board.card)

    reply = _call(board, {"task_id": board.number, "status": "done", "note": AUTOS_NOTE},
                  {"conversation_id": str(uuid4())})

    assert reply["success"] is False and SIGNATURE_REFUSAL in reply["error"]
    board.handler.assert_not_called()


# ── The rule set, and the other checks keep their order ─────────────────────────────

def test_only_the_signature_rule_is_skipped_for_an_owner_only_call():
    from modules.tools.discovery import follows_the_owner as guard

    driven = _owners_chat()
    assert guard._rules_for(MOVE, {"status": "done"}, driven) == guard.CARD_RULES
    assert guard._not_their_words not in guard.CARD_RULES
    assert set(guard.RULES) - set(guard.CARD_RULES) == {guard._not_their_words}
    assert guard._rules_for(MOVE, {"status": "in_progress"}, driven) == guard.RULES
    assert guard._rules_for(MOVE, {"status": "done"}, {"conversation_id": str(uuid4())}) == guard.RULES


def test_a_wrong_target_is_still_refused_before_any_card(board, says):
    """The other rules still run: a card's number sent to a mission's call is refused, not asked about."""
    says(f"Approve {board.number} — looks good", board.card)

    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        reply = asyncio.run(board.executor.execute("platform_approve_mission",
                                                   {"mission_id": board.number, "_agent_id": board.auto},
                                                   _owners_chat()))

    assert reply["success"] is False and "not a mission" in reply["error"]
    assert "requires_confirmation" not in reply


def test_a_rate_limited_approval_is_refused_not_asked_about(board, says):
    """The rate limit (and the hierarchy check and backstop) still come before the ask."""
    from fastapi import HTTPException

    from core.models.approval_grants import ApprovalGrant

    says(f"Approve {board.number} — looks good", board.card)
    limited = AsyncMock(side_effect=HTTPException(status_code=429, detail="slow down"))
    with patch("core.security.rate_limiter.check_rate_limit", new=limited):
        reply = asyncio.run(board.executor.execute(
            MOVE, {"task_id": board.number, "status": "done", "note": AUTOS_NOTE, "_agent_id": board.auto},
            _owners_chat()))

    assert reply.get("rate_limited") is True and "requires_confirmation" not in reply
    assert board.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == board.ws).count() == 0
    board.handler.assert_not_called()
