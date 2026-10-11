"""P256-FIX-RVW-10 (FX-013 vs D1): a card that closes is never re-briefed on the click.

``owner_only.is_owner_only`` reads platform_update_task with a closing status (done,
cancelled) as owner-only, so the card asked to "approve (move to Done)"; on the click,
``ticket_edit_moves.edited_then_moved`` saw a new brief on a worked card and FX-013's
``rebriefed_without_its_status`` dropped the status and sent the card back to its agent.
The owner approved closing the card; the card reopened.

Now a closing status beside a new brief on a worked card is refused before any card is
raised (``rebrief_that_closes``, read by ``owner_only.platform_ask``), naming both
choices, as before FX-013; the handler refuses it on every other lane too. FX-013's drop
applies only to a status that does not close the card.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from modules.tools.discovery import owner_only
from tests import test_f241_e_a_new_card_never_copies_one_and_notes_are_the_owners as f241

shop, in_review, notes_in_this_session = f241.shop, f241.in_review, f241.notes_in_this_session

EDIT = "platform_update_task"


class _Executor:
    """PlatformActionExecutor._run_cleared's shape: the gates have cleared, the handler runs
    in the test's session and workspace."""

    def __init__(self, card):
        self.card = card

    @owner_only.asks_the_owner_first
    async def _run_cleared(self, action_name, params, caller_context, cleared, handler):
        return await handler(self.card.db, self.card.ws, params)


def _owners_chat():
    return {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}


def _run(card, params, caller_context):
    """platform_update_task as the executor runs it after its gates, with the real handler
    (the executor injects the driving user's ``_user_id`` on a person's turn)."""
    params = {**params, "_user_id": "user_owner"} if caller_context else params
    return asyncio.run(_Executor(card)._run_cleared(EDIT, params, caller_context, None,
                                                    card.handlers.update_board_task))


def _grants(card):
    from core.models.approval_grants import ApprovalGrant

    return card.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == card.ws).count()


@pytest.mark.parametrize("status", ["done", "cancelled", "approved"])
def test_a_new_brief_that_closes_the_card_is_refused_before_any_card(in_review, notes_in_this_session, status):
    from modules.tools.discovery.ticket_edit_moves import CLOSE_OR_REBRIEF

    before = (in_review.card.status, in_review.card.description)

    out = _run(in_review, {"task_id": in_review.number, "description": f241.NEW_BRIEF, "status": status},
               _owners_chat())

    assert out == {"success": False, "error": CLOSE_OR_REBRIEF}
    assert "re-brief it" in CLOSE_OR_REBRIEF and "close it" in CLOSE_OR_REBRIEF   # both choices named
    assert _grants(in_review) == 0                                                  # no card, no grant
    in_review.db.refresh(in_review.card)
    assert (in_review.card.status, in_review.card.description) == before            # nothing changed


def test_a_new_brief_that_does_not_close_the_card_re_briefs_it_with_no_card(in_review, notes_in_this_session):
    """FX-013 stands for a status that does not close the card (A440's call)."""
    out = _run(in_review, {"task_id": in_review.number, "description": f241.NEW_BRIEF, "status": "in_progress"},
               _owners_chat())

    assert out["success"] is True and out["status_ignored"] is True
    assert "requires_confirmation" not in out and _grants(in_review) == 0
    in_review.db.refresh(in_review.card)
    assert (in_review.card.status, in_review.card.description) == ("assigned", f241.NEW_BRIEF)


def test_an_agents_run_is_refused_the_same_way(in_review, notes_in_this_session):
    """No driving user: no card is asked, and the handler refuses the close beside a re-brief."""
    from modules.tools.discovery.ticket_edit_moves import CLOSE_OR_REBRIEF

    out = _run(in_review, {"task_id": in_review.number, "description": f241.NEW_BRIEF, "status": "done"}, None)

    assert out == {"success": False, "error": CLOSE_OR_REBRIEF}
    in_review.db.refresh(in_review.card)
    assert in_review.card.status == "review"


def test_a_close_with_no_new_brief_still_asks_the_owner(in_review, monkeypatch):
    """The refusal is the re-brief's alone: approving the card from chat still raises the card."""
    asked = []
    monkeypatch.setattr(owner_only, "_ask", lambda db, ws, action, params, ctx, **kw: asked.append(kw["act"]) or {
        "success": False, "requires_confirmation": True, "owner_only": True})

    out = _run(in_review, {"task_id": in_review.number, "status": "done", "note": "Looks good."}, _owners_chat())

    assert out["requires_confirmation"] is True and asked and asked[0].startswith("approve (move to Done)")


def test_the_refusal_reads_only_a_rebrief_that_closes_a_worked_card(shop):
    """An unworked card's new brief is a plain edit, then the move; other actions are never read."""
    from modules.tools.discovery.ticket_edit_moves import rebrief_that_closes

    closing = {"task_id": shop.number, "description": "New", "status": "done"}
    assert rebrief_that_closes(shop.db, shop.ws, EDIT, closing) is None                    # card in the Inbox
    assert rebrief_that_closes(shop.db, shop.ws, "platform_update_task_status", closing) is None
    assert rebrief_that_closes(shop.db, shop.ws, EDIT, {**closing, "description": " "}) is None
