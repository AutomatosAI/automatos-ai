"""P256-FIX-N17-3 (F439, night 16): a card met again is named with its time, never "Card raised".

Night 16 row 29: "I've raised a fresh run card for playbook #81 just now", over #2607, raised the
night before; rows 28/162/248/342: "click … on the board". Each reading of a reused card said
"Card raised" with no age and no place.

Now a chat turn that meets the card an earlier turn raised (``already_waiting``: its id and when)
is told "Not run: this exact call already waits on its approval card, raised at …"; the receipt
says "still waiting on its card"; a claim that a card was raised this turn is unbacked and cut in
place, while "the card I raised at 19:53 on 14 Oct" and "still waiting on the card" stand. Every
card's tool result says where it is: under Approvals, beside the reply, not a ticket on the board.
"""
from __future__ import annotations

import string
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

from consumers.chatbot.claim_removed import without_unbacked_claims
from consumers.chatbot.receipts import receipt, unbacked_sentences
from modules.tools.discovery.my_actions_reads import at_said, iso_utc
from modules.tools.execution.card_raised import (
    ACCOUNT_LINE, ACT, FOR_THE_MODEL, RECEIPT_EFFECT, TOOL_END, account_line, for_the_model, tool_end_summary,
)
from modules.tools.execution.stale_cards import ALREADY_WAITING

ACTION = "platform_execute_playbook"
RUN_NOW = {"playbook_id": 81}
RUN_ACT = "run a playbook 'Monday green-coffee reorder' (playbook #81)"
ASK = {"success": False, "requires_confirmation": True, "owner_only": True, ACT: RUN_ACT,
       "message": f"Waiting for the owner's click: {RUN_ACT}"}
RAISED = "2026-10-14T04:12:35+00:00"
FRESH = {**ASK, "grant_id": 2607}
MET_AGAIN = {**FRESH, ALREADY_WAITING: {"grant_id": 2607, "since": RAISED}}
NIGHT_16 = "I've raised a fresh run card for playbook #81 just now."


@pytest.fixture
def desk(db_session, seed_workspace, monkeypatch):
    import modules.tools.execution.tool_grants as tool_grants

    monkeypatch.setattr(tool_grants, "_notify_approval_pending", lambda grant, ws: None)
    return NS(db=db_session, ws=UUID(seed_workspace()))


def _chat(conversation, turn):
    return {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": conversation, "turn_id": turn}


def _ask(desk, ctx):
    from modules.tools.execution.tool_grants import attach_ask_grant

    return attach_ask_grant(desk.db, desk.ws, action=ACTION, params=RUN_NOW, ask=dict(ASK), permission_level="write",
                            description="Run a playbook", caller_context=ctx)


def _row(desk, grant_id):
    from core.models.approval_grants import ApprovalGrant

    desk.db.expire_all()
    return desk.db.get(ApprovalGrant, grant_id)


# ── The same chat, a later turn: the card it raised, named with its time ─────────────

def test_a_later_turn_meets_its_card_named_with_when_it_was_raised(desk):
    chat = str(uuid4())
    first = _ask(desk, _chat(chat, "t-1"))["grant_id"]
    row = _row(desk, first)
    row.requested_at = datetime.now(timezone.utc) - timedelta(hours=16)
    desk.db.flush()
    since = _row(desk, first).requested_at

    again = _ask(desk, _chat(chat, "t-2"))

    assert again["grant_id"] == first
    assert again[ALREADY_WAITING] == {"grant_id": first, "since": iso_utc(since)}
    told = for_the_model(again, ACTION)
    assert told.startswith("Not run: this exact call already waits on its approval card")
    assert "No new card was raised" in told and at_said(since) in told and "do not retry the call" in told
    waiting = receipt(ACTION, RUN_NOW, again)
    assert waiting["status"] == "waiting" and waiting[ALREADY_WAITING] is True
    assert waiting["effect"].startswith("still waiting on its card (raised at ")
    assert tool_end_summary(again).startswith("Still waiting on its card, raised at ")
    assert account_line(again, ACTION).startswith("- Still waiting on its card, raised at ")


def test_the_same_turn_twice_is_one_card_and_no_already_waiting(desk):
    from core.models.approval_grants import SUBJECT_TOOL_CALL, ApprovalGrant

    ctx = _chat(str(uuid4()), "t-1")
    first, second = _ask(desk, ctx), _ask(desk, ctx)

    assert first["grant_id"] == second["grant_id"]
    assert ALREADY_WAITING not in first and ALREADY_WAITING not in second
    assert desk.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == desk.ws,
                                               ApprovalGrant.subject_type == SUBJECT_TOOL_CALL).count() == 1
    assert for_the_model(second).startswith("Card raised: ")


# ── Honesty: a fresh card is claimed only when this turn raised one ───────────────────

def test_a_fresh_card_claimed_over_a_card_met_again_is_unbacked_and_cut_in_place():
    found = unbacked_sentences(NIGHT_16, [(ACTION, RUN_NOW, MET_AGAIN)])

    assert [verb for verb, _sentence in found] == ["raised"] and "fresh run card" in found[0][1]
    removed = without_unbacked_claims(f"{NIGHT_16} Click Approve when you're ready.", [(ACTION, RUN_NOW, MET_AGAIN)])
    assert removed is not None and "fresh run card" not in removed.text
    assert removed.text.startswith("Just to be clear: ")


def test_the_same_claim_over_a_card_this_turn_raised_is_backed():
    assert unbacked_sentences(NIGHT_16, [(ACTION, RUN_NOW, FRESH)]) == []
    assert without_unbacked_claims(NIGHT_16, [(ACTION, RUN_NOW, FRESH)]) is None


@pytest.mark.parametrize("said", ["It's still the card I raised at 19:53 on 14 Oct.",
                                  "The run is still waiting on the card."])
def test_naming_the_card_met_again_is_no_claim(said):
    assert unbacked_sentences(said, [(ACTION, RUN_NOW, MET_AGAIN)]) == []


# ── Where the card lives, and the templates keep their one placeholder ───────────────

def test_every_card_says_it_is_under_approvals_not_a_ticket_and_not_to_retry():
    told = for_the_model(FRESH, ACTION)

    assert told.startswith(f"Card raised: {RUN_ACT}. Nothing changes until the owner clicks; say so in one line")
    assert "under Approvals" in told and "not a ticket on the board" in told and told.endswith(
        "do not retry the call.")
    assert tool_end_summary(FRESH) == TOOL_END.format(act=RUN_ACT)


@pytest.mark.parametrize("template", [FOR_THE_MODEL, TOOL_END, ACCOUNT_LINE, RECEIPT_EFFECT])
def test_a_raised_cards_wording_has_one_placeholder_and_no_number(template):
    fields = [field for _text, field, _spec, _conv in string.Formatter().parse(template) if field is not None]
    assert fields == ["act"] and not any(ch.isdigit() for ch in template)
