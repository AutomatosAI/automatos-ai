"""F241, F280, F281, F289 (night 8): Auto acts only on the card the owner names, the way they asked.

Every call below is one Auto made on night 8, with the owner's words that came before it.
"""
from __future__ import annotations

from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

CHAT = {"conversation_id": str(uuid4())}
WS = uuid4()


def _card(ref, title, status="review", source_type="user", **fields):
    from modules.tools.discovery.owner_turn import NamedCard

    seq, _, step = ref.lstrip("#").partition(".")
    task = NS(id=900 + int(seq), title=title, status=status, source_type=source_type, description="", **fields)
    return NamedCard(ref=ref, seq=int(seq), step=int(step) if step else None, task=task)


@pytest.fixture
def turn(monkeypatch):
    """The owner's words this turn (and the cards they name), as owner_turn would read them."""
    import modules.tools.discovery.follows_the_owner as guard
    from modules.tools.discovery.owner_turn import OwnerTurn

    said = {}

    def _set(latest, *cards, earlier=""):
        said["turn"] = OwnerTurn(latest=latest, earlier=earlier, cards=tuple(cards))

    monkeypatch.setattr(guard, "owner_turn", lambda db, ws, ctx: said.get("turn") if ctx else None)
    monkeypatch.setattr(guard, "_the_cards_words", lambda db, ws, params: ())
    return _set


def _refused(action, **params):
    from modules.tools.discovery.follows_the_owner import refusal_for

    return refusal_for(None, WS, action, params, CHAT)


# ── A card's number never goes to something that isn't a card (F241, F281) ──────────

def test_approve_by_number_is_never_a_social_post(turn):
    turn("Approve #0201 with this note: Good, that's the tone I want with Hannah.",
         _card("#0201", "Reply to Hannah at Mill Lane Kitchen"))
    refusal = _refused("platform_submit_social_post", post_id="0201", note="Good, that's the tone I want with Hannah.")
    assert "#0201" in refusal and "not a social post" in refusal
    assert 'platform_update_task_status {task_id: "#0201", status: "done"' in refusal


def test_cancel_by_number_is_never_a_scheduled_task(turn):
    turn("Cancel #0209, I've already written to Jess myself.", _card("#0209", "Thank-you note to Bramble & Co", "inbox"))
    refusal = _refused("platform_cancel_scheduled_task", task_id=209)
    assert "not a scheduled task" in refusal and 'status: "cancelled"' in refusal


def test_give_by_number_never_gives_a_tool_called_after_the_card(turn):
    turn("Give #0382 to the Content Creator, please.", _card("#0382", "Front page line: Guji", "inbox"))
    refusal = _refused("platform_assign_tool_to_agent", agent_name="Content Creator", app_name="#0382")
    assert "not a tool" in refusal and "platform_assign_task" in refusal


def test_what_is_a_card_doing_is_never_an_agent(turn):
    turn("What is #0226 doing?", _card("#0226", "Counter card: Kenya Kiambu AA", "inbox"))
    assert "platform_get_task" in _refused("platform_get_agent", agent_id=226)


def test_an_update_by_number_never_rewrites_an_agent(turn):
    """#0296: agent #323's own description became a product blurb."""
    turn("Update #0296: Shop description for the Peru Cajamarca, about 100 words.",
         _card("#0296", "Shop description: Peru Cajamarca"))
    refusal = _refused("platform_update_agent", agent_id=323, description="Milk chocolate, red apple and honey…")
    assert "an agent isn't what they asked about" in refusal and "platform_update_task" in refusal


def test_a_social_post_the_owner_asks_for_alongside_a_card_is_theirs(turn):
    turn("Approve #0201, and draft an Instagram post about the Guji.", _card("#0201", "Reply to Hannah"))
    assert _refused("platform_create_social_post", title="Guji", copy={"default": "Guji"}) is None


# ── No copies (F289) ────────────────────────────────────────────────────────────────

def test_give_by_number_never_makes_a_new_card(turn):
    """#0287 'Club Pause' was made for "Give #0285 to …"."""
    turn("Give #0285 to the Shopify Operations Manager: it's about a club pause.",
         _card("#0285", "Reply to Tom Becker about a pause"))
    refusal = _refused("platform_create_task", title="Club Pause", assigned_agent_name="Shopify Operations Manager")
    assert "already on the owner's board" in refusal and "platform_assign_task" in refusal


def test_a_new_card_the_owner_asks_for_is_made(turn):
    turn("Make a new card like #0285 for the Quayside Pantry.", _card("#0285", "Reply to Tom Becker about a pause"))
    assert _refused("platform_create_task", title="Reply to Tess at Quayside Pantry") is None


def test_changing_a_mission_never_makes_a_second_one(turn, monkeypatch):
    """#0268 was made for "set #0267 so every step stops"."""
    turn("Set #0267 so every step stops and waits for me.", _card("#0267", "Mission: price rise", "inbox",
                                                                   source_type="orchestration"))
    assert "platform_update_mission_plan" in _refused("platform_create_mission", goal="Price rise letter")


def test_start_it_again_keeps_the_missions_own_goal(turn, monkeypatch):
    """#0437: "cancel #0433 and start it again" became a competitor-research mission."""
    import modules.tools.discovery.follows_the_owner as guard

    goal = "Write a reusable welcome pack for new wholesale cafés, with gaps for each café's details."
    monkeypatch.setattr(guard, "_mission_goal", lambda db, card: goal)
    turn("So yes: cancel #0433 and start it again.", _card("#0433", "Mission: welcome pack", "in_progress",
                                                              source_type="orchestration"))
    refusal = _refused("platform_create_mission", goal="Research our top 5 competitors, analyze their pricing.")
    assert goal in refusal
    assert _refused("platform_create_mission", goal=goal) is None


# ── Decide only what the owner decided (F280) ───────────────────────────────────────

def test_naming_a_card_is_not_approving_it(turn):
    """#0329 was approved 'by you' when the owner only said which card they meant."""
    turn("#0329 is a ticket on my board, in Review: the reply to Raj Patel about skipping November.",
         _card("#0329", "Reply to Raj Patel"), earlier="Let's talk about #0329.")
    assert "hasn't said to approve #0329" in _refused("platform_update_task_status", task_id=329, status="done")


def test_cancel_is_never_an_approval(turn):
    """#0422: 'Cancel #0422' became Done with 'User chalked it up themselves.' in the owner's name."""
    turn("Cancel #0422, please: I've chalked it up myself.", _card("#0422", "Chalkboard line"))
    refusal = _refused("platform_update_task_status", task_id=422, status="done", note="User chalked it up themselves.")
    assert "said cancel, not approve" in refusal and 'status: "cancelled"' in refusal


def test_before_i_approve_is_not_an_approval(turn):
    turn("Before I approve #0410: is it set to stop after each step?", _card("#0410", "Mission: Easter box",
                                                                            source_type="orchestration"))
    assert "yet" in _refused("platform_approve_mission", mission_id="#0410")


def test_the_owners_approval_with_their_note_goes_through(turn):
    turn("Approve #0201 with this note: Good, that's the tone I want with Hannah. I'll send it myself.",
         _card("#0201", "Reply to Hannah"))
    assert _refused("platform_update_task_status", task_id=201, status="done",
                    note="Good, that's the tone I want with Hannah. I'll send it myself.") is None


def test_a_yes_to_autos_question_is_the_owners_go_ahead(turn):
    turn("Yes, go ahead.", earlier="Start a mission to get my wholesale price list ready for December.")
    assert _refused("platform_approve_mission", mission_id=str(uuid4())) is None


# ── Words signed as the owner's are theirs (F280, F279) ─────────────────────────────

def test_a_note_signed_as_the_owner_is_in_their_words(turn):
    turn("Send #0347 back: take out the line It starts with To:. Nothing at all before To:.",
         _card("#0347", "Invoice reminder to Fernhill Bakery"))
    refusal = _refused("platform_update_task_status", task_id=347, status="assigned",
                       note="Remove the line that starts with 'To:'. Ensure clean formatting.")
    assert "their own words" in refusal and "take out the line It starts with To:" in refusal
    assert _refused("platform_update_task_status", task_id=347, status="assigned",
                    note="take out the line It starts with To:. Nothing at all before To:.") is None


def test_a_new_brief_is_the_owners_words_not_autos(turn):
    """#0256 and #0451: Auto's own email went in as the brief."""
    turn("Update #0451: Reply to club member Ana Lucas. She asked to pause her club box for April.",
         _card("#0451", "Reply to Ana Lucas"))
    autos = "To: ana@lucas.example\nHi Ana, your club box has been successfully paused. We hope you enjoy May!"
    assert "A new brief is the owner's words" in _refused("platform_update_task", task_id=451, description=autos)
    assert _refused("platform_update_task", task_id=451,
                    description="Reply to club member Ana Lucas. She asked to pause her club box for April.") is None


def test_a_brief_the_owner_took_from_auto_goes_on(turn, monkeypatch):
    """Discuss (PRD-252 R2): 'Yes, that's it, with about 90 words added. Put that brief on #0204.'"""
    import modules.tools.discovery.follows_the_owner as guard

    proposed = "Draft the club newsletter's opening: Gerard chatting to members about the October coffees."
    monkeypatch.setattr(guard, "autos_proposal", lambda db, ws, turn: proposed)
    turn("Yes, that's it, with 'about 90 words' added. Put that brief on #0204 and send it back.",
         _card("#0204", "Newsletter opening"))
    assert _refused("platform_update_task", task_id=204, description=proposed + " About 90 words.") is None


def test_in_progress_is_not_a_send_back(turn):
    """#0425: moved to In progress, the card ran its old brief again without the owner's words."""
    turn("Send #0425 back: the length is right this time, keep it. But no indulge or delightful.",
         _card("#0425", "Brazil Cerrado description"))
    refusal = _refused("platform_update_task_status", task_id=425, status="in_progress")
    assert 'status: "assigned", note: the owner\'s own words' in refusal


def test_nothing_is_checked_outside_the_owners_chat(turn):
    from modules.tools.discovery.follows_the_owner import refusal_for

    turn("Cancel #0422", _card("#0422", "Chalkboard line"))
    assert refusal_for(None, WS, "platform_update_task_status", {"task_id": 422, "status": "done"}, None) is None


# ── The turn: the cards the owner names, found on their board ───────────────────────

def test_the_turn_finds_the_cards_the_owner_named(db_session, seed_workspace, monkeypatch):
    import modules.tools.discovery.handlers_board_task_review as review
    from core.models.core import BoardTask
    from modules.tools.discovery.owner_turn import owner_turn

    ws = UUID(seed_workspace())
    card = BoardTask(workspace_id=ws, title="Reply to Hannah", status="review", source_type="user")
    db_session.add(card)
    db_session.flush()
    number = f"#{card.workspace_seq:04d}"
    monkeypatch.setattr(review, "owner_words", lambda db, w, chat: [f"Approve {number} and #9999 please", "Hi"])

    turn = owner_turn(db_session, ws, CHAT)

    assert (turn.latest, turn.earlier) == (f"Approve {number} and #9999 please", "Hi")
    assert [(c.ref, c.task.id if c.task else None) for c in turn.cards] == [(number, card.id), ("#9999", None)]
    assert owner_turn(db_session, ws, {}) is None                               # no chat, nothing to check


# ── A new card starts in the Inbox, never in Review (F289) ──────────────────────────

def test_a_new_card_is_never_filed_in_review():
    """#0251 and #0386 went straight into Review with no work done, and Needs you counted them."""
    import asyncio

    from modules.tools.discovery.new_card_checks import STARTS_IN_THE_INBOX, checks_the_new_card

    filed = []

    async def create(db, workspace_id, params):
        filed.append(params)
        return {"success": True, "task_id": 7}

    out = asyncio.run(checks_the_new_card(create)(None, WS, {"title": "Price card for the market stall",
                                                             "status": "review"}))

    assert "status" not in filed[0] and out["status_note"] == STARTS_IN_THE_INBOX
