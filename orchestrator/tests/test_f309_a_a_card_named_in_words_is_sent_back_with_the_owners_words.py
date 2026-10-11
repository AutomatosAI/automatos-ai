"""F309 (night 9): Auto acts on a card the owner names in words, and a send-back carries their words.

Night 9 (build 13), each in a fresh chat:
- "Card 1869 needs to go back. Correction: you forgot the roast loss …" became
  platform_ask_human about card 1869: the card went to Blocked with ask #1460, and its
  correction was nowhere on it (times_sent_back 0);
- "Please approve card 1879 with this note: …" became platform_approve_mission
  {mission_id: 1879}: "I couldn't find a mission with ID 1879";
- "Send card 27.2 back …" became platform_get_social_post {post_id: "27.2"};
- "It's the board card #0027.2 … Send it back to the writer with that correction." The
  paraphrased note was refused, quoting only that latest message, which held no
  correction: four messages to send one card back.
"""
from __future__ import annotations

from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

CHAT = {"conversation_id": str(uuid4())}
WS = uuid4()
ROAST_LOSS = ("you forgot the roast loss - every 250 g bag needs about 294 g of green - and 55 kg is the "
              "1 September figure. Use today's stock from the shop system and do the sum again.")
SEND_1869_BACK = f"Card 1869 needs to go back. Correction: {ROAST_LOSS}"
BANNED = ("'delightful' and 'exquisite' are banned in our brand voice, no exclamation marks, 85 to 95 words, "
          "plain and warm, and say the coffee was roasted Tuesday.")
SEND_27_2_BACK = f"Send card 27.2 back - the Harbour Log intro. Correction: {BANNED}"
THAT_CORRECTION = ("It's not a social post. It's the board card #0027.2, step 2 of my October club box mission. "
                   "Send it back to the writer with that correction.")
FINE = "Fine — Auto got these figures in chat, I'll use those."


def _card(ref, title, status="review", source_type="user", task_id=None, by_id=False):
    from modules.tools.discovery.owner_turn import NamedCard

    seq, _, step = ref.lstrip("#").partition(".")
    task = NS(id=task_id or 900 + int(seq), title=title, status=status, source_type=source_type, description="")
    return NamedCard(ref=ref, seq=int(seq), step=int(step) if step else None, task=task, by_id=by_id)


@pytest.fixture
def turn(monkeypatch):
    """The owner's words this turn (and the cards they name), as owner_turn would read them.
    PRD-256 US-004: Auto's last reply is no longer patched: the guard read it only to judge a
    "yes" as an approval, which is the owner's click now."""
    import modules.tools.discovery.follows_the_owner as guard
    from modules.tools.discovery.owner_turn import OwnerTurn

    said = {}

    def _set(latest, *cards, earlier="", recent=()):
        said.update(turn=OwnerTurn(latest=latest, earlier=earlier, cards=tuple(cards)), recent=recent)

    monkeypatch.setattr(guard, "owner_turn", lambda db, ws, ctx: said.get("turn") if ctx else None)
    monkeypatch.setattr(guard, "_the_cards_words", lambda db, ws, params: ())
    monkeypatch.setattr(guard, "owners_recent_words", lambda db, ws, turn: said.get("recent", ()))
    return _set


def _refused(action, **params):
    from modules.tools.discovery.follows_the_owner import refusal_for

    return refusal_for(None, WS, action, params, CHAT)


# ── 1. A send-back with a correction is the board's Reject, never a question ───────

def test_a_send_back_with_a_correction_is_never_a_question_to_the_owner(turn):
    turn(SEND_1869_BACK, _card("#0022", "Enough Kirinyaga for 120 Christmas boxes?", task_id=1869, by_id=True))

    asked = _refused("platform_ask_human", subject_type="board_task", subject_id="1869",
                     question="Can you provide the current stock of green coffee from the shop system?")
    parked = _refused("platform_update_task_status", task_id=1869, status="blocked", blocked_reason="needs stock")

    for refusal in (asked, parked):
        assert "goes back to its agent" in refusal and "Nothing was done" in refusal
        assert 'platform_update_task_status {task_id: "#0022", status: "assigned"' in refusal
        assert "you forgot the roast loss - every 250 g bag" in refusal          # the call carries their words
    assert _refused("platform_update_task_status", task_id="#0022", status="assigned", note=ROAST_LOSS) is None


def test_go_back_with_a_correction_is_a_send_back():
    from modules.tools.discovery.owner_turn import SEND_BACK, SENDS_IT_BACK

    for said in (SEND_1869_BACK, "It goes back to the Watchdog.", "Correction: 41 kg, not 55."):
        assert SEND_BACK.search(said) and SENDS_IT_BACK.search(said)
    assert not SENDS_IT_BACK.search("Please give card 1859 to the Shopify Business Analyst instead.")


# ── 3. A card named in words is the card, never a mission or a social post ──────────

def test_approving_card_1879_is_never_a_mission(turn):
    turn(f"Please approve card 1879 with this note: {FINE}",
         _card("#0029", "Top three cafés by kg, June–August (second try)", task_id=1879, by_id=True))

    refusal = _refused("platform_approve_mission", mission_id=1879)

    assert "#0029 (id 1879, as the owner named it)" in refusal and "not a mission" in refusal
    assert 'platform_update_task_status {task_id: "#0029", status: "done"' in refusal and FINE in refusal
    assert _refused("platform_update_task_status", task_id=1879, status="done", note=FINE) is None


def test_a_missions_own_card_still_takes_the_mission_call(turn):
    turn("Approve card 1874, please.", _card("#0027", "Mission: October club box", "awaiting_approval",
                                             source_type="orchestration", task_id=1874, by_id=True))
    assert _refused("platform_approve_mission", mission_id="#0027") is None


def test_card_27_2_is_never_a_social_post(turn):
    turn(SEND_27_2_BACK, _card("#0027.2", "Draft Harbour Log intro for October box", source_type="orchestration_task"))

    refusal = _refused("platform_get_social_post", post_id="27.2")

    assert "#0027.2" in refusal and "not a social post" in refusal
    assert 'status: "assigned"' in refusal and BANNED in refusal


def test_giving_a_card_to_another_agent_instead_is_a_reassign_not_a_send_back(turn):
    from modules.tools.discovery.follows_the_owner import right_call
    from modules.tools.discovery.owner_turn import OwnerTurn

    card = _card("#0012", "Late club boxes in September", task_id=1859, by_id=True)
    said = "Please give card 1859 to the Shopify Business Analyst instead - the Analyst couldn't find the data."

    assert "platform_assign_task" in right_call(OwnerTurn(latest=said, earlier="", cards=(card,)), card)


# ── 4. A refused paraphrase says the owner's words, and those words go through ──────

def test_a_paraphrase_is_refused_with_the_words_the_owner_gave_a_message_before(turn):
    """#0027.2: the correction was in the owner's message before 'with that correction'."""
    turn(THAT_CORRECTION, _card("#0027.2", "Draft Harbour Log intro for October box", source_type="orchestration_task"),
         earlier=SEND_27_2_BACK, recent=(THAT_CORRECTION, SEND_27_2_BACK))
    paraphrase = ("Please revise the Harbour Log intro. Remove 'delightful' and 'exquisite', avoid exclamation marks, "
                  "keep it between 85 and 95 words, maintain a plain and warm tone, and mention the coffee was "
                  "roasted on Tuesday.")

    refusal = _refused("platform_update_task_status", task_id="0027.2", status="assigned", note=paraphrase)

    assert "word for word" in refusal and f'"{BANNED}"' in refusal
    assert _refused("platform_update_task_status", task_id="0027.2", status="assigned", note=BANNED) is None


def test_an_approval_through_the_edit_tool_waits_for_the_owners_click_not_their_words(turn):
    """platform_update_task with a status moves the card now. PRD-256 US-004 changed this test:
    the guard no longer judges an approval by the owner's words ("hasn't said to approve"); the
    move to Done is owner-only and waits for their click on the approval card (owner_only)."""
    from modules.tools.discovery.owner_only import is_owner_only

    turn("Let's talk about card 1866.", _card("#0019", "Margin on a 250 g bag of Kirinyaga AA", task_id=1866,
                                               by_id=True))
    assert _refused("platform_update_task", task_id=1866, status="done") is None
    assert is_owner_only("platform_update_task", {"task_id": 1866, "status": "done"})


# ── The turn reads cards named in words from the board ─────────────────────────────

@pytest.fixture
def board(db_session, seed_workspace):
    from core.models.core import BoardTask
    from core.models.ticket_numbers import STEP_SOURCE

    ws = UUID(seed_workspace())

    def card(title, **fields):
        made = BoardTask(workspace_id=ws, title=title, status=fields.pop("status", "review"), **fields)
        db_session.add(made)
        db_session.flush()
        return made

    plain = card("Enough Kirinyaga for 120 Christmas boxes?", source_type="user")
    mission = card("Mission: October club box", source_type="orchestration", status="in_progress")
    card("Verify green coffee", source_type=STEP_SOURCE, parent_task_id=mission.id)
    step = card("Draft Harbour Log intro", source_type=STEP_SOURCE, parent_task_id=mission.id)
    return NS(db=db_session, ws=ws, plain=plain, mission=mission, step=step)


def test_cards_named_in_words_are_found_by_id_or_by_number(board):
    from modules.tools.discovery.owner_turn import cards_named

    plain_no, mission_no = f"#{board.plain.workspace_seq:04d}", f"#{board.mission.workspace_seq:04d}"
    words = (f"Card {board.plain.id} needs to go back, and send card {board.mission.workspace_seq}.2 back too. "
             f"Ticket {mission_no[1:]} as well.")

    cards = cards_named(board.db, board.ws, words)

    assert [(c.ref, c.task.id, c.by_id) for c in cards] == [
        (plain_no, board.plain.id, True), (f"{mission_no}.2", board.step.id, False), (mission_no, board.mission.id, False)]


def test_a_number_with_no_card_word_is_not_a_card(board):
    from modules.tools.discovery.owner_turn import cards_named

    said = f"120 boxes of {board.plain.workspace_seq} kg, step 2 of the plan, {board.plain.id} grams."
    assert cards_named(board.db, board.ws, said) == ()


def test_the_turn_tells_auto_the_send_back_with_the_owners_words(board):
    from modules.tools.discovery.card_note import cards_note

    note = cards_note(board.db, board.ws, f"Card {board.plain.id} needs to go back. Correction: {ROAST_LOSS}")

    number = f"#{board.plain.workspace_seq:04d}"
    assert f"{number} (id {board.plain.id}, as the owner named it) ('Enough Kirinyaga" in note
    assert f'platform_update_task_status {{task_id: "{number}", status: "assigned"' in note and ROAST_LOSS in note
