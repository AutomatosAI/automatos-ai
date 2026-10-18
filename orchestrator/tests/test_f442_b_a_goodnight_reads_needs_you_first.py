"""F442 (nights 16-17): "Thanks for tonight, that's been useful." got "Nothing's pending on my side
that needs you tonight" with no call and no receipt, and two minutes later "201 items need you". A
sign-off asks nothing, so the board's Needs you was never read, and the all-clear check had no
count to hold the answer to.

Now a sign-off reads Needs you's count before the model answers: the note gives the number, the
read is on the activity trail and in the turn's receipts, and an answer that still says nothing is
pending gains the count. The sign-offs and answers below are shaped like the night's.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from consumers.chatbot import needs_you_turn
from consumers.chatbot.needs_you_turn import READ_TOOL, asks_what_needs_me, signs_off
from consumers.chatbot.streaming import StreamingHandler
from tests import test_f307_a_what_needs_me_reads_the_boards_needs_you as f307

board = f307.board
THANKS = "Thanks for tonight, that's been useful."
SIGN_OFFS = (THANKS, "Thanks, that's me done for tonight.", "Good night!", "Cheers, that's all for today.",
             "That's me done, signing off.", "Thank you, goodnight.")
NOT_SIGN_OFFS = ("Thanks, can you send the price list to Declan?", "List them all, every single card in review.",
                 "Thanks for the draft. Now send it to Declan and copy the café's manager on it.",
                 "Send card 27.2 back.", "Take #1601 off Scout and give it to OPS.")
NIGHT_16_ANSWERS = ("Nothing's pending on my side that needs you tonight.",
                    "Nothing’s waiting for you tonight. Sleep well!",
                    "You're welcome! There's no pending work for you right now, so enjoy your evening.")
ROW_369_ANSWER = "Nothing is waiting on changes right now."
GIVES_THE_COUNT = "Goodnight! 3 things are waiting for you on Needs you; they'll keep till morning."


def _additions(answer):
    from consumers.chatbot.service import StreamingChatService

    return lambda: StreamingChatService._answer_additions(None, NS(content=answer, finish_reason="stop"))


def _clear_line(board):
    from modules.tools.discovery.board_waiting import whats_waiting

    waiting = whats_waiting(board.db, board.ws)
    return "\n\n" + needs_you_turn.CLEAR_LINE.format(total=waiting["total"],
                                                     by_kind=needs_you_turn._by_kind(waiting["by_kind"]))


@pytest.mark.parametrize("said", SIGN_OFFS)
def test_each_sign_off_signs_off_and_asks_nothing(said):
    assert signs_off(said) is True and asks_what_needs_me(said) is False


@pytest.mark.parametrize("said", NOT_SIGN_OFFS)
def test_other_messages_do_not_sign_off(said):
    assert signs_off(said) is False


def test_a_sign_off_reads_needs_you_before_the_model_answers(board):
    from consumers.chatbot.receipts import build_receipts
    from services.needs_you import needs_you_counts

    turn = f307._turn(f307._chat(board), THANKS)

    total = needs_you_counts(board.db, board.ws)["total"]
    assert total == 3
    note = turn.messages[-1]
    assert note["role"] == "system" and f"Needs you holds {total} for them right now" in note["content"]
    assert "never say nothing is waiting or pending unless it is 0" in note["content"]
    assert turn.prefetched == [(READ_TOOL, {"automatic": True, "needs_you": True})]
    starts = [json.loads(f[2:])["data"] for f in turn.frames if '"tool-start"' in f]
    assert [s["toolName"] for s in starts] == [READ_TOOL] and turn.frames[-1] == "retrieval"
    assert [r["effect"] for r in build_receipts(None, turn.prefetched)] == ["read the board"]


@pytest.mark.parametrize("answer", NIGHT_16_ANSWERS)
def test_a_nothing_pending_answer_at_bedtime_gains_the_count(board, answer):
    turn = f307._turn(f307._chat(board), THANKS, then=_additions(answer))

    assert turn.after == [_clear_line(board)]
    assert turn.after[0].startswith("\n\nJust to be clear: your board's Needs you has 3 waiting for you right now")


def test_a_bedtime_answer_that_gives_the_count_is_left_alone(board):
    turn = f307._turn(f307._chat(board), THANKS, then=_additions(GIVES_THE_COUNT))

    assert turn.after == []


def test_a_widget_visitors_sign_off_reads_nothing(board):
    turn = f307._turn(f307._chat(board, widget=True), THANKS, then=_additions(NIGHT_16_ANSWERS[0]))

    assert turn.frames == ["retrieval"] and len(turn.messages) == 2 and turn.prefetched == []
    assert turn.after == []


def test_a_sign_off_whose_read_fails_says_nothing_about_what_waits(board, monkeypatch):
    from services import needs_you

    def broken(db, workspace_id, **kwargs):
        raise RuntimeError("database went away")

    monkeypatch.setattr(needs_you, "needs_you_counts", broken)
    turn = f307._turn(f307._chat(board), THANKS, then=_additions(NIGHT_16_ANSWERS[0]))

    assert turn.messages[-1] == {"role": "system", "content": needs_you_turn.SIGN_OFF_UNREAD}
    assert turn.prefetched == [] and turn.frames == ["retrieval"] and turn.after == []


@pytest.mark.parametrize("said, answer", [("Has Scout's price-list redo come back yet?", ROW_369_ANSWER),
                                          ("What do we charge a café for delivery on 10 kg?", NIGHT_16_ANSWERS[0])])
def test_a_turn_that_does_not_sign_off_reads_nothing_and_gains_nothing(board, said, answer):
    turn = f307._turn(f307._chat(board), said, then=_additions(answer))

    assert turn.frames == ["retrieval"] and len(turn.messages) == 2 and turn.prefetched == []
    assert turn.after == []


def test_an_empty_boards_nothing_waiting_is_left_alone(db_session, seed_workspace):
    empty = NS(db=db_session, workspace_id=UUID(seed_workspace()), widget_mode=False,
               streaming_handler=StreamingHandler())

    turn = f307._turn(empty, THANKS, then=_additions("Nothing's waiting on you. Goodnight!"))

    assert "Needs you holds 0 for them right now" in turn.messages[-1]["content"]
    assert turn.after == []


def test_an_asking_sign_off_keeps_the_list(board):
    turn = f307._turn(f307._chat(board), "Thanks. What needs me before I go?")

    assert "in all" in turn.messages[-1]["content"] and "signing off" not in turn.messages[-1]["content"]
    assert "Delivery charge on 10 kg" in turn.messages[-1]["content"]


def test_the_night_9_answer_is_still_all_clear():
    assert needs_you_turn._ALL_CLEAR.search(f307.NIGHT_9_ANSWER)
