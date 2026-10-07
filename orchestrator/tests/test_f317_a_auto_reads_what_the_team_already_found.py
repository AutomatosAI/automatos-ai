"""F317 (night 9b): Auto answered as if the team had found nothing. "The club box count is on the
board — what number did they give?" got "neither the Analyst nor the Shopify Business Analyst's
completed tasks directly address the total number of club boxes" while the Q5 cards said 63
(b8d81ba9); "check what the team has already found on the board" got a count of cards by
status; "Do I need to reorder any Kirinyaga?" came from the 1 September paper while the
Watchdog's card sat on the board (378c1e41). Now a question reads the cards in review or done
that already answer it, and their answers reach the model by number, as an agent's, not the
owner's facts.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from consumers.chatbot.streaming import get_streaming_handler
from consumers.chatbot.team_findings import (
    ASKED_RULE, NONE_FOUND, NOTE_RULE, READ_TOOL, reads_what_the_team_found,
)
from services.team_findings import team_findings

Q5 = "How many Harvest Club boxes go out on Monday 5 October?"
SIXTY_THREE = "Perfect! Now I have the count. 63 active Harvest Club subscribers, so 63 boxes go out on Monday."
KIRINYAGA = "Kirinyaga AA: 41 kg in green stock today. At 26 kg a week, reorder from Tidewater now."


@pytest.fixture
def board(db_session, seed_workspace):
    from core.models import Agent
    from core.models.core import BoardTask

    ws = UUID(seed_workspace())
    agents = {}
    for name in ("Shopify Business Analyst", "Shopify Inventory Watchdog"):
        agents[name] = Agent(name=name, agent_type="custom", description="", status="active", configuration={},
                             workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
        db_session.add(agents[name])
    db_session.flush()

    def card(title, result, status="done", agent="Shopify Business Analyst", workspace=ws):
        made = BoardTask(workspace_id=workspace, title=title, status=status, priority="medium", result=result,
                         assigned_agent_id=agents[agent].id,
                         completed_at=datetime(2026, 10, 4, 16, 1, tzinfo=timezone.utc))
        db_session.add(made)
        db_session.flush()
        return made

    return NS(db=db_session, ws=ws, card=card)


def _label(board, card):
    """The card as the note names it: its number (#0042), or "ticket <id>" without one."""
    from services.ticket_numbers import ticket_numbers

    return ticket_numbers(board.db, board.ws, [card]).get(card.id) or f"ticket {card.id}"


def test_the_cards_that_answer_come_back_by_number_with_their_answer(board):
    q5 = board.card(Q5, SIXTY_THREE)
    board.card("Harvest Club cancellations, April to September", "Total cancellations: 11 members.")
    board.card(Q5, "A draft that was never finished.", status="assigned")       # no answer yet: not a finding

    found = team_findings(board.db, board.ws, Q5)

    assert found[0]["number"] == _label(board, q5)
    assert found[0]["agent"] == "Shopify Business Analyst" and found[0]["status"] == "done"
    assert "63 active Harvest Club subscribers" in found[0]["excerpt"]
    assert all("never finished" not in f["excerpt"] for f in found)
    assert all("cancellations" not in f["title"] for f in found)                # shares too few of its words


def test_another_workspaces_cards_never_come_back(board, seed_workspace):
    board.card(Q5, SIXTY_THREE, workspace=UUID(seed_workspace()))

    assert team_findings(board.db, board.ws, Q5) == []


def _turn(board, said, widget_mode=False):
    chat = NS(db=board.db, workspace_id=str(board.ws), widget_mode=widget_mode,
              streaming_handler=get_streaming_handler())
    messages, prefetched = [{"role": "user", "content": said}], []

    async def retrieval_first(self, latest_text, llm_messages, agent_runtime, chat_id, ran):
        llm_messages.append({"role": "system", "content": "passages from green-coffee-list-1-sep-2026.csv"})
        yield "searched"

    async def run():
        return [f async for f in reads_what_the_team_found(retrieval_first)(chat, said, messages, None, "c",
                                                                            prefetched)]
    return asyncio.run(run()), messages, prefetched


def test_a_question_the_board_answers_reaches_the_model_after_the_passages(board):
    watchdog = board.card("Do I have plenty of Kirinyaga for Christmas?", KIRINYAGA, status="review",
                          agent="Shopify Inventory Watchdog")

    frames, messages, prefetched = _turn(board, "Do I need to reorder any Kirinyaga?")

    note = messages[-1]["content"]                                      # last: after the 1 Sept list's passages
    assert messages[-2]["content"].startswith("passages from green-coffee-list")
    assert _label(board, watchdog) in note and "Shopify Inventory Watchdog" in note
    assert "41 kg in green stock today" in note and "in review since" in note
    assert "not the owner's facts" in note and NOTE_RULE in note and ASKED_RULE not in note
    assert prefetched == [(READ_TOOL, {"automatic": True, "answers": True, "status": "review,done"})]
    assert '"tool-start"' in frames[0] and frames[-1] == "searched"


def test_check_the_board_reads_the_cards_not_a_count_of_them(board):
    board.card(Q5, SIXTY_THREE)

    _frames, messages, _ran = _turn(board, "How many club boxes go out on Monday? Check what the team has already "
                                           "found on the board first.")
    assert "63 active Harvest Club subscribers" in messages[-1]["content"] and ASKED_RULE in messages[-1]["content"]

    _frames, messages, _ran = _turn(board, "Check what the team found on the Yirgacheffe roast loss.")
    assert messages[-1]["content"] == NONE_FOUND and "never from a count of cards by status" in NONE_FOUND


def test_a_question_nothing_on_the_board_answers_or_a_widget_turn_reads_nothing_more(board):
    board.card(Q5, SIXTY_THREE)

    frames, messages, prefetched = _turn(board, "What do we charge a café for delivery on 10 kg?")
    assert (frames, prefetched, len(messages)) == (["searched"], [], 2)
    frames, messages, prefetched = _turn(board, Q5, widget_mode=True)          # a visitor: the board is the owner's
    assert (frames, prefetched, len(messages)) == (["searched"], [], 2)


def test_the_chat_runs_retrieval_first_through_it():
    from consumers.chatbot.service import StreamingChatService

    inner = StreamingChatService._retrieval_first.__wrapped__.__wrapped__.__wrapped__.__wrapped__.__wrapped__
    assert inner.__code__ is reads_what_the_team_found(lambda: None).__code__
