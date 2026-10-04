"""F307 (night 9): "What needs me right now?" got "You're all clear!" with 16 cards in
review (chat 2223df4d: its one call read the agents' status), then the same sentence
in two fresh chats with no call at all (5950e0e1; 834c2e25 with mission step #1888
waiting). Now the turn reads the board's Needs you before the model answers, the
answer is that list, and an answer that still says "all clear" gains the count.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

from consumers.chatbot import needs_you_turn
from consumers.chatbot.needs_you_turn import READ_TOOL, answers_what_needs_you, asks_what_needs_me
from consumers.chatbot.streaming import StreamingHandler

ASKED = "What needs me right now?"
NIGHT_9_ANSWER = ("It looks like all agents are currently idle, and there are no anomalies or urgent tasks that "
                  "require your immediate attention. You're all clear!")


def _step_of_a_cancelled_mission(db, ws):
    """Mission #0035, cancelled after its first step was sent back: step card in review (#1888)."""
    run = str(db.execute(text(
        "INSERT INTO orchestration_runs (workspace_id, goal, created_by, state, state_type) "
        "VALUES (CAST(:w AS uuid), 'Top three cafés by kg', 'owner', 'cancelled', 'terminal') RETURNING id"),
        {"w": str(ws)}).scalar())
    step = str(db.execute(text(
        "INSERT INTO orchestration_tasks (run_id, title, sequence_number, state) "
        "VALUES (CAST(:r AS uuid), 'Kilos per café, June to August', 1, 'pending') RETURNING id"),
        {"r": run}).scalar())
    card = db.execute(text(
        "INSERT INTO board_tasks (workspace_id, workspace_seq, title, status, source_type, orchestration_run_id) "
        "VALUES (CAST(:w AS uuid), 35, 'Mission: top three cafés', 'cancelled', 'orchestration', "
        "CAST(:r AS uuid)) RETURNING id"), {"w": str(ws), "r": run}).scalar()
    db.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, status, source_type, parent_task_id, orchestration_task_id) "
        "VALUES (CAST(:w AS uuid), 'Kilos per café, June to August', 'review', 'orchestration_task', :card, "
        "CAST(:step AS uuid))"), {"w": str(ws), "card": card, "step": step})


@pytest.fixture
def board(db_session, seed_workspace):
    from core.models.core import BoardTask

    ws = UUID(seed_workspace())
    for title, status in (("Delivery charge on 10 kg", "review"), ("Club cancellations, April to September",
                                                                   "review"), ("Rota for October", "done")):
        # F293 (on main): a card in Review counts once someone worked on it, so it carries an answer.
        db_session.add(BoardTask(workspace_id=ws, title=title, status=status, source_type="user",
                                 result="Draft ready." if status == "review" else None))
    db_session.flush()
    _step_of_a_cancelled_mission(db_session, ws)
    db_session.flush()
    return NS(db=db_session, ws=ws)


def _chat(board, *, widget=False):
    return NS(db=board.db, workspace_id=board.ws, widget_mode=widget, streaming_handler=StreamingHandler())


def _turn(chat, said, then=None):
    """The wrapped ``_retrieval_first`` for ``said``: frames, the turn's messages, what ran, and ``then()``'s
    result, called in the same task after it (the saved answer's additions)."""
    messages = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": said}]
    prefetched = []

    async def retrieval_first(self, latest_text, llm_messages, agent_runtime, chat_id, ran):
        yield "retrieval"

    async def run():
        frames = [f async for f in answers_what_needs_you(retrieval_first)(chat, said, messages, None, "c", prefetched)]
        return frames, (then() if then else None)

    frames, after = asyncio.run(run())
    return NS(frames=frames, messages=messages, prefetched=prefetched, after=after)


@pytest.mark.parametrize("said", [ASKED, "What's waiting for me?", "Anything for me today?",
                                  "Is there anything that needs my attention?", "What should I look at next?"])
def test_each_way_of_asking_is_caught(said):
    assert asks_what_needs_me(said) is True


@pytest.mark.parametrize("said", ["What do we charge a café for delivery on 10 kg?",
                                  "What do I need to do to connect Shopify?", "Send card 27.2 back."])
def test_other_questions_are_not(said):
    assert asks_what_needs_me(said) is False


def test_the_turn_reads_needs_you_before_the_model_answers(board):
    from services.needs_you import needs_you_counts

    turn = _turn(_chat(board), ASKED)

    total = needs_you_counts(board.db, board.ws)["total"]
    assert total == 3                                                     # two cards in review + the mission's step
    note = turn.messages[-1]["content"]
    assert turn.messages[-1]["role"] == "system" and f"{total} in all" in note
    for title in ("Delivery charge on 10 kg", "Club cancellations, April to September",
                  "Kilos per café, June to August"):
        assert title in note
    assert "Rota for October" not in note
    assert "Never say nothing needs them, or that they are all clear, unless the total is 0." in note
    assert turn.prefetched == [(READ_TOOL, {"automatic": True, "needs_you": True})]
    starts = [json.loads(f[2:])["data"] for f in turn.frames if '"tool-start"' in f]
    assert [s["toolName"] for s in starts] == [READ_TOOL] and turn.frames[-1] == "retrieval"


def test_an_all_clear_answer_gains_the_count(board):
    from consumers.chatbot.service import StreamingChatService

    turn = _turn(_chat(board), ASKED, then=lambda: StreamingChatService._answer_additions(
        None, NS(content=NIGHT_9_ANSWER, finish_reason="stop")))

    from modules.tools.discovery.board_waiting import whats_waiting

    waiting = whats_waiting(board.db, board.ws)
    (line,) = turn.after
    assert line == "\n\n" + needs_you_turn.CLEAR_LINE.format(total=3, by_kind=needs_you_turn._by_kind(
        waiting["by_kind"]))
    assert line.startswith("\n\nJust to be clear: your board's Needs you has 3 waiting for you right now (2 in review")


def test_an_answer_that_gives_the_count_is_left_alone(board):
    from consumers.chatbot.service import StreamingChatService

    turn = _turn(_chat(board), ASKED, then=lambda: StreamingChatService._answer_additions(
        None, NS(content="3 things need you: two cards in review and a mission step.", finish_reason="stop")))

    assert turn.after == []


def test_a_board_that_cannot_be_read_is_never_all_clear(board, monkeypatch):
    from modules.tools.discovery import board_waiting

    def broken(db, workspace_id):
        raise RuntimeError("database went away")

    monkeypatch.setattr(board_waiting, "whats_waiting", broken)
    turn = _turn(_chat(board), ASKED)

    assert turn.messages[-1]["content"] == needs_you_turn.NOTE_UNREAD
    assert turn.prefetched == [] and turn.frames == ["retrieval"]


def test_a_widget_visitor_or_another_question_reads_nothing(board):
    for chat, said in ((_chat(board, widget=True), ASKED), (_chat(board), "What do we charge for 10 kg?")):
        turn = _turn(chat, said)
        assert turn.frames == ["retrieval"] and len(turn.messages) == 2 and turn.prefetched == []


def test_the_chat_runs_retrieval_first_through_it():
    from consumers.chatbot.service import StreamingChatService

    inner = StreamingChatService._retrieval_first.__wrapped__              # under F241's card note
    assert inner.__code__ is answers_what_needs_you(lambda: None).__code__
