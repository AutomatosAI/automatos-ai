"""F241 (night 8): the turn tells Auto which cards the owner named, and the call for each.

17 of 95 first tries landed on the card the owner named. Nothing in the turn said that
#0201 was a card: "Approve #0201" went to platform_submit_social_post, and "Let's talk
about #0329." got "What is #0329? Is it a task, a report, or something else?".
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

from core.models.core import BoardTask
from modules.tools.discovery.card_note import CARDS_NAMED, cards_note, grounds_the_cards


def _card(db_session, ws, title, status="review", source_type="user"):
    card = BoardTask(workspace_id=ws, title=title, status=status, source_type=source_type)
    db_session.add(card)
    db_session.flush()
    return card, f"#{card.workspace_seq:04d}"


def test_the_note_names_each_card_and_the_call_for_the_owners_verb(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    _, number = _card(db_session, ws, "Reply to Hannah at Mill Lane Kitchen")

    note = cards_note(db_session, ws, f"Approve {number} with this note: Good, that's the tone. And #9999?")

    assert note.startswith(CARDS_NAMED)
    assert f"{number} ('Reply to Hannah at Mill Lane Kitchen', review), a card." in note
    assert f'platform_update_task_status {{task_id: "{number}", status: "done"' in note
    assert "#9999 (no card on the board has that number)" in note


def test_a_missions_card_is_named_as_one(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    _, number = _card(db_session, ws, "Mission: Easter box", status="in_progress", source_type="orchestration")

    note = cards_note(db_session, ws, f"Cancel {number}, please.")

    assert "a mission's own card" in note and f'platform_cancel_mission {{mission_id: "{number}"}}' in note


def test_no_card_number_no_note(db_session, seed_workspace):
    assert cards_note(db_session, UUID(seed_workspace()), "What's waiting for me?") == ""


def _turn(chat, text):
    """``_retrieval_first`` wrapped, run for ``text``: the frames it yields and the turn's messages."""
    messages = [{"role": "system", "content": "the stable prompt"}]

    async def retrieval_first(self, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
        yield "frame"

    async def run():
        return [frame async for frame in grounds_the_cards(retrieval_first)(chat, text, messages, None, "c", [])]

    return asyncio.run(run()), messages


def test_the_owners_turn_carries_the_note_last(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    _, number = _card(db_session, ws, "Reply to Raj Patel")

    frames, messages = _turn(NS(db=db_session, workspace_id=ws, widget_mode=False), f"Let's talk about {number}.")

    assert frames == ["frame"]
    assert messages[-1]["role"] == "system" and f"{number} ('Reply to Raj Patel', review)" in messages[-1]["content"]
    assert f'platform_get_task {{task_id: "{number}"}}' in messages[-1]["content"]


def test_a_widget_visitor_gets_no_note(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    _, number = _card(db_session, ws, "Reply to Raj Patel")

    frames, messages = _turn(NS(db=db_session, workspace_id=ws, widget_mode=True), f"What is {number}?")

    assert frames == ["frame"] and len(messages) == 1


def test_the_chat_runs_retrieval_first_through_it():
    from consumers.chatbot.service import StreamingChatService

    assert StreamingChatService._retrieval_first.__wrapped__.__name__ == "_retrieval_first"
    assert StreamingChatService._retrieval_first.__wrapped__.__code__ is grounds_the_cards(  # under PRD-256's receipts
        lambda: None).__code__
