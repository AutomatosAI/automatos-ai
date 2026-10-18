"""P256-FIX-N17-3 (F439, nights 16–17): "did it run?" names the cards still waiting for the click.

"Did the stock check run?" (night 16 rows 168 and 347; night 17 row 460, two minutes after the
run-now raised #2828) was answered from the board's finished cards and never named the run that
waited for the owner's click.

Now a did-it-run question reads the asking person's call cards still pending and not expired, from
the last day, in this workspace (``services/waiting_asks``), and the turn gives the model one note:
each card's words and when it was raised, and the rule. Another person's cards, expired ones, older
ones, board tickets' approvals and questions are left out; no waiting card, no note; a widget
visitor's turn and any other question read nothing.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from consumers.chatbot.waiting_asks_turn import (
    NOTE_RULE, asks_did_it_run, reads_the_waiting, reads_the_waiting_asks, waiting_note,
)

NOW = datetime.now(timezone.utc)
ASKED = "Did the stock check run?"
STOCK_CHECK = "Run a playbook 'Daily stock check' (playbook #122)"
AT = re.compile(r"raised at \d{2}:\d{2} on \d{1,2} [A-Z][a-z]{2}")


def _person(db, name):
    return db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                      {"e": f"{name}-{uuid4().hex[:8]}@harbourline.test", "u": f"{name}-{uuid4().hex[:8]}"}).scalar()


def _card(db, ws, user_id, question, *, raised=NOW - timedelta(minutes=2), ttl=timedelta(hours=24),
          subject_type=None, kind=None):
    from core.models.approval_grants import KIND_APPROVAL, SUBJECT_TOOL_CALL
    from core.services.approval_grants import create_grant

    return create_grant(db, ws, subject_type=subject_type or SUBJECT_TOOL_CALL,
                        subject_id=f"platform_execute_playbook:{uuid4().hex[:32]}", tool_name="platform_execute_playbook",
                        question_md=f"{question}:", kind=kind or KIND_APPROVAL, now=raised,
                        ttl_seconds=int(ttl.total_seconds()),
                        details={"lane": "chat", "caller_context": {"driving_user_id": str(user_id)}})


@pytest.fixture
def night(db_session, seed_workspace):
    """The owner's chat; their run-now waiting two minutes; their expired card, their card from
    yesterday, a stranger's card, a board ticket's approval and a question beside it."""
    from core.models.approval_grants import KIND_QUESTION, SUBJECT_BOARD_TASK
    from core.models.core import Chat

    db, ws = db_session, UUID(seed_workspace())
    owner, stranger = _person(db, "owner"), _person(db, "stranger")
    chat = Chat(id=uuid4(), user_id=owner, workspace_id=ws, title="Stock", visibility="private")
    db.add(chat)
    db.flush()
    _card(db, ws, owner, STOCK_CHECK)
    _card(db, ws, owner, "Run a playbook 'Expired one' (playbook #80)", raised=NOW - timedelta(hours=3),
          ttl=timedelta(hours=1))
    _card(db, ws, owner, "Run a playbook 'Yesterday' (playbook #79)", raised=NOW - timedelta(hours=30),
          ttl=timedelta(hours=48))
    _card(db, ws, stranger, "Run a playbook 'Not theirs' (playbook #78)")
    _card(db, ws, owner, "Approve ticket #0412", subject_type=SUBJECT_BOARD_TASK)
    _card(db, ws, owner, "Which supplier?", kind=KIND_QUESTION)
    db.flush()
    return NS(db=db, ws=ws, owner=owner, chat=chat, turn=NS(db=db, workspace_id=ws, widget_mode=False))


def test_did_it_run_names_only_the_owners_live_card_with_its_time(night):
    note = reads_the_waiting(night.turn, ASKED, str(night.chat.id))

    assert note is not None and STOCK_CHECK in note and AT.search(note)
    assert note.endswith(NOTE_RULE)
    for left_out in ("Expired one", "Yesterday", "Not theirs", "#0412", "Which supplier"):
        assert left_out not in note


def test_no_waiting_card_means_no_note(db_session, seed_workspace):
    from core.models.core import Chat

    ws = UUID(seed_workspace())
    chat = Chat(id=uuid4(), user_id=_person(db_session, "owner"), workspace_id=ws, title="Quiet", visibility="private")
    db_session.add(chat)
    db_session.flush()

    assert reads_the_waiting(NS(db=db_session, workspace_id=ws, widget_mode=False), ASKED, str(chat.id)) is None


@pytest.mark.parametrize("widget, said", [(True, ASKED), (False, "What do we charge for 10 kg of Guji?")])
def test_a_widget_visitor_or_another_question_reads_nothing(widget, said):
    db = MagicMock(name="db")

    assert reads_the_waiting(NS(db=db, workspace_id=uuid4(), widget_mode=widget), said, str(uuid4())) is None
    db.query.assert_not_called()


@pytest.mark.parametrize("said, asks", [
    ("Did the stock check run?", True), ("Has the reorder gone through?", True), ("Did it go through?", True),
    ("Was the Monday run finished?", True), ("did the green-coffee reorder happen", True),
    ("Run the stock check now", False), ("How much Guji is left?", False), ("Has Declan replied?", False),
])
def test_what_reads_as_did_it_run(said, asks):
    assert asks_did_it_run(said) is asks


def test_the_note_goes_in_after_the_turns_reads(night):
    async def retrieval_first(chat, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
        llm_messages.append({"role": "system", "content": "document passages"})
        yield "retrieval"

    async def run():
        messages = [{"role": "user", "content": ASKED}]
        frames = [f async for f in reads_the_waiting_asks(retrieval_first)(
            night.turn, ASKED, messages, None, str(night.chat.id), [])]
        return frames, messages

    frames, messages = asyncio.run(run())

    assert frames == ["retrieval"]
    assert messages[-2]["content"] == "document passages"
    assert messages[-1]["role"] == "system" and STOCK_CHECK in messages[-1]["content"]


def test_the_note_names_each_card_then_the_rule():
    note = waiting_note([{"card": STOCK_CHECK, "at": "at 04:12 on 14 Oct"}])

    assert f"- {STOCK_CHECK}, raised at 04:12 on 14 Oct" in note.splitlines()
    assert note.splitlines()[-1] == NOTE_RULE and "a finished run of something else is not their run" in NOTE_RULE


def test_the_chat_runs_retrieval_first_through_it_over_the_teams_findings():
    """Directly over F317's read, so the note comes after the finished cards it answers."""
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.team_findings import reads_what_the_team_found

    inner, codes = StreamingChatService._retrieval_first, []
    while hasattr(inner, "__wrapped__"):
        codes.append(inner.__code__)
        inner = inner.__wrapped__
    mine = codes.index(reads_the_waiting_asks(lambda: None).__code__)
    assert codes[mine + 1] is reads_what_the_team_found(lambda: None).__code__
