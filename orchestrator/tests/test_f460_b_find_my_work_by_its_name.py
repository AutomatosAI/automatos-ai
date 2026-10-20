"""F460 (night 18): find my work without its number.

"Show me the Taunton quote." read no card (F317's board read runs for questions only) and printed
the wholesale accounts sheet, which mentions Taunton, while #1836, the quote, waited in Review.
"Did the Bell Yard invoice come back?" read the board, but an old card whose answer held "income"
and "feedback" outranked the card titled for the Bell Yard invoice: title and answer were one
string, matched part-word, and "come" and "back" counted as what the question was about.

Now the work the owner names reaches the model as ONE note naming each card that holds it, and the
board's findings match whole words, weigh a title above an answer and put ties newest first.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

from consumers.chatbot.named_work_note import NOTE_HEAD, NOTE_RULE
from consumers.chatbot.streaming import get_streaming_handler
from consumers.chatbot.team_findings import reads_what_the_team_found
from modules.tools.discovery.described_work import work_named
from services.team_findings import NO_ANSWER, team_findings, terms_of

TAUNTON_TITLE = "Draft wholesale quote: new café in Taunton (8 kg/week Harbour Blend)"
TAUNTON_FILE = "sessions/3125/taunton-quote.md"
ACCOUNTS_PASSAGE = "passages from harbourline-wholesale-accounts.csv: Harrow Lane Deli, Taunton, 30 days"
BELL_YARD = "Did the Bell Yard invoice come back?"
OLD_ANSWER = ("Wholesale accounts, September: Bell Yard Coffee owes invoice HL-2026-0147. Café income is down 4% "
              "and the feedback from Bell Yard on the new roast was good.")
NEW_ANSWER = "Draft October invoice for Bell Yard Coffee: 12 kg Harbour Blend at £23.00/kg. Draft only, not sent."


def _at(day):
    return datetime(2026, 10, day, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def board(db_session, seed_workspace):
    from core.models import Agent
    from core.models.core import BoardTask
    from tests import test_f161_a_later_step_reads_an_earlier_steps_file as f161

    db_session.execute(text(f161.OUTPUTS_VIEW))
    ws = UUID(seed_workspace())
    agents = {}
    for name in ("WholesalePricingAgent", "INVOICE DESK", "Shopify Business Analyst"):
        agents[name] = Agent(name=name, agent_type="custom", description="", status="active", configuration={},
                             workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
        db_session.add(agents[name])
    db_session.flush()

    def card(title, seq, agent, status="review", result=None, day=19):
        made = BoardTask(workspace_id=ws, title=title, status=status, priority="medium", result=result,
                         workspace_seq=seq, assigned_agent_id=agents[agent].id, created_at=_at(day),
                         completed_at=_at(day) if result else None)
        db_session.add(made)
        db_session.flush()
        return made

    def file(made, path):
        db_session.execute(text(
            "INSERT INTO deliverables (workspace_id, source_type, source_id, artifact_type, title, file_path, "
            "created_at) VALUES (CAST(:ws AS uuid), 'task', :card, 'document', :title, :path, :made)"),
            {"ws": str(ws), "card": str(made.id), "title": path.rsplit("/", 1)[-1], "path": path, "made": _at(19)})
        db_session.flush()

    return NS(db=db_session, ws=ws, card=card, file=file)


def _turn(board, said, widget_mode=False):
    chat = NS(db=board.db, workspace_id=str(board.ws), widget_mode=widget_mode,
              streaming_handler=get_streaming_handler())
    messages, prefetched = [{"role": "user", "content": said}], []

    async def retrieval_first(self, latest_text, llm_messages, agent_runtime, chat_id, ran):
        llm_messages.append({"role": "system", "content": ACCOUNTS_PASSAGE})
        yield "searched"

    async def run():
        return [f async for f in reads_what_the_team_found(retrieval_first)(chat, said, messages, None, "c",
                                                                            prefetched)]
    return asyncio.run(run()), messages, prefetched


def _named_notes(messages):
    return [m["content"] for m in messages if m["content"].startswith(NOTE_HEAD)]


# ── the work the owner names reaches the model by its card ────────────────────

def test_show_me_the_taunton_quote_names_its_card_after_the_documents(board):
    quote = board.card(TAUNTON_TITLE, 1836, "WholesalePricingAgent")
    board.file(quote, TAUNTON_FILE)
    board.card("Harrow Lane Deli reorder reminder", 1790, "Shopify Business Analyst", status="done",
               result="Harrow Lane Deli in Taunton is due a reorder.", day=12)

    _frames, messages, prefetched = _turn(board, "Show me the Taunton quote.")

    notes = _named_notes(messages)
    assert len(notes) == 1 and messages[-1]["content"] == notes[0]
    assert messages[-2]["content"] == ACCOUNTS_PASSAGE                     # after the document passages
    assert f"- #1836 '{TAUNTON_TITLE}', WholesalePricingAgent, review, file: {TAUNTON_FILE}" in notes[0]
    assert "#1790" not in notes[0] and notes[0].endswith(NOTE_RULE)
    assert ("platform_list_tasks", {"automatic": True, "search": "taunton quote"}) in prefetched


def test_a_long_name_and_the_new_kinds_of_work_are_read():
    assert work_named("Show me the Harbour Lights Box launch wording draft") == ["harbour", "lights", "box",
                                                                                 "launch", "wording"]
    assert work_named("Did the club box count come back?") == ["club", "box", "count"]
    assert work_named("Show me the Christmas box packing list") == ["christmas", "box", "packing", "list"]
    assert work_named("Where is the morning roast routine?") == ["morning", "roast", "routine"]
    assert work_named("Show me the quote.") == [] and work_named("What's on my board?") == []


def test_new_work_a_visitor_or_no_named_work_gets_no_note(board):
    board.card(TAUNTON_TITLE, 1836, "WholesalePricingAgent")

    for said, widget in (("Draft the Taunton quote for Harrow Lane. Draft only.", False),
                         ("Show me the Taunton quote.", True), ("What's the time?", False)):
        _frames, messages, _ran = _turn(board, said, widget_mode=widget)
        assert _named_notes(messages) == [], said

    _frames, messages, _ran = _turn(board, "Draft the Taunton quote again, but shorter.")
    assert "#1836" in _named_notes(messages)[0]                             # a change to it: the card is named


# ── the board's findings: whole words, title first, newest first ──────────────

def test_the_card_titled_for_the_work_outranks_an_old_answer_that_mentions_it(board):
    board.card("Wholesale accounts check, September", 1702, "Shopify Business Analyst", status="done",
               result=OLD_ANSWER, day=2)
    invoice = board.card("Bell Yard invoice", 1848, "INVOICE DESK", result=NEW_ANSWER, day=19)

    found = team_findings(board.db, board.ws, BELL_YARD)

    assert found and found[0]["number"] == "#1848" and found[0]["title"] == invoice.title
    _frames, messages, _ran = _turn(board, BELL_YARD)
    team_note = next(m["content"] for m in messages if "already found on the board" in m["content"])
    assert team_note.index("#1848") < team_note.index("#1702")              # both answer; the titled one first
    assert "#1848" in _named_notes(messages)[0]


def test_how_the_owner_asks_is_not_what_about():
    assert terms_of(BELL_YARD) == ["bell", "yard", "invoice"]
    assert "count" in terms_of("What's the club box count?")


def test_a_card_titled_for_the_work_comes_back_before_it_has_an_answer(board):
    board.card("Bell Yard invoice, October", 1849, "INVOICE DESK", result=None)

    found = team_findings(board.db, board.ws, BELL_YARD)

    assert [f["number"] for f in found] == ["#1849"] and found[0]["excerpt"] == NO_ANSWER


def test_part_words_never_match(board):
    board.card("Courtyard Café invoices", 1703, "Shopify Business Analyst", status="done",
               result="Two invoices raised for Courtyard Café.", day=3)

    assert team_findings(board.db, board.ws, "What did the yard invoice say?") == []      # "yard" is not "Courtyard"
