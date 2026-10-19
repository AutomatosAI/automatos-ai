"""P256-FIX-N17-9 (night 17, F448) (A) — a ticket reads the files on a card its owner names.

CELLAR's redo of #1754 carried the owner's "the quote you cannot find is card #1751", and
SHOPFRONT was answered "It is card #1751 … the PDF is in Deliverables". Neither session
could open #1751's quote: ``read_step_file`` refused any ticket outside a mission, and the
quote was a 'generated' PDF with no text. CELLAR wrote it again from memory notes.

Only the owner's words name a card: the brief, their answers and this ticket's own
corrections; never the agent's question, an answer's "question #…", a standing rule from
another card, a colour, the ticket itself or another workspace's number. The claim lists
the named cards' files newest first, with each card's status, keeps them on the ticket,
and ``read_step_file`` reads them by id: a text as text, a PDF as a copy in the folder,
read from its own storage. Through the real claim and the real JSON-RPC tools/call.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from sqlalchemy import text

import modules.documents.generation_service as gs
from core.models import Agent
from modules.documents.models import GeneratedDocument
from modules.tools.execution import session_document_folder
from services import cli_host_service as svc
from services import cli_ticket_lane as lane
from services import session_tools as st
from services import session_tools_rpc as rpc
from services.deliverable_service import DeliverableService
from services.named_cards import claimed_files, named_cards, owner_text
from services.step_files import CARD_STATUS_WORDS, NAMED_FILES_HEADER, NAMED_FILES_PER_CARD, READ_FRAME_LINE
from services.ticket_numbers import format_number
from services.ticket_redo import BRIEF_AGREED
from tests import test_f161_a_later_step_reads_an_earlier_steps_file as f161
from tests import test_f370_a_later_step_opens_an_earlier_steps_picture as f370
from tests.helpers_mission_lane import quiet_board

PDF = b"%PDF-1.7 Tidewater Kitchen, 8 kg fortnightly"
PRICES = "# Price notes\n\nHarbour Blend £22.00 a kilo wholesale.\n"
OTHER_NUMBER = 4242       # a card number only another workspace has


def _now(hours: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _card(db, ws, agent, title, status, source_id):
    card = lane.file_cli_ticket(db, workspace_id=ws, agent_id=agent.id, title=title, prompt=f"Work on: {title}",
                                source_type="chat", source_id=source_id)
    card.status = status
    db.flush()
    return card


def _text_file(db, ws, card, agent, name, content):
    path = f"sessions/{card.id}/{name}"
    f161._Worker.files = {**f161._Worker.files, path: content}
    return DeliverableService(db, str(ws)).register(
        file_path=path, source_type="task", source_id=str(card.id), agent_id=agent.id, agent_name=agent.name,
        artifact_type="document", file_size_bytes=len(content))["deliverable_id"]


def _generated(db, ws, card, agent, filename, hours_ago):
    """A PDF generate_document made on ``card``'s session: storage 'generated', no text."""
    made = GeneratedDocument(path=f"/generated/{filename}", format="pdf", filename=filename, size=len(PDF))
    saved = gs.DocumentGenerationService(db, ws).register_as_deliverable(
        made, title=f"Tidewater quote ({filename})", source_type="task", source_id=str(card.id),
        agent_id=agent.id, agent_name=agent.name)
    db.execute(text("UPDATE deliverables SET created_at = :at WHERE id = CAST(:id AS uuid)"),
               {"at": datetime.now(timezone.utc) - timedelta(hours=hours_ago), "id": saved["deliverable_id"]})
    return saved["deliverable_id"]


def _number(card) -> str:
    return format_number(card.workspace_seq)


@pytest.fixture
def board(db_session, seed_workspace, monkeypatch):
    """CELLAR's workspace: review card A with two generated PDFs (the newer second), done card
    D with a text file, cards E, F and G with files nobody's own words name, card C carrying a
    standing note that names G, and the redo ticket B whose owner's words name A and D."""
    from modules.tools.discovery import handlers_deliverables
    from modules.tools.execution import unified_executor

    quiet_board(monkeypatch)
    monkeypatch.setattr(lane, "host_online", lambda db, ws: True)
    monkeypatch.setattr(lane, "no_cli_host_reason_for", lambda db, ws, cli: None)
    monkeypatch.setattr("services.deliverable_service.WorkspaceClient", f161._Worker)
    monkeypatch.setattr(f161._Worker, "files", {})
    monkeypatch.setattr(gs, "copy_when_registered", lambda *args, **kwargs: False)
    monkeypatch.setattr(session_document_folder, "WorkspaceClient", f370._Folder)
    monkeypatch.setattr(f370._Folder, "written", {})
    stored = {}
    monkeypatch.setattr("modules.documents.thumbnails.sources.read_source",
                        lambda ws, storage, path: stored.get((str(ws), storage, path)))
    db_session.execute(text(f161.OUTPUTS_VIEW))
    dispatched = []

    class _Executor:
        def __init__(self, db):
            self.db = db

        async def execute_tool(self, *, tool_name, parameters, agent_id, workspace_id, trace_id, caller_context):
            dispatched.append((tool_name, parameters["action"]))
            return await handlers_deliverables.get_deliverable(self.db, workspace_id, parameters["params"])

    monkeypatch.setattr(unified_executor, "UnifiedToolExecutor", _Executor)
    ws = UUID(seed_workspace())
    agent = Agent(name="CELLAR", agent_type="chatbot", description="", status="active",
                  configuration={"runtime": "cli", "provider": "claude", "model": "sonnet"}, model_config=None,
                  workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db_session.add(agent)
    db_session.flush()
    cards = {key: _card(db_session, ws, agent, title, status, f"f448-{key}") for key, title, status in (
        ("A", "Quote for Tidewater Kitchen", "review"), ("D", "Harbourline price notes", "done"),
        ("E", "Cafe list", "done"), ("F", "Roast log", "done"), ("G", "Green coffee list", "done"),
        ("C", "Wholesale email", "done"))}
    files = {"a_old": _generated(db_session, ws, cards["A"], agent, "quote-v1.pdf", hours_ago=2),
             "a_new": _generated(db_session, ws, cards["A"], agent, "quote-v2.pdf", hours_ago=1),
             "d": _text_file(db_session, ws, cards["D"], agent, "price-notes.md", PRICES)}
    for key in ("E", "F", "G"):
        files[key.lower()] = _text_file(db_session, ws, cards[key], agent, f"{key.lower()}.md", f"card {key}'s file")
    stored[(str(ws), "generated", "generated/quote-v2.pdf")] = PDF
    other = _another_workspaces_card(db_session, seed_workspace(), agent)
    cards["C"].planning_data = {"owner_corrections": [
        {"note": f"Always quote from the green coffee list, never from memory ({_number(cards['G'])}).",
         "by": "user:1", "at": _now(-3)}]}
    cards["B"] = _redo_ticket(db_session, ws, agent, cards)
    return {"db": db_session, "ws": ws, "agent": agent, "cards": cards, "files": files, "other": other,
            "dispatched": dispatched, "stored": stored}


def _another_workspaces_card(db, other_ws, agent):
    """A card numbered #4242 with a file, in another workspace: never this ticket's to read."""
    other = lane.file_cli_ticket(db, workspace_id=UUID(other_ws), agent_id=agent.id, title="Not yours",
                                 prompt="Elsewhere.", source_type="chat", source_id="f448-other")
    other.workspace_seq, other.status = OTHER_NUMBER, "done"
    db.flush()
    DeliverableService(db, other_ws).register(file_path=f"sessions/{other.id}/theirs.md", source_type="task",
                                              source_id=str(other.id), artifact_type="document")
    return other


def _redo_ticket(db, ws, agent, cards):
    """#1754's shape: the owner's correction names A, their answer names D (and 'question #F'),
    the agent's own question names E, and the brief quotes colours and another workspace's card."""
    num = {key: _number(card) for key, card in cards.items()}
    ticket = lane.file_cli_ticket(
        db, workspace_id=ws, agent_id=agent.id, title="Redo the Tidewater quote", source_type="chat",
        source_id="f448-B",
        prompt=(f"Write the Tidewater Kitchen quote, 8 kg fortnightly. Headings in #222222, rules in #333. "
                f"Their supplier's card #{OTHER_NUMBER} is not ours."))
    note = f"The quote you cannot find is card {num['A']}."
    ticket.planning_data = {
        "owner_corrections": [{"note": note, "by": "user:1", "at": _now(-1)}],
        "human_qa": [{"q": f"Is the quote on {num['E']}?",
                      "a": f"It is card {num['D']}, and question {num['F']} was mine."}],
    }
    ticket.review_feedback = note
    ticket.runtime_ref = {"session_asks": [{"grant_id": 1, "question": f"Should I use {num['E']}?",
                                            "answer": "No. Read the quote the owner named.", "answered_at": _now(-1)}]}
    db.flush()
    return ticket


def _claim(db, ws):
    host, code, _ = svc.create_pairing_code(db, ws, "laptop")
    host, _token = svc.pair_host(db, code)
    (ticket,) = svc.claim_for_host(db, host, 1)["tasks"]
    return host, ticket


def _read(board, ticket, file_id):
    ctx = st.SessionContext(task_id=ticket.id, agent_id=board["agent"].id, agent_name="CELLAR",
                            workspace_id=board["ws"])
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "read_step_file", "arguments": {"file_id": file_id}}}
    reply = asyncio.run(rpc.handle_message(
        message, ctx, server_version="1.0",
        call=lambda tool, params, c: st.call_tool(board["db"], tool, params, c)))
    return reply["result"]["content"][0]["text"], reply["result"]["isError"]


def test_only_the_owners_own_words_name_a_card(board):
    cards, db = board["cards"], board["db"]
    texts = owner_text(cards["B"])

    named = named_cards(db, board["ws"], cards["B"].id, texts + [f"This redo is card {_number(cards['B'])}."])

    assert set(named) == {cards["A"].id, cards["D"].id}
    assert not any(_number(cards["E"]) in t for t in texts)                       # the agent's questions
    assert not any(_number(cards["G"]) in t for t in texts)                       # another card's standing rule
    assert board["other"].id not in named                                         # #4242 is another workspace's


def test_a_mission_steps_brief_is_the_coordinators_and_names_no_card(board):
    """The coordinator lists the other steps' files as "(ticket #0051.3, …)" in a step's brief:
    those are listed as the mission's already, never again as cards the owner named."""
    db, cards = board["db"], board["cards"]
    run, (step,) = f161._mission(db, board["ws"], board["agent"], "Tidewater quote", 1)
    ticket = f161._claimed(db, run, step, board["agent"])
    ticket.raw_prompt = ticket.description = (
        f"Work on the quote.\n\n- `{board['files']['a_new']}` quote-v2.pdf (ticket {_number(cards['A'])}, \"Quote\")")
    ticket.planning_data = {"human_qa": [{"q": "Which price notes?", "a": f"It is card {_number(cards['D'])}."}]}
    db.flush()

    assert named_cards(db, board["ws"], ticket.id, owner_text(ticket)) == [cards["D"].id]


def test_the_claim_lists_the_named_cards_files_newest_first(board):
    db, cards, files = board["db"], board["cards"], board["files"]
    _host, ticket = _claim(db, board["ws"])

    assert ticket["task_id"] == cards["B"].id
    block = ticket["prompt"].split(NAMED_FILES_HEADER, 1)[1]
    lines = {key: next(line for line in block.splitlines() if files[key] in line) for key in ("a_new", "a_old", "d")}
    assert block.index(files["a_new"]) < block.index(files["a_old"])
    assert "not approved" in lines["a_new"] and "the newest" in lines["a_new"] and _number(cards["A"]) in lines["a_new"]
    assert "an earlier one" in lines["a_old"]
    assert CARD_STATUS_WORDS["done"] in lines["d"]
    assert not any(files[key] in ticket["prompt"] for key in ("e", "f", "g"))
    db.refresh(cards["B"])
    rows = cards["B"].runtime_ref["named_card_files"]
    assert {row["task_id"] for row in rows} == {cards["A"].id, cards["D"].id}
    assert {row["storage_type"] for row in rows if row["task_id"] == cards["A"].id} == {"generated"}


def test_a_second_claim_still_lists_both_cards(board):
    db, cards, files = board["db"], board["cards"], board["files"]
    host, _first = _claim(db, board["ws"])
    db.refresh(cards["B"])
    cards["B"].status = "assigned"                                     # the owner's answer re-queued it
    db.flush()

    (again,) = svc.claim_for_host(db, host, 1)["tasks"]

    assert again["task_id"] == cards["B"].id
    assert all(files[key] in again["prompt"] for key in ("a_new", "a_old", "d"))


def test_a_brief_agreed_anew_with_no_naming_text_lists_none(board):
    db, cards, files = board["db"], board["cards"], board["files"]
    agreed = _card(db, board["ws"], board["agent"], "The agreed brief", "assigned", "f448-agreed")
    agreed.planning_data = {
        "previous_briefs": [{"description": "the old brief", "by": "user:1", "at": _now(-1)}],
        "owner_corrections": [{"note": f"It is card {_number(cards['A'])}.", "by": "user:1", "at": _now(-2)}],
    }
    agreed.review_feedback = BRIEF_AGREED
    db.flush()
    carried = [{"task_id": cards["A"].id, "number": _number(cards["A"]), "deliverable_id": files["a_new"],
                "storage_type": "generated", "file_path": "generated/quote-v2.pdf", "name": "quote-v2.pdf",
                "created_at": _now(-1), "card_status": CARD_STATUS_WORDS["review"], "title": "Quote"}]

    listed = claimed_files(db, agreed, carried)

    assert (listed.block, listed.rows) == ("", ())                   # the earlier claim's rows go too
    kept = claimed_files(db, cards["E"], carried)                      # any other claim keeps them
    assert [row["deliverable_id"] for row in kept.rows] == [files["a_new"]] and files["a_new"] in kept.block


def test_a_card_with_more_files_than_the_cap_says_how_many_more(board):
    db, cards = board["db"], board["cards"]
    for n in range(NAMED_FILES_PER_CARD + 1):                          # D held one file; now twelve
        _text_file(db, board["ws"], cards["D"], board["agent"], f"note-{n}.md", f"note {n}")

    listed = claimed_files(db, cards["B"], ())

    on_d = [line for line in listed.block.splitlines() if f"ticket {_number(cards['D'])}" in line]
    assert len(on_d) == NAMED_FILES_PER_CARD + 1
    assert on_d[-1] == f"- …and 2 more on ticket {_number(cards['D'])}, not listed here."
    assert {row["files_on_card"] for row in listed.rows if row["task_id"] == cards["D"].id} == {12}


def test_the_named_pdf_is_copied_into_the_tickets_folder_from_its_own_storage(board):
    db, cards, files = board["db"], board["cards"], board["files"]
    _claim(db, board["ws"])

    body, is_error = _read(board, cards["B"], files["a_new"])

    copy = f"sessions/{cards['B'].id}/card-{cards['A'].id}-quote-v2.pdf"
    assert not is_error, body
    assert f"a copy is in your folder: {copy}" in body
    assert f"saved on ticket {_number(cards['A'])}" in body and "has not approved" in body.splitlines()[0]
    assert f370._Folder.written == {(str(board["ws"]), copy): PDF}
    assert set(board["dispatched"]) == {("platform_execute", "platform_get_deliverable")}


def test_a_named_text_file_comes_back_as_text(board):
    db, cards = board["db"], board["cards"]
    _claim(db, board["ws"])

    body, is_error = _read(board, cards["B"], board["files"]["d"])

    assert not is_error and PRICES in body and READ_FRAME_LINE in body.splitlines()[0]
    assert f370._Folder.written == {}


def test_a_card_nobody_named_and_a_ticket_naming_nothing_are_refused(board):
    db, cards = board["db"], board["cards"]
    _claim(db, board["ws"])
    nothing = _card(db, board["ws"], board["agent"], "Heartbeat", "done", "f448-nothing")

    body, is_error = _read(board, cards["B"], board["files"]["e"])
    assert is_error and "not one of the files listed on your ticket" in body, body
    body, is_error = _read(board, nothing, board["files"]["a_new"])
    assert is_error and "not a step of a mission" in body
    assert board["dispatched"] == []
