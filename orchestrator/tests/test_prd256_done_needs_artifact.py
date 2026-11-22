"""PRD-256 US-005 (F380): an agent's card moves to Done only when the thing it was for exists.

Night 11's cards closed Done with nothing made: #2158 ("September numbers card for
Instagram") with a one-line caption, and paperwork cards with instructions for the owner.
One rule (``services.done_needs_an_artifact``) says what a card lacks: a social post card
needs a post its agent saved that has rendered, a document card (and a card Auto filed from
chat to make a document) a Deliverable on the card. A card that asks for nothing (a question
answered in text), a Playbook's card and a mission's move as before.

Who moves the card decides what the rule does (#0065: the owner's own Approve was refused on a
card whose post Auto had made, and he had no way to close it):

* Auto's status tool, and any caller that is not a person, is refused: the refusal says what is
  missing and the card stays where it was;
* a person (the board's Approve, its drag and its PATCH, on a Clerk sign-in or a paired phone
  or browser; and the owner's click on the chat's approval card) always closes the card, and the
  card carries a note saying what was missing. A request with no credential is not a person,
  the local edition's included: an agent's ``http_request`` to localhost arrives that way.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from services import done_needs_an_artifact as dna
from services import social_ticket_post as stp
from services.ticket_numbers import ticket_label
from tests.test_f094_a_step_card_ends_on_its_sessions_outcome import quiet as _quiet

quiet = _quiet  # the board's fan-out on an approval is its own suites' business

STARTED = datetime(2026, 10, 7, 1, 8, 0, tzinfo=timezone.utc)
SOCIAL = ("September numbers card for Instagram",
          "A stats card: 1,240 bags roasted, 412 subscribers, 31 wholesale cafés. Draft only.")
DOCUMENT = ("Welcome letter for Rosa", "Write a welcome letter to Rosa at Lantern Kitchen for the wholesale club.")
FROM_CHAT = ("Lantern Kitchen, September", "OBJECTIVE: the September statement for Lantern Kitchen.\n"
             "TOOLS: generate_document with template_id 7 (Harbourline Statement).")
QUESTION = ("How many bags did we roast in September?", "Read the roast log and answer in one line.")
ANSWER = "Done: the work is on the card."
NO_LETTER_NOTE = "Closed by you without a letter: no Deliverable is linked to this card."
NO_POST_NOTE = "Closed by you without a Socials post saved for this card."
CLOSED_BY_YOU = "Closed by you"
CLICKER = "user:clicker-7"                      # who the owner's click signs as (owner_only._clicker)
STATUS_TOOL = "platform_update_task_status"


# ── which cards need what (pure) ─────────────────────────────────────────────

@pytest.mark.parametrize("brief, kind, what", [
    (SOCIAL, dna.POST, "a Socials post"),
    (DOCUMENT, dna.DELIVERABLE, "a letter"),
    (("Invoice", "Draft an invoice for Salt Kitchen: 8 kg Harbour Blend."), dna.DELIVERABLE, "an invoice"),
    (FROM_CHAT, dna.DELIVERABLE, dna.A_DOCUMENT),
])
def test_a_cards_brief_says_what_it_is_for(brief, kind, what):
    need = dna.artifact_need(NS(title=brief[0], description=brief[1], source_type="user"))
    assert need == dna.ArtifactNeed(kind, what)


@pytest.mark.parametrize("title, description, source_type", [
    (*QUESTION, "user"),
    ("Summarize last week's Instagram posts", "Which did best?", "user"),
    ("Recipe: Weekly Instagram posts", "The Playbook's run.", "recipe"),     # its engine files the work
    ("Draft the offer letter", "Step 2 of the Christmas box mission.", "orchestration_task"),
    (*DOCUMENT, "mission"),
])
def test_a_card_that_asks_for_nothing_or_that_an_engine_runs_needs_nothing(title, description, source_type):
    assert dna.artifact_need(NS(title=title, description=description, source_type=source_type)) is None


# ── the one rule, on fakes ───────────────────────────────────────────────────

class _Db:
    """A session whose deliverables lookup finds ``found``; records what it asked."""

    def __init__(self, found):
        self.found, self.asked = found, []

    def execute(self, statement, params):
        self.asked.append((str(statement), params))
        return NS(first=lambda: (1,) if self.found else None)


def _card(brief, **over):
    base = dict(id=2158, workspace_id=UUID("febae41b-374b-4580-a5ef-f698bdd382e4"), title=brief[0],
                description=brief[1], status="review", source_type="user", workspace_seq=2158,
                assigned_agent_id=348, started_at=STARTED, created_at=STARTED)
    return NS(**{**base, **over})


def _post(**over):
    base = dict(id=uuid4(), title="September at a Glance", status="draft", media={}, template_id=uuid4(),
                format="image", review_log=[])
    return NS(**{**base, **over})


@pytest.fixture
def socials(monkeypatch):
    """Socials on, and the posts the card's agent saved during its run."""
    saved, asked = [], []
    monkeypatch.setattr(stp, "socials_on", lambda db, ws: NS(id=ws))

    def _runs_posts(db, ws, actor, since):
        asked.append((ws, actor, since))
        return list(saved)

    monkeypatch.setattr(stp, "runs_posts", _runs_posts)
    return saved, asked


def test_a_document_card_with_no_deliverable_stays_in_review_and_says_so():
    db = _Db(found=False)
    said = dna.missing_artifact(db, _card(DOCUMENT))
    assert said == ("Ticket #2158 can't be Done: it asks for a letter, and no Deliverable is linked to it. "
                    "It stays in Review. Send it back so its agent makes a letter, then approve the card.")
    statement, params = db.asked[0]
    assert "deleted_at IS NULL" in statement and "workspace_id = CAST(:workspace_id AS uuid)" in statement
    assert params == {"workspace_id": "febae41b-374b-4580-a5ef-f698bdd382e4", "source_type": "task",
                      "source_id": "2158"}


def test_a_document_card_with_its_deliverable_and_a_question_card_go_to_done():
    assert dna.missing_artifact(_Db(found=True), _card(DOCUMENT)) is None
    assert dna.missing_artifact(_Db(found=True), _card(FROM_CHAT)) is None
    nothing_asked = _Db(found=False)
    assert dna.missing_artifact(nothing_asked, _card(QUESTION)) is None
    assert nothing_asked.asked == []                                  # a question's card reads nothing


def test_a_social_card_with_no_post_says_none_was_saved(socials):
    _saved, asked = socials
    said = dna.missing_artifact(_Db(found=False), _card(SOCIAL))
    assert said.startswith("Ticket #2158 can't be Done: it asks for a Socials post, and its agent has saved no post")
    assert "It stays in Review." in said
    assert asked[0][1] == "agent:348"                                 # the actor its Socials tools write


@pytest.mark.parametrize("post, why", [
    (_post(), dna.DID_NOT_RENDER),
    (_post(status="failed"), dna.DID_NOT_RENDER),
    (_post(status="rendering"), dna.STILL_RENDERING),
])
def test_a_social_cards_post_that_has_not_rendered_is_named(socials, post, why):
    socials[0].append(post)
    said = dna.missing_artifact(_Db(found=False), _card(SOCIAL))
    assert f'its Socials post "September at a Glance" {why}' in said


@pytest.mark.parametrize("post", [
    _post(media={"4:5": [{"deliverable_id": "d1"}]}),                   # rendered
    _post(template_id=None, format="text"),                             # a text post needs no render
    _post(status="approved"),                                           # a person has it
])
def test_a_social_card_whose_post_is_made_goes_to_done(socials, post):
    socials[0].append(post)
    assert dna.missing_artifact(_Db(found=False), _card(SOCIAL)) is None


def test_with_socials_off_a_social_card_is_not_held(monkeypatch):
    monkeypatch.setattr(stp, "socials_on", lambda db, ws: None)
    assert dna.missing_artifact(_Db(found=False), _card(SOCIAL)) is None


def test_a_title_with_braces_is_said_as_it_is(socials):
    socials[0].append(_post(title="{label} {stays} September"))
    assert '"{label} {stays} September" has not rendered' in dna.missing_artifact(_Db(False), _card(SOCIAL))


def test_only_a_move_to_done_is_judged():
    db = _Db(found=False)
    assert dna.done_refusal(db, _card(DOCUMENT), "review") is None and db.asked == []
    assert dna.done_refusal(db, _card(DOCUMENT), "done") is not None


# ── #0065: who moves the card (pure) ─────────────────────────────────────────

@pytest.mark.parametrize("edition", ["local", "saas"])
def test_a_person_is_told_by_the_lane_the_request_came_in_on(monkeypatch, edition):
    from config import config as app_config

    monkeypatch.setattr(app_config, "AUTH_EDITION", edition)
    for person in ("clerk", "device"):                                 # a sign-in, or a paired phone or browser
        assert dna.a_persons_request(NS(auth_type=person, user_id="2")) is True
    for not_one in (None, NS(user_id="2"), NS(auth_type="api_key"), NS(auth_type="sdk_key"),
                    NS(auth_type="anonymous", user_id="2")):           # no credential: an agent can send that
        assert dna.a_persons_request(not_one) is False


def test_a_persons_move_to_done_is_never_refused_and_any_other_is():
    db = _Db(found=False)
    assert dna.done_refusal(db, _card(DOCUMENT), "done", ctx=NS(auth_type="clerk")) is None and db.asked == []
    said = dna.done_refusal(db, _card(DOCUMENT), "done", ctx=NS(user_id="2"))
    assert said == dna.missing_artifact(_Db(found=False), _card(DOCUMENT))
    assert said.startswith("Ticket #2158 can't be Done: it asks for a letter")


def test_what_a_persons_close_notes_on_a_document_card():
    assert dna.closed_without(_Db(found=False), _card(DOCUMENT)) == NO_LETTER_NOTE
    assert dna.closed_without(_Db(found=False), _card(FROM_CHAT)) == dna.CLOSED_WITHOUT_A_DELIVERABLE.format(
        what=dna.A_DOCUMENT)
    assert dna.closed_without(_Db(found=True), _card(DOCUMENT)) is None
    nothing_asked = _Db(found=False)
    assert dna.closed_without(nothing_asked, _card(QUESTION)) is None and nothing_asked.asked == []


def test_what_a_persons_close_notes_on_a_social_card_with_no_post(socials):
    assert dna.closed_without(_Db(found=False), _card(SOCIAL)) == NO_POST_NOTE
    assert NO_POST_NOTE == dna.CLOSED_WITHOUT_A_POST.format(what="a Socials post")


@pytest.mark.parametrize("post, why", [
    (_post(), dna.DID_NOT_RENDER),
    (_post(status="rendering"), dna.STILL_RENDERING),
])
def test_a_persons_close_names_the_post_that_was_not_made(socials, post, why):
    socials[0].append(post)
    assert dna.closed_without(_Db(found=False), _card(SOCIAL)) == (
        f'Closed by you while its Socials post "September at a Glance" {why}.')


def test_a_persons_close_of_a_card_with_its_post_notes_nothing(socials):
    socials[0].append(_post(media={"4:5": [{"deliverable_id": "d1"}]}))
    assert dna.closed_without(_Db(found=False), _card(SOCIAL)) is None


# ── through the tool, the board's Approve, its drag and its PATCH (real schema) ──

@pytest.fixture
def board(db_session, seed_workspace, monkeypatch):
    """A workspace with an agent; ``card`` files a ticket with a result, ``deliverable``
    registers a Deliverable on a card the way a board run does."""
    from core.database import database
    from core.models import Agent
    from core.models.core import BoardTask

    @contextmanager
    def this_session():                    # the ticket's change notes, in the test's own transaction
        yield db_session

    async def _not_filed(db, workspace_id, task):
        return None

    monkeypatch.setattr(database, "get_db_session", this_session)
    monkeypatch.setattr("services.report_knowledge.file_done_ticket", _not_filed)
    ws = UUID(seed_workspace())
    agent = Agent(name="Social Media Director", agent_type="chatbot", description="", status="active",
                  configuration={}, model_config=None, workspace_id=ws, created_by="test",
                  owner_type="workspace", owner_id=str(ws))
    db_session.add(agent)
    db_session.flush()

    def card(brief, status="review"):
        task = BoardTask(workspace_id=ws, title=brief[0], description=brief[1], status=status, priority="medium",
                         review_mode="human", source_type="user", assigned_agent_id=agent.id, result=ANSWER,
                         started_at=STARTED)
        db_session.add(task)
        db_session.flush()
        return task

    def deliverable(task, workspace=None):
        db_session.execute(text(
            "INSERT INTO deliverables (workspace_id, source_type, source_id, artifact_type, title, file_path) "
            "VALUES (CAST(:ws AS uuid), 'task', :card, 'document', 'Welcome letter', :path)"),
            {"ws": str(workspace or ws), "card": str(task.id), "path": f"documents/{uuid4().hex}.pdf"})
        db_session.flush()

    return NS(ws=ws, card=card, deliverable=deliverable, agent=agent)


def _move(db, ws, task, status="done"):
    from modules.tools.discovery.handlers_board_task_done import update_board_task_status

    return asyncio.run(update_board_task_status(db, ws, {"task_id": task.id, "status": status, "_user_id": "2"}))


def _owner(ws):
    return NS(workspace_id=ws, user_id="2", auth_type="anonymous", user=NS(id="2"))


PERSONS = ("clerk", "device")


def _person(ws, lane="clerk"):
    """A person's request: a Clerk sign-in, or a paired phone or browser (``device``)."""
    return NS(workspace_id=ws, user_id="2", auth_type=lane, user=NS(id="2"))


def _not_a_person(ws):
    """A call with no person's lane: the context ``mission_step_verdicts`` hands Approve for Auto."""
    return NS(workspace_id=ws, user_id="2")


def _an_env_key(ws):
    return NS(workspace_id=ws, user_id=None, auth_type="api_key", user=NS(id="api_key"))


def _notes(db, task):
    """The card's notes as its view lists them: who, and what."""
    db.expire_all()
    db.refresh(task)
    return [(n["by"], n["note"]) for n in (task.runtime_ref or {}).get("session_notes") or []]


def _body(payload):
    async def _json():
        return payload
    return NS(json=_json)


def test_the_tool_refuses_a_document_card_with_no_deliverable(db_session, board):
    task = board.card(DOCUMENT)
    out = _move(db_session, board.ws, task)
    db_session.refresh(task)
    assert out["success"] is False and "no Deliverable is linked to it" in out["error"]
    assert "It stays in Review." in out["error"] and task.status == "review"


def test_the_tool_moves_a_document_card_with_its_deliverable_and_a_question_card(db_session, board):
    letter, question = board.card(DOCUMENT), board.card(QUESTION)
    board.deliverable(letter)
    assert _move(db_session, board.ws, letter)["success"] is True
    assert _move(db_session, board.ws, question)["success"] is True
    db_session.refresh(letter)
    db_session.refresh(question)
    assert (letter.status, question.status) == ("done", "done")
    for task in (letter, question):                                   # the tool notes its move, never a lack
        assert not [note for _, note in _notes(db_session, task) if note.startswith(CLOSED_BY_YOU)]


def test_another_workspaces_deliverable_is_never_this_cards(db_session, seed_workspace, board):
    task = board.card(DOCUMENT)
    board.deliverable(task, workspace=UUID(seed_workspace()))
    assert _move(db_session, board.ws, task)["success"] is False


def test_the_tool_refuses_a_social_card_with_no_rendered_post_and_moves_one_with(db_session, board, socials):
    task = board.card(SOCIAL)
    out = _move(db_session, board.ws, task)
    assert out["success"] is False and "saved no post for it in Socials" in out["error"]
    socials[0].append(_post(media={"4:5": [{"deliverable_id": "d1"}]}))
    assert _move(db_session, board.ws, task)["success"] is True


def test_a_bulk_move_refuses_only_the_card_without_its_artifact(db_session, board):
    from modules.tools.discovery.handlers_board_task_done import update_board_task_status

    letter, question = board.card(DOCUMENT), board.card(QUESTION)
    out = asyncio.run(update_board_task_status(db_session, board.ws, {
        "task_ids": [letter.id, question.id], "status": "done", "_user_id": "2"}))
    db_session.refresh(letter)
    db_session.refresh(question)
    assert (letter.status, question.status) == ("review", "done")
    assert out["success"] is False and out["updated"] == [question.id]
    assert "no Deliverable is linked to it" in out["failed"][0]["error"]


@pytest.mark.parametrize("lane", PERSONS)
def test_a_persons_approve_closes_a_document_card_with_no_deliverable_and_says_so_on_it(db_session, board, quiet,
                                                                                         lane):
    from api.board_tasks import approve_task

    task = board.card(DOCUMENT)
    out = asyncio.run(approve_task(task.id, _body({}), ctx=_person(board.ws, lane), db=db_session))
    assert out["success"] is True and out["status"] == "done"
    assert _notes(db_session, task) == [("you", NO_LETTER_NOTE)]
    assert task.status == "done" and task.completed_at is not None


def test_a_persons_approve_closes_a_social_card_with_no_post_and_says_so_on_it(db_session, board, quiet, socials):
    from api.board_tasks import approve_task

    task = board.card(SOCIAL)                                   # #0065: Auto made the post, in the owner's chat
    out = asyncio.run(approve_task(task.id, _body({}), ctx=_person(board.ws), db=db_session))
    assert out["success"] is True and out["status"] == "done"
    assert _notes(db_session, task) == [("you", NO_POST_NOTE)]


def test_the_local_operators_approve_with_no_credential_is_refused_as_today(db_session, board, quiet, monkeypatch):
    """An unpaired local browser and an agent's ``http_request`` to localhost look the same: no
    credential. The owner closes a card without its artifact from a paired browser or phone."""
    from api.board_tasks import approve_task
    from config import config as app_config

    monkeypatch.setattr(app_config, "AUTH_EDITION", "local")
    task = board.card(DOCUMENT)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(approve_task(task.id, _body({}), ctx=_owner(board.ws), db=db_session))
    db_session.refresh(task)
    assert refused.value.status_code == 409 and refused.value.detail == dna.NO_DELIVERABLE.format(
        label=ticket_label(task, capital=True), what="a letter", stays=dna.STAYS_IN_REVIEW)
    assert task.status == "review" and task.completed_at is None and _notes(db_session, task) == []


@pytest.mark.parametrize("route", ["approve_task", "update_task_status", "update_task"])
def test_an_agents_http_request_to_the_local_api_is_refused_and_moves_nothing(db_session, board, quiet, monkeypatch,
                                                                             route):
    """The agent's ``http_request`` tool may call localhost (``exec_shell``), with its own mark
    (``agent_request.AGENT_REQUEST_HEADER``) and no credential: the local edition's anonymous lane."""
    from api import board_tasks as bt
    from config import config as app_config
    from modules.tools.execution.agent_request import AGENT_REQUEST_HEADER, AGENT_REQUEST_MARK

    monkeypatch.setattr(app_config, "AUTH_EDITION", "local")
    task = board.card(DOCUMENT, status="review" if route == "approve_task" else "blocked")
    asked = NS(json=_body({} if route == "approve_task" else {"status": "done"}).json,
               headers={AGENT_REQUEST_HEADER: AGENT_REQUEST_MARK})
    was = task.status
    with pytest.raises(HTTPException) as refused:
        asyncio.run(getattr(bt, route)(task.id, asked, ctx=_owner(board.ws), db=db_session))
    db_session.refresh(task)
    assert refused.value.status_code == 409 and "no Deliverable is linked to it" in refused.value.detail
    assert task.status == was and _notes(db_session, task) == []


def test_a_deliverable_the_approvals_own_action_files_is_not_said_missing(db_session, board, quiet, monkeypatch):
    """What the card lacked is read once the approval's action has run (``approve_task``)."""
    from api import board_tasks as bt

    task = board.card(DOCUMENT)
    task.planning_data = {"approval_action": {"type": "publish_blog", "post_id": "p1"}}
    db_session.flush()

    async def _files_the_letter(db, ctx, action):
        board.deliverable(task)
        return {"type": "publish_blog", "title": "Welcome letter"}

    monkeypatch.setattr(bt, "_run_approval_action", _files_the_letter)
    out = asyncio.run(bt.approve_task(task.id, _body({}), ctx=_person(board.ws), db=db_session))
    assert out["status"] == "done" and _notes(db_session, task) == []


def test_a_persons_approve_note_is_kept_beside_what_the_card_lacked(db_session, board, quiet):
    from api.board_tasks import approve_task

    task = board.card(DOCUMENT)
    asyncio.run(approve_task(task.id, _body({"note": "I sent it myself."}), ctx=_person(board.ws), db=db_session))
    assert _notes(db_session, task) == [("you", NO_LETTER_NOTE), ("you", "Approved: I sent it myself.")]


def test_what_the_card_lacked_is_kept_only_while_the_approval_stands(db_session, board):
    from services.ticket_verdict import record_approval

    task = board.card(DOCUMENT, status="done")
    task.completed_at = STARTED
    db_session.flush()
    moved_since = datetime(2026, 10, 8, 9, 0, 0, tzinfo=timezone.utc)   # another decision's time, not this one's
    assert record_approval(db_session, task.id, workspace_id=board.ws, decided_at=moved_since, action_result=None,
                           missing=NO_LETTER_NOTE) is False
    assert _notes(db_session, task) == []
    assert record_approval(db_session, task.id, workspace_id=board.ws, decided_at=STARTED, action_result=None,
                           missing=NO_LETTER_NOTE) is True
    assert _notes(db_session, task) == [("you", NO_LETTER_NOTE)]


@pytest.mark.parametrize("brief, refusal", [
    (DOCUMENT, lambda label: dna.NO_DELIVERABLE.format(label=label, what="a letter", stays=dna.STAYS_IN_REVIEW)),
    (SOCIAL, lambda label: dna.NO_POST.format(label=label, stays=dna.STAYS_IN_REVIEW)),
])
def test_approve_without_a_persons_lane_is_refused_as_today(db_session, board, quiet, socials, brief, refusal):
    from api.board_tasks import approve_task

    task = board.card(brief)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(approve_task(task.id, _body({}), ctx=_not_a_person(board.ws), db=db_session))
    db_session.refresh(task)
    assert refused.value.status_code == 409 and refused.value.detail == refusal(ticket_label(task, capital=True))
    assert task.status == "review" and task.completed_at is None and _notes(db_session, task) == []


@pytest.mark.parametrize("brief, refusal", [
    (DOCUMENT, lambda label: dna.NO_DELIVERABLE.format(label=label, what="a letter", stays=dna.STAYS_IN_REVIEW)),
    (SOCIAL, lambda label: dna.NO_POST.format(label=label, stays=dna.STAYS_IN_REVIEW)),
])
def test_autos_status_tool_is_refused_on_the_same_cards_in_todays_words(db_session, board, socials, brief, refusal):
    from modules.tools.discovery.handlers_board_task_done import update_board_task_status

    task = board.card(brief)
    out = asyncio.run(update_board_task_status(db_session, board.ws, {
        "task_id": task.id, "status": "done", "_user_id": "2"}))           # no click: Auto's call alone
    db_session.refresh(task)
    assert out["success"] is False and out["error"] == refusal(ticket_label(task, capital=True))
    assert task.status == "review" and _notes(db_session, task) == []


@pytest.mark.parametrize("brief, said", [(DOCUMENT, NO_LETTER_NOTE), (SOCIAL, NO_POST_NOTE)])
def test_a_move_the_owner_clicked_closes_the_same_cards_and_says_what_they_lacked(db_session, board, socials, brief,
                                                                                  said):
    """#0065: the owner's click on the chat's card is a person's close (``owner_only.signed_by``)."""
    from modules.tools.discovery.handlers_board_task_done import update_board_task_status

    task = board.card(brief)
    out = asyncio.run(update_board_task_status(db_session, board.ws, {
        "task_id": task.id, "status": "done", "_user_id": "2", "_clicked_by": CLICKER}))
    notes = _notes(db_session, task)
    assert out["success"] is True and task.status == "done"
    assert (CLICKER, said) in notes and len([n for _, n in notes if n.startswith(CLOSED_BY_YOU)]) == 1


def test_a_clicked_move_of_a_card_with_its_artifact_says_nothing_of_it(db_session, board):
    from modules.tools.discovery.handlers_board_task_done import update_board_task_status

    task = board.card(DOCUMENT)
    board.deliverable(task)
    out = asyncio.run(update_board_task_status(db_session, board.ws, {
        "task_id": task.id, "status": "done", "_user_id": "2", "_clicked_by": CLICKER}))
    assert out["success"] is True
    assert not [note for _, note in _notes(db_session, task) if note.startswith(CLOSED_BY_YOU)]


def test_a_click_on_a_card_already_done_notes_nothing_again(db_session, board):
    """A repeat click (or a retry on a click given back) closes nothing, so it says nothing."""
    from modules.tools.discovery.handlers_board_task_done import update_board_task_status

    task = board.card(DOCUMENT, status="done")
    asyncio.run(update_board_task_status(db_session, board.ws, {
        "task_id": task.id, "status": "done", "_user_id": "2", "_clicked_by": CLICKER}))
    assert not [note for _, note in _notes(db_session, task) if note.startswith(CLOSED_BY_YOU)]


# ── the chat: Auto's call asks, the owner's click closes it (Auto's executor) ────

@pytest.fixture
def autos_chat(db_session, board):
    """Auto's executor on the board's workspace, as the chat runs it: Auto is the actor the
    hierarchy check reads (``_agent_id``), and the dial is on (an owner-only call still asks)."""
    from core.models import Agent
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    auto = Agent(name="Auto", agent_type="system", description="", status="active", configuration={},
                 workspace_id=board.ws, created_by="test", owner_type="workspace", owner_id=str(board.ws),
                 is_system_agent=True)
    db_session.add(auto)
    db_session.flush()
    executor = PlatformActionExecutor(db_session, board.ws)
    executor._full_autonomy = lambda: True

    def call(params, caller_context):
        with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
            return asyncio.run(executor.execute(STATUS_TOOL, {**params, "_agent_id": auto.id}, caller_context))
    return call


def _owners_chat():
    return {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}


def _number(task):
    return f"#{task.workspace_seq:04d}"


def test_in_the_owners_chat_autos_call_asks_and_the_owners_click_closes_it(db_session, board, quiet, autos_chat):
    from core.models.approval_grants import ApprovalGrant
    from core.services.approval_grants import grant_grant

    task = board.card(DOCUMENT)
    params, chat = {"task_id": _number(task), "status": "done"}, _owners_chat()
    asked = autos_chat(params, chat)
    db_session.refresh(task)
    assert asked["requires_confirmation"] is True and task.status == "review"   # Auto's call alone moves nothing
    grant_grant(db_session.get(ApprovalGrant, asked["grant_id"]), granted_by=CLICKER)
    db_session.flush()

    ran = autos_chat(params, chat)                                         # the click's run of the same call
    notes = _notes(db_session, task)
    assert ran["success"] is True and ran["approved_via_grant_id"] == asked["grant_id"]
    assert task.status == "done" and (CLICKER, NO_LETTER_NOTE) in notes


@pytest.mark.parametrize("signer", [{}, {"_clicked_by": CLICKER}], ids=["autos-call", "a-signer-the-model-sent"])
def test_autos_own_call_with_no_one_driving_it_is_refused_whatever_it_says(db_session, board, quiet, autos_chat,
                                                                          signer):
    """An agent's run, a playbook step or a heartbeat: no click, so no person; a ``_clicked_by``
    the model sends is dropped before the handler (``owner_only._without_a_signer``)."""
    task = board.card(DOCUMENT)
    out = autos_chat({"task_id": _number(task), "status": "done", **signer}, None)
    db_session.refresh(task)
    assert out["success"] is False and out["error"] == dna.NO_DELIVERABLE.format(
        label=ticket_label(task, capital=True), what="a letter", stays=dna.STAYS_IN_REVIEW)
    assert task.status == "review" and _notes(db_session, task) == []


def test_approve_closes_a_card_with_its_artifact_and_a_question_card(db_session, board, quiet):
    from api.board_tasks import approve_task

    letter, question = board.card(DOCUMENT), board.card(QUESTION)
    board.deliverable(letter)
    for task in (letter, question):
        out = asyncio.run(approve_task(task.id, _body({}), ctx=_owner(board.ws), db=db_session))
        assert out["status"] == "done"
        assert _notes(db_session, task) == []                          # nothing was missing, so nothing is said


@pytest.mark.parametrize("ctx_of", [_person, _not_a_person], ids=["a-person", "not-a-person"])
def test_a_card_with_its_artifact_closes_for_anyone_with_no_note(db_session, board, quiet, ctx_of):
    from api import board_tasks as bt

    letter, dragged, question = board.card(DOCUMENT), board.card(DOCUMENT, status="blocked"), board.card(QUESTION)
    board.deliverable(letter)
    board.deliverable(dragged)
    for task in (letter, question):
        assert asyncio.run(bt.approve_task(task.id, _body({}), ctx=ctx_of(board.ws), db=db_session))["status"] == "done"
    out = asyncio.run(bt.update_task_status(dragged.id, _body({"status": "done"}), ctx=ctx_of(board.ws), db=db_session))
    assert out["status"] == "done"
    assert [_notes(db_session, task) for task in (letter, dragged, question)] == [[], [], []]


@pytest.mark.parametrize("lane", PERSONS)
@pytest.mark.parametrize("brief, said", [(DOCUMENT, NO_LETTER_NOTE), (SOCIAL, NO_POST_NOTE)])
@pytest.mark.parametrize("route", ["update_task_status", "update_task"])
def test_a_persons_drag_or_patch_to_done_closes_it_and_says_what_it_lacked(db_session, board, quiet, socials, route,
                                                                           brief, said, lane):
    from api import board_tasks as bt

    task = board.card(brief, status="blocked")                   # a drag from Review is Approve's (F259)
    out = asyncio.run(getattr(bt, route)(task.id, _body({"status": "done"}), ctx=_person(board.ws, lane),
                                         db=db_session))
    assert out["status"] == "done"
    assert _notes(db_session, task) == [("you", said)] and task.status == "done"


def test_a_persons_patch_keeps_its_own_note_beside_what_the_card_lacked(db_session, board, quiet):
    from api.board_tasks import update_task

    task = board.card(DOCUMENT, status="blocked")
    asyncio.run(update_task(task.id, _body({"status": "done", "note": "Posted it from my phone."}),
                            ctx=_person(board.ws, "device"), db=db_session))
    assert _notes(db_session, task) == [("you", "Posted it from my phone."), ("you", NO_LETTER_NOTE)]


@pytest.mark.parametrize("ctx_of", [_not_a_person, _an_env_key], ids=["no-lane", "env-key"])
@pytest.mark.parametrize("route", ["update_task_status", "update_task"])
def test_a_drag_or_patch_that_is_not_a_persons_keeps_the_rule(db_session, board, quiet, route, ctx_of):
    from api import board_tasks as bt

    task = board.card(DOCUMENT, status="blocked")
    with pytest.raises(HTTPException) as refused:
        asyncio.run(getattr(bt, route)(task.id, _body({"status": "done"}), ctx=ctx_of(board.ws), db=db_session))
    db_session.refresh(task)
    assert refused.value.status_code == 409 and "It stays where it is." in refused.value.detail
    assert task.status == "blocked" and _notes(db_session, task) == []
