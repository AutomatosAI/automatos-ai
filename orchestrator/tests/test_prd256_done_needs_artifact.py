"""PRD-256 US-005 (F380): a card moves to Done only when the thing it was for exists.

Night 11's cards closed Done with nothing made: #2158 ("September numbers card for
Instagram") with a one-line caption, and paperwork cards with instructions for the owner.
Now Auto's status tool, the board's Approve, its drag and its PATCH all ask one rule
(``services.done_needs_an_artifact.missing_artifact``): a social post card needs a post
its agent saved that has rendered, a document card (and a card Auto filed from chat to
make a document) a Deliverable on the card. The refusal says what is missing and the card
stays where it was. A card that asks for nothing (a question answered in text), a
Playbook's card and a mission's move as before.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from services import done_needs_an_artifact as dna
from services import social_ticket_post as stp
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


def test_approve_refuses_a_document_card_with_no_deliverable(db_session, board, quiet):
    from api.board_tasks import approve_task

    task = board.card(DOCUMENT)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(approve_task(task.id, _body({}), ctx=_owner(board.ws), db=db_session))
    db_session.refresh(task)
    assert refused.value.status_code == 409 and "no Deliverable is linked to it" in refused.value.detail
    assert task.status == "review" and task.completed_at is None


def test_approve_refuses_a_social_card_with_no_post(db_session, board, quiet, socials):
    from api.board_tasks import approve_task

    task = board.card(SOCIAL)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(approve_task(task.id, _body({}), ctx=_owner(board.ws), db=db_session))
    db_session.refresh(task)
    assert refused.value.status_code == 409 and "Socials post" in refused.value.detail
    assert task.status == "review"


def test_approve_closes_a_card_with_its_artifact_and_a_question_card(db_session, board, quiet):
    from api.board_tasks import approve_task

    letter, question = board.card(DOCUMENT), board.card(QUESTION)
    board.deliverable(letter)
    for task in (letter, question):
        out = asyncio.run(approve_task(task.id, _body({}), ctx=_owner(board.ws), db=db_session))
        assert out["status"] == "done"


@pytest.mark.parametrize("route", ["update_task_status", "update_task"])
def test_a_drag_or_patch_to_done_keeps_the_same_rule(db_session, board, quiet, route):
    from api import board_tasks as bt

    task = board.card(DOCUMENT, status="blocked")               # a drag from Review is Approve's (F259)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(getattr(bt, route)(task.id, _body({"status": "done"}), ctx=_owner(board.ws), db=db_session))
    db_session.refresh(task)
    assert refused.value.status_code == 409 and "It stays where it is." in refused.value.detail
    assert task.status == "blocked"
