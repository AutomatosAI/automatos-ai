"""P256-FIX-SOC-4 (F431-a, 14 Oct socials day): the ticket note is a fact for the owner, never an order.

The platform's F380 check closed a run that saved no post with "Make the post with
platform_create_social_post", which read as the agent's own sign-off, and which a Reject fed back
to the redo's agent as "Your last attempt:". #2962's brief named an existing post
(f1b1719a-…) and said "do not create a new one"; the agent rightly left it alone, and the note
told it to make a new one. The Done check refused that card with "make the post in the Socials
tab" even after the named post had rendered.

Now the note is a neutral fact. A brief that names an existing post by its id gets that post's
title, id and status and what it still needs (its template's empty fields included), and a card
naming a post that has rendered may move to Done.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

from core.models.socials import SocialPost
from core.services import ticket_reasons as tr
from modules.documents.template_service import DocumentTemplateService
from modules.socials import service
from services import done_needs_an_artifact as dna
from services import social_ticket_post as stp
from services import ticket_redo
from tests.test_f380_a_a_social_ticket_is_done_only_with_its_post import finish as _finish

finish = _finish  # F380's completion writer: a Social Media Director ticket ending through finalize_board_task_run

WS = UUID("febae41b-374b-4580-a5ef-f698bdd382e4")
POST_ID = UUID("f1b1719a-14df-44ac-bd9d-93d64cef573a")
TITLE_2962 = "Finish the Guji stats card"
BRIEF_2962 = (f"The stats card post {POST_ID} is in Socials already: fill in its numbers and render it. "
              "Do not create a new one.")
ANSWER_2962 = "I've looked at the stats card; I'll leave it as it is."
TEMPLATE = NS(id=uuid4(), format="social_image", blocks={
    "html": "<div>{{ stat_value }} {{ stat_label }}</div>",
    "variables_schema": {"stat_value": {"type": "text"}, "stat_label": {"type": "text"},
                         "footer": {"type": "text", "default": "Harbourline"}},
})


def _named_post(**fields):
    base = dict(id=POST_ID, workspace_id=WS, created_by="user:owner", title="Guji stats card", status="draft",
                format="image", template_id=TEMPLATE.id, copy={"base": "Our Guji, by the numbers."},
                variables={"stat_label": {"value": "bags roasted", "claim": False}}, sources={}, media={},
                review_log=[], content_hash="h" * 64, override_unsourced=False)
    return SocialPost(**{**base, **fields})


@pytest.fixture
def named(monkeypatch):
    """The workspace's posts by id, and its template, as the note reads them."""
    posts = {}
    monkeypatch.setattr(service, "get_post", lambda db, ws, post_id: posts.get(post_id) if ws == WS else None)
    monkeypatch.setattr(DocumentTemplateService, "get_template",
                        lambda self, template_id, ws: TEMPLATE if template_id == TEMPLATE.id else None)
    return posts


# ── (1) the note says a fact, and no tool ────────────────────────────────────────────

def test_the_no_post_note_is_a_fact_with_no_tool_in_it():
    note = stp.NO_POST_NOTE.format(image="")

    assert note.startswith(tr.NOTHING_DONE_NOTE_PREFIX)
    assert "no Socials post was saved in this run" in note
    assert "platform_create_social_post" not in note and "Make the post" not in note
    assert "platform_" not in stp.NAMED_POST_NOTE + stp.NAMED_POST + stp.NAMED_POST_DONE


# ── (2) #2962: a ticket naming an existing post, a run that changed nothing ──────────

def test_2962s_ticket_names_its_post_and_what_it_still_needs(finish, named):
    named[POST_ID] = _named_post()

    task = finish(ANSWER_2962, title=TITLE_2962, description=BRIEF_2962)

    assert task.status == "review" and tr.review_reason(task) == tr.NOTHING_DONE
    note = task.result[len(ANSWER_2962):]
    assert '"Guji stats card"' in note and str(POST_ID) in note and "it is draft" in note
    assert "stat_value" in note and "stat_label" not in note and "footer" not in note   # only what is empty
    assert "a render: nothing is rendered yet" in note
    assert "create" not in note.lower() and "platform_" not in note


def test_a_named_post_of_another_workspace_is_never_read(finish, named):
    task = finish(ANSWER_2962, title=TITLE_2962, description=BRIEF_2962)   # get_post finds nothing here

    assert task.result.endswith(stp.NO_POST_NOTE.format(image=""))


def test_the_template_cannot_be_read_and_the_fields_are_left_out(finish, named, monkeypatch):
    named[POST_ID] = _named_post()
    monkeypatch.setattr(DocumentTemplateService, "get_template", lambda self, template_id, ws: None)

    task = finish(ANSWER_2962, title=TITLE_2962, description=BRIEF_2962)

    assert task.status == "review" and str(POST_ID) in task.result and "fields filled in" not in task.result


# ── (3) a Reject feeds the note back as a fact ───────────────────────────────────────

def test_after_reject_the_redo_reads_the_note_with_no_tool_instruction(finish, named):
    from api.board_tasks import keep_previous_run

    named[POST_ID] = _named_post()
    task = finish(ANSWER_2962, title=TITLE_2962, description=BRIEF_2962)
    task.planning_data = None
    keep_previous_run(task, why=ticket_redo.SENT_BACK, by="user:owner", now=datetime.now(timezone.utc))
    task.review_feedback = "Fill in the number from the September roast log."

    block = ticket_redo._redo(task)

    assert "Your last attempt:" in block and str(POST_ID) in block
    assert "platform_create_social_post" not in block and "Make the post" not in block
    assert "create" not in block[block.index("Your last attempt:"):].lower()


# ── (4) the Done check reads the named post ──────────────────────────────────────────

def _card(**over):
    base = dict(id=2962, workspace_id=WS, title=TITLE_2962, description=BRIEF_2962, status="review",
                source_type="user", workspace_seq=2962, assigned_agent_id=348,
                started_at=datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc), created_at=None)
    return NS(**{**base, **over})


@pytest.fixture
def board(named, monkeypatch):
    monkeypatch.setattr(stp, "socials_on", lambda db, ws: NS(id=ws))
    monkeypatch.setattr(stp, "runs_posts", lambda db, ws, actor, since: [])
    return named


RENDERED_SINCE = [{"at": "2026-10-14T12:30:00+00:00", "by": "user:owner", "action": "render_done", "comment": None}]


def test_a_card_naming_a_post_that_has_rendered_since_may_be_done(board):
    board[POST_ID] = _named_post(status="needs_approval", media={"4:5": [{"deliverable_id": "d1"}]},
                                 review_log=RENDERED_SINCE)

    assert dna.missing_artifact(object(), _card()) is None


def test_a_post_named_as_an_example_that_nothing_touched_is_not_the_cards_work(board):
    """A brief that names a post made long ago ("one like f1b1719a-…, our best") is not done by it."""
    board[POST_ID] = _named_post(status="approved", media={"4:5": [{"deliverable_id": "d1"}]},
                                 review_log=[{"at": "2026-09-01T09:00:00+00:00", "by": "user:owner",
                                              "action": "approve", "comment": None}])

    said = dna.missing_artifact(object(), _card())

    assert said is not None and '"Guji stats card" has not changed since this card\'s run started' in said
    assert "make the post" not in said.lower()


def test_a_card_naming_a_post_still_in_draft_names_that_post(board):
    board[POST_ID] = _named_post()

    said = dna.missing_artifact(object(), _card())

    assert said.startswith("Ticket #2962 can't be Done") and '"Guji stats card" has not rendered' in said
    assert "make the post" not in said.lower()


def test_a_card_that_names_no_post_still_says_none_was_saved(board):
    said = dna.missing_artifact(object(), _card(description="A stats card: 1,240 bags roasted. Draft only."))

    assert "has saved no post" in said and "make the post in the Socials tab" in said


# ── named_posts, on the real schema: this workspace's posts only, in the order named ──

def test_named_posts_reads_this_workspaces_posts_by_the_ids_the_brief_names(db_session, seed_workspace):
    ws, other = UUID(seed_workspace()), UUID(seed_workspace())

    def _saved(workspace, title):
        post = SocialPost(workspace_id=workspace, created_by="user:owner", title=title, content_hash="h" * 64)
        db_session.add(post)
        db_session.flush()
        return post

    second, first, theirs = _saved(ws, "Second"), _saved(ws, "First"), _saved(other, "Theirs")
    task = NS(title=f"Fix {first.id}", description=f"Then {str(second.id).upper()}, not {theirs.id} or {uuid4()}.")

    assert [post.title for post in stp.named_posts(db_session, ws, task)] == ["First", "Second"]
    assert stp.named_posts(db_session, ws, NS(title="No ids here", description=None)) == []
