"""F180, reopened (2 Oct, night 6 iterations 11-14): when the owner says the work
waits for them, the ticket Auto files waits for them, whatever mode Auto passed.

#1205 and #1206 ("Drafts only - nothing gets sent without me") and #1209 ("set to
wait for me") were filed with no review mode. They closed Done on their own while
Auto told the owner they would wait. #1214's first try said "human_review" and was
refused. The persona: "lets it finish without you, however clearly you said you
wanted to see it first."
"""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

from modules.tools.discovery.handlers_board_task_review import CLOSES_BY_ITSELF, OWNER_ASKED, WAITS_FOR_OWNER
from services.owner_review import asks_to_see_it_first

NIGHT_6_ASKS = [
    "Drafts only - nothing gets sent without me.",
    "Yes please. And remember: it waits for me before it’s done.",
    "Do it the same way: one job on the board, on the Content Creator, set to wait for me. No mission.",
    "Put it on the board for whichever helper does it, set to wait for me, because I want to look at it "
    "before I print it.",
    "Please do it now: job 1214, on the Content Creator, for Monday 5 October, waiting for me.",
]
NOT_ASKING = [
    "I already said yes. Go ahead, and tell me the job number when it's on the board.",
    "No need to wait for me, just run it.",
    "Don’t wait for me.",
    "I want to see the report from Monday.",
    "Can you give me what it said, word for word?",
]


@pytest.mark.parametrize("words", NIGHT_6_ASKS)
def test_the_owners_ways_of_asking_to_see_it_first(words):
    assert asks_to_see_it_first(words)


@pytest.mark.parametrize("words", NOT_ASKING)
def test_words_that_do_not_ask(words):
    assert not asks_to_see_it_first(words)


def test_the_names_a_model_reaches_for_are_the_boards():
    from api.board_tasks import board_review_mode

    assert board_review_mode("human_review") == "human" and board_review_mode("review") == "human"
    assert board_review_mode("automatic") == "auto" and board_review_mode("strict") is None


@pytest.fixture
def chat(db_session, seed_workspace, monkeypatch):
    """A workspace with the Content Creator, and the owner's chat with Auto."""
    from modules.tools.discovery import handlers_board_tasks as handlers

    ws = UUID(seed_workspace())
    db_session.execute(
        text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
             "VALUES ('Content Creator', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json))"),
        {"w": str(ws)})
    tag = uuid.uuid4().hex[:8]
    user_id = db_session.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                                 {"e": f"f180-{tag}@example.test", "u": f"f180-{tag}"}).scalar()
    chat_id = uuid.uuid4()
    db_session.execute(
        text("INSERT INTO chats (id, user_id, workspace_id, title, visibility) "
             "VALUES (CAST(:c AS uuid), :u, CAST(:w AS uuid), 'night 6', 'private')"),
        {"c": str(chat_id), "u": user_id, "w": str(ws)})
    monkeypatch.setattr(handlers, "_notify_dispatch_safe", lambda *a, **k: None)
    return NS(db=db_session, ws=ws, chat_id=chat_id)


def _say(chat, role, words, minutes_ago):
    chat.db.execute(
        text("INSERT INTO messages (id, chat_id, workspace_id, role, parts, created_at) VALUES "
             "(CAST(:i AS uuid), CAST(:c AS uuid), CAST(:w AS uuid), :r, CAST(:p AS jsonb), "
             "NOW() - make_interval(mins => :m))"),
        {"i": str(uuid.uuid4()), "c": str(chat.chat_id), "w": str(chat.ws), "r": role,
         "p": json.dumps([{"type": "text", "text": words}]), "m": minutes_ago})


def _file(chat, **params):
    from modules.tools.discovery.handlers_board_task_review import create_board_task

    return asyncio.run(create_board_task(chat.db, chat.ws, {
        "title": "Draft Product Description for Kiambu AA Coffee", "description": "In our brand voice.",
        "assigned_agent_name": "Content Creator", "_origin_chat_id": str(chat.chat_id), **params}))


def _review_mode(chat, task_id):
    return chat.db.execute(text("SELECT review_mode FROM board_tasks WHERE id = :i"), {"i": task_id}).scalar()


def test_set_to_wait_for_me_waits_even_when_auto_leaves_the_mode_out(chat):
    """#1209: no review mode passed, and the owner had just said "set to wait for me"."""
    _say(chat, "user", NIGHT_6_ASKS[2], 0)
    reply = _file(chat)
    assert _review_mode(chat, reply["task_id"]) == "human"           # night 6: 'auto', it closed by itself
    assert reply["review"] == WAITS_FOR_OWNER and reply["review_set_because"] == OWNER_ASKED


def test_a_confirmation_carries_the_request_from_the_message_before(chat):
    """The owner asked, then confirmed: "I already said yes. Go ahead"."""
    _say(chat, "user", NIGHT_6_ASKS[1], 2)
    _say(chat, "assistant", "Let me set that up.", 1)
    _say(chat, "user", NOT_ASKING[0], 0)
    assert _review_mode(chat, _file(chat)["task_id"]) == "human"


def test_the_owners_words_win_over_the_mode_auto_passed(chat):
    _say(chat, "user", NIGHT_6_ASKS[0], 0)
    assert _review_mode(chat, _file(chat, review_mode="auto")["task_id"]) == "human"


def test_without_a_request_it_closes_by_itself_and_says_so(chat):
    _say(chat, "user", "Draft the Kiambu AA description in our brand voice.", 0)
    reply = _file(chat)
    assert _review_mode(chat, reply["task_id"]) == "auto" and reply["review"] == CLOSES_BY_ITSELF
    assert "review_set_because" not in reply


def test_an_older_request_does_not_reach_a_new_job(chat):
    _say(chat, "user", "Set it to wait for me.", 3)
    _say(chat, "user", "Thanks.", 2)
    _say(chat, "user", "Now draft the Kiambu AA description.", 0)
    assert _review_mode(chat, _file(chat)["task_id"]) == "auto"


def test_a_caller_with_no_conversation_is_unchanged(chat):
    """A session, a heartbeat or a playbook step files with no chat behind it."""
    _say(chat, "user", NIGHT_6_ASKS[0], 0)
    assert _review_mode(chat, _file(chat, _origin_chat_id=None)["task_id"]) == "auto"


def test_the_platform_action_runs_through_it():
    from modules.tools.discovery import platform_executor
    from modules.tools.discovery.handlers_board_task_review import create_board_task

    assert platform_executor.create_board_task is create_board_task
