"""Auto's last reply, as the call guards read it, is never a failed turn's note.

``owner_turn._chat_words`` reads the newest assistant messages for ``autos_last_reply``,
``autos_proposal`` (F241: "Yes, that's it, put that brief on #0204" takes Auto's brief) and
``mission_owner_words.autos_last_words`` (F308). A failed turn leaves an assistant note,
"⚠️ Auto could not finish this reply…" (source origin ``turn_error``): read as Auto's last
reply, the owner's "yes" took the error sentence as the brief, and the call was refused.
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

from modules.tools.discovery.owner_turn import OwnerTurn, autos_last_reply, owners_recent_words

BRIEF = "Brief for #0204: a 600-word guide to cold brew at home, in the brand's voice, with two recipes."
FAILED = "⚠️ Auto could not finish this reply: the AI provider had a problem on its side."
NOTE_SOURCE = {"origin": "turn_error", "label": "Error", "code": "provider_failed"}


@pytest.fixture
def chat(db_session, seed_workspace):
    """The owner's chat with Auto, in a workspace of its own."""
    ws = UUID(seed_workspace())
    tag = uuid.uuid4().hex[:8]
    user_id = db_session.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                                 {"e": f"note-{tag}@example.test", "u": f"note-{tag}"}).scalar()
    chat_id = uuid.uuid4()
    db_session.execute(
        text("INSERT INTO chats (id, user_id, workspace_id, title, visibility) "
             "VALUES (CAST(:c AS uuid), :u, CAST(:w AS uuid), 'brief', 'private')"),
        {"c": str(chat_id), "u": user_id, "w": str(ws)})
    return NS(db=db_session, ws=ws, chat_id=chat_id)


def _say(chat, role, words, minutes_ago, source=None):
    chat.db.execute(
        text("INSERT INTO messages (id, chat_id, workspace_id, role, parts, source, created_at) VALUES "
             "(CAST(:i AS uuid), CAST(:c AS uuid), CAST(:w AS uuid), :r, CAST(:p AS jsonb), CAST(:s AS jsonb), "
             "NOW() - make_interval(mins => :m))"),
        {"i": str(uuid.uuid4()), "c": str(chat.chat_id), "w": str(chat.ws), "r": role,
         "p": json.dumps([{"type": "text", "text": words}]), "s": json.dumps(source) if source else None,
         "m": minutes_ago})


def _turn(chat, latest):
    return OwnerTurn(latest=latest, earlier="", cards=(), chat_id=chat.chat_id)


def test_autos_last_reply_skips_the_note_a_failed_turn_left(chat):
    _say(chat, "user", "Write me a brief for the cold brew guide", 4)
    _say(chat, "assistant", BRIEF, 3)
    _say(chat, "user", "Put that on #0204", 2)
    _say(chat, "assistant", FAILED, 1, source=NOTE_SOURCE)

    assert autos_last_reply(chat.db, chat.ws, _turn(chat, "Yes, that's it, put that brief on #0204")) == BRIEF


def test_a_background_post_and_autos_reply_are_still_read(chat):
    """Only a failed turn's note is left out: a reply with any other source is Auto's words."""
    _say(chat, "user", "Write me a brief for the cold brew guide", 3)
    _say(chat, "assistant", BRIEF, 2, source={"origin": "voice", "label": "Auto · voice"})

    assert autos_last_reply(chat.db, chat.ws, _turn(chat, "Yes, use that")) == BRIEF


def test_the_owners_words_are_unchanged(chat):
    _say(chat, "user", "Write me a brief for the cold brew guide", 3)
    _say(chat, "assistant", FAILED, 2, source=NOTE_SOURCE)
    _say(chat, "user", "Put that on #0204", 1)

    assert owners_recent_words(chat.db, chat.ws, _turn(chat, "Put that on #0204")) == (
        "Put that on #0204", "Write me a brief for the cold brew guide")
