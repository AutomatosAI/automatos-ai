"""F091-C2 (night 3) — a "no" in chat offers to withdraw Auto's pending request.

The owner said no in chat; Auto's delete card (#600) stayed pending for its
whole 24 h. When the owner's message reads as a refusal and this conversation
still has a gated request of Auto's waiting on a yes, the reply carries a card
offering to withdraw it. Nothing is withdrawn without the owner's click.
"""
from __future__ import annotations

import inspect
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from services.withdraw_offers import is_refusal, withdraw_offer


@pytest.mark.parametrize("said", ["No", "no, keep it", "Don't delete that", "do not", "Cancel that",
                                  "never mind", "Stop"])
def test_a_refusal_reads_as_one(said):
    assert is_refusal(said)


@pytest.mark.parametrize("said", ["Nothing to add — go ahead", "Yes please", "Knowledge base first", "", None])
def test_anything_else_does_not(said):
    assert not is_refusal(said)


@pytest.fixture
def db(test_engine):
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        session.execute(text("DROP TABLE IF EXISTS pg_temp.approval_grants"))
        session.execute(text("CREATE TEMP TABLE approval_grants (LIKE public.approval_grants INCLUDING DEFAULTS)"))
        yield session
        session.rollback()
        session.execute(text("DROP TABLE IF EXISTS pg_temp.approval_grants"))
        session.commit()
        session.close()


def _grant(db, gid, ws, chat, *, status="pending", kind="approval"):
    db.execute(text(
        "INSERT INTO approval_grants (id, workspace_id, subject_type, subject_id, tool_name, kind, status, reason, "
        "details, requested_at) VALUES (:id, CAST(:ws AS uuid), 'tool_call', :sid, 'platform_delete_document', "
        ":kind, :status, 'Delete christmas-box-2026.csv (document #716)', CAST(:details AS jsonb), now())"),
        {"id": gid, "ws": ws, "sid": f"call-{gid}", "kind": kind, "status": status,
         "details": '{"conversation_id": "%s"}' % chat})


def test_the_pending_request_of_this_conversation_is_offered(db):
    ws, chat = str(uuid4()), str(uuid4())
    _grant(db, 600, ws, chat)
    _grant(db, 601, ws, chat, status="denied")                       # already settled
    _grant(db, 602, ws, str(uuid4()))                                 # another conversation
    _grant(db, 603, ws, chat, kind="question")                        # a question is answered, not withdrawn
    offer = withdraw_offer(db, ws, chat, "No — don't delete it")
    assert [r["grant_id"] for r in offer["requests"]] == [600]
    assert offer["requests"][0]["action"] == "platform_delete_document"
    assert withdraw_offer(db, ws, chat, "Yes, go ahead") is None       # not a refusal
    assert withdraw_offer(db, str(uuid4()), chat, "No") is None        # not this workspace


def test_the_chat_turn_carries_the_offer():
    from consumers.chatbot import service

    source = inspect.getsource(service.StreamingChatService._stream_response_with_agent_scoped)
    assert "withdraw_offer(self.db, self.workspace_id, chat_id, latest_text)" in source
    assert 'format_aisdk_data("withdraw_offer", _offer)' in source
