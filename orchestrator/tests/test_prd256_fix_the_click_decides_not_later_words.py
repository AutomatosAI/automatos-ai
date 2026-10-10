"""PRD-256 P256-FIX-RVW-18: the owner's click is not refused by words they wrote after the card.

The click's resume (``_resume_tool_call``) replayed the stored call with the chat's
context, so ``follows_the_owner`` and ``answers_the_question_first`` judged it against the
owner's LATEST message at the time of the click. "Change GREEN BUYER's heartbeat to 15
min" raised the card; the owner then asked "how's #0960?", clicked, and the click was
refused ("The owner named #0960 … an agent isn't what they asked about"): nothing ran.

A call resumed from the owner's granted click (the grant, set by the server in the
context, checked against its row) is judged by the click. The same call made fresh in
that turn, with no click, is judged by the word rules as before.

Every call runs the whole stack: the unified executor, the question guard, the word
rules, the role gates, the hierarchy check, the owner's-click gate and the handler.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

HEARTBEAT = "platform_configure_agent_heartbeat"
ASKED_FOR, WAS = 15, 60
NOT_THE_CARD = "isn't what they asked about"
A_QUESTION = "The owner asked a question"
# What the owner says after the card was raised, before they click, and the rule that refuses
# a fresh call in that turn.
LATER = {
    "names_another_card": (lambda cafe: f"How's {cafe.number} going?", NOT_THE_CARD),
    "asks_how": (lambda cafe: "How do I change an agent's heartbeat myself?", A_QUESTION),
}


def _person(db, ws):
    clerk = f"user_owner_{uuid4().hex[:8]}"
    user = db.execute(text("INSERT INTO users (email, username, clerk_user_id) VALUES (:e, :u, :c) RETURNING id"),
                      {"e": f"owner-{uuid4().hex[:8]}@cafe.test", "u": f"owner-{uuid4().hex[:8]}", "c": clerk}).scalar()
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :user, 'owner', TRUE)"), {"ws": str(ws), "user": user})
    return NS(id=user, clerk=clerk)


@pytest.fixture
def cafe(db_session, seed_workspace, monkeypatch):
    """Auto, GREEN BUYER on an hourly heartbeat, a card in Review and the owner's chat."""
    from core.models.core import Agent, BoardTask, Chat

    db, ws = db_session, UUID(seed_workspace())
    owner = _person(db, ws)
    auto = Agent(name="Auto", agent_type="system", description="", status="active", configuration={},
                 workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws), is_system_agent=True)
    buyer = Agent(name="GREEN BUYER", agent_type="chatbot", description="Buys green coffee", status="active",
                  configuration={"heartbeat": {"enabled": True, "interval_minutes": WAS}}, workspace_id=ws,
                  created_by="test", owner_type="workspace", owner_id=str(ws))
    card = BoardTask(workspace_id=ws, title="Oat milk reorder", status="review", source_type="user")
    chat = Chat(id=uuid4(), user_id=owner.id, workspace_id=ws, title="Heartbeats", visibility="private")
    db.add_all([auto, buyer, card, chat])
    db.flush()
    monkeypatch.setattr("modules.tools.execution.tool_grants._notify_approval_pending", lambda grant, workspace_id: None)
    return NS(db=db, ws=ws, owner=owner, auto=auto, buyer=buyer, card=card, chat=chat,
              number=f"#{card.workspace_seq:04d}", said=[])


def _says(cafe, words):
    """The owner's next message in the chat: one transaction's now() is one instant, so each is a second on."""
    from core.models.core import Message

    cafe.said.append(words)
    cafe.db.add(Message(chat_id=cafe.chat.id, workspace_id=cafe.ws, role="user",
                        parts=[{"type": "text", "text": words}],
                        created_at=datetime(2026, 10, 8, 9, 0) + timedelta(seconds=len(cafe.said))))
    cafe.db.flush()


def _owners_chat(cafe):
    return {"user_id": cafe.owner.clerk, "driving_user_id": str(cafe.owner.id),
            "conversation_id": str(cafe.chat.id), "turn_id": "t-1"}


def _call(cafe):
    """Auto's call in the owner's chat, as the chat's tool loop makes it."""
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(UnifiedToolExecutor(db_session=cafe.db).execute_tool(
            tool_name=HEARTBEAT, parameters={"agent_id": cafe.buyer.id, "interval_minutes": ASKED_FOR},
            agent_id=cafe.auto.id, workspace_id=cafe.ws, caller_context=_owners_chat(cafe)))


def _raised_and_granted(cafe):
    """'Change GREEN BUYER's heartbeat to 15 min': the card is raised, and the owner says yes on it."""
    from core.models.approval_grants import ApprovalGrant
    from core.services.approval_grants import grant_grant

    _says(cafe, "Change GREEN BUYER's heartbeat to 15 min")
    asked = _call(cafe)
    assert asked["success"] is False and asked.get("requires_confirmation") is True, asked
    grant = cafe.db.get(ApprovalGrant, asked["grant_id"])
    grant_grant(grant, granted_by=f"user:{cafe.owner.id}")
    cafe.db.flush()
    return grant


def _clicked(cafe, grant):
    from api.approval_grants import _resume_tool_call

    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        asyncio.run(_resume_tool_call(cafe.db, grant))


def _interval(cafe):
    cafe.db.refresh(cafe.buyer)
    return cafe.buyer.configuration["heartbeat"]["interval_minutes"]


@pytest.mark.parametrize("later", list(LATER), ids=list(LATER))
def test_the_click_runs_the_change_the_card_showed_whatever_the_owner_said_since(cafe, later):
    from core.models.approval_grants import GrantStatus
    from modules.tools.execution.tool_grants import GRANT_CONSUMED_BY

    grant = _raised_and_granted(cafe)
    _says(cafe, LATER[later][0](cafe))

    _clicked(cafe, grant)

    result = grant.details["executed_result"]
    assert result["success"] is True and result["error"] is None, result
    assert _interval(cafe) == ASKED_FOR
    assert (grant.status, grant.revoked_by) == (GrantStatus.REVOKED.value, GRANT_CONSUMED_BY)   # one click, one run


@pytest.mark.parametrize("later", list(LATER), ids=list(LATER))
def test_the_same_call_made_fresh_in_that_turn_is_still_judged_by_the_rules(cafe, later):
    from core.models.approval_grants import GrantStatus

    grant = _raised_and_granted(cafe)
    words, refused_by = LATER[later]
    _says(cafe, words(cafe))

    reply = _call(cafe)

    assert reply["success"] is False and refused_by in reply["error"], reply
    assert "requires_confirmation" not in reply
    assert _interval(cafe) == WAS
    assert grant.status == GrantStatus.GRANTED.value                  # the owner's yes is still theirs to use


def test_a_resume_mark_is_honoured_only_for_the_granted_call(cafe):
    """The mark names a grant; the rules skip only the call that grant is the live yes for."""
    from modules.tools.discovery.click_resume import RESUMED_GRANT, on_the_click, resumed_context

    grant = _raised_and_granted(cafe)
    params = {"agent_id": cafe.buyer.id, "interval_minutes": ASKED_FOR}
    marked = resumed_context(_owners_chat(cafe), grant.id)

    assert marked[RESUMED_GRANT] == grant.id and resumed_context(None, grant.id) is None
    assert on_the_click(cafe.db, cafe.ws, HEARTBEAT, params, marked) is True
    assert on_the_click(cafe.db, cafe.ws, HEARTBEAT, {**params, "interval_minutes": 5}, marked) is False
    assert on_the_click(cafe.db, cafe.ws, "platform_delete_agent", {"agent_id": cafe.buyer.id}, marked) is False
    assert on_the_click(cafe.db, uuid4(), HEARTBEAT, params, marked) is False
    assert on_the_click(cafe.db, cafe.ws, HEARTBEAT, params, _owners_chat(cafe)) is False
    assert on_the_click(cafe.db, cafe.ws, HEARTBEAT, params, {**marked, RESUMED_GRANT: "not-a-grant"}) is False


def test_a_mark_on_a_grant_no_longer_granted_is_not_a_click(cafe):
    from core.services.approval_grants import revoke_grant
    from modules.tools.discovery.click_resume import on_the_click, resumed_context

    grant = _raised_and_granted(cafe)
    revoke_grant(grant, revoked_by=f"user:{cafe.owner.id}")
    cafe.db.flush()
    params = {"agent_id": cafe.buyer.id, "interval_minutes": ASKED_FOR}

    assert on_the_click(cafe.db, cafe.ws, HEARTBEAT, params, resumed_context(_owners_chat(cafe), grant.id)) is False
