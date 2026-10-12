"""PRD-256 P256-FIX-N13-4 (night 13, F411): the owner's click changes what Auto knows.

Night 13: approving a card moved it to Done with the note, and Auto said itself (ask 416) that
"it doesn't update what I know"; later chats (asks 392, 415) did not know the decision. Now the
approve and deny routes (api/approval_grants, wrapped by ``services.click_memory``) store the
decision as ONE owner-attributed memory through the store-memory tool's handler, after the
click has run: the card's words with its number, the day, the note and what came of it. A
decline is stored too; a second click on the same grant is refused and writes nothing more.
The card itself says, on its last line, what approving does and that Auto will remember it.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from modules.tools.discovery.owner_only import REMEMBERED
from modules.tools.execution.card_raised import ACT
from services.click_memory import APPROVED, DECLINED, decision_memory

MOVE = "platform_update_task_status"
OWNER_ID = 7
NOTE = "Ship it with the autumn prices"


class _Memory:
    """The long-term store the store-memory handler writes to: each write recorded."""

    is_durable_configured = True

    def __init__(self):
        self.stored = []

    async def store_long_term(self, workspace_id, content, agent_id=None, metadata=None, **_):
        self.stored.append(NS(workspace_id=workspace_id, content=content, metadata=metadata or {}))
        return {"facts_extracted": 1}

    def namespace(self, workspace_id):
        return NS(resolve=lambda agent_id: f"ws_{workspace_id}")


@pytest.fixture
def board(db_session, seed_workspace, monkeypatch):
    """A workspace with one card in Review, Auto's executor (its move handler recorded), the
    approval routes with the resume run on that executor, and a recording memory store."""
    import api.approval_grants as ag
    import modules.memory.unified_memory_service as memory_service
    from core.models.core import Agent, BoardTask
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    ws = UUID(seed_workspace())
    auto = Agent(name="Auto", agent_type="system", description="", status="active", configuration={},
                 workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws), is_system_agent=True)
    card = BoardTask(workspace_id=ws, title="Autumn tier launch copy", status="review", source_type="user")
    db_session.add_all([auto, card])
    db_session.flush()
    executor = PlatformActionExecutor(db_session, ws)
    executor._full_autonomy = lambda: True
    handler = AsyncMock(return_value={"success": True, "task_id": card.id, "status": "done"})
    executor._handlers[MOVE] = handler
    memory = _Memory()
    monkeypatch.setattr(memory_service, "get_unified_memory_service", lambda: memory)
    monkeypatch.setattr(ag, "_audit", lambda *a, **k: None)
    monkeypatch.setattr(ag, "said_in_the_chat", lambda *a, **k: None)
    state = NS(db=db_session, ws=ws, card=card, number=f"#{card.workspace_seq:04d}", executor=executor,
               handler=handler, auto=auto.id, memory=memory, chat=_owners_chat(), ag=ag, calls=[])
    monkeypatch.setattr(ag, "_requeue_subject", lambda db, grant: _resume(state, grant))
    return state


def _owners_chat():
    return {"driving_user_id": str(OWNER_ID), "user_id": "user_owner", "conversation_id": str(uuid4()),
            "turn_id": "t-1"}


def _ctx(board):
    """The owner's request, as the approval routes read it (``actor_ref``: 'user:7')."""
    return NS(workspace_id=board.ws, user=NS(id=OWNER_ID, clerk_user_id=None, email="owner@harbourline.test"))


def _move(board, params):
    """The call as the chat makes it (Auto as ``_agent_id``), through the owner's gate."""
    board.calls.append(params)
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(board.executor.execute(MOVE, {**params, "_agent_id": board.auto}, board.chat))


async def _resume(board, grant):
    """The click's resume: the call that raised the card runs again in the owner's chat and
    finds the grant (single-use: the run retires it); its outcome lands on the grant, as the
    real resume (``_resume_tool_call``) leaves it."""
    from services.click_results import executed_summary

    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        raw = await board.executor.execute(MOVE, {**board.calls[-1], "_agent_id": board.auto}, board.chat)
    grant.details = {**dict(grant.details or {}), "executed_result": executed_summary(raw)}
    return True


def _approve(board, grant_id):
    return asyncio.run(board.ag.grant_approval(grant_id, ctx=_ctx(board), db=board.db))


def _deny(board, grant_id):
    return asyncio.run(board.ag.deny_approval(grant_id, ctx=_ctx(board), db=board.db))


# ── The card says what approving does, and that Auto will remember it ─────────────────

def test_the_card_says_what_approving_does_and_that_auto_remembers_it(board):
    asked = _move(board, {"task_id": board.number, "status": "approved", "note": NOTE})

    assert asked["requires_confirmation"] is True and board.handler.await_count == 0
    lines = asked["question_md"].splitlines()
    assert board.number in lines[0]
    assert lines[-1] == REMEMBERED.format(act=asked[ACT])
    assert lines[-1].startswith("Approving will approve (move to Done) 'Autumn tier launch copy'")
    assert board.number in lines[-1]
    assert "Auto will remember your decision, and a decline too." in lines[-1]


# ── A clicked approve: one memory, the owner's, naming the card ──────────────────────

def test_a_clicked_approve_is_one_memory_owned_by_who_clicked_naming_the_card(board):
    asked = _move(board, {"task_id": board.number, "status": "approved", "note": NOTE})

    _approve(board, asked["grant_id"])

    assert board.handler.await_count == 1                                    # the click ran the move once
    assert len(board.memory.stored) == 1
    memory = board.memory.stored[0]
    assert memory.workspace_id == str(board.ws)
    assert memory.content.startswith('The owner approved the card "Approve (move to Done) ')
    assert board.number in memory.content and f"note: {NOTE}" in memory.content and "It ran" in memory.content
    assert memory.metadata["owner"] == f"user:{OWNER_ID}"                    # owner-attributed
    assert memory.metadata["type"] == "decision" and memory.metadata["scope"] == "workspace"
    assert memory.metadata["source_type"] == "platform_verified"
    assert memory.metadata["evidence_uri"] == f"approval_grant:{asked['grant_id']}"


def test_the_decision_is_the_routes_not_the_grants_status(board):
    """A single-use yes is retired by the run it allowed (revoked, system:consumed): still an approval."""
    from core.models.approval_grants import ApprovalGrant
    from modules.tools.execution.tool_grants import GRANT_CONSUMED_BY

    asked = _move(board, {"task_id": board.number, "status": "approved", "note": NOTE})
    _approve(board, asked["grant_id"])

    grant = board.db.get(ApprovalGrant, asked["grant_id"])
    assert grant.revoked_by == GRANT_CONSUMED_BY and grant.granted_by == f"user:{OWNER_ID}"
    assert [m.content.split(" the card ")[0] for m in board.memory.stored] == ["The owner approved"]


# ── A second click on the same grant writes nothing more ─────────────────────────────

def test_a_second_click_on_the_same_grant_writes_nothing_more(board):
    asked = _move(board, {"task_id": board.number, "status": "approved", "note": NOTE})
    _approve(board, asked["grant_id"])

    with pytest.raises(HTTPException) as again:                              # a double click
        _approve(board, asked["grant_id"])

    assert again.value.status_code == 422 and "not pending" in again.value.detail
    assert len(board.memory.stored) == 1 and board.handler.await_count == 1


# ── A deny: one memory saying it was declined ──────────────────────────────────────

def test_a_deny_is_one_memory_saying_it_was_declined(board):
    asked = _move(board, {"task_id": board.number, "status": "cancelled"})

    _deny(board, asked["grant_id"])

    assert board.handler.await_count == 0                                    # nothing ran
    assert len(board.memory.stored) == 1
    memory = board.memory.stored[0]
    assert memory.content.startswith('The owner declined the card "Cancel ')
    assert board.number in memory.content and memory.content.endswith("Nothing ran.")
    assert memory.metadata["owner"] == f"user:{OWNER_ID}" and memory.metadata["type"] == "decision"


# ── What the memory says when the run did not go through; only what the card showed ───

def test_an_approval_that_did_not_go_through_says_so():
    from datetime import datetime, timezone

    grant = NS(id=3, question_md="Change an agent 'CLUB DESK' (agent #12):\n- model: a → b", tool_name="platform_update_agent",
               details={"params": {"agent_id": 12, "note": "a note the card never showed"},
                        "executed_result": {"success": False, "error": "agent is archived"}},
               granted_at=datetime(2026, 10, 12, 9, tzinfo=timezone.utc), revoked_at=None)

    said = decision_memory(grant, APPROVED)

    assert said == ('The owner approved the card "Change an agent \'CLUB DESK\' (agent #12)" on 12 Oct 2026. '
                    "It did not go through: agent is archived")
    assert decision_memory(NS(**{**grant.__dict__, "revoked_at": grant.granted_at}), DECLINED).endswith(
        "on 12 Oct 2026. Nothing ran.")
