"""PRD-256 FX-004 (night 12, C2, F391): an ask for the owner's click is waiting, never a failure.

Night 12: an owner-only call from chat returned its ask (``requires_confirmation: True``,
``success: False``) and everything downstream read it as a refusal. The receipt said "I tried
to change the agent and it didn't go through: Waiting for the owner's click…", the model was
handed "Tool platform_update_agent failed: Waiting for the owner's click…" and told the owner
the change "didn't go through" (A124, A473; 39 such receipts in one night).

Now the ask is a third receipt status, ``waiting`` ("card raised: <verb> <subject>", no reason,
no "I tried to" line), the model reads "Card raised: … Nothing changes until the owner clicks;
say so in one line and do not retry the call.", and the chat's tool-end line says the same. A
genuine refusal is still refused, and prose that calls the waiting change done still gets the
not-done line.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from consumers.chatbot.receipts import (
    NOTHING_DONE_LINE, REFUSED, WAITING, WRITE, build_receipts, honesty_lines,
)
from consumers.chatbot.tool_summary import tool_result_summary
from modules.tools.execution.card_raised import ACT, FOR_THE_MODEL, TOOL_END
from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker

MOVE = "platform_update_task_status"
UPDATE_AGENT = "platform_update_agent"
GRANT_ID = 42
TRIED = "I tried to"
SAID_DONE = "I've changed Scout's model to Sonnet, so it's ready for tomorrow."
SAID_WAITING = "The approval card is up: click Approve and Scout moves to Sonnet."


def _owners_chat():
    return {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}


def _with_a_card(db, workspace_id, *, ask, **_kwargs):
    """tool_grants.attach_ask_grant with its grant issued: the ask, enriched as the card needs."""
    return {**ask, "grant_id": GRANT_ID, "requires_approval": True}


def _agent_ask():
    """The owner-only ask for update_agent, as owner_only raises it (the grant issued)."""
    from modules.tools.discovery import owner_only

    params = {"agent_id": 12, "name": "Scout", "model": "claude-sonnet-5-5"}
    with patch("modules.tools.execution.tool_grants.attach_ask_grant", new=_with_a_card):
        act = f"{owner_only.VERBS[UPDATE_AGENT]} 'Scout' (agent #12)"
        return owner_only._ask(None, "ws", UPDATE_AGENT, params, _owners_chat(),
                               act=act, what="'Scout' (agent #12)", asked=f"{act}.")


def _routed(ask, tool=UPDATE_AGENT):
    """The tool router's envelope for a call that returned ``ask``: what the model reads."""
    from modules.tools.tool_router import ToolRouter

    with patch("modules.tools.tool_router.execute_tool", new=AsyncMock(return_value=ask)), \
            patch.object(ToolRouter, "_record_tool_signal", lambda *a, **k: None):
        return asyncio.run(ToolRouter().execute_and_format(tool, {"agent_id": 12}, agent_id=1))


def _chats(envelope):
    """The chat's tool callback hands the loop this shape (service.py ``_tool_callback``)."""
    return {"success": envelope["success"], "llm_context": envelope["llm_context"],
            "raw_result": envelope["raw_result"], "frontend_data": envelope["frontend_data"]}


def _receipts(*calls):
    tracker = ToolExecutionTracker()
    for action, params, result in calls:
        tracker.record_outcome("platform_execute", {"action": action, "params": params}, result)
    return build_receipts(tracker)


# ── The ask: one waiting receipt, no refused-write line ─────────────────────────────

def test_the_owner_only_ask_says_what_its_card_asks():
    ask = _agent_ask()
    assert ask["requires_confirmation"] is True and ask["grant_id"] == GRANT_ID
    assert ask[ACT] == "change an agent 'Scout' (agent #12)"


def test_an_owner_only_ask_is_one_waiting_receipt_and_no_refused_line():
    ask = _agent_ask()
    receipts = _receipts((UPDATE_AGENT, {"agent_id": 12, "name": "Scout"}, _chats(_routed(ask))))

    assert len(receipts) == 1
    waiting = receipts[0]
    assert (waiting["kind"], waiting["status"], waiting["reason"], waiting["link"]) == (WRITE, WAITING, None, None)
    assert waiting["effect"] == "card raised: change an agent 'Scout' (agent #12)"
    assert honesty_lines(receipts, SAID_WAITING) == []
    assert not any(line.startswith(TRIED) for line in honesty_lines(receipts, SAID_DONE))


def test_the_model_reads_card_raised_never_failed():
    envelope = _routed(_agent_ask())

    assert envelope["llm_context"] == FOR_THE_MODEL.format(act="change an agent 'Scout' (agent #12)")
    assert envelope["llm_context"].startswith("Card raised: change an agent 'Scout'")
    assert "do not retry the call" in envelope["llm_context"]
    assert "failed" not in envelope["llm_context"].lower()
    assert envelope["frontend_data"]["tool_approval"]["grant_id"] == GRANT_ID   # the card still rides


def test_the_tool_end_line_says_the_same():
    summary = tool_result_summary(_chats(_routed(_agent_ask())))

    assert summary == TOOL_END.format(act="change an agent 'Scout' (agent #12)")
    assert summary != "Failed"


def test_an_ask_without_a_card_keeps_its_own_words_for_the_model():
    """tool_grants' fail-safe floor: no grant, no card, so no 'Card raised' is claimed; still not failed."""
    ask = {k: v for k, v in _agent_ask().items() if k != "grant_id"}
    envelope = _routed(ask)

    assert "Card raised" not in envelope["llm_context"]
    assert "Waiting for the owner's click" in envelope["llm_context"]
    assert "failed" not in envelope["llm_context"].lower()
    assert "do not retry the call" in envelope["llm_context"]
    assert tool_result_summary(_chats(envelope)).startswith("Asked for the owner's OK: change an agent")
    receipt = _receipts((UPDATE_AGENT, {"agent_id": 12}, _chats(envelope)))[0]
    assert receipt["status"] == WAITING and receipt["effect"].startswith("asked for your OK: change an agent")


# ── A genuine refusal is still a refusal; prose that calls the ask done is not believed ──

def test_a_genuine_refusal_is_still_refused():
    refused = {"success": False, "llm_context": "Tool platform_update_agent failed: no agent #99",
               "raw_result": {"success": False, "error": "No agent #99 in this workspace."}}
    receipts = _receipts((UPDATE_AGENT, {"agent_id": 99}, refused))

    assert receipts[0]["status"] == REFUSED
    assert receipts[0]["reason"] == "No agent #99 in this workspace."
    assert honesty_lines(receipts, SAID_WAITING)[0].startswith(TRIED)


def test_saying_the_waiting_change_is_done_gets_the_nothing_done_line():
    receipts = _receipts((UPDATE_AGENT, {"agent_id": 12}, _chats(_routed(_agent_ask()))))

    assert honesty_lines(receipts, SAID_DONE) == [NOTHING_DONE_LINE]


# ── End to end through the executor: a closing move from the owner's chat ───────────

@pytest.fixture
def board(db_session, seed_workspace, monkeypatch):
    """A workspace with Auto and a card in Review; the move's handler is recorded."""
    import modules.tools.discovery.follows_the_owner as guard
    from core.models.core import Agent, BoardTask
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    ws = UUID(seed_workspace())
    auto = Agent(name="Auto", agent_type="system", description="", status="active", configuration={},
                 workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws),
                 is_system_agent=True)
    card = BoardTask(workspace_id=ws, title="Wholesale reply to Larder & Loaf", status="review", source_type="user")
    db_session.add_all([auto, card])
    db_session.flush()
    monkeypatch.setattr(guard, "owner_turn", lambda db, w, ctx: None)
    executor = PlatformActionExecutor(db_session, ws)
    executor._full_autonomy = lambda: True   # the dial on: an owner-only call still asks
    handler = AsyncMock(return_value={"success": True, "task_id": card.id})
    executor._handlers[MOVE] = handler
    return NS(card=card, number=f"#{card.workspace_seq:04d}", executor=executor, handler=handler, auto=auto.id)


def test_an_approve_from_chat_is_a_waiting_receipt_end_to_end(board):
    params = {"task_id": board.number, "status": "done"}
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        ask = asyncio.run(board.executor.execute(MOVE, {**params, "_agent_id": board.auto}, _owners_chat()))
    assert ask.get("requires_confirmation") is True and isinstance(ask.get("grant_id"), int), ask
    board.handler.assert_not_called()

    envelope = _routed(ask, tool=MOVE)
    receipts = _receipts((MOVE, params, _chats(envelope)))

    assert envelope["llm_context"].startswith("Card raised: approve (move to Done)")
    assert board.card.title in envelope["llm_context"]
    assert [r["status"] for r in receipts] == [WAITING]
    assert receipts[0]["effect"].startswith("card raised: approve (move to Done)")
    assert honesty_lines(receipts, "Done. #0001 is approved and moved to Done.") == [NOTHING_DONE_LINE]
