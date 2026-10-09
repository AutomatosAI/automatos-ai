"""PRD-256 US-004 (FR-5): an owner-only action from a person's chat waits for their click.

Night 8: "Cancel #0422" became Done with "User chalked it up themselves." signed "you";
#0329 was approved when the owner had only named it. The night-8 fix read their words;
words are not a click. Now an action on Decision D1's list, called in a turn a person
drives (``caller_context.driving_user_id``), returns the existing approval card's ask
(``tool_grants.attach_ask_grant``), whatever the full-autonomy dial says; the owner's
click grants it and the same call runs once on that grant. Agent runs, playbook steps
and heartbeats are unchanged, and what the move writes on the card is signed by the user
who clicked.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from modules.tools.discovery.owner_only import (
    CLICKED_BY, OWNER_ONLY_ACTIONS, OWNERS_OWN_DECISION, asks_before_a_send, is_owner_only,
)

CLICKER = "user:clicker-7"
CANCEL = "platform_update_task_status"


# ── The list: Decision D1 ───────────────────────────────────────────────────────────

D1 = {"platform_update_task_status", "platform_update_task", "platform_assign_tool_to_agent",
      "platform_unassign_tool_from_agent", "platform_update_agent", "platform_update_system_setting",
      "platform_create_mission", "platform_approve_mission", "platform_cancel_mission",
      "platform_publish_blog_post", "platform_submit_social_post",
      # D1 amended 8 Oct (FX-010): every agent-setting change, and a playbook made, timed or deleted.
      "platform_configure_agent_heartbeat", "platform_delete_agent", "platform_assign_skill_to_agent",
      "platform_unassign_skill_from_agent", "platform_assign_plugin_to_agent", "platform_create_playbook",
      "platform_schedule_playbook", "platform_delete_playbook",
      # P256-FIX-RVW-14: a timer set through an update; a plugin or skill taken from every agent.
      "platform_update_playbook", "platform_uninstall_plugin", "platform_delete_workspace_skill",
      "platform_update_skill",
      # P256-FIX-RVW-23: an agent's timer, as a playbook's timer.
      "platform_schedule_task"}
# Owner-only only when the call closes the card, or (an update) sets the playbook's timer.
CONDITIONAL = {"platform_update_task_status", "platform_update_task", "platform_update_playbook"}


def test_the_list_is_decision_d1():
    assert OWNER_ONLY_ACTIONS == D1


@pytest.mark.parametrize("status, owner_only", [
    ("done", True), ("cancelled", True), ("approved", True), ("in_progress", False), ("assigned", False),
    ("blocked", False), ("send back", False), (None, False),
])
def test_a_card_move_is_owner_only_when_it_closes_the_card(status, owner_only):
    for action in ("platform_update_task_status", "platform_update_task"):
        assert is_owner_only(action, {"task_id": 422, "status": status}) is owner_only


def test_every_other_d1_action_is_owner_only_and_a_read_is_not():
    for action in D1 - CONDITIONAL:
        assert is_owner_only(action, {}) is True
    assert is_owner_only("platform_update_playbook", {"playbook_id": 3, "schedule_config": {"enabled": False}})
    assert is_owner_only("platform_update_playbook", {"playbook_id": 3, "name": "Daily Digest"}) is False
    assert is_owner_only("platform_get_task", {"task_id": 422}) is False
    assert is_owner_only("platform_create_task", {"title": "Reorder oat milk"}) is False


@pytest.mark.parametrize("slug, sends", [
    ("GMAIL_SEND_EMAIL", True), ("SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL", True),
    ("LINKEDIN_CREATE_LINKED_IN_POST", True), ("GMAIL_REPLY_TO_THREAD", True), ("INSTAGRAM_MEDIA_PUBLISH", True),
    ("GMAIL_FETCH_EMAILS", False), ("LINKEDIN_GET_POST", False), ("GMAIL_CREATE_EMAIL_DRAFT", False),
    ("TWITTER_POST_DELETE_BY_POST_ID", True),   # not a read: fails toward asking
])
def test_every_composio_send_or_publish_is_owner_only(slug, sends):
    assert is_owner_only(slug, {}, composio=True) is sends


# ── The executor: the ask, the click, the run ───────────────────────────────────────

@pytest.fixture
def board(db_session, seed_workspace):
    """A workspace with one card in Review, and Auto's executor whose handlers are recorded.
    Auto is the actor the hierarchy check reads, as in the chat (exec_platform injects the
    running agent's ``_agent_id``), so the gate's own assertions run."""
    from core.models.core import Agent, BoardTask
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    ws = UUID(seed_workspace())
    auto = Agent(name="Auto", agent_type="system", description="", status="active", configuration={},
                 workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws),
                 is_system_agent=True)
    card = BoardTask(workspace_id=ws, title="Chalkboard line for the Guji", status="review", source_type="user")
    db_session.add_all([auto, card])
    db_session.flush()
    executor = PlatformActionExecutor(db_session, ws)
    executor._full_autonomy = lambda: True   # the dial on: an owner-only action still asks
    handler = AsyncMock(return_value={"success": True, "task_id": card.id, "status": "cancelled"})
    executor._handlers[CANCEL] = handler
    return NS(db=db_session, ws=ws, card=card, number=f"#{card.workspace_seq:04d}", executor=executor,
              handler=handler, auto=auto.id)


def _owners_chat():
    return {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}


def _as_auto(board, params):
    """The call as the chat makes it: exec_platform injects the running agent (Auto) as ``_agent_id``."""
    return {**params, "_agent_id": board.auto}


def _call(board, params, caller_context):
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(board.executor.execute(CANCEL, _as_auto(board, params), caller_context))


def _grant(board, grant_id):
    from core.models.approval_grants import ApprovalGrant

    return board.db.get(ApprovalGrant, grant_id)


def test_cancel_from_chat_asks_naming_the_card_and_the_verb_and_moves_nothing(board):
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    reply = _call(board, {"task_id": board.number, "status": "cancelled"}, _owners_chat())

    assert reply["success"] is False and reply["requires_confirmation"] is True and reply["owner_only"] is True
    assert board.number in reply["message"] and "cancel" in reply["message"]
    assert isinstance(reply["grant_id"], int)
    board.handler.assert_not_called()
    card = ToolResultFormatter.format_for_frontend(reply, CANCEL)["tool_approval"]   # the existing card
    assert card["grant_id"] == reply["grant_id"] and board.number in card["message"]


def test_the_owners_click_runs_the_call_once_signed_by_who_clicked(board):
    from core.services.approval_grants import grant_grant

    params = {"task_id": board.number, "status": "cancelled"}
    asked = _call(board, params, _owners_chat())
    grant_grant(_grant(board, asked["grant_id"]), granted_by=CLICKER)
    board.db.flush()

    ran = _call(board, params, _owners_chat())

    assert ran["success"] is True and ran["approved_via_grant_id"] == asked["grant_id"]
    board.handler.assert_called_once()
    assert board.handler.call_args.args[2][CLICKED_BY] == CLICKER
    assert _grant(board, asked["grant_id"]).status == "revoked"               # one click, one run (claimed)
    again = _call(board, params, _owners_chat())
    assert again["requires_confirmation"] is True and again["grant_id"] != asked["grant_id"]
    board.handler.assert_called_once()                                          # a second "cancel it" did nothing


def test_typed_words_never_stand_in_for_the_click(board):
    """'Approve it' typed after the ask makes the same call: it asks again, on the same card."""
    params = {"task_id": board.number, "status": "done"}
    first = _call(board, params, _owners_chat())
    second = _call(board, params, _owners_chat())
    assert first["grant_id"] == second["grant_id"] and "approve" in second["message"]
    board.handler.assert_not_called()


@pytest.mark.parametrize("lane", [None, {}, {"board_task_id": 85}, {"playbook_execution_id": "exec-1"},
                                  {"conversation_id": "heartbeat"}],
                         ids=["no-context", "empty", "board-ticket", "playbook-step", "no-driving-user"])
def test_an_agent_run_a_playbook_step_or_a_heartbeat_is_unchanged(board, lane):
    reply = _call(board, {"task_id": board.number, "status": "cancelled"}, lane)
    assert reply["success"] is True and "requires_confirmation" not in reply
    board.handler.assert_called_once()


def test_the_harness_approve_is_the_admins_own_decision(board):
    reply = _call(board, {"task_id": board.number, "status": "cancelled"},
                  {"driving_user_id": "7", OWNERS_OWN_DECISION: True})
    assert reply["success"] is True
    board.handler.assert_called_once()


def test_a_move_to_in_progress_is_not_owner_only(board):
    reply = _call(board, {"task_id": board.number, "status": "in_progress"}, _owners_chat())
    assert reply["success"] is True and "requires_confirmation" not in reply
    board.handler.assert_called_once()


def test_a_signer_in_the_call_is_dropped(board):
    """Only a click signs: a ``_clicked_by`` the model sends never reaches the handler."""
    _call(board, {"task_id": board.number, "status": "in_progress", CLICKED_BY: "user:forged"}, _owners_chat())
    assert CLICKED_BY not in board.handler.call_args.args[2]


def test_a_click_given_back_when_the_call_did_nothing(board):
    """A refused handler (F193): the click is given back, so the retry runs on it without a new card."""
    from core.services.approval_grants import grant_grant

    params = {"task_id": board.number, "status": "cancelled"}
    asked = _call(board, params, _owners_chat())
    grant_grant(_grant(board, asked["grant_id"]), granted_by=CLICKER)
    board.db.flush()
    board.handler.return_value = {"success": False, "error": "A running ticket waits."}

    assert _call(board, params, _owners_chat())["success"] is False
    assert _grant(board, asked["grant_id"]).status == "granted"


def test_a_call_refused_by_the_other_checks_is_never_asked_about(board):
    """The ask comes after the hierarchy check, the rate limit and the backstop: a
    rate-limited cancel is refused, and no card is staged for a click that could not run it."""
    from fastapi import HTTPException

    from core.models.approval_grants import ApprovalGrant

    limited = AsyncMock(side_effect=HTTPException(status_code=429, detail="slow down"))
    with patch("core.security.rate_limiter.check_rate_limit", new=limited):
        reply = asyncio.run(board.executor.execute(
            CANCEL, _as_auto(board, {"task_id": board.number, "status": "cancelled"}), _owners_chat()))
    assert reply.get("rate_limited") is True and "requires_confirmation" not in reply
    assert board.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == board.ws).count() == 0
    board.handler.assert_not_called()


def test_a_card_not_on_the_board_is_never_asked_about(board):
    reply = _call(board, {"task_id": "#9999", "status": "cancelled"}, _owners_chat())
    assert reply["success"] is False and "requires_confirmation" not in reply and "nothing was asked" in reply["error"]
    board.handler.assert_not_called()


# ── The note a move writes: signed by who clicked ───────────────────────────────────

class _Session:
    def commit(self):
        pass


@pytest.fixture
def notes(monkeypatch):
    import services.cli_host_service as host
    import services.ticket_verdict as verdict

    written = []
    monkeypatch.setattr(host, "append_session_note", lambda db, **kw: written.append(("note", kw["note"], kw["by"])))
    monkeypatch.setattr(verdict, "keep_approval_note",
                        lambda db, **kw: written.append(("approval", kw["note"], kw["by"])))
    return written


def test_an_approvals_note_is_signed_by_the_user_who_clicked(notes):
    from modules.tools.discovery.ticket_moves import _keep_the_note

    _keep_the_note(_Session(), uuid4(), {"status": "done", "note": "Good, that's the tone.", "_user_id": "user_owner",
                                         CLICKED_BY: CLICKER}, {"success": True, "task_id": 422})
    assert notes == [("approval", "Good, that's the tone.", CLICKER)]


def test_a_note_on_autos_call_alone_is_never_signed_as_the_owner(notes):
    """F280 (#0422): a note in the owner's name they never clicked for is an agent's."""
    from modules.tools.discovery.ticket_moves import AN_AGENTS_NOTE_BY, _keep_the_note

    _keep_the_note(_Session(), uuid4(), {"status": "done", "note": "User chalked it up themselves.",
                                         "_user_id": "user_owner"}, {"success": True, "task_id": 422})
    assert notes == [("note", "User chalked it up themselves.", AN_AGENTS_NOTE_BY)]


def test_a_cancel_of_a_running_card_is_signed_by_who_clicked(monkeypatch):
    import services.run_cancel as run_cancel
    from modules.tools.discovery import ticket_cancel

    signed = []
    monkeypatch.setattr(run_cancel, "cancel_ticket", lambda db, card, by, may: signed.append(by) or ("no", "stop"))
    monkeypatch.setattr(ticket_cancel, "_runs_live", lambda db, card: False)
    monkeypatch.setattr(ticket_cancel, "_may", lambda db, ws, driver: lambda what: True)

    ticket_cancel._cancel(None, uuid4(), {7: NS(id=7)}, "user_owner", CLICKER)
    ticket_cancel._cancel(None, uuid4(), {7: NS(id=7)}, "user_owner")
    assert signed == [CLICKER, ticket_cancel.BY_AN_AGENT]


# ── A Composio send asks through the same card ──────────────────────────────────────

class _Tools:
    """UnifiedToolExecutor's shape: it resolves a Composio call and runs it."""

    db = None

    def __init__(self, slug):
        self.slug, self.ran = slug, []

    def _resolve_effective_call(self, tool_name, parameters):
        return self.slug, parameters, True

    @asks_before_a_send
    async def execute_tool(self, tool_name, parameters, agent_id=0, tenant_id=None, workspace_id=None,
                           trace_id=None, caller_context=None):
        self.ran.append(tool_name)
        return {"success": True}


@pytest.fixture
def grants(monkeypatch):
    import modules.tools.discovery.owner_only as owner_only
    import modules.tools.execution.tool_grants as tool_grants

    said = {"grant": None, "given_back": []}
    monkeypatch.setattr(owner_only, "_the_click", lambda db, ws, action, params: said["grant"])
    monkeypatch.setattr(tool_grants, "attach_ask_grant", lambda db, ws, **kw: {**kw["ask"], "grant_id": 31})
    monkeypatch.setattr(tool_grants, "give_back_unused",
                        lambda db, grant_id, result: result.get("success") is False and said["given_back"].append(grant_id))
    return said


def test_a_send_from_chat_asks_and_sends_nothing(grants):
    tools = _Tools("GMAIL_SEND_EMAIL")
    reply = asyncio.run(tools.execute_tool("composio_execute", {"action": "GMAIL_SEND_EMAIL", "params": {}},
                                           workspace_id=uuid4(), caller_context=_owners_chat()))
    assert reply["requires_confirmation"] is True and "GMAIL_SEND_EMAIL" in reply["message"]
    assert tools.ran == []


def test_a_send_runs_on_the_click_and_a_read_never_asks(grants):
    grants["grant"] = NS(id=31, granted_by=CLICKER, status="granted")
    tools = _Tools("GMAIL_SEND_EMAIL")
    assert asyncio.run(tools.execute_tool("GMAIL_SEND_EMAIL", {}, caller_context=_owners_chat()))["success"] is True
    assert tools.ran == ["GMAIL_SEND_EMAIL"] and grants["given_back"] == []
    grants["grant"] = None
    reader = _Tools("GMAIL_FETCH_EMAILS")
    assert asyncio.run(reader.execute_tool("GMAIL_FETCH_EMAILS", {}, caller_context=_owners_chat()))["success"] is True
    assert reader.ran == ["GMAIL_FETCH_EMAILS"]


def test_an_agents_own_send_is_unchanged(grants):
    tools = _Tools("GMAIL_SEND_EMAIL")
    assert asyncio.run(tools.execute_tool("GMAIL_SEND_EMAIL", {}, caller_context=None))["success"] is True
    assert tools.ran == ["GMAIL_SEND_EMAIL"]
