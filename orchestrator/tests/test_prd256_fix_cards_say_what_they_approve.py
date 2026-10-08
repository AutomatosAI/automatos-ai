"""PRD-256 FX-008 (B2, F388): an approval card says what it approves.

Night 12, every owner-only card said a verb and no more ('change an agent', 'start a
mission'), and ``question_md`` was empty on every one: CLUB DESK's description was wiped
through a card that said 'change an agent' (A462). The card's question now carries the
subject and the change: each changed field 'field: from → to' (the current value read
from the workspace's row, the new one from the call), a new mission's goal and steps, a
mission's title and number, a send's recipient, subject and first line. The grant stores
it (the approvals queue reads it) and the chat card's frame carries it. Another
workspace's row is never read onto a card.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

from modules.tools.discovery import owner_only
from modules.tools.discovery.card_question import platform_question, send_question
from modules.tools.discovery.card_question_text import EMPTY, question, shown

UPDATE_AGENT = "platform_update_agent"
CREATE_MISSION = "platform_create_mission"
OLD_BRIEF = "Runs the club desk: bookings, the members' list and the Friday tasting."
NEW_BRIEF = "Takes bookings only."


def _owners_chat():
    return {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}


def _agent(db, ws, name, description):
    from core.models.core import Agent

    agent = Agent(name=name, agent_type="chatbot", description=description, status="active", configuration={},
                  model_config={"model_id": "claude-sonnet-5-5"}, workspace_id=ws, created_by="test",
                  owner_type="workspace", owner_id=str(ws))
    db.add(agent)
    db.flush()
    return agent


@pytest.fixture
def desk(db_session, seed_workspace):
    """A workspace with CLUB DESK, and a second workspace with its own agent and ticket."""
    from core.models.core import BoardTask

    ws, other = UUID(seed_workspace()), UUID(seed_workspace())
    club = _agent(db_session, ws, "CLUB DESK", OLD_BRIEF)
    theirs = _agent(db_session, other, "BRAVO DESK", "Bravo's private brief: the Harbourline margins.")
    their_card = BoardTask(workspace_id=other, title="Bravo's private ticket", description="Bravo's own brief",
                           status="review", source_type="user")
    db_session.add(their_card)
    db_session.flush()
    return NS(db=db_session, ws=ws, other=other, club=club, theirs=theirs,
              their_number=f"#{their_card.workspace_seq:04d}")


def _grant(db, grant_id):
    from core.models.approval_grants import ApprovalGrant

    return db.get(ApprovalGrant, grant_id)


# ── An agent change: each field, from → to ──────────────────────────────────────────

def test_an_update_agent_card_lists_the_description_from_and_to(desk):
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    params = {"agent_id": desk.club.id, "description": NEW_BRIEF, "model_id": "claude-opus-5-5"}
    ask = owner_only.platform_ask(desk.db, desk.ws, UPDATE_AGENT, params, _owners_chat())

    asked = ask["question_md"]
    assert asked.startswith("Change an agent 'CLUB DESK'")
    assert f"- description: {OLD_BRIEF} → {NEW_BRIEF}" in asked
    assert "- model: claude-sonnet-5-5 → claude-opus-5-5" in asked
    assert _grant(desk.db, ask["grant_id"]).question_md == asked          # the approvals queue reads it
    card = ToolResultFormatter.format_for_frontend(ask, UPDATE_AGENT)["tool_approval"]
    assert card["question_md"] == asked                                     # the chat card renders it


def test_a_wipe_says_the_field_goes_empty(desk):
    """A462: CLUB DESK's description was wiped through a card that said 'change an agent'."""
    asked = platform_question(desk.db, desk.ws, UPDATE_AGENT, {"agent_id": desk.club.id, "description": ""},
                              "change an agent 'CLUB DESK'")
    assert f"- description: {OLD_BRIEF} → {EMPTY}" in asked


def test_long_text_is_one_line_each():
    persona = "You are CLUB DESK.\n\n" + "Answer every member warmly. " * 20
    line = shown(persona)
    assert "\n" not in line and line.startswith("You are CLUB DESK. Answer") and line.endswith("…")
    assert question("change an agent", ["- a", "- b"]) == "Change an agent:\n- a\n- b"
    assert question("cancel a mission", []) == "Cancel a mission."


def test_a_tool_card_lists_the_agents_tools_before_and_after(desk):
    from core.models.composio_cache import AgentAppAssignment

    desk.db.add(AgentAppAssignment(agent_id=desk.club.id, app_name="GMAIL", is_active=True))
    desk.db.flush()
    asked = platform_question(desk.db, desk.ws, "platform_assign_tool_to_agent",
                              {"agent_id": desk.club.id, "app_name": "slack"}, "give a tool to an agent")
    assert "- tools: GMAIL → GMAIL, SLACK" in asked
    taken = platform_question(desk.db, desk.ws, "platform_unassign_tool_from_agent",
                              {"agent_id": desk.club.id, "app_name": "GMAIL"}, "take a tool from an agent")
    assert f"- tools: GMAIL → {EMPTY}" in taken


def test_a_ticket_card_lists_its_move_and_its_brief(db_session, seed_workspace):
    from core.models.core import BoardTask

    ws = UUID(seed_workspace())
    card = BoardTask(workspace_id=ws, title="Chalkboard line for the Guji", description="Two lines, chalk.",
                     status="review", source_type="user")
    db_session.add(card)
    db_session.flush()
    number = f"#{card.workspace_seq:04d}"                                   # the number the owner uses
    asked = platform_question(db_session, ws, "platform_update_task_status",
                              {"task_id": number, "status": "approved", "note": "Lovely"}, "approve (move to Done)")
    assert "- status: review → done" in asked and "- note: Lovely" in asked
    rebriefed = platform_question(db_session, ws, "platform_update_task",
                                  {"task_id": number, "description": "Three lines."}, "change a ticket")
    assert "- description: Two lines, chalk. → Three lines." in rebriefed


# ── A mission: its goal and steps, or its title and number ─────────────────────────

def test_a_create_mission_card_lists_the_goal_and_the_steps(desk):
    params = {"goal": "Plan the spring menu", "steps": ["Price the oat milk", "Draft the board"],
              "staffing": [{"agent": "CLUB DESK", "does": "the members' note"}]}
    ask = owner_only.platform_ask(desk.db, desk.ws, CREATE_MISSION, params, _owners_chat())

    lines = ask["question_md"].splitlines()
    assert lines[0] == "Start a mission:"
    assert lines[1:5] == ["- goal: Plan the spring menu", "- steps:", "  1. Price the oat milk", "  2. Draft the board"]
    assert "- CLUB DESK does: the members' note" in lines
    assert _grant(desk.db, ask["grant_id"]).question_md == ask["question_md"]


def test_a_mission_with_no_steps_says_so(desk):
    asked = platform_question(desk.db, desk.ws, CREATE_MISSION, {"goal": "Plan the spring menu"}, "start a mission")
    assert "- steps: none given" in asked


def test_an_approve_or_cancel_card_names_the_mission_and_its_number(db_session, seed_workspace):
    from core.models.core import BoardTask
    from core.models.orchestration import OrchestrationRun
    from core.models.orchestration_enums import RunState
    from services.orchestration_board_bridge import create_mission_board_task
    from services.ticket_numbers import ticket_number

    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal="A spring menu for the cafés", created_by="user_test",
                           state=RunState.AWAITING_APPROVAL.value, config={})
    db_session.add(run)
    db_session.flush()
    create_mission_board_task(db_session, run)
    card = db_session.query(BoardTask).filter(BoardTask.orchestration_run_id == run.id).first()
    number = ticket_number(db_session, card)

    for action, said in (("platform_approve_mission", str(run.id)), ("platform_cancel_mission", number)):
        asked = platform_question(db_session, ws, action, {"mission_id": said}, "approve a mission's plan")
        assert f"- mission: '{card.title}' ({number})" in asked


# ── A Composio send: to whom, about what, its first line ───────────────────────────

class _Tools:
    """UnifiedToolExecutor's shape: composio_execute nests the action's own params."""

    db = None

    def __init__(self):
        self.ran = []

    def _resolve_effective_call(self, tool_name, parameters):
        return parameters["action"], parameters["params"], True

    @owner_only.asks_before_a_send
    async def execute_tool(self, tool_name, parameters, agent_id=0, tenant_id=None, workspace_id=None,
                           trace_id=None, caller_context=None):
        self.ran.append(tool_name)
        return {"success": True}


def test_a_send_card_lists_the_recipient_the_subject_and_the_first_line(monkeypatch):
    import modules.tools.execution.tool_grants as tool_grants

    stored = {}
    monkeypatch.setattr(owner_only, "_the_click", lambda db, ws, action, params: None)
    monkeypatch.setattr(tool_grants, "attach_ask_grant",
                        lambda db, ws, **kw: stored.update(kw) or {**kw["ask"], "grant_id": 31})
    sent = {"recipient_email": "ana@harbourline.test", "subject": "Spring order",
            "body": "Hi Ana,\nPlease send 12 kilos of the Guji."}
    tools = _Tools()
    reply = asyncio.run(tools.execute_tool("composio_execute", {"action": "GMAIL_SEND_EMAIL", "params": sent},
                                           workspace_id=uuid4(), caller_context=_owners_chat()))

    asked = reply["question_md"]
    assert asked.startswith("Send, publish or order through GMAIL_SEND_EMAIL:")
    assert "- to: ana@harbourline.test" in asked and "- subject: Spring order" in asked
    assert "- first line: Hi Ana," in asked and "12 kilos" not in asked
    assert stored["question_md"] == asked and tools.ran == []


def test_a_send_with_nothing_to_name_still_says_the_action():
    assert send_question("send or publish through SLACK_SENDS_A_MESSAGE", {}) == (
        "Send or publish through SLACK_SENDS_A_MESSAGE.")


# ── Tenancy: a card never reads another workspace's row ─────────────────────────────

def test_a_card_never_leaks_another_workspaces_row(desk):
    by_id = owner_only.platform_ask(desk.db, desk.ws, UPDATE_AGENT,
                                    {"agent_id": desk.theirs.id, "description": "x"}, _owners_chat())
    assert by_id["success"] is False and "question_md" not in by_id      # F091: nothing is asked

    for action, params in ((UPDATE_AGENT, {"agent_name": "BRAVO", "description": "x"}),
                           ("platform_assign_tool_to_agent", {"agent_name": "BRAVO", "app_name": "SLACK"}),
                           ("platform_update_task", {"task_id": desk.their_number, "description": "x"}),
                           ("platform_update_task_status", {"task_id": desk.their_number, "status": "done"})):
        asked = platform_question(desk.db, desk.ws, action, params, "change something")
        assert "Bravo" not in asked and "Harbourline" not in asked
        assert asked == "Change something."
