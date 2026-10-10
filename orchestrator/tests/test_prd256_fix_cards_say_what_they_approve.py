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
from modules.tools.execution.unified_executor import UnifiedToolExecutor

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

    _resolve_effective_call = UnifiedToolExecutor._resolve_effective_call   # the executor's own reading

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


# ── P256-FIX-RVW-26: every recipient the click mails, and the files it carries ─────────

SUPPLIER = "orders@kerbside.example"
WIDE_SEND = {"recipient_email": SUPPLIER, "extra_recipients": ["ana@kerbside.example", "bea@kerbside.example"],
             "cc": ["callum@club.test"], "bcc": ["x@elsewhere.test"], "subject": "Order confirmation",
             "body": "Hi Kerbside,\nPlease confirm.",
             "attachment": {"name": "order-0412.pdf", "mimetype": "application/pdf", "s3key": "ws/7/order-0412.pdf"}}


def _ask_for(monkeypatch, sent, call=None):
    """The reply to a GMAIL_SEND_EMAIL in the owner's chat, and what reached the grant (``stored``);
    ``call``: the whole composio_execute call, when it is not ``{action, params: sent}``."""
    import modules.tools.execution.tool_grants as tool_grants

    stored = {}
    monkeypatch.setattr(owner_only, "_the_click", lambda db, ws, action, params: None)
    monkeypatch.setattr(tool_grants, "attach_ask_grant",
                        lambda db, ws, **kw: stored.update(kw) or {**kw["ask"], "grant_id": 32})
    tools = _Tools()
    reply = asyncio.run(tools.execute_tool("composio_execute", call or {"action": "GMAIL_SEND_EMAIL", "params": sent},
                                           workspace_id=uuid4(), caller_context=_owners_chat()))
    return reply, stored, tools


def test_a_send_card_lists_every_recipient_cc_bcc_and_the_file(monkeypatch):
    reply, stored, tools = _ask_for(monkeypatch, WIDE_SEND)

    asked = reply["question_md"]
    for line in (f"- to: {SUPPLIER}", "- to: ana@kerbside.example", "- to: bea@kerbside.example",
                 "- cc: callum@club.test", "- bcc: x@elsewhere.test",
                 "- attachment: order-0412.pdf (from ws/7/order-0412.pdf)",
                 "- subject: Order confirmation", "- first line: Hi Kerbside,"):
        assert line in asked.splitlines()
    assert stored["question_md"] == asked and tools.ran == []


def test_a_long_recipient_list_is_listed_in_full_never_cut():
    everyone = [f"member{n:02d}@club-members.example" for n in range(12)]
    asked = send_question("send, publish or order through GMAIL_SEND_EMAIL", {"bcc": everyone})

    assert [line for line in asked.splitlines() if line.startswith("- bcc: ")] == [f"- bcc: {who}" for who in everyone]


def test_an_attachment_is_named_by_the_file_it_attaches():
    files = ["reports/q3/margins.xlsx", {"name": "invoice.pdf", "s3key": "ws/7/payroll.xlsx"},
             {"name": "menu.pdf", "s3key": "ws/7/menu.pdf"}]
    asked = send_question("send through GMAIL_SEND_EMAIL", {"to": SUPPLIER, "attachments": files}).splitlines()

    assert "- attachment: reports/q3/margins.xlsx" in asked                     # a path whole
    assert "- attachment: invoice.pdf (from ws/7/payroll.xlsx)" in asked        # a name that is not its file
    assert "- attachment: menu.pdf (from ws/7/menu.pdf)" in asked             # its source, always


def test_a_long_attachment_path_is_shown_whole_never_cut():
    deep = "/".join(["workspace-files"] * 12) + "/payroll-2026.xlsx"
    asked = send_question("send through GMAIL_SEND_EMAIL", {"to": SUPPLIER, "attachment": {"name": "menu.pdf", "s3key": deep}})

    assert len(deep) > 120 and f"- attachment: menu.pdf (from {deep})" in asked.splitlines()


def test_every_other_field_the_send_carries_is_on_the_card():
    """Fail closed: a recipient under a name the card does not know is still shown."""
    hidden = [f"member{n:02d}@club-members.example" for n in range(8)]
    asked = send_question("send through OUTLOOK_SEND_EMAIL", {"to_recipients": hidden, "subject": "Hi",
                                                              "body": "Hello\nmore", "is_html": False,
                                                              "x\n- to: decoy@club.test": "y"}).splitlines()

    assert all(f"- to_recipients: {who}" in asked for who in hidden)           # each, uncut
    assert "- is_html: False" in asked and "- to: decoy@club.test" not in asked  # a key is one line
    assert not any(line.startswith(("- subject: Hi", "- body:")) for line in asked[2:])   # each shown once


@pytest.mark.parametrize("field, value", [("recipient_email", {"email": SUPPLIER}), ("bcc", ["a@b.test", {"x": 1}]),
                                          ("cc", True)])
def test_a_recipient_the_card_cannot_show_is_refused_before_any_grant(monkeypatch, field, value):
    reply, stored, tools = _ask_for(monkeypatch, {**WIDE_SEND, field: value})

    assert reply["success"] is False and not reply.get("requires_confirmation")
    assert f"cannot show the recipient in '{field}'" in reply["error"]
    assert stored == {} and tools.ran == []          # no grant asked for, nothing sent


# ── P256-FIX-RVW-31: the card is read from what the send carries, not params alone ─────

BCC = "x@other.test"


@pytest.mark.parametrize("call", [
    {"action": "GMAIL_SEND_EMAIL", "params": {"recipient_email": SUPPLIER, "subject": "Hi"}, "bcc": BCC},
    {"action": "GMAIL_SEND_EMAIL", "parameters": {"recipient_email": SUPPLIER, "subject": "Hi", "bcc": BCC}},
], ids=["bcc-beside-params", "under-parameters"])
def test_a_send_card_lists_a_recipient_passed_beside_params_or_under_parameters(monkeypatch, call):
    from modules.tools.execution.composio_params import sent_params

    reply, stored, tools = _ask_for(monkeypatch, None, call)

    asked = reply["question_md"].splitlines()
    assert f"- to: {SUPPLIER}" in asked and f"- bcc: {BCC}" in asked
    assert not any(line.startswith("- parameters:") for line in asked)
    assert stored["question_md"] == reply["question_md"] and tools.ran == []    # nothing sent before the click
    assert stored["params"] == call                                              # the click runs this call...
    assert sent_params(call) == {"recipient_email": SUPPLIER, "subject": "Hi", "bcc": BCC}   # ...sending what it shows


def test_an_unreadable_recipient_beside_params_is_refused_before_any_grant(monkeypatch):
    call = {"action": "GMAIL_SEND_EMAIL", "params": {"recipient_email": SUPPLIER}, "bcc": {"email": BCC}}

    reply, stored, tools = _ask_for(monkeypatch, None, call)

    assert reply["success"] is False and "cannot show the recipient in 'bcc'" in reply["error"]
    assert stored == {} and tools.ran == []


# ── A playbook named by name: bound to one row before the card (P256-FIX-RVW-45) ─────

SCHEDULE, DELETE_PLAYBOOK = "platform_schedule_playbook", "platform_delete_playbook"
STOCK_TIMER = {"cron_expression": "0 8 * * 1", "timezone": "Europe/London"}


def _playbook(db, ws, name, steps=()):
    from core.models.core import WorkflowTemplate

    playbook = WorkflowTemplate(template_id=f"rvw45-{uuid4().hex[:8]}", name=name, description="Count the stock.",
                                workspace_id=ws, owner_type="workspace", owner_id=str(ws), created_by="test",
                                steps=list(steps), schedule_config={"type": "manual"},
                                template_definition={"steps": list(steps), "agents": [], "config": {}, "variables": []})
    db.add(playbook)
    db.flush()
    return playbook


@pytest.fixture
def pantry(db_session, seed_workspace):
    """'Monday Stock Check' only: 'Stock Check' is a name it contains, not its own."""
    ws = UUID(seed_workspace())
    return NS(db=db_session, ws=ws, monday=_playbook(db_session, ws, "Monday Stock Check", [{"name": "Count"}]))


def _grants(db, ws):
    from core.models.approval_grants import ApprovalGrant

    return db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == ws).count()


def test_a_timer_by_a_name_one_playbook_contains_names_that_playbook_and_binds_its_id(pantry):
    ask = owner_only.platform_ask(pantry.db, pantry.ws, SCHEDULE, {"playbook_name": "Stock Check", **STOCK_TIMER},
                                  _owners_chat())

    assert ask["requires_confirmation"] is True
    assert f"- playbook: 'Monday Stock Check' (playbook #{pantry.monday.id})" in ask["question_md"]
    assert "- runs at (cron): (empty) → 0 8 * * 1" in ask["question_md"]
    assert "playbook_name" not in ask["question_md"]
    assert ask["params"]["playbook_id"] == pantry.monday.id and "playbook_name" not in ask["params"]
    stored = _grant(pantry.db, ask["grant_id"]).details["params"]
    assert stored["playbook_id"] == pantry.monday.id and "playbook_name" not in stored


def test_of_two_namesakes_the_one_with_steps_gets_the_timer_and_the_card_names_it(pantry):
    _playbook(pantry.db, pantry.ws, "Stock Check")
    staffed = _playbook(pantry.db, pantry.ws, "stock check ", [{"name": "Count"}])

    ask = owner_only.platform_ask(pantry.db, pantry.ws, SCHEDULE, {"playbook_name": "Stock Check", **STOCK_TIMER},
                                  _owners_chat())

    assert f"(playbook #{staffed.id})" in ask["question_md"] and ask["params"]["playbook_id"] == staffed.id


def test_namesakes_none_can_be_picked_from_are_refused_listing_them_with_no_grant(pantry):
    first, second = _playbook(pantry.db, pantry.ws, "Stock Check"), _playbook(pantry.db, pantry.ws, "Stock Check")

    for action, params in ((SCHEDULE, {"playbook_name": "Stock Check", **STOCK_TIMER}),
                           (DELETE_PLAYBOOK, {"playbook_name": "Stock Check"})):
        reply = owner_only.platform_ask(pantry.db, pantry.ws, action, params, _owners_chat())
        assert reply["success"] is False and "requires_confirmation" not in reply, action
        assert f"#{first.id} 'Stock Check'" in reply["error"] and f"#{second.id} 'Stock Check'" in reply["error"]
    assert _grants(pantry.db, pantry.ws) == 0


def test_a_delete_takes_a_whole_name_only_never_one_it_contains(pantry):
    reply = owner_only.platform_ask(pantry.db, pantry.ws, DELETE_PLAYBOOK, {"playbook_name": "Stock Check"},
                                    _owners_chat())
    assert reply["success"] is False and f"#{pantry.monday.id} 'Monday Stock Check'" in reply["error"]
    assert _grants(pantry.db, pantry.ws) == 0

    ask = owner_only.platform_ask(pantry.db, pantry.ws, DELETE_PLAYBOOK, {"playbook_name": "monday stock check"},
                                  _owners_chat())
    assert f"'Monday Stock Check' (playbook #{pantry.monday.id}) is deleted for good" in ask["question_md"]
    assert ask["params"] == {"playbook_id": pantry.monday.id}


def test_a_playbook_name_of_another_workspace_is_never_bound(pantry, seed_workspace):
    _playbook(pantry.db, UUID(seed_workspace()), "Bravo Stock Take")
    reply = owner_only.platform_ask(pantry.db, pantry.ws, SCHEDULE, {"playbook_name": "Bravo Stock Take", **STOCK_TIMER},
                                    _owners_chat())
    assert reply["success"] is False and "Bravo" not in reply.get("question_md", "")
    assert _grants(pantry.db, pantry.ws) == 0


def test_the_click_times_the_playbook_the_card_named(pantry, monkeypatch):
    """A playbook made with the very name after the card does not move the click."""
    from unittest.mock import AsyncMock, patch

    from sqlalchemy import text

    from core.models.approval_grants import ApprovalGrant
    from core.services.approval_grants import grant_grant
    from modules.tools.discovery import handlers_playbooks
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    owner = pantry.db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                              {"e": f"rvw45-{uuid4().hex[:8]}@harbourline.test", "u": f"rvw45-{uuid4().hex[:8]}"}).scalar()
    pantry.db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                           "VALUES (CAST(:ws AS uuid), :user, 'owner', TRUE)"), {"ws": str(pantry.ws), "user": owner})
    monkeypatch.setattr(handlers_playbooks, "_sync_schedule", lambda playbook: (None, None))
    executor = PlatformActionExecutor(pantry.db, pantry.ws)
    executor._full_autonomy = lambda: False
    chat = {"driving_user_id": str(owner), "conversation_id": str(uuid4())}

    def run(params):
        with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
            return asyncio.run(executor.execute(SCHEDULE, params, chat))

    ask = run({"playbook_name": "Stock Check", **STOCK_TIMER})
    assert ask["requires_confirmation"] is True and ask["params"]["playbook_id"] == pantry.monday.id
    newcomer = _playbook(pantry.db, pantry.ws, "Stock Check", [{"name": "Count"}])
    grant_grant(pantry.db.get(ApprovalGrant, ask["grant_id"]), granted_by=f"user:{owner}")
    pantry.db.flush()

    done = run(ask["params"])

    assert done["success"] is True and done["playbook_id"] == pantry.monday.id
    pantry.db.refresh(pantry.monday)
    pantry.db.refresh(newcomer)
    assert pantry.monday.schedule_config["cron_expression"] == "0 8 * * 1"
    assert newcomer.schedule_config == {"type": "manual"}


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
