"""PRD-256 FX-011 (Decision D7): an agent's send on a ticket Auto wrote waits for the owner's click.

Night 12: ticket 2318 ("Confirm the order with the supplier"), written by Auto from the
owner's chat, had its CLI agent run GMAIL_SEND_EMAIL 57 s after it started. Now the send
raises the grant card (FX-008's text), nothing goes out, the session's ticket parks on the
card; the click sends once, notes it on the ticket and sends the ticket back to work; a no
fails it. A read never asks, and a ticket a person wrote sends as before. A playbook step,
or any run on Auto's ticket (a heartbeat's included), asks the same way.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

from modules.tools.discovery.agent_sends import AGENT_SEND, CARD_RAISED
from modules.tools.discovery.owner_only import asks_before_a_send

SUPPLIER = "orders@kerbside.example"
SUBJECT = "Order confirmation: 40 Christmas boxes"
SEND = {"action": "GMAIL_SEND_EMAIL",
        "params": {"recipient_email": SUPPLIER, "subject": SUBJECT, "body": "Hi Kerbside,\nPlease confirm the order."}}
OWNER = "user:7"


class _Composio:
    """UnifiedToolExecutor's shape over a fake Composio: it records every action that ran."""

    sent: list = []

    def __init__(self, db):
        self.db = db

    def _resolve_effective_call(self, tool_name, parameters):
        return parameters.get("action"), parameters.get("params"), True

    @asks_before_a_send
    async def execute_tool(self, tool_name, parameters, agent_id=0, tenant_id=None, workspace_id=None,
                           trace_id=None, caller_context=None):
        _Composio.sent.append(parameters.get("action"))
        return {"success": True, "data": {"id": "msg-1"}}


@pytest.fixture
def desk(db_session, seed_workspace, monkeypatch):
    """A workspace with Auto, the agent that buys, and a ticket Auto wrote from the chat."""
    import modules.tools.execution.tool_grants as tool_grants
    import modules.tools.execution.unified_executor as unified_executor
    import services.cli_host_service as host
    from core.models.core import Agent, BoardTask

    monkeypatch.setattr(tool_grants, "_notify_approval_pending", lambda grant, ws: None)
    monkeypatch.setattr(host, "publish_note_line", lambda ws, task_id, note: None)
    monkeypatch.setattr(unified_executor, "UnifiedToolExecutor", _Composio)
    monkeypatch.setattr(_Composio, "sent", [])
    ws = UUID(seed_workspace())
    common = dict(description="", status="active", configuration={}, workspace_id=ws, created_by="test",
                  owner_type="workspace", owner_id=str(ws))
    auto = Agent(name="Auto", slug=f"auto-{ws}", agent_type="system", is_system_agent=True, **common)
    buyer = Agent(name="CHRISTMAS BOX", agent_type="chatbot", **common)
    db_session.add_all([auto, buyer])
    db_session.flush()
    ticket = _ticket(db_session, ws, created_by_type="agent", created_by_id=str(auto.id), agent=buyer.id)
    return NS(db=db_session, ws=ws, auto=auto, buyer=buyer, ticket=ticket, BoardTask=BoardTask)


def _ticket(db, ws, *, created_by_type, created_by_id, agent, **extra):
    from core.models.core import BoardTask

    row = BoardTask(workspace_id=ws, title="Confirm the order with the supplier", status="in_progress",
                    assigned_agent_id=agent, created_by_type=created_by_type, created_by_id=created_by_id,
                    runtime_ref={}, **extra)
    db.add(row)
    db.flush()
    return row


def _send(desk, context, send=SEND):
    return asyncio.run(_Composio(desk.db).execute_tool(
        "composio_execute", send, agent_id=desk.buyer.id, workspace_id=desk.ws, caller_context=context))


def _grants(desk):
    from core.models.approval_grants import SUBJECT_TOOL_CALL, ApprovalGrant

    return desk.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == desk.ws,
                                               ApprovalGrant.subject_type == SUBJECT_TOOL_CALL).all()


def _session(desk):
    return {"session_task_id": desk.ticket.id}


def _click(desk, grant):
    from api.approval_grants import _resume_tool_call
    from core.services.approval_grants import grant_grant

    grant_grant(grant, granted_by=OWNER)
    desk.db.flush()
    asyncio.run(_resume_tool_call(desk.db, grant))
    desk.db.flush()


def _park(desk):
    from services.cli_host_service import _park_for_answer

    desk.db.refresh(desk.ticket)
    return _park_for_answer(desk.db, desk.ticket, dict(desk.ticket.runtime_ref or {}))


# ── The session's send raises the card; nothing goes out ───────────────────────────────

def test_a_sessions_send_on_autos_ticket_raises_the_card_and_sends_nothing(desk):
    reply = _send(desk, _session(desk))

    assert _Composio.sent == []
    assert reply["requires_confirmation"] is True
    assert reply["message"] == CARD_RAISED.format(subject=SUBJECT, recipient=SUPPLIER)
    assert reply["message"].startswith(f"Card raised: send {SUBJECT} to {SUPPLIER}. Nothing goes out")
    grants = _grants(desk)
    assert len(grants) == 1
    card = grants[0].question_md
    assert SUPPLIER in card and SUBJECT in card and "Hi Kerbside," in card
    assert grants[0].details[AGENT_SEND]["task_id"] == desk.ticket.id
    assert grants[0].agent_id == desk.buyer.id


def test_the_sessions_ticket_parks_on_the_card_naming_it(desk):
    reply = _send(desk, _session(desk))
    _send(desk, _session(desk))   # the same send again: the same card, one entry on the ledger

    assert len(_grants(desk)) == 1
    assert _park(desk) == "blocked"
    assert f"#{reply['grant_id']}" in desk.ticket.blocked_reason
    assert "send card" in desk.ticket.blocked_reason


# ── The click sends once; a no fails the ticket ───────────────────────────────────────

def test_the_owners_click_sends_once_notes_it_and_resumes_the_ticket(desk):
    from core.models.approval_grants import ApprovalGrant
    from modules.tools.execution.tool_grants import GRANT_CONSUMED_BY

    reply = _send(desk, _session(desk))
    _park(desk)
    grant = desk.db.get(ApprovalGrant, reply["grant_id"])
    _click(desk, grant)

    assert _Composio.sent == ["GMAIL_SEND_EMAIL"]
    assert grant.details["executed_result"]["success"] is True
    assert grant.revoked_by == GRANT_CONSUMED_BY            # single-use: claimed by the one run
    desk.db.refresh(desk.ticket)
    assert desk.ticket.status == "assigned"                   # back to work: the session resumes
    notes = [entry["note"] for entry in desk.ticket.runtime_ref.get("session_notes", [])]
    assert any(note.startswith("Sent on the owner's click at ") and note.endswith(f": {SUPPLIER}, {SUBJECT}")
               for note in notes)
    from api.approval_grants import _resume_tool_call

    asyncio.run(_resume_tool_call(desk.db, grant))            # the same click replayed: claimed, nothing sends
    assert _Composio.sent == ["GMAIL_SEND_EMAIL"] and len(_grants(desk)) == 1


def test_a_declined_card_fails_the_ticket_with_the_reason(desk):
    from api.approval_grants import _fail_subject
    from core.models.approval_grants import ApprovalGrant
    from core.services.approval_grants import deny_grant

    reply = _send(desk, _session(desk))
    _park(desk)
    grant = desk.db.get(ApprovalGrant, reply["grant_id"])
    deny_grant(grant, revoked_by=OWNER)
    _fail_subject(desk.db, grant)

    assert _Composio.sent == []
    assert desk.ticket.status == "failed"
    assert desk.ticket.error_message == f"The owner declined the send: {SUBJECT} to {SUPPLIER}"


# ── What never asks, and the other lanes on Auto's brief ──────────────────────────────

def test_a_read_never_asks(desk):
    reply = _send(desk, _session(desk), {"action": "GMAIL_FETCH_EMAILS", "params": {"query": "from:kerbside"}})

    assert reply["success"] is True and _Composio.sent == ["GMAIL_FETCH_EMAILS"]
    assert _grants(desk) == []


def test_a_send_on_a_ticket_a_person_wrote_sends_directly(desk):
    theirs = _ticket(desk.db, desk.ws, created_by_type="user", created_by_id="7", agent=desk.buyer.id)
    reply = _send(desk, {"session_task_id": theirs.id})

    assert reply["success"] is True and _Composio.sent == ["GMAIL_SEND_EMAIL"]
    assert _grants(desk) == []


def test_another_agents_ticket_is_not_autos(desk):
    theirs = _ticket(desk.db, desk.ws, created_by_type="agent", created_by_id=str(desk.buyer.id), agent=desk.buyer.id)

    assert _send(desk, {"session_task_id": theirs.id})["success"] is True
    assert _Composio.sent == ["GMAIL_SEND_EMAIL"]


def test_a_board_run_on_autos_ticket_asks_the_same_way(desk):
    """An API agent's run of the ticket (the dispatcher's, a heartbeat's review ticket's)."""
    reply = _send(desk, {"source": "heartbeat", "board_task_id": desk.ticket.id})

    assert _Composio.sent == [] and reply["message"].startswith("Card raised: send ")
    assert _grants(desk)[0].details[AGENT_SEND]["lane"] == "board"


def _auto_run(desk, triggered_by="platform_action"):
    """A playbook run Auto's platform call started (PRD-204 S9: triggered_by 'platform_action')."""
    from core.models.core import RecipeExecution, WorkflowTemplate

    playbook = WorkflowTemplate(template_id=f"fx011-{uuid4().hex[:8]}", name="Supplier orders", description="",
                                workspace_id=desk.ws, template_definition={"steps": []}, steps=[], created_by="test")
    desk.db.add(playbook)
    desk.db.flush()
    run = f"exec-{uuid4().hex[:12]}"
    desk.db.add(RecipeExecution(execution_id=run, recipe_id=playbook.id, workspace_id=desk.ws, status="running",
                                input_data={}, attempt_count=1, triggered_by=triggered_by))
    desk.db.flush()
    return run


def test_a_playbook_step_auto_started_asks_the_same_way(desk):
    run = _auto_run(desk)
    _ticket(desk.db, desk.ws, created_by_type="recipe", created_by_id=None, agent=desk.buyer.id,
            source_type="recipe", source_id=run)
    reply = _send(desk, {"playbook_execution_id": run, "playbook_step": 2})

    assert _Composio.sent == [] and reply["message"].startswith("Card raised: send ")
    assert _grants(desk)[0].details[AGENT_SEND]["lane"] == "playbook"


def test_a_playbook_run_with_no_card_still_asks(desk):
    run = _auto_run(desk)
    reply = _send(desk, {"playbook_execution_id": run, "playbook_step": 1})

    assert _Composio.sent == [] and reply["message"].startswith("Card raised: send ")
    assert _grants(desk)[0].details[AGENT_SEND]["task_id"] is None


def test_a_cli_agents_playbook_step_on_an_auto_started_run_asks_and_parks(desk):
    """A CLI agent's step is a session ticket the runner files (api/recipe_executor: recipe:<run>:<step>)."""
    run = _auto_run(desk)
    step = _ticket(desk.db, desk.ws, created_by_type="system", created_by_id=None, agent=desk.buyer.id,
                   source_type="recipe", source_id=f"recipe:{run}:2")
    reply = _send(desk, {"session_task_id": step.id})

    assert _Composio.sent == [] and reply["message"].startswith("Card raised: send ")
    desk.db.refresh(step)
    assert [entry["grant_id"] for entry in step.runtime_ref["session_asks"]] == [reply["grant_id"]]


def test_a_playbook_run_a_person_started_sends_directly(desk):
    run = _auto_run(desk, triggered_by="user@example.com")

    assert _send(desk, {"playbook_execution_id": run})["success"] is True
    assert _Composio.sent == ["GMAIL_SEND_EMAIL"]


def test_one_send_asked_on_two_tickets_keeps_the_first_tickets_card(desk):
    from modules.tools.discovery.agent_sends import ON_ANOTHER_TICKET

    first = _send(desk, _session(desk))
    again = _ticket(desk.db, desk.ws, created_by_type="agent", created_by_id=str(desk.auto.id), agent=desk.buyer.id)
    second = _send(desk, {"session_task_id": again.id})

    assert _Composio.sent == [] and second["grant_id"] == first["grant_id"]
    assert second["message"] == ON_ANOTHER_TICKET.format(task_id=desk.ticket.id)
    assert _grants(desk)[0].details[AGENT_SEND]["task_id"] == desk.ticket.id
    desk.db.refresh(again)
    assert not (again.runtime_ref or {}).get("session_asks")


def test_a_ticket_in_another_workspace_is_never_read(desk, seed_workspace):
    other = UUID(seed_workspace())
    theirs = _ticket(desk.db, other, created_by_type="agent", created_by_id=str(desk.auto.id), agent=None)

    assert _send(desk, {"session_task_id": theirs.id})["success"] is True
    assert _Composio.sent == ["GMAIL_SEND_EMAIL"]


class _Reader:
    """An executor built without a session (the F088 fixture's shape): it has no ``db``."""

    ran: list = []

    def _resolve_effective_call(self, tool_name, parameters):
        return tool_name, parameters, False

    @asks_before_a_send
    async def execute_tool(self, tool_name, parameters, agent_id=0, tenant_id=None, workspace_id=None,
                           trace_id=None, caller_context=None):
        _Reader.ran.append(tool_name)
        return {"success": True}


def test_a_call_that_is_not_a_send_never_reads_the_executors_session(monkeypatch):
    monkeypatch.setattr(_Reader, "ran", [])
    reader = _Reader.__new__(_Reader)

    out = asyncio.run(reader.execute_tool("search_knowledge", {"query": "Callum"}, agent_id=1,
                                          caller_context={"session_task_id": 1}))

    assert out == {"success": True} and _Reader.ran == ["search_knowledge"]
