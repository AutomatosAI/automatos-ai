"""F341 (night 10): a document an agent generates while working a card is on that card.

Ticket #2099 (Ops Manager, a Claude Code session) called generate_document through
its session tools. The PDF became Deliverable 837dd5fa with ``source_type:
"agent_output"`` and ``source_id: None``, so ``GET /api/deliverables?source_type=task
&source_id=2099`` returned nothing: the card never showed the document it made, and
the completion checks could not see it. An API agent's generate_document on a board
card dropped its card the same way (the executor never passed the caller context on).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from services import session_tools as st

TICKET = 2099
WS = "6d0b5c1e-8f1a-4c2b-9d3e-0a1b2c3d4e61"
CTX = st.SessionContext(task_id=TICKET, agent_id=341, agent_name="Ops Manager", workspace_id=WS)
PARAMS = {"title": "Invoice HL-2026-0145", "format": "pdf",
          "data": {"sections": [{"title": "Invoice", "content": "Two bags of Guji, 2 x 12.50."}]}}


def _capture_executor(monkeypatch):
    from modules.tools.execution import unified_executor

    calls = []

    class _Executor:
        def __init__(self, db):
            pass

        async def execute_tool(self, **kwargs):
            calls.append(kwargs)
            return {"success": True, "results": [{"status": "success", "filename": "invoice.pdf"}]}

    monkeypatch.setattr(unified_executor, "UnifiedToolExecutor", _Executor)
    return calls


def test_a_sessions_generate_document_call_names_its_ticket(monkeypatch):
    from modules.tools.execution.generate_document_tool import SESSION_CARD_KEY, card_of

    calls = _capture_executor(monkeypatch)
    tool = st.get_tool("generate_document")
    asyncio.run(st.call_tool(None, tool, st.resolve_parameters(tool, PARAMS, CTX), CTX))

    assert calls[0]["tool_name"] == "generate_document"
    assert calls[0]["caller_context"] == {SESSION_CARD_KEY: TICKET}
    assert card_of(calls[0]["caller_context"]) == TICKET


def test_a_mission_sessions_call_keeps_its_field_beside_the_ticket(monkeypatch):
    from modules.tools.execution.generate_document_tool import SESSION_CARD_KEY

    calls = _capture_executor(monkeypatch)
    ctx = st.SessionContext(task_id=TICKET, agent_id=341, agent_name="Ops Manager", workspace_id=WS,
                            mission_field_id="f-224")
    tool = st.get_tool("generate_document")
    asyncio.run(st.call_tool(None, tool, st.resolve_parameters(tool, PARAMS, ctx), ctx))

    assert calls[0]["caller_context"] == {"field_context": {"field_id": "f-224"}, SESSION_CARD_KEY: TICKET}


def test_other_session_tools_carry_no_more_context_than_before(monkeypatch):
    calls = _capture_executor(monkeypatch)
    tool = st.get_tool("list_templates")
    asyncio.run(st.call_tool(None, tool, st.resolve_parameters(tool, {"format": "pdf"}, CTX), CTX))

    assert calls[0]["tool_name"] == st.PLATFORM_DISPATCHER and calls[0]["caller_context"] is None


@pytest.mark.parametrize("context, card", [
    ({"session_task_id": TICKET}, TICKET),
    ({"board_task_id": 612, "field_context": {"field_id": "f-1"}}, 612),
    ({"board_task_id": "612"}, 612),
    ({"conversation_id": "c-1"}, None),
    ({"board_task_id": True}, None),
    ({"board_task_id": "not-a-card"}, None),
    (None, None),
])
def test_the_card_is_read_from_the_server_built_context_only(context, card):
    from modules.tools.execution.generate_document_tool import card_of

    assert card_of(context) == card


def test_the_executor_hands_the_card_on_to_the_document_tool():
    from modules.tools.execution import exec_document

    seen = []

    async def execute_tool(**kwargs):
        seen.append(kwargs)
        return {"success": True, "results": [{"status": "success", "filename": "invoice.pdf"}]}

    db = NS(get=lambda model, key: NS(settings={}), query=lambda *a, **k: None)
    executor = NS(db=db, platform_tools=NS(execute_tool=execute_tool))
    asyncio.run(exec_document.execute_generate_document(
        executor, "generate_document", PARAMS, 341, workspace_id=WS,
        caller_context={"session_task_id": TICKET}))

    assert seen == [{"tool_name": "generate_document", "parameters": PARAMS, "agent_id": 341, "card_id": TICKET}]


@pytest.fixture
def registrations(monkeypatch):
    """generate_document's render and registration, faked; returns what register_as_deliverable was asked."""
    import modules.documents.generation_service as generation_service
    from modules.tools.execution import generate_document_tool as gdt

    asked = []

    class _Service:
        def __init__(self, db, workspace_id):
            pass

        async def generate(self, **kwargs):
            return NS(filename="invoice.pdf", format="pdf", download_url="/api/documents/generated/invoice.pdf",
                      size=2048, content="# Invoice", template_id=None, template_name=None)

        def register_as_deliverable(self, result, **kwargs):
            asked.append(kwargs)
            return {"success": True, "deliverable_id": "837dd5fa"}

        def share_link(self, result):
            return None

    async def _no_ingest(*args, **kwargs):
        return None

    monkeypatch.setattr(generation_service, "DocumentGenerationService", _Service)
    monkeypatch.setattr(gdt, "_ingest", _no_ingest)
    return asked


def _make(card_id=None):
    from modules.tools.execution import generate_document_tool as gdt

    agent = NS(id=341, name="Ops Manager", user_id=None)
    request = gdt.document_request(PARAMS)
    return asyncio.run(gdt.make_document(None, request, agent, WS, card_id=card_id))


def test_the_document_is_the_cards_deliverable(registrations):
    answer = _make(card_id=TICKET)

    assert (registrations[0]["source_type"], registrations[0]["source_id"]) == ("task", str(TICKET))
    assert answer["results"][0]["deliverable_id"] == "837dd5fa"


def test_without_a_card_the_document_stays_the_agents_own_output(registrations):
    _make()

    assert (registrations[0]["source_type"], registrations[0]["source_id"]) == ("agent_output", None)
