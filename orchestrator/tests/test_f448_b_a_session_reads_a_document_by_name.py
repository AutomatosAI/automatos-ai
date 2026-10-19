"""P256-FIX-N17-9 (night 17, F448) (B) — a session reads a knowledge-base document whole, by its name.

WRITER's brief on #1756 named harbourline-green-coffee-price-list.csv. Its session had
only search_documents, which timed out twice; Auto had read the same CSV with
platform_read_document an hour before. ``read_document`` is now a core session tool, right
after search_documents: it forwards only the document's id or filename and the page, runs
platform_read_document through the dispatcher as the ticket's agent (so the agent's team
scoping applies), and puts a spreadsheet's copy in the ticket's own folder to count with code.
"""
from __future__ import annotations

import asyncio

import pytest

from modules.tools.execution import session_document_folder
from services import session_tool_groups as groups
from services import session_tools as st
from services.session_file_reads import NO_SPREADSHEET_COPY
from services.spreadsheet_workspace import COUNT_WITH_CODE
from tests import test_f370_a_later_step_opens_an_earlier_steps_picture as f370

CTX = st.SessionContext(task_id=3044, agent_id=268, agent_name="WRITER", workspace_id="ws-harbourline")
CSV = b"coffee,price\nHarbour Blend,22.00\nQuay Decaf,24.50\n"
SHEET = {"success": True, "document_id": 708, "filename": "x.csv", "file_type": "csv", "page": 0,
         "total_pages": 1, "workspace_path": "knowledge/x.csv", "row_count": 2,
         "count_with_code": COUNT_WITH_CODE, "content": "coffee,price\nHarbour Blend,22.00"}


@pytest.fixture
def executor(monkeypatch):
    """The API agents' executor, answering platform_read_document with ``answer``; the
    worker's read of the stored copy and the folder write, faked."""
    from modules.tools.execution import unified_executor

    seen = {"calls": [], "answer": dict(SHEET), "stored": {("ws-harbourline", "workspace", "knowledge/x.csv"): CSV}}

    class _Executor:
        def __init__(self, db):
            pass

        async def execute_tool(self, *, tool_name, parameters, agent_id, workspace_id, trace_id, caller_context):
            seen["calls"].append({"tool_name": tool_name, "parameters": parameters, "agent_id": agent_id,
                                  "workspace_id": workspace_id, "caller_context": caller_context})
            return dict(seen["answer"])

    monkeypatch.setattr(unified_executor, "UnifiedToolExecutor", _Executor)
    monkeypatch.setattr("modules.documents.thumbnails.sources.read_source",
                        lambda ws, storage, path: seen["stored"].get((str(ws), storage, path)))
    monkeypatch.setattr(session_document_folder, "WorkspaceClient", f370._Folder)
    monkeypatch.setattr(f370._Folder, "written", {})
    return seen


def _call(arguments):
    tool = st.get_tool("read_document")
    return asyncio.run(st.call_tool(None, tool, st.resolve_parameters(tool, arguments, CTX), CTX))


def test_read_document_is_a_core_tool_right_after_search_documents():
    names = st.tool_names()
    assert names.index("read_document") == names.index("search_documents") + 1
    assert "read_document" in groups.CORE_TOOLS
    assert st.get_tool("read_document").action == "platform_read_document" and st.get_tool("read_document").reads_only
    assert st.equivalent_of("platform_read_document") == "read_document"     # a skill naming it is pointed here
    description = st.get_tool("read_document").description
    assert "filename" in description and "a copy in your folder" in description


def test_the_scope_forwards_only_the_document_and_the_page():
    tool = st.get_tool("read_document")
    asked = {"document_name": "x.csv", "page": 2, "query": "prices", "workspace_id": "another", "_agent_id": 1}
    assert st.resolve_parameters(tool, asked, CTX) == {"filename": "x.csv", "page": 2}
    assert st.resolve_parameters(tool, {"document_id": 708}, CTX) == {"document_id": 708}
    with pytest.raises(st.SessionToolRefused, match="filename"):
        st.resolve_parameters(tool, {"page": 1}, CTX)


def test_it_runs_platform_read_document_through_the_dispatcher_as_the_tickets_agent(executor):
    executor["answer"] = {"success": True, "document_id": 12, "filename": "brief.md", "content": "The brief."}

    result = _call({"filename": "brief.md"})

    assert result["content"] == "The brief."
    (call,) = executor["calls"]
    assert call["tool_name"] == "platform_execute"
    assert call["parameters"] == {"action": "platform_read_document", "params": {"filename": "brief.md"}}
    assert (call["agent_id"], call["workspace_id"], call["caller_context"]) == (268, "ws-harbourline", None)


def test_a_spreadsheet_arrives_as_a_copy_in_the_tickets_folder(executor):
    result = _call({"filename": "x.csv"})

    copy = "sessions/3044/doc-708-x.csv"
    assert result["workspace_path"] == copy and result["count_with_code"] == COUNT_WITH_CODE
    assert f370._Folder.written == {("ws-harbourline", copy): CSV}


def test_a_spreadsheet_that_cannot_be_copied_says_to_read_every_page(executor):
    executor["stored"].clear()

    result = _call({"filename": "x.csv"})

    assert "workspace_path" not in result and "count_with_code" not in result
    assert result["note"] == NO_SPREADSHEET_COPY and result["row_count"] == 2
    assert f370._Folder.written == {}
