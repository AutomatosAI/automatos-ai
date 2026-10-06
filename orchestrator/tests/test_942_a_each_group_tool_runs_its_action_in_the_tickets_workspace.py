"""#942 (A): every tool a group adds runs the API agents' own action, through the
executor, as the ticket's agent in the ticket's workspace.

Platform actions go through the ``platform_execute`` dispatcher; ``generate_document``
is an executor tool, not a platform action, so it is routed by name, as an API
agent's call is. Either way the executor is handed the ticket's workspace and agent,
and a workspace, agent or extra field in the call never reaches the action.
"""
from __future__ import annotations

import asyncio

import pytest

from services import session_tools as st

CTX = st.SessionContext(task_id=2001, agent_id=268, agent_name="Business Analyst", workspace_id="ws-shop",
                        mission_field_id="field-7")
SMUGGLED = {"workspace_id": "ws-other", "agent_id": 1, "_agent_id": 1, "field_id": "field-other", "task_id": 9}

CASES = [
    # Brand kit at generation (night 10 prep): the documents group reads the owner's templates.
    ("list_templates", {"format": "docx", "category": "letter"}, "platform_list_templates",
     {"format": "docx", "category": "letter"}),
    ("get_template_schema", {"template_id": "6f1c0d1e-2b7a-4c55-9a1e-1d2f3a4b5c6d"}, "platform_get_template_schema",
     {"template_id": "6f1c0d1e-2b7a-4c55-9a1e-1d2f3a4b5c6d"}),
    ("list_playbooks", {"status": "active"}, "platform_list_playbooks", {"status_filter": "active"}),
    ("get_playbook", {"playbook_name": "Weekly digest"}, "platform_get_playbook", {"playbook_name": "Weekly digest"}),
    ("run_playbook", {"playbook_id": 12, "inputs": {"input": "October"}, "wait_for_me": True},
     "platform_execute_playbook", {"playbook_id": 12, "input_data": {"input": "October"}, "wait_for_me": True}),
    ("get_latest_report", {"agent_name": "RESEARCHER", "report_type": "research"},
     "platform_get_latest_report", {"agent_name": "RESEARCHER", "report_type": "research"}),
    ("list_missions", {"state": "running", "limit": 5}, "platform_list_missions", {"state": "running", "limit": 5}),
    ("get_mission", {"mission_id": "#0188"}, "platform_get_mission", {"mission_id": "#0188"}),
    ("search_mission_findings", {"query": "supplier prices", "limit": 4},
     "platform_field_query", {"query": "supplier prices", "top_k": 4}),
]


class _Executor:
    calls: list = []

    def __init__(self, db):
        pass

    async def execute_tool(self, **kwargs):
        type(self).calls.append(kwargs)
        return {"success": True, "answer": "ok"}


@pytest.fixture
def executor(monkeypatch):
    from modules.tools.execution import unified_executor

    monkeypatch.setattr(_Executor, "calls", [])
    monkeypatch.setattr(unified_executor, "UnifiedToolExecutor", _Executor)
    return _Executor


def _call(name, arguments):
    tool = st.get_tool(name)
    # What a tool doesn't declare is smuggled in, and must be dropped. A field it does
    # declare is its own: get_latest_report's agent_id names whose report, not the caller.
    declared = set(tool.input_schema.get("properties") or {})
    smuggled = {key: value for key, value in SMUGGLED.items() if key not in declared}
    return asyncio.run(st.call_tool(None, tool, st.resolve_parameters(tool, {**arguments, **smuggled}, CTX), CTX))


def _only(executor):
    (call,) = executor.calls
    assert call["workspace_id"] == "ws-shop" and call["agent_id"] == 268    # the ticket's, never the call's
    return call


@pytest.mark.parametrize("name, arguments, action, params", CASES)
def test_a_group_tool_runs_its_platform_action_through_the_dispatcher(executor, name, arguments, action, params):
    _call(name, arguments)
    call = _only(executor)
    assert call["tool_name"] == st.PLATFORM_DISPATCHER
    assert call["parameters"] == {"action": action, "params": params}
    assert call["trace_id"] == f"session:2001:{name}"
    # the mission's field comes from the ticket's run, never from the call
    assert call["caller_context"] == {"field_context": {"field_id": "field-7"}}


def test_generate_document_is_routed_by_name_as_the_tickets_agent(executor):
    _call("generate_document", {"title": "October sales", "format": "pdf", "data": {"total": "1,204"},
                                "template_name": "Basic Report"})
    call = _only(executor)
    assert call["tool_name"] == "generate_document"
    assert call["parameters"] == {"title": "October sales", "format": "pdf", "data": {"total": "1,204"},
                                  "template_name": "Basic Report"}


def test_every_platform_action_a_group_tool_names_is_registered():
    from modules.tools.discovery import get_action_registry

    registry = get_action_registry()
    for _name, _arguments, action, _params in CASES:
        assert registry.get(action) is not None, action


def test_the_writes_say_so_and_the_rest_only_read():
    writes = {t.name for t in st.SESSION_TOOLS[10:] if not t.reads_only}
    # PRD-255 US-013: create_template / update_template write the workspace's templates.
    assert writes == {"generate_document", "run_playbook", "create_template", "update_template"}


@pytest.mark.parametrize("name, arguments, words", [
    ("get_playbook", {}, "Name the playbook"),
    ("run_playbook", {"inputs": {"input": "x"}}, "Name the playbook"),
    ("get_latest_report", {"report_type": "audit"}, "needs the agent"),
    ("get_mission", {}, "needs the mission"),
    ("search_mission_findings", {"query": "  "}, "needs a query"),
    ("generate_document", {"title": "t"}, "needs a title, a format"),
    ("get_template_schema", {"template_id": " "}, "needs the template"),
])
def test_a_call_missing_what_it_needs_is_refused_and_runs_nothing(executor, name, arguments, words):
    with pytest.raises(st.SessionToolRefused) as refused:
        st.resolve_parameters(st.get_tool(name), arguments, CTX)
    assert words in str(refused.value)
    assert executor.calls == []


def test_the_document_formats_the_tool_names_are_the_ones_the_platform_renders():
    from core.models.core import DOCUMENT_TEMPLATE_FORMATS

    text = st.get_tool("generate_document").input_schema["properties"]["format"]["description"]
    for fmt in DOCUMENT_TEMPLATE_FORMATS:
        assert fmt in text, fmt


def test_the_documents_group_reads_the_templates_beside_generate_document():
    from core.models.core import DOCUMENT_TEMPLATE_FORMATS
    from services import session_tool_groups as groups

    documents = next(g for g in groups.SESSION_TOOL_GROUPS if g.id == "documents")
    assert documents.tools == ("generate_document", "list_templates", "get_template_schema", "render_preview",
                               "create_template", "update_template")
    assert st.get_tool("list_templates").reads_only and st.get_tool("get_template_schema").reads_only
    listed = st.get_tool("list_templates").input_schema["properties"]["format"]["enum"]
    assert tuple(listed) == DOCUMENT_TEMPLATE_FORMATS
