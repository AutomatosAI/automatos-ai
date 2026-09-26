"""F155 — a widget-born mission's task is offered only its widget key's tools.

A widget-born mission's tasks run under the widget origin, and the tool
executor's gate refuses what the key's scopes do not grant. But
AgentFactory.execute_with_prompt still offered the task the agent's whole tool
list, and injected hints naming the owner's connected apps and their actions.
It also handled a call to the memory tool itself, before the executor,
reading and writing the workspace's durable memory with no gate. On a widget
turn the task is now offered what the key's scopes allow and no connected
apps, and the memory tool is refused.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

WORKSPACE = "11111111-1111-1111-1111-111111111111"


def _schema(name, actions=None):
    parameters = {"type": "object", "properties": {"action": {"type": "string", "enum": actions}}} if actions else {}
    return {"type": "function", "function": {"name": name, "parameters": parameters}}


OFFERED = [
    _schema("search_knowledge"),
    _schema("memory"),
    _schema("composio_execute"),
    _schema("platform_execute", ["platform_list_documents", "platform_create_agent"]),
]


def _widget_turn():
    from core.security.surface import WIDGET, turn_surface

    return turn_surface(WIDGET, ("chat", "documents:read"), None)


def _answer(content="Done.", tool_calls=None):
    return NS(content=content, tool_calls=tool_calls or [], finish_reason="stop", model="stub-model",
              provider="stub", usage={"total_tokens": 3, "prompt_tokens": 2, "completion_tokens": 1})


class _LLM:
    def __init__(self, responses):
        self.responses, self.offered, self.config = list(responses), [], NS(model="stub-model")

    async def generate_response(self, messages, tools=None, **kwargs):
        self.offered.append(tools)
        return self.responses[min(len(self.offered) - 1, len(self.responses) - 1)]


def _run_task(monkeypatch, responses):
    """Drive one task through the real execute_with_prompt; return what the model
    was offered first, the hint injections, and the memory tool's calls."""
    import modules.agents.factory.agent_factory as af

    hints, remembered = [], []

    class _MemoryTool:
        def __init__(self, store, workspace_id):
            pass

        async def handle(self, args):
            remembered.append(args)
            return "stored"

    monkeypatch.setattr(af, "get_monitoring_service", lambda: NS(record_agent_execution=lambda **kw: None))
    monkeypatch.setattr("modules.tools.tool_router.get_tools_for_agent_async",
                        AsyncMock(side_effect=lambda **kw: [dict(schema) for schema in OFFERED]))

    async def _inject_composio_hints(self, *a, **k):
        hints.append("hints")

    monkeypatch.setattr(af.AgentFactory, "_inject_composio_hints", _inject_composio_hints)
    monkeypatch.setattr("modules.memory.memory_tool.MemoryToolBackend", _MemoryTool)
    monkeypatch.setattr("modules.memory.memory_tool.DurableMemoryStoreBackend", lambda: None)

    factory = af.AgentFactory.__new__(af.AgentFactory)
    factory.db_session, factory.active_agents, factory.logger = None, {}, logging.getLogger("f155")
    llm = _LLM(responses)
    factory.active_agents[101] = af.AgentRuntime(
        agent_id=101, metadata=af.AgentMetadata(name="Scribe", agent_type="worker"), llm_manager=llm,
        lifecycle_state=af.AgentLifecycle.ACTIVE, created_at=datetime.now(timezone.utc),
        tools=[{"provider": "Composio", "name": "GMAIL"}],
        tool_executor=NS(execute_tool=AsyncMock(return_value={"ok": True})), workspace_id=WORKSPACE)
    asyncio.run(factory.execute_with_prompt(agent=101, prompt="Draft the reply.", system_prompt="You are Scribe.",
                                            use_memory=False, max_retries=1))
    return llm.offered[0], hints, remembered


def test_a_widget_born_task_is_offered_only_its_keys_tools_and_no_connected_apps(monkeypatch):
    with _widget_turn():
        offered, hints, _remembered = _run_task(monkeypatch, [_answer()])
    assert [schema["function"]["name"] for schema in offered] == ["search_knowledge", "platform_execute"]
    assert offered[1]["function"]["parameters"]["properties"]["action"]["enum"] == ["platform_list_documents"]
    assert hints == []
    offered, hints, _remembered = _run_task(monkeypatch, [_answer()])
    assert [schema["function"]["name"] for schema in offered] == [schema["function"]["name"] for schema in OFFERED]
    assert hints == ["hints"]


def test_a_widget_born_task_cannot_reach_the_memory_tool(monkeypatch):
    call = {"id": "call-1", "type": "function",
            "function": {"name": "memory", "arguments": json.dumps({"command": "view", "path": "/memories"})}}
    with _widget_turn():
        _offered, _hints, remembered = _run_task(monkeypatch, [_answer("", [call]), _answer()])
    assert remembered == []
    _offered, _hints, remembered = _run_task(monkeypatch, [_answer("", [call]), _answer()])
    assert remembered == [{"command": "view", "path": "/memories"}]
