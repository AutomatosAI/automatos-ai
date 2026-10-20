"""PRD-256 F430 (B): each Socials session tool runs the API agents' own action, as the ticket's
agent in the ticket's workspace, with exactly the fields that action declares.

The tools go through the ``platform_execute`` dispatcher (``session_tools.call_tool``), so the
executor's registry check, the handler's Socials switches and its review log all apply as they do
to an API agent's call. A workspace or agent in the call never reaches the action. A session is no
human driving the turn (no caller context), so it drafts, changes and submits as an API agent's
board run does, and approves, schedules and publishes nothing.

ONE DEFINITION: every Socials session tool's input schema is its registry action's own
(``core.social_post_schema``), so a field a session sends is one the action takes, and render,
note, status, queue and limit all reach it.

End to end (the US-116 SQLite harness): a session's draft and its submit, scoped here, run
through ``PlatformActionExecutor`` with the agent keys ``exec_platform`` mints, and the post's
review log names the agent.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251w1_social_post_tools as harness  # noqa: E402
from services import session_tools as st  # noqa: E402
from services.session_socials_tools import SOCIAL_TOOL_SPECS  # noqa: E402
from tests.test_prd251w1_social_post_tools import ACTOR, _created, _row, _tool  # noqa: E402

env = harness.env  # the US-116 fixture: SQLite, the real router and executor

CTX = st.SessionContext(task_id=2001, agent_id=268, agent_name="Social Media Director", workspace_id="ws-shop")
SMUGGLED = {"workspace_id": "ws-other", "agent_id": 1, "_agent_id": 1, "_agent_name": "Someone else"}
POST = "6f1c0d1e-2b7a-4c55-9a1e-1d2f3a4b5c6d"

CASES = [
    ("create_social_post",
     {"title": "Harvest Club", "template": "Quote card", "variables": {"quote": "Two coffees, £32."},
      "copy": {"base": "October's box."}, "render": False},
     "platform_create_social_post"),
    ("update_social_post", {"post_id": POST, "variables": {"quote": "Three coffees."}, "render": True},
     "platform_update_social_post"),
    ("submit_social_post", {"post_id": POST, "note": "Check the price."}, "platform_submit_social_post"),
    ("get_social_post", {"post_id": POST}, "platform_get_social_post"),
    ("list_social_posts", {"status": ["draft"], "queue": "open", "limit": 5}, "platform_list_social_posts"),
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
    return asyncio.run(st.call_tool(None, tool, st.resolve_parameters(tool, {**arguments, **SMUGGLED}, CTX), CTX))


@pytest.mark.parametrize("name, arguments, action", CASES)
def test_a_socials_tool_runs_its_action_as_the_tickets_agent(executor, name, arguments, action):
    _call(name, arguments)
    (call,) = executor.calls
    assert call["tool_name"] == st.PLATFORM_DISPATCHER
    assert call["parameters"] == {"action": action, "params": arguments}     # exactly these: render, note, …
    assert call["workspace_id"] == "ws-shop" and call["agent_id"] == 268      # the ticket's, never the call's
    assert call["trace_id"] == f"session:2001:{name}"
    assert call["caller_context"] is None                                    # an agent, no human driving


@pytest.mark.parametrize("name, arguments, words", [
    ("create_social_post", {"copy": {"base": "x"}}, "needs a title"),
    ("create_social_post", {"title": "   "}, "needs a title"),
    ("update_social_post", {"render": True}, "needs the post_id"),
    ("submit_social_post", {"note": "please"}, "needs the post_id"),
    ("get_social_post", {}, "needs the post_id"),
])
def test_a_call_missing_what_it_needs_is_refused_and_runs_nothing(executor, name, arguments, words):
    with pytest.raises(st.SessionToolRefused) as refused:
        st.resolve_parameters(st.get_tool(name), arguments, CTX)
    assert words in str(refused.value)
    assert executor.calls == []


def test_each_socials_tool_takes_exactly_what_its_action_declares():
    from modules.tools.discovery import get_action_registry

    registry = get_action_registry()
    for spec in SOCIAL_TOOL_SPECS:
        tool = st.get_tool(spec["name"])
        declared = registry.get(tool.action).parameters
        assert tool.input_schema["properties"] == declared["properties"], tool.name
        assert tool.input_schema["required"] == list(declared.get("required") or []), tool.name


def test_the_writes_say_so_and_the_reads_only_read():
    reads = {spec["name"]: st.get_tool(spec["name"]).reads_only for spec in SOCIAL_TOOL_SPECS}
    assert reads == {"create_social_post": False, "update_social_post": False, "submit_social_post": False,
                     "get_social_post": True, "list_social_posts": True}


def test_a_sessions_draft_and_its_submit_run_through_the_platform_as_its_agent(env):
    create = st.get_tool("create_social_post")
    scoped = st.resolve_parameters(create, {**harness._draft_fields(), "render": False, **SMUGGLED}, CTX)
    assert not set(SMUGGLED) & set(scoped)
    post = _created(_tool(env, create.action, **scoped))         # exec_platform's agent keys, as the dispatcher adds
    assert _row(env, post["id"]).status == "draft"

    submit = st.get_tool("submit_social_post")
    sent = _tool(env, submit.action, **st.resolve_parameters(submit, {"post_id": post["id"], "note": "Check it."}, CTX))
    assert sent["success"] is True, sent
    row = _row(env, post["id"])
    assert row.status == "needs_approval"
    assert [(entry["action"], entry["by"]) for entry in row.review_log] == [("draft", ACTOR), ("submit", ACTOR)]
    assert row.review_log[-1]["comment"] == "Check it."
