"""PRD-229 US-002 — ask_orchestrator: the tool, task-execution-only, time-boxed.

Pure tests, no DB / LLM / network:
  * the mode gate (excluded_tool_names + strip_actions_from_surface) proves the
    tool is present in TASK_EXECUTION and stripped from CHATBOT — on the callable
    surface AND the prompt action catalog, both directions;
  * the handler is driven with a stubbed US-001 answer_clarification to prove the
    answer path, the time-box → cannot_answer path, and caller resolution from
    SERVER context (never a tool param).
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import modules.tools.discovery.handlers_clarify as hc  # noqa: E402
import services.orchestrator_answers as oa  # noqa: E402
from config import Config  # noqa: E402
from modules.context.modes import (  # noqa: E402
    ContextMode,
    EXECUTION_ONLY_TOOLS,
    excluded_tool_names,
    strip_actions_from_surface,
)
from modules.tools.discovery.action_registry import get_action_registry  # noqa: E402
from modules.tools.discovery.handlers_clarify import ask_orchestrator  # noqa: E402


# ---------------------------------------------------------------------------
# config — the time-box fits inside the task execution envelope
# ---------------------------------------------------------------------------

def test_time_box_fits_inside_smallest_execution_envelope():
    # _POWER_MODE_DEFAULTS: light=120s is the smallest task asyncio.wait_for
    # envelope; the answer round must be well inside it.
    assert isinstance(Config.CLARIFICATION_ANSWER_TIMEOUT, int)
    assert 0 < Config.CLARIFICATION_ANSWER_TIMEOUT < 120


# ---------------------------------------------------------------------------
# registration — the 3-file platform pattern
# ---------------------------------------------------------------------------

def test_ask_orchestrator_registered():
    action = get_action_registry().get("ask_orchestrator")
    assert action is not None
    assert action.name == "ask_orchestrator"
    # question is the only required param; category is the optional governance mark
    assert action.parameters["required"] == ["question"]
    assert set(action.parameters["properties"]["category"]["enum"]) == {
        "destructive", "spend", "scope",
    }
    # not admin/su gated — an executing worker must be able to call it
    assert not action.admin_only and not action.super_admin_only


# ---------------------------------------------------------------------------
# mode gate — visible in TASK_EXECUTION, ABSENT from CHATBOT (both directions)
# ---------------------------------------------------------------------------

def test_excluded_tool_names_both_directions():
    assert "ask_orchestrator" in EXECUTION_ONLY_TOOLS
    assert excluded_tool_names(ContextMode.CHATBOT) == frozenset({"ask_orchestrator"})
    assert excluded_tool_names(ContextMode.TASK_EXECUTION) == frozenset()
    # accepts the string value too (what flows through SectionContext)
    assert excluded_tool_names("chatbot") == frozenset({"ask_orchestrator"})
    assert excluded_tool_names("task_execution") == frozenset()


def _surface():
    return [
        {"type": "function", "function": {
            "name": "platform_execute",
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["ask_orchestrator", "platform_list_agents"]},
                "params": {"type": "object"},
            }, "required": ["action", "params"]},
        }},
        {"type": "function", "function": {"name": "ask_orchestrator", "parameters": {}}},
        {"type": "function", "function": {"name": "workspace_read_file", "parameters": {}}},
    ]


def test_chatbot_surface_strips_ask_orchestrator():
    tools = _surface()
    stripped = strip_actions_from_surface(tools, excluded_tool_names(ContextMode.CHATBOT))
    names = [t["function"]["name"] for t in stripped]
    assert "ask_orchestrator" not in names          # first-class schema dropped
    assert "platform_execute" in names
    dispatcher = next(t for t in stripped if t["function"]["name"] == "platform_execute")
    enum = dispatcher["function"]["parameters"]["properties"]["action"]["enum"]
    assert "ask_orchestrator" not in enum           # pruned from the dispatcher enum
    assert "platform_list_agents" in enum


def test_task_execution_surface_keeps_ask_orchestrator():
    tools = _surface()
    kept = strip_actions_from_surface(tools, excluded_tool_names(ContextMode.TASK_EXECUTION))
    names = [t["function"]["name"] for t in kept]
    assert "ask_orchestrator" in names
    dispatcher = next(t for t in kept if t["function"]["name"] == "platform_execute")
    assert "ask_orchestrator" in dispatcher["function"]["parameters"]["properties"]["action"]["enum"]


def test_strip_is_rebuild_not_mutate():
    tools = _surface()
    original_enum = list(tools[0]["function"]["parameters"]["properties"]["action"]["enum"])
    strip_actions_from_surface(tools, frozenset({"ask_orchestrator"}))
    # the input surface is untouched
    assert tools[0]["function"]["parameters"]["properties"]["action"]["enum"] == original_enum
    assert [t["function"]["name"] for t in tools] == [
        "platform_execute", "ask_orchestrator", "workspace_read_file",
    ]


def test_prompt_catalog_excludes_in_chat_keeps_in_execution():
    registry = get_action_registry()
    full = registry.build_prompt_summary(exclude_admin=True, exclude_promoted=True)
    assert "ask_orchestrator" in full  # execution lanes advertise it
    chat = registry.build_prompt_summary(
        exclude_admin=True, exclude_promoted=True,
        exclude_names=list(excluded_tool_names(ContextMode.CHATBOT)),
    )
    assert "ask_orchestrator" not in chat  # chat catalog does not


# ---------------------------------------------------------------------------
# handler — answer path, time-box path, caller-from-server-context
# ---------------------------------------------------------------------------

def _server_params(**over):
    p = {
        "question": "Which upstream output is canonical?",
        "_run_id": "run-abc",
        "_task_id": "task-xyz",
        "_agent_id": 7,
        "_field_id": "field-1",
    }
    p.update(over)
    return p


@pytest.mark.asyncio
async def test_answer_path_returns_grounded_answer(monkeypatch):
    async def _stub(db, subject, question, *, category=None):
        return {"answer": "Use output B.", "sources": [{"type": "upstream_task", "task_id": "42"}]}

    monkeypatch.setattr(oa, "answer_clarification", _stub)
    result = await ask_orchestrator(MagicMock(), uuid4(), _server_params())

    assert result["success"] is True
    assert result["answer"] == "Use output B."
    assert result["sources"] == [{"type": "upstream_task", "task_id": "42"}]
    assert "proceed_with_assumption" not in result


@pytest.mark.asyncio
async def test_cannot_answer_escalates(monkeypatch):
    # US-003 replaced US-002's proceed-with-assumption branch: cannot_answer now
    # escalates to a human ask and the task parks.
    import services.clarification_ladder as cl

    async def _stub(db, subject, question, *, category=None):
        return {"cannot_answer": True, "reason": "unretrievable"}

    async def _escalate(db, subject, question, *, category=None, partial_output=None, agent_name=None):
        return {"parked": True, "ask_id": 3, "message": "parked"}

    monkeypatch.setattr(oa, "answer_clarification", _stub)
    monkeypatch.setattr(cl, "escalate_clarification", _escalate)
    result = await ask_orchestrator(MagicMock(), uuid4(), _server_params())

    assert result["success"] is True
    assert result["parked"] is True
    assert result["ask_id"] == 3
    assert result["detail"]["reason"] == "unretrievable"
    assert "answer" not in result


@pytest.mark.asyncio
async def test_time_box_expiry_takes_cannot_answer_path(monkeypatch):
    # A slow answer round is time-boxed → cannot_answer(timeout) → escalation.
    import services.clarification_ladder as cl
    seen = {}

    async def _slow(db, subject, question, *, category=None):
        await asyncio.sleep(0.2)
        return {"answer": "too late"}

    async def _escalate(db, subject, question, *, category=None, partial_output=None, agent_name=None):
        seen["called"] = True
        return {"parked": True, "ask_id": 4, "message": "parked"}

    monkeypatch.setattr(oa, "answer_clarification", _slow)
    monkeypatch.setattr(cl, "escalate_clarification", _escalate)
    monkeypatch.setattr(Config, "CLARIFICATION_ANSWER_TIMEOUT", 0.01)
    result = await ask_orchestrator(MagicMock(), uuid4(), _server_params())

    assert result["success"] is True
    assert seen.get("called") is True          # the time-box fired → escalated
    assert result["parked"] is True
    assert result["detail"]["reason"] == "timeout"


@pytest.mark.asyncio
async def test_caller_resolved_from_server_context_not_tool_param(monkeypatch):
    seen = {}

    async def _capture(db, subject, question, *, category=None):
        seen["subject"] = subject
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(oa, "answer_clarification", _capture)
    # A tool call that SMUGGLES run_id/task_id — must be ignored; only the
    # server-injected _run_id/_task_id win.
    params = _server_params(run_id="SPOOF-run", task_id="SPOOF-task")
    await ask_orchestrator(MagicMock(), uuid4(), params)

    subject = seen["subject"]
    assert subject.run_id == "run-abc"
    assert subject.task_id == "task-xyz"
    assert subject.run_id != "SPOOF-run"


@pytest.mark.asyncio
async def test_no_run_context_guides_without_answer_attempt(monkeypatch):
    called = {"n": 0}

    async def _stub(db, subject, question, *, category=None):
        called["n"] += 1
        return {"answer": "should not run"}

    monkeypatch.setattr(oa, "answer_clarification", _stub)
    # No _run_id / _task_id (e.g. not a mission execution lane).
    result = await ask_orchestrator(MagicMock(), uuid4(), {"question": "Anything?"})

    assert result["success"] is True
    assert "proceed_with_assumption" in result
    assert called["n"] == 0  # never attempted an answer without run context


@pytest.mark.asyncio
async def test_missing_question_is_rejected():
    result = await ask_orchestrator(MagicMock(), uuid4(), {"_run_id": "r", "_task_id": "t"})
    assert result["success"] is False
    assert "question" in result["error"]
