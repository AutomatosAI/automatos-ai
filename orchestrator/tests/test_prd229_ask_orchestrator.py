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
from modules.tools.discovery.platform_executor import (  # noqa: E402
    _bind_ask_orchestrator_context,
)


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
    action = get_action_registry().get("platform_ask_orchestrator")
    assert action is not None
    assert action.name == "platform_ask_orchestrator"
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
    assert "platform_ask_orchestrator" in EXECUTION_ONLY_TOOLS
    assert excluded_tool_names(ContextMode.CHATBOT) == frozenset({"platform_ask_orchestrator"})
    assert excluded_tool_names(ContextMode.TASK_EXECUTION) == frozenset()
    # accepts the string value too (what flows through SectionContext)
    assert excluded_tool_names("chatbot") == frozenset({"platform_ask_orchestrator"})
    assert excluded_tool_names("task_execution") == frozenset()


def _surface():
    return [
        {"type": "function", "function": {
            "name": "platform_execute",
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["platform_ask_orchestrator", "platform_list_agents"]},
                "params": {"type": "object"},
            }, "required": ["action", "params"]},
        }},
        {"type": "function", "function": {"name": "platform_ask_orchestrator", "parameters": {}}},
        {"type": "function", "function": {"name": "workspace_read_file", "parameters": {}}},
    ]


def test_chatbot_surface_strips_ask_orchestrator():
    tools = _surface()
    stripped = strip_actions_from_surface(tools, excluded_tool_names(ContextMode.CHATBOT))
    names = [t["function"]["name"] for t in stripped]
    assert "platform_ask_orchestrator" not in names          # first-class schema dropped
    assert "platform_execute" in names
    dispatcher = next(t for t in stripped if t["function"]["name"] == "platform_execute")
    enum = dispatcher["function"]["parameters"]["properties"]["action"]["enum"]
    assert "platform_ask_orchestrator" not in enum           # pruned from the dispatcher enum
    assert "platform_list_agents" in enum


def test_task_execution_surface_keeps_ask_orchestrator():
    tools = _surface()
    kept = strip_actions_from_surface(tools, excluded_tool_names(ContextMode.TASK_EXECUTION))
    names = [t["function"]["name"] for t in kept]
    assert "platform_ask_orchestrator" in names
    dispatcher = next(t for t in kept if t["function"]["name"] == "platform_execute")
    assert "platform_ask_orchestrator" in dispatcher["function"]["parameters"]["properties"]["action"]["enum"]


def test_strip_is_rebuild_not_mutate():
    tools = _surface()
    original_enum = list(tools[0]["function"]["parameters"]["properties"]["action"]["enum"])
    strip_actions_from_surface(tools, frozenset({"platform_ask_orchestrator"}))
    # the input surface is untouched
    assert tools[0]["function"]["parameters"]["properties"]["action"]["enum"] == original_enum
    assert [t["function"]["name"] for t in tools] == [
        "platform_execute", "platform_ask_orchestrator", "workspace_read_file",
    ]


def test_prompt_catalog_excludes_in_chat_keeps_in_execution():
    registry = get_action_registry()
    full = registry.build_prompt_summary(exclude_admin=True, exclude_promoted=True)
    assert "platform_ask_orchestrator" in full  # execution lanes advertise it
    chat = registry.build_prompt_summary(
        exclude_admin=True, exclude_promoted=True,
        exclude_names=list(excluded_tool_names(ContextMode.CHATBOT)),
    )
    assert "platform_ask_orchestrator" not in chat  # chat catalog does not


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
    # A tool call that SMUGGLES the subject two ways: plain run_id/task_id (the
    # handler reads only _-prefixed keys, so these are inert) AND _-prefixed
    # _run_id/_task_id (which the EXECUTOR binding strips before injecting the
    # real server field_context). Neither smuggle survives — only server wins.
    smuggled = _server_params(
        run_id="SPOOF-run", task_id="SPOOF-task",     # non-underscore: handler ignores
        _run_id="SPOOF-run", _task_id="SPOOF-task",   # underscore: executor strips
    )
    ctx = {"field_context": {"run_id": "run-abc", "task_id": "task-xyz", "field_id": "field-1"}}
    bound = _bind_ask_orchestrator_context(smuggled, ctx)
    await ask_orchestrator(MagicMock(), uuid4(), bound)

    subject = seen["subject"]
    assert subject.run_id == "run-abc"
    assert subject.task_id == "task-xyz"
    assert subject.run_id != "SPOOF-run"
    assert subject.task_id != "SPOOF-task"


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


# ---------------------------------------------------------------------------
# P229-RVW-2 — broken object-level authorization, closed.
#
# Two guarantees: (1) the executor binding STRIPS any smuggled _run_id/_task_id/
# _field_id before injecting the server field_context, so a prompt-injected call
# in a non-mission lane cannot point Auto at a foreign task; (2) the task read is
# tenant-scoped, so even a same-shaped foreign task_id can never load a
# cross-run / cross-workspace row.
# ---------------------------------------------------------------------------

import operator as _operator  # noqa: E402
from sqlalchemy.sql.operators import in_op  # noqa: E402


def _row_matches(row, expr) -> bool:
    """Evaluate a simple SQLAlchemy eq / in_ predicate against a fake row."""
    key = getattr(getattr(expr, "left", None), "key", None)
    if key is None:
        return True  # not a column predicate we model → don't filter
    actual = getattr(row, key, None)
    op = getattr(expr, "operator", None)
    if op is _operator.eq:
        return actual == expr.right.value
    if op is in_op:
        return actual in (expr.right.value or [])
    return True


class _ScopedQuery:
    """Filter-AWARE fake query: evaluates eq / in_ predicates so tenant scoping
    is actually exercised (unlike the filter-ignoring US-001 fake)."""

    def __init__(self, rows):
        self._rows = list(rows)

    def join(self, *a, **k):
        return self

    def filter(self, *exprs):
        rows = self._rows
        for e in exprs:
            rows = [r for r in rows if _row_matches(r, e)]
        return _ScopedQuery(rows)

    def order_by(self, *a, **k):
        return self

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def count(self):
        return len(self._rows)


class _ScopedSession:
    def __init__(self, by_model):
        self._by_model = by_model

    def query(self, model):
        return _ScopedQuery(self._by_model.get(model, []))


def _task_row(task_id="task-1", run_id="run-A"):
    return SimpleNamespace(id=task_id, run_id=run_id)


def _run_row(run_id="run-A", workspace_id="ws-A"):
    return SimpleNamespace(id=run_id, workspace_id=workspace_id)


# --- executor binding: strip-then-inject ---

def test_bind_strips_smuggled_subject_keys_when_no_field_context():
    # Non-mission lane: field_context is empty. A smuggled _run_id/_task_id/
    # _field_id must NOT survive — the binding strips them and injects nothing.
    smuggled = {
        "question": "what?",
        "_run_id": "victim-run",
        "_task_id": "victim-task",
        "_field_id": "victim-field",
    }
    bound = _bind_ask_orchestrator_context(smuggled, {})
    assert bound == {"question": "what?"}
    # rebuild-don't-mutate: the caller's dict is untouched
    assert smuggled["_run_id"] == "victim-run"


def test_bind_server_field_context_overrides_smuggled_keys():
    smuggled = {"question": "q", "_run_id": "SPOOF", "_task_id": "SPOOF", "_field_id": "SPOOF"}
    ctx = {"field_context": {"run_id": "real-run", "task_id": "real-task", "field_id": "real-field"}}
    bound = _bind_ask_orchestrator_context(smuggled, ctx)
    assert bound["_run_id"] == "real-run"
    assert bound["_task_id"] == "real-task"
    assert bound["_field_id"] == "real-field"
    assert "SPOOF" not in set(bound.values())


@pytest.mark.asyncio
async def test_smuggled_underscore_keys_stripped_then_handler_proceeds(monkeypatch):
    # Full non-mission smuggle path: the executor binding strips the smuggled
    # _run_id/_task_id (empty field_context); the handler then sees no run context
    # and falls to proceed_with_assumption — it NEVER attempts an answer, so no
    # foreign task is ever loaded.
    called = {"n": 0}

    async def _stub(db, subject, question, *, category=None):
        called["n"] += 1
        return {"answer": "should not run"}

    monkeypatch.setattr(oa, "answer_clarification", _stub)
    smuggled = _server_params(_run_id="victim-run", _task_id="victim-task")
    bound = _bind_ask_orchestrator_context(smuggled, {})  # non-mission lane
    result = await ask_orchestrator(MagicMock(), uuid4(), bound)

    assert result["success"] is True
    assert "proceed_with_assumption" in result
    assert called["n"] == 0


# --- handler task read: tenant-scoped (_load_task) ---

def test_load_task_loads_in_scope_task():
    from core.models.orchestration import OrchestrationRun, OrchestrationTask

    db = _ScopedSession({
        OrchestrationTask: [_task_row("task-1", "run-A")],
        OrchestrationRun: [_run_row("run-A", "ws-A")],
    })
    task = hc._load_task(db, "run-A", "ws-A", "task-1")
    assert task is not None and task.id == "task-1"


def test_load_task_scopes_out_foreign_run():
    from core.models.orchestration import OrchestrationRun, OrchestrationTask

    # The task belongs to run-A; the server subject says run-B → no row loads.
    db = _ScopedSession({
        OrchestrationTask: [_task_row("task-1", "run-A")],
        OrchestrationRun: [_run_row("run-A", "ws-A"), _run_row("run-B", "ws-A")],
    })
    assert hc._load_task(db, "run-B", "ws-A", "task-1") is None


def test_load_task_scopes_out_foreign_workspace():
    from core.models.orchestration import OrchestrationRun, OrchestrationTask

    # task-1 is in run-A which belongs to ws-A; the subject workspace is ws-B.
    db = _ScopedSession({
        OrchestrationTask: [_task_row("task-1", "run-A")],
        OrchestrationRun: [_run_row("run-A", "ws-A")],
    })
    assert hc._load_task(db, "run-A", "ws-B", "task-1") is None


def test_load_task_guards_missing_subject():
    from core.models.orchestration import OrchestrationRun, OrchestrationTask

    db = _ScopedSession({
        OrchestrationTask: [_task_row("task-1", "run-A")],
        OrchestrationRun: [_run_row("run-A", "ws-A")],
    })
    assert hc._load_task(db, None, "ws-A", "task-1") is None   # no run
    assert hc._load_task(db, "run-A", None, "task-1") is None  # no workspace
    assert hc._load_task(db, "run-A", "ws-A", None) is None    # no task


# ---------------------------------------------------------------------------
# P229-RVW-8 — per-task round cap: N rounds cannot approach the task envelope
# ---------------------------------------------------------------------------

def test_clarification_max_rounds_per_task_config():
    # a real constant in config.py (no os.getenv outside config.py), and the
    # cumulative bound stays well inside the smallest (light=120s) envelope:
    # cap × single-round timeout ≤ half the envelope.
    assert isinstance(Config.CLARIFICATION_MAX_ROUNDS_PER_TASK, int)
    assert Config.CLARIFICATION_MAX_ROUNDS_PER_TASK >= 1
    assert (
        Config.CLARIFICATION_MAX_ROUNDS_PER_TASK * Config.CLARIFICATION_ANSWER_TIMEOUT
        <= 120 // 2
    )


class _EventCountDB:
    """Counts preset OrchestrationEvent rows for any filtered query."""

    def __init__(self, n):
        self._n = n

    def query(self, *a, **k):
        return self

    def filter(self, *a, **k):
        return self

    def count(self):
        return self._n


def test_task_clarification_rounds_counts_trail():
    subj = oa.ClarificationSubject(run_id="r", workspace_id="w", task_id="t")
    assert oa.task_clarification_rounds(_EventCountDB(2), subj) == 2
    # a run-level question (no task) never accrues a per-task count
    assert oa.task_clarification_rounds(_EventCountDB(5), oa.ClarificationSubject(run_id="r", workspace_id="w")) == 0


@pytest.mark.asyncio
async def test_under_round_cap_still_answers(monkeypatch):
    # below the cap, the answer round runs normally (the cap does not break the
    # happy path).
    async def _answer(db, subject, question, *, category=None):
        return {"answer": "Use output B.", "sources": []}

    monkeypatch.setattr(oa, "answer_clarification", _answer)
    monkeypatch.setattr(oa, "task_clarification_rounds", lambda db, subject: 0)
    monkeypatch.setattr(Config, "CLARIFICATION_MAX_ROUNDS_PER_TASK", 2)

    result = await ask_orchestrator(MagicMock(), uuid4(), _server_params())
    assert result["answer"] == "Use output B."


@pytest.mark.asyncio
async def test_per_task_round_cap_short_circuits_across_calls(monkeypatch):
    # AC2 — drive N+1 real handler calls: the trail grows on each ANSWERED round,
    # and the (cap+1)th call short-circuits to escalation WITHOUT entering the
    # answer round (answer_clarification is never called on that call — no 30s
    # round). Genuine guard: without the cap, call 3 would answer (trail→3).
    import services.clarification_ladder as cl

    trail = {"answered": 0}

    async def _answer(db, subject, question, *, category=None):
        trail["answered"] += 1                      # an answered round hits the trail
        return {"answer": f"A{trail['answered']}", "sources": []}

    escalated = {"n": 0}

    async def _escalate(db, subject, question, *, category=None, partial_output=None, agent_name=None):
        escalated["n"] += 1
        return {"parked": True, "ask_id": 9, "message": "parked"}

    monkeypatch.setattr(oa, "answer_clarification", _answer)
    monkeypatch.setattr(oa, "task_clarification_rounds", lambda db, subject: trail["answered"])
    monkeypatch.setattr(cl, "escalate_clarification", _escalate)
    monkeypatch.setattr(Config, "CLARIFICATION_MAX_ROUNDS_PER_TASK", 2)

    db = MagicMock()
    r1 = await ask_orchestrator(db, uuid4(), _server_params())
    r2 = await ask_orchestrator(db, uuid4(), _server_params())
    assert r1["answer"] == "A1" and r2["answer"] == "A2"

    # call 3 (>= cap) short-circuits: no answer round, escalate with reason round_cap
    r3 = await ask_orchestrator(db, uuid4(), _server_params())
    assert "answer" not in r3
    assert r3["parked"] is True
    assert r3["detail"]["reason"] == "round_cap"
    assert trail["answered"] == 2                   # answer round NOT entered on call 3
    assert escalated["n"] == 1
