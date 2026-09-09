"""Usage attribution scope — which LANE is spending, and on whose behalf.

``llm_usage`` rows used to say ``request_type='orchestrator'`` for chat, board
tickets, missions and heartbeats alike, and ``execution_id`` was NULL on 99% of
rows: the manager only knew what it was constructed with, and one manager is
shared by every run of an agent. This module carries the attribution on a
``ContextVar`` instead, so it is task-local (two tickets of the same agent
running at once never overwrite each other) and every ``LLMManager`` used
inside the scope — the agent's own, a helper's, a fallback's — books the same
lane and execution.

Vocabulary (``request_type`` — the LANE that spent):

* ``chat``            a conversation turn (page, studio, widget, voice)
* ``board_task``      a Command Centre ticket run by an API agent
* ``mission``         a mission task (coordinator)
* ``heartbeat`` / ``scheduled_task`` / ``channel`` / ``webhook`` / ``composio``
* ``recipe`` / ``watch`` / ``planner`` / ``verifier`` / ``digest`` / …  (helpers keep
  the names they already used)
* ``session``         a Claude Code session on the user's own subscription
* ``embedding`` / ``rerank``  retrieval infrastructure

``execution_id`` is ``<kind>:<id>`` (``mission:42``, ``board_task:97``,
``chat:<conversation>``) so a lane's spend can be joined back to the thing
that spent it.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Mapping, Optional

LANE_CHAT = "chat"
LANE_BOARD_TASK = "board_task"
LANE_MISSION = "mission"
LANE_SESSION = "session"
LANE_EMBEDDING = "embedding"
LANE_RERANK = "rerank"

# Context ``source`` values that name a lane directly (the callers of
# ``execute_with_prompt`` already pass these); anything else is passed through
# as-is so a new lane shows up under its own name rather than "orchestrator".
_SOURCE_ALIASES = {
    "manual": "manual_run",
    "cli": LANE_SESSION,
}

_scope_var: ContextVar[Optional[Mapping[str, Any]]] = ContextVar("llm_usage_scope", default=None)


def current_usage_scope() -> Mapping[str, Any]:
    """The attribution in force for this task (empty when nothing set it)."""
    return _scope_var.get() or {}


@contextmanager
def usage_scope(
    *,
    request_type: Optional[str] = None,
    execution_id: Optional[str] = None,
    agent_id: Optional[int] = None,
    workspace_id: Any = None,
) -> Iterator[Dict[str, Any]]:
    """Attribute every LLM call made inside the block.

    Fields given as ``None`` inherit the enclosing scope, so a helper deep in a
    mission task (the verifier, a memory distil) keeps the mission's
    ``execution_id`` while naming its own ``request_type``.
    """
    merged: Dict[str, Any] = dict(current_usage_scope())
    for key, value in (
        ("request_type", request_type),
        ("execution_id", execution_id),
        ("agent_id", agent_id),
        ("workspace_id", workspace_id),
    ):
        if value is not None:
            merged[key] = value
    token = _scope_var.set(merged)
    try:
        yield merged
    finally:
        try:
            _scope_var.reset(token)
        except ValueError:
            # An async generator closed from another task cannot reset a token
            # minted in its own context; clearing is the honest fallback.
            _scope_var.set(None)


def lane_for_context(context: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The ``request_type`` an ``execute_with_prompt`` context implies."""
    if not isinstance(context, Mapping):
        return None
    source = context.get("source")
    if isinstance(source, str) and source.strip():
        key = source.strip().lower()
        return _SOURCE_ALIASES.get(key, key)
    if context.get("mission_id") or context.get("run_id"):
        return LANE_MISSION
    if context.get("task_id"):
        return LANE_BOARD_TASK
    return None


def execution_ref_for_context(context: Optional[Mapping[str, Any]]) -> Optional[str]:
    """``<kind>:<id>`` for the thing this context executes on behalf of."""
    if not isinstance(context, Mapping):
        return None
    lane = lane_for_context(context)
    if context.get("mission_id"):
        return f"mission:{context['mission_id']}"
    if context.get("run_id"):
        return f"mission:{context['run_id']}"
    if context.get("execution_id"):
        return str(context["execution_id"])
    if context.get("task_id"):
        kind = lane if lane in (LANE_BOARD_TASK, "scheduled_task", "heartbeat") else "task"
        return f"{kind}:{context['task_id']}"
    for key in ("heartbeat_id", "schedule_id", "conversation_id", "chat_id"):
        if context.get(key):
            return f"{key.replace('_id', '')}:{context[key]}"
    return None
