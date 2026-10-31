"""PRD-268 §6.7 (K14) — what an eval run takes from the platform: its workspace and a tool's schema.

The runners that call the model run on one workspace, ``config.PROACTIVE_EVAL_WORKSPACE_ID``: its
model settings and credentials answer, and its usage rows record the calls. A tool is offered as
the platform offers it, its schema read from the action registry, never restated here. Either one
missing means the eval cannot run (``EvalCannotRun``, the command line's exit 2), said in a message
that names what to fix.
"""
from __future__ import annotations

from typing import Any, Dict
from uuid import UUID

from config import config

__all__ = ["WORKSPACE_SETTING", "EvalCannotRun", "eval_workspace", "tool_schema"]

WORKSPACE_SETTING = "PROACTIVE_EVAL_WORKSPACE_ID"


class EvalCannotRun(Exception):
    """The eval cannot run to a result; the message says why and what to fix."""


def eval_workspace() -> str:
    """The workspace the eval's model calls run on, as a UUID string; refused when unset or not an id."""
    value = str(config.PROACTIVE_EVAL_WORKSPACE_ID or "").strip()
    if not value:
        raise EvalCannotRun(
            f"{WORKSPACE_SETTING} is not set. Name the workspace whose model settings and credits the eval "
            "runs on.")
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise EvalCannotRun(f"{WORKSPACE_SETTING} is {value!r}, which is not a workspace id.") from exc


def tool_schema(name: str) -> Dict[str, Any]:
    """The platform action ``name`` as a model is offered it (``ActionDefinition.to_openai_schema``)."""
    from modules.tools.discovery.action_registry import get_action_registry

    action = get_action_registry().get(name)
    if action is None:
        raise EvalCannotRun(f"The platform has no action {name}: the eval offers it as the platform does.")
    return action.to_openai_schema()
