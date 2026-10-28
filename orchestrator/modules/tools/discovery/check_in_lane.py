"""PRD-268 US-C02b (§6.2.3, FR-1, K8): a check-in can only read.

A check-in is a background turn that looks and suggests. It never sends, edits or
deletes. Its runner (US-C03d) offers only read platform actions and
``platform_suggest_next_step``, and makes every call through
``PlatformActionExecutor.execute`` with ``caller_context["lane"] = "check_in"``.

The lane is enforced in depth, failing closed, by two decorators:

- ``reads_only_in_a_check_in`` wraps ``PlatformActionExecutor.clear``, the gates every
  platform call clears (the workspace tools clear them too, ``exec_workspace``). In the
  lane a call clears only when its registered definition is positively ``read``, or it is
  ``platform_suggest_next_step``. An unknown action, a write and a destructive action are
  refused before any other gate runs, and so is any action whose definition cannot be read.
- ``platform_actions_only_in_a_check_in`` wraps ``UnifiedToolExecutor.execute_tool``. In the
  lane, a tool that is not a platform action (Composio, a plugin, a workspace tool, the
  ``platform_execute`` dispatcher) is refused before anything routes it. Composio calls
  never reach the platform executor, so this layer is the one that stops them.

What a check-in is shown matches what it may run: ``offered_in_a_check_in`` is the rule
the ``read_only`` tool strategy (``ContextMode.CHECK_IN``, US-C02c) loads by.

``caller_context`` is the plain dict the server builds for a call. Its ``lane`` key is
new here: ``"check_in"`` marks a check-in's call. With any other value, or none, both
executors run exactly as they did before the lane existed.
"""
from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

LANE_KEY = "lane"
CHECK_IN_LANE = "check_in"
READ_LEVEL = "read"
# The one write a check-in makes: it records a suggestion (US-C03b), and changes nothing else.
SUGGEST_NEXT_STEP = "platform_suggest_next_step"
PLATFORM_PREFIX = "platform_"
PLATFORM_DISPATCHER = "platform_execute"
CHECK_IN_REFUSAL = "Not available in a check-in, which can only read. Suggest it as a next step instead."

ExecuteTool = Callable[..., Awaitable[Dict[str, Any]]]


def in_a_check_in(caller_context: Any) -> bool:
    """Whether the call is a check-in's: ``caller_context["lane"] == "check_in"``."""
    return isinstance(caller_context, dict) and caller_context.get(LANE_KEY) == CHECK_IN_LANE


def check_in_refusal() -> Dict[str, Any]:
    """The lane's refusal. Nothing ran."""
    return {"success": False, "error": CHECK_IN_REFUSAL}


def may_clear_in_a_check_in(action_name: str) -> bool:
    """Whether a check-in may run ``action_name``: ``platform_suggest_next_step``, or an
    action registered as positively ``read``. Unknown, write and destructive actions are
    refused, and so is one whose definition cannot be read."""
    if action_name == SUGGEST_NEXT_STEP:
        return True
    try:
        from modules.tools.discovery import get_action_registry

        action_def = get_action_registry().get(action_name)
    except Exception:
        logger.exception("[check_in_lane] could not read the definition of %s; refused", action_name)
        return False
    return action_def is not None and getattr(action_def, "permission_level", None) == READ_LEVEL


def is_a_platform_action(tool_name: Any) -> bool:
    """Whether ``tool_name`` is a platform action, which ``execute_tool`` routes to
    ``PlatformActionExecutor``: not the ``platform_execute`` dispatcher, a Composio tool,
    a plugin or a workspace tool."""
    return isinstance(tool_name, str) and tool_name.startswith(PLATFORM_PREFIX) and tool_name != PLATFORM_DISPATCHER


def offered_in_a_check_in(action: Any) -> bool:
    """PRD-268 US-C02c: whether a check-in's tool surface (``ContextMode.CHECK_IN``) offers
    ``action``, a registered definition. It must be a platform action that is positively
    ``read``, or ``platform_suggest_next_step``; needs no admin; and can run on this server.
    The surface offers nothing the two decorators below would refuse, so the model is
    never shown a tool it may not call."""
    name = getattr(action, "name", None)
    if not is_a_platform_action(name):
        return False
    if getattr(action, "admin_only", False) or getattr(action, "super_admin_only", False):
        return False
    if name != SUGGEST_NEXT_STEP and getattr(action, "permission_level", None) != READ_LEVEL:
        return False
    from modules.tools.discovery.action_registry import action_is_available

    return action_is_available(action)


def reads_only_in_a_check_in(clear: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``PlatformActionExecutor.clear``: in a check-in, a call that is not positively
    read (or ``platform_suggest_next_step``) is refused before any other gate. Outside
    the lane, the call clears as before."""
    @functools.wraps(clear)
    def wrapped(self: Any, action_name: str, params: Any, caller_context: Optional[Dict[str, Any]] = None,
                *, card_subject: str = "") -> Any:
        if in_a_check_in(caller_context) and not may_clear_in_a_check_in(action_name):
            logger.warning("[check_in_lane] %s refused in a check-in (workspace=%s)", action_name,
                           getattr(self, "workspace_id", None))
            return check_in_refusal()
        return clear(self, action_name, params, caller_context, card_subject=card_subject)
    return wrapped


def platform_actions_only_in_a_check_in(execute_tool: ExecuteTool) -> ExecuteTool:
    """Wrap ``UnifiedToolExecutor.execute_tool``: in a check-in, a tool that is not a
    platform action is refused before anything routes it. Outside the lane, the call
    runs as before."""
    signature = inspect.signature(execute_tool)

    @functools.wraps(execute_tool)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        call = signature.bind(self, *args, **kwargs).arguments
        tool_name = call.get("tool_name")
        if in_a_check_in(call.get("caller_context")) and not is_a_platform_action(tool_name):
            logger.warning("[check_in_lane] tool %s refused in a check-in (workspace=%s)", tool_name,
                           call.get("workspace_id"))
            return {**check_in_refusal(), "tool": tool_name}
        return await execute_tool(self, *args, **kwargs)
    return wrapped


__all__ = [
    "CHECK_IN_LANE",
    "CHECK_IN_REFUSAL",
    "LANE_KEY",
    "SUGGEST_NEXT_STEP",
    "check_in_refusal",
    "in_a_check_in",
    "is_a_platform_action",
    "may_clear_in_a_check_in",
    "offered_in_a_check_in",
    "platform_actions_only_in_a_check_in",
    "reads_only_in_a_check_in",
]
