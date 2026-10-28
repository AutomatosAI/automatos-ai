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

PRD-268 US-C03d: a check-in reads the workspace, and nothing beyond it. The reads that
reach outside the platform (the ``web`` category: ``platform_web_fetch``,
``platform_web_search``) are labelled read, yet a check-in reads untrusted data, and a fetch
or a search would carry it out of the workspace. They are neither offered nor cleared in
the lane; everywhere else they run as before.

``caller_context`` is the plain dict the server builds for a call. Its ``lane`` key is
new here: ``"check_in"`` marks a check-in's call. With any other value, or none, both
executors run exactly as they did before the lane existed.

PRD-268 US-C03b: a check-in's runner also puts its own facts in ``caller_context``:
``run_id`` (the ``proactive_runs`` id of this run), ``check_in_reads`` (the ``{type, id}``
refs of the reads this run made that succeeded, so far) and ``check_in_agent_id`` (the
agent whose check-in it is; absent for Auto's workspace check-in).
``carries_the_check_in`` wraps ``PlatformActionExecutor.execute`` and hands them to
``platform_suggest_next_step`` as the server-side ``_check_in`` parameter. A
caller-supplied ``_check_in`` is stripped from every call, so a model can neither forge
its run nor its reads, and the tool refuses a call that carries none.

PRD-268 P268W1-RVW-1 (K17): ``read`` means read in the lane. A few read actions keep a
record of their own as they read (a search's log row, the onboarding funnel's stamp, a
spreadsheet's copy in the workspace, the NL2SQL audit and training rows, a preview's
picture). ``carries_the_check_in`` runs every check-in call but the suggestion inside
``core.read_only_call.reading_only()``, and each of those records is skipped there.
"""
from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from core.read_only_call import reading_only

logger = logging.getLogger(__name__)

LANE_KEY = "lane"
CHECK_IN_LANE = "check_in"
READ_LEVEL = "read"
# The one write a check-in makes: it records a suggestion (US-C03b), and changes nothing else.
SUGGEST_NEXT_STEP = "platform_suggest_next_step"
PLATFORM_PREFIX = "platform_"
PLATFORM_DISPATCHER = "platform_execute"
CHECK_IN_REFUSAL = "Not available in a check-in, which can only read. Suggest it as a next step instead."
# PRD-268 US-C03d: read categories that reach outside the platform, never in a check-in.
OUTSIDE_CATEGORIES = frozenset({"web"})
# The check-in's own facts in caller_context (US-C03b), and the server-side parameter
# that hands them to platform_suggest_next_step.
RUN_ID_KEY = "run_id"
READS_KEY = "check_in_reads"
AGENT_KEY = "check_in_agent_id"
CHECK_IN_PARAM = "_check_in"

ExecuteTool = Callable[..., Awaitable[Dict[str, Any]]]


def in_a_check_in(caller_context: Any) -> bool:
    """Whether the call is a check-in's: ``caller_context["lane"] == "check_in"``."""
    return isinstance(caller_context, dict) and caller_context.get(LANE_KEY) == CHECK_IN_LANE


def check_in_refusal() -> Dict[str, Any]:
    """The lane's refusal. Nothing ran."""
    return {"success": False, "error": CHECK_IN_REFUSAL}


def reads_the_workspace(action: Any) -> bool:
    """Whether ``action`` is positively ``read`` and reads inside the platform (not one of
    ``OUTSIDE_CATEGORIES``)."""
    return (getattr(action, "permission_level", None) == READ_LEVEL
            and getattr(action, "category", None) not in OUTSIDE_CATEGORIES)


def may_clear_in_a_check_in(action_name: str) -> bool:
    """Whether a check-in may run ``action_name``: ``platform_suggest_next_step``, or an
    action registered as positively ``read`` that reads inside the platform. Unknown, write
    and destructive actions are refused, a web read too, and so is one whose definition
    cannot be read."""
    if action_name == SUGGEST_NEXT_STEP:
        return True
    try:
        from modules.tools.discovery import get_action_registry

        action_def = get_action_registry().get(action_name)
    except Exception:
        logger.exception("[check_in_lane] could not read the definition of %s; refused", action_name)
        return False
    return action_def is not None and reads_the_workspace(action_def)


def is_a_platform_action(tool_name: Any) -> bool:
    """Whether ``tool_name`` is a platform action, which ``execute_tool`` routes to
    ``PlatformActionExecutor``: not the ``platform_execute`` dispatcher, a Composio tool,
    a plugin or a workspace tool."""
    return isinstance(tool_name, str) and tool_name.startswith(PLATFORM_PREFIX) and tool_name != PLATFORM_DISPATCHER


def offered_in_a_check_in(action: Any) -> bool:
    """PRD-268 US-C02c: whether a check-in's tool surface (``ContextMode.CHECK_IN``) offers
    ``action``, a registered definition. It must be a platform action that is positively
    ``read`` and reads inside the platform, or ``platform_suggest_next_step``; needs no
    admin; and can run on this server.
    The surface offers nothing the two decorators below would refuse, so the model is
    never shown a tool it may not call."""
    name = getattr(action, "name", None)
    if not is_a_platform_action(name):
        return False
    if getattr(action, "admin_only", False) or getattr(action, "super_admin_only", False):
        return False
    if name != SUGGEST_NEXT_STEP and not reads_the_workspace(action):
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


def check_in_of(caller_context: Any) -> Optional[Dict[str, Any]]:
    """A check-in's own facts, from the server-built ``caller_context``: ``run_id``,
    ``reads`` (a new list) and ``agent_id``. None outside the lane."""
    if not in_a_check_in(caller_context):
        return None
    reads = caller_context.get(READS_KEY)
    return {
        "run_id": caller_context.get(RUN_ID_KEY),
        "reads": list(reads) if isinstance(reads, (list, tuple)) else [],
        "agent_id": caller_context.get(AGENT_KEY),
    }


def check_in_params(action_name: str, params: Any, caller_context: Any) -> Any:
    """``params`` without a caller-supplied ``_check_in``; ``platform_suggest_next_step``
    called in a check-in gets the run's own, from ``caller_context``."""
    from modules.tools.execution.params_text import params_object

    params = params_object(params)
    if not isinstance(params, dict):
        return params
    kept = {key: value for key, value in params.items() if key != CHECK_IN_PARAM}
    check_in = check_in_of(caller_context) if action_name == SUGGEST_NEXT_STEP else None
    return {**kept, CHECK_IN_PARAM: check_in} if check_in is not None else kept


def carries_the_check_in(execute: ExecuteTool) -> ExecuteTool:
    """Wrap ``PlatformActionExecutor.execute`` so the check-in's run and reads reach the
    suggestion tool server-side, and never from the model. Every other call in the lane
    runs ``reading_only()`` (P268W1-RVW-1): a read's own records are skipped."""
    @functools.wraps(execute)
    async def wrapped(self: Any, action_name: str, params: Any, caller_context: Any = None) -> Dict[str, Any]:
        params = check_in_params(action_name, params, caller_context)
        if not in_a_check_in(caller_context) or action_name == SUGGEST_NEXT_STEP:
            return await execute(self, action_name, params, caller_context)
        with reading_only():
            return await execute(self, action_name, params, caller_context)
    return wrapped


__all__ = [
    "AGENT_KEY",
    "CHECK_IN_LANE",
    "CHECK_IN_PARAM",
    "CHECK_IN_REFUSAL",
    "LANE_KEY",
    "OUTSIDE_CATEGORIES",
    "READS_KEY",
    "RUN_ID_KEY",
    "SUGGEST_NEXT_STEP",
    "carries_the_check_in",
    "check_in_of",
    "check_in_params",
    "check_in_refusal",
    "in_a_check_in",
    "is_a_platform_action",
    "may_clear_in_a_check_in",
    "offered_in_a_check_in",
    "platform_actions_only_in_a_check_in",
    "reads_only_in_a_check_in",
    "reads_the_workspace",
]
