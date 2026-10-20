"""P256-FIX-N17-10 (F449): the executor's own refusals say why in the owner's words.

Night 17: 66 of the 141 refused receipts were a call's shape ("'platform_board_snapshot' does not take
['status'] … Fix the call and send it again", "Missing required params for …"), and 3 named an action the
platform does not have; two of those were ``search_knowledge`` sent through ``platform_execute``, a real
tool through the wrong door. ``UnifiedToolExecutor.execute_tool`` makes those refusals inside a body far
over the code-shape limit, so they are tagged on its seam (``says_why_it_refused``): each one starts with
the words its builder writes (below; F449's tests run every builder through the seam, so a builder that
changes its words fails them), and gets ``owner_reason``. A refusal whose text carries its owner's reason
(``owner_reasons.Refusal``: an agent-name clash from a handler) has it copied onto the result. The model's
``error`` is untouched, and a refusal that already says its owner's reason keeps it.
"""
from __future__ import annotations

import functools
import inspect
import re
from typing import Any, Awaitable, Callable, Dict, Optional

from modules.tools.execution.owner_reasons import NO_SUCH_ACTION, OWNER_REASON, SENT_WRONG, owner_fields
from modules.tools.execution.params_text import PLATFORM_DISPATCHER

Execute = Callable[..., Awaitable[Dict[str, Any]]]

ACTION_KEY = "action"
# How each of execute_tool's refusals starts (unified_executor, params_text, direct_contract).
MISSING_ACTION = "Missing required field: action"       # platform_execute with no action
UNKNOWN_ACTION = "Unknown platform action:"             # unknown_action_error, PlatformActionExecutor
UNKNOWN_TOOL = "Unknown tool:"                          # UnifiedToolExecutor._unknown_tool_error
MISSING_PARAMS = "Missing required params for '"        # missing_params_error, missing_on_a_direct_call
NOT_AN_OBJECT = "'s params must be an object of its parameters"   # params_refusal_text
NOT_TAKEN = re.compile(r"'[^'\n]+' does not take \[")   # unknown_params_error (F182)


def executor_reason(error: str, wrong_door: bool = False) -> Optional[str]:
    """The owner's reason for one of the executor's own refusals, read from how it starts; None for any
    other text. ``wrong_door``: the unknown action is a real tool sent through ``platform_execute``."""
    said = error.strip()
    first_line = said.split("\n", 1)[0]
    if said == MISSING_ACTION or said.startswith(MISSING_PARAMS) or NOT_TAKEN.match(said) \
            or NOT_AN_OBJECT in first_line:
        return SENT_WRONG
    if said.startswith(UNKNOWN_ACTION):
        return SENT_WRONG if wrong_door else NO_SUCH_ACTION
    if said.startswith(UNKNOWN_TOOL):
        return NO_SUCH_ACTION
    return None


def is_a_tool(executor: Any, name: Any) -> bool:
    """Whether ``name`` is one of the executor's own tools (``tool_routes``) or in its tool registry."""
    if not isinstance(name, str) or not name.strip():
        return False
    if name.strip() in (getattr(executor, "tool_routes", None) or {}):
        return True
    registry = getattr(executor, "tool_registry", None)
    return registry is not None and registry.get_tool(name.strip()) is not None


def in_the_owners_words(result: Any, tool_name: str, parameters: Any, executor: Any = None) -> Any:
    """``result`` with its owner's reason when it is a refusal that has none yet and says why in a way
    the owner can be told; any other result as it came. A new dict, never the caller's."""
    if not isinstance(result, dict) or result.get("success") is not False or result.get(OWNER_REASON):
        return result
    error = result.get("error")
    if not isinstance(error, str):
        return result
    inner = parameters.get(ACTION_KEY) if tool_name == PLATFORM_DISPATCHER and isinstance(parameters, dict) else None
    wrong_door = error.strip().startswith(UNKNOWN_ACTION) and is_a_tool(executor, inner)
    reason = getattr(error, OWNER_REASON, None) or executor_reason(error, wrong_door)
    fields = owner_fields(error, reason)
    return {**result, **fields} if fields else result


def says_why_it_refused(execute: Execute) -> Execute:
    """Wrap ``UnifiedToolExecutor.execute_tool``: its refusals carry the owner's reason
    (``in_the_owners_words``), read against the call as it was made."""
    signature = inspect.signature(execute)

    @functools.wraps(execute)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        result = await execute(self, *args, **kwargs)
        call = signature.bind(self, *args, **kwargs).arguments
        return in_the_owners_words(result, call.get("tool_name"), call.get("parameters"), self)
    return wrapped


__all__ = ["MISSING_ACTION", "NOT_AN_OBJECT", "NOT_TAKEN", "UNKNOWN_ACTION", "UNKNOWN_TOOL", "executor_reason",
           "in_the_owners_words", "is_a_tool", "says_why_it_refused"]
