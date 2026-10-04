"""A dispatcher's ``params`` sent as JSON text is read as the object it holds.

F181 (night 6): Gemini sent platform_execute's ``params`` as a JSON string, and
the chat's tool loop learned to decode it. F321 (night 9b): run #0084 of the
"Monday green stock" playbook failed on "platform_submit_report's params must be
an object of its parameters … not str", and runs #0085 and #0102 lost their
report the same way (six refused submit_report calls each, then the agent saved
it to the scratchpad instead). A playbook step parses its own tool calls, so the
chat's decoding never reached it, nor a mission's or a heartbeat's.

So the decoding lives at the one boundary every caller crosses: the unified
executor's ``execute_tool``. A ``params`` string that holds a JSON object runs
as that object, then is validated as before; anything else is refused in plain
words and never runs.
"""
from __future__ import annotations

import functools
import json
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

PLATFORM_DISPATCHER = "platform_execute"
COMPOSIO_DISPATCHER = "composio_execute"
# The dispatchers whose ``params`` is an object of the inner action's parameters.
NESTED_PARAMS_TOOLS = frozenset({PLATFORM_DISPATCHER, COMPOSIO_DISPATCHER})
# A model that double-encodes sends JSON text inside JSON text: read two layers.
MAX_DECODE_LAYERS = 2
PARAMS_KEY = "params"


def params_object(raw: Any) -> Any:
    """``raw`` as an object when it is JSON text holding one, else ``raw`` as it came.

    Literal newlines inside the text's strings are accepted (``strict=False``):
    a model writing a long report into ``content`` often leaves them unescaped.
    """
    value = raw
    for _ in range(MAX_DECODE_LAYERS):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value, strict=False)
        except json.JSONDecodeError:
            return raw
    return value if isinstance(value, dict) else raw


def nested_params_decoded(name: str, args: Any) -> Any:
    """``args`` with a dispatcher's ``params`` decoded when it is JSON text of an
    object; any other ``args`` unchanged. A new dict, never the caller's."""
    raw = args.get(PARAMS_KEY) if isinstance(args, dict) else None
    if name not in NESTED_PARAMS_TOOLS or not isinstance(raw, str):
        return args
    decoded = params_object(raw)
    return {**args, PARAMS_KEY: decoded} if isinstance(decoded, dict) else args


def _kind_of(params: Any) -> str:
    """What the params came as, in words a model and an owner both read."""
    if isinstance(params, str):
        return "text that is not a JSON object" if params.strip() else "empty text"
    if isinstance(params, list):
        return "a list"
    if isinstance(params, bool) or isinstance(params, (int, float)):
        return "a single value"
    return "something other than an object"


def params_refusal_text(action_name: str, params: Any) -> str:
    """The refusal for a ``params`` that is no object, in plain words. It starts
    as F181's did, so a model that learned it still recognises it."""
    return (f"{action_name}'s params must be an object of its parameters, e.g. "
            f"{{\"action\": \"{action_name}\", \"params\": {{...}}}}. They came as "
            f"{_kind_of(params)}, so nothing ran: send params as an object.")


def _composio_refusal(parameters: Dict[str, Any]) -> Optional[str]:
    """composio_execute ran a ``params`` it could not read as ``{}``: the action
    went out with none of the model's values. It is refused instead."""
    raw = parameters.get(PARAMS_KEY)
    if not isinstance(raw, str) or not raw.strip():
        return None
    action = str(parameters.get("action") or COMPOSIO_DISPATCHER)
    return params_refusal_text(action, raw)


def decodes_nested_params(execute: Callable[..., Awaitable[Dict[str, Any]]]) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """Wrap ``UnifiedToolExecutor.execute_tool``: a dispatcher's ``params`` sent as
    JSON text of an object runs as that object (logged, to be counted); a
    composio_execute ``params`` that is text of anything else is refused."""
    @functools.wraps(execute)
    async def wrapped(self: Any, tool_name: str, parameters: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        decoded = nested_params_decoded(tool_name, parameters)
        if decoded is not parameters:
            logger.info("[F321] %s params came as JSON text; read as the object it holds (action %s)",
                        tool_name, decoded.get("action"))
        elif tool_name == COMPOSIO_DISPATCHER and isinstance(parameters, dict):
            refused = _composio_refusal(parameters)
            if refused:
                return {"success": False, "error": refused, "tool": tool_name}
        return await execute(self, tool_name, decoded, *args, **kwargs)
    return wrapped


__all__ = [
    "NESTED_PARAMS_TOOLS",
    "decodes_nested_params",
    "nested_params_decoded",
    "params_object",
    "params_refusal_text",
]
