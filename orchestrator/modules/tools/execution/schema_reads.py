"""PRD-256 FX-013 (night 12, M2, F390): a strict tool reads what the model reliably sends.

US-006 made Auto's writes first-class tools with strict schemas, and made their refusals
honest. Night 12 showed which refusals were the same wasted turn every time:

- 31 ``platform_query_data`` calls sent the question as ``query`` (A255, A319, A457, A475,
  A529), and were refused as a key the tool does not take. ``query`` is read as ``question``,
  and ``platform_query_database`` (the name of the NL2SQL tool the chat no longer carries, with
  the platform's prefix) runs ``platform_query_data``.
- 10 social posts sent ``variables`` or ``copy`` as JSON text ("Input should be a valid
  dictionary"). A promoted (first-class) action's object or array parameter sent as JSON text
  is read as the object or array it holds, before anything checks it, through the executor's
  one decoder (``params_text``).

Both run where every caller crosses: ``UnifiedToolExecutor.execute_tool``, through
``params_text.decodes_nested_params`` (a direct call, and the action inside ``platform_execute``).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from modules.tools.execution.params_text import PARAMS_KEY, PLATFORM_DISPATCHER, json_container

logger = logging.getLogger(__name__)

TRACE = "FX-013"
ACTION_KEY = "action"
QUERY_DATA = "platform_query_data"
# Names a model reaches for that are another action: the action they run.
ACTION_NAMES: Dict[str, str] = {"platform_query_database": QUERY_DATA}
# Per action, a key models send for one of its parameters: {action: {sent: parameter}}.
KEYS_SENT: Dict[str, Dict[str, str]] = {QUERY_DATA: {"query": "question"}}
CONTAINER_TYPES = frozenset({"object", "array"})
# A parameter that also takes text keeps text as text: only a container-only parameter is decoded.
TEXT_TYPE = "string"


def _action(name: str) -> Any:
    from modules.tools.discovery.action_registry import get_action_registry

    return get_action_registry().get(name)


def _takes_a_container(prop: Any) -> bool:
    kind = prop.get("type") if isinstance(prop, dict) else None
    kinds = set(kind) if isinstance(kind, list) else {kind}
    return bool(kinds & CONTAINER_TYPES) and TEXT_TYPE not in kinds


def containers_decoded(action_def: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """``params`` with each object or array parameter of a promoted action that came as JSON
    text read as what it holds; any other value as it came. A new dict, never the caller's."""
    if not getattr(action_def, "promoted", False):
        return params
    props = (getattr(action_def, "parameters", None) or {}).get("properties") or {}
    decoded = {key: json_container(value) if isinstance(value, str) and _takes_a_container(props.get(key))
               else value for key, value in params.items()}
    for key in (key for key in params if decoded[key] is not params[key]):
        logger.info("[%s] %s's %s came as JSON text; read as the %s it holds", TRACE, action_def.name, key,
                    type(decoded[key]).__name__)
    return decoded


def keys_read(action_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """``params`` with a key models send for one of the action's parameters under that
    parameter's name, when the call did not send the parameter itself."""
    renamed = dict(params)
    for sent, name in KEYS_SENT.get(action_name, {}).items():
        if sent in renamed and renamed.get(name) in (None, ""):
            renamed[name] = renamed.pop(sent)
    return renamed


def action_params_read(action_name: str, params: Any) -> Tuple[str, Any]:
    """The action a call runs and its params, read as its schema takes them."""
    name = ACTION_NAMES.get(action_name, action_name)
    if not isinstance(params, dict):
        return name, params
    params = keys_read(name, params)
    action_def = _action(name)
    return name, containers_decoded(action_def, params) if action_def is not None else params


def as_the_schema_reads(tool_name: str, parameters: Any) -> Tuple[str, Any]:
    """The tool and its parameters as the action's schema reads them: a direct platform call
    by its own name, or the action inside ``platform_execute``. Anything else as it came."""
    if tool_name == PLATFORM_DISPATCHER and isinstance(parameters, dict):
        inner = _inner_read(parameters)
        return tool_name, inner if inner is not None else parameters
    if not str(tool_name).startswith("platform_"):
        return tool_name, parameters
    name, params = action_params_read(tool_name, parameters)
    if name != tool_name:
        logger.info("[%s] '%s' is not a tool: running %s", TRACE, tool_name, name)
    return name, params


def _inner_read(parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """``platform_execute``'s call with its action and params read, or None when it names no action."""
    action = parameters.get(ACTION_KEY)
    if not isinstance(action, str) or not action:
        return None
    name, params = action_params_read(action, parameters.get(PARAMS_KEY))
    read = {**parameters, ACTION_KEY: name}
    return {**read, PARAMS_KEY: params} if PARAMS_KEY in parameters else read


__all__ = ["ACTION_NAMES", "KEYS_SENT", "as_the_schema_reads", "containers_decoded", "keys_read"]
