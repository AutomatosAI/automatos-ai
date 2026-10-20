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

P256-FIX-N17-10 (F449, night 17): 17 direct platform_* calls came in the dispatcher's envelope
(``platform_get_task({"action": "platform_get_task", "params": {...}})``), each refused and retried.
A direct call whose only keys are that ``action`` (the tool's own name) and ``params`` holding an
object of keys the action takes runs with that object (``envelope_opened``); an action that declares
``params`` or ``action`` itself is never unwrapped, and ``platform_execute`` is untouched (F182's
night-6 refusal of inputs nested under ``params`` stands). Three more keys Auto reaches for are read
as the parameter they mean: platform_read_document's ``file_name``/``name``/``document`` (15
refusals), platform_recommend_agent's ``request``/``query`` (7) and platform_grep_documents'
``max_results``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from modules.tools.execution.params_text import PARAMS_KEY, PLATFORM_DISPATCHER, json_container, params_object

logger = logging.getLogger(__name__)

TRACE = "FX-013"
ENVELOPE_TRACE = "F449"
ACTION_KEY = "action"
PLATFORM_PREFIX = "platform_"
QUERY_DATA = "platform_query_data"
READ_DOCUMENT = "platform_read_document"
RECOMMEND_AGENT = "platform_recommend_agent"
GREP_DOCUMENTS = "platform_grep_documents"
# Names a model reaches for that are another action: the action they run.
ACTION_NAMES: Dict[str, str] = {"platform_query_database": QUERY_DATA}
# Per action, a key models send for one of its parameters: {action: {sent: parameter}}.
KEYS_SENT: Dict[str, Dict[str, str]] = {
    QUERY_DATA: {"query": "question"},
    READ_DOCUMENT: {"file_name": "filename", "name": "filename", "document": "filename"},   # F449
    RECOMMEND_AGENT: {"request": "objective", "query": "objective"},                         # F449
    GREP_DOCUMENTS: {"max_results": "limit"},                                                # F449
}
# The keys of the dispatcher's envelope: a direct call sent in it has these and no others.
ENVELOPE_KEYS = frozenset({ACTION_KEY, PARAMS_KEY})
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
    parameter's name, when the call did not send the parameter itself; each one logged."""
    renamed = dict(params)
    for sent, name in KEYS_SENT.get(action_name, {}).items():
        if sent in renamed and renamed.get(name) in (None, ""):
            renamed[name] = renamed.pop(sent)
            logger.info("[%s] %s's %s read as %s", ENVELOPE_TRACE, action_name, sent, name)
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
    if not str(tool_name).startswith(PLATFORM_PREFIX):
        return tool_name, parameters
    name, params = action_params_read(tool_name, envelope_opened(tool_name, parameters))
    if name != tool_name:
        logger.info("[%s] '%s' is not a tool: running %s", TRACE, tool_name, name)
    return name, params


def envelope_opened(tool_name: str, parameters: Any) -> Any:
    """F449: a direct call sent in the dispatcher's envelope, ``{"action": <this tool>, "params": {...}}``,
    as the object inside it, when the action declares neither ``params`` nor ``action`` and takes every key
    inside (after ``keys_read``); else ``parameters`` as they came."""
    name, inner = ACTION_NAMES.get(tool_name, tool_name), _enveloped(tool_name, parameters)
    action_def = _action(name) if inner is not None else None
    if action_def is None or ENVELOPE_KEYS & _declared(action_def):
        return parameters
    from modules.tools.execution.unified_executor import undeclared_params

    inner = keys_read(name, inner)
    if undeclared_params(action_def, inner):
        return parameters
    logger.info("[%s] %s came in platform_execute's envelope: run with the params inside it", ENVELOPE_TRACE,
                tool_name)
    return inner


def _enveloped(tool_name: str, parameters: Any) -> Optional[Dict[str, Any]]:
    """The object a direct call's envelope holds, when its keys are only ``action`` (naming this tool) and
    ``params`` (an object, or the text of one); else None."""
    if not isinstance(parameters, dict) or set(parameters) != ENVELOPE_KEYS:
        return None
    inner = params_object(parameters.get(PARAMS_KEY))
    said = parameters.get(ACTION_KEY)
    return inner if isinstance(inner, dict) and isinstance(said, str) and _names(said, tool_name) else None


def _names(said: str, tool_name: str) -> bool:
    """Whether the envelope's ``action`` names the tool it was sent to, with or without the platform's prefix."""
    named = said.strip()
    named = named if named.startswith(PLATFORM_PREFIX) else PLATFORM_PREFIX + named
    return ACTION_NAMES.get(named, named) == ACTION_NAMES.get(tool_name, tool_name)


def _declared(action_def: Any) -> frozenset:
    """The keys an action takes by its own name: its schema's and the ones its handler reads."""
    props = (getattr(action_def, "parameters", None) or {}).get("properties") or {}
    return frozenset(props) | frozenset(getattr(action_def, "accepts", ()) or ())


def _inner_read(parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """``platform_execute``'s call with its action and params read, or None when it names no action."""
    action = parameters.get(ACTION_KEY)
    if not isinstance(action, str) or not action:
        return None
    name, params = action_params_read(action, parameters.get(PARAMS_KEY))
    read = {**parameters, ACTION_KEY: name}
    return {**read, PARAMS_KEY: params} if PARAMS_KEY in parameters else read


__all__ = ["ACTION_NAMES", "KEYS_SENT", "as_the_schema_reads", "containers_decoded", "envelope_opened", "keys_read"]
