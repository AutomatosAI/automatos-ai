"""PRD-264 EUAI-07 (Guidelines ¶36): an agent's email or message says at the top that an AI wrote it, and for whom.

Every Composio send or publish an agent starts (``owner_only.is_composio_send``) carries, as the
first line of its body, "Written and sent by an AI agent on behalf of {principal}." (or "... of
the sender." when the workspace has no public name, FR-15: ``core.public_principal``); an HTML
body gets it as its first paragraph. The rules and the body map live in
``modules.policy.ai_disclosure``; this module puts the line on the call, where the executor
reads the action's params:

* ``composio_execute``: its ``params`` object (else ``parameters``), then the stray top-level
  keys beside it (``composio_params.sent_params``: the explicit object wins);
* a per-action tool (GMAIL_SEND_EMAIL, composio_SLACK_SEND_MESSAGE): its own params, then a
  ``params`` object nested in them (``exec_composio.execute_composio_tool``).

``owner_only.asks_before_a_send`` calls :func:`told_call` before the owner's card is built, so
the card shows the body that goes out and the grant's params hash covers it; the click runs that
call again, and the line, already there, is not added twice. The owner's click never removes it
(D9). A person's own send (the ``user_direct`` actor of ``api/widget_email.py``) is unchanged,
unless an agent's ``http_request`` made it (:func:`told_if_an_agent_asked`).

A Composio call asked for under a name that is not a send, which the executor then resolves onto
one, can't have its body changed there (``core/composio/tool_executor.py`` resolves it after this
gate): it is refused (:func:`untold_resolved_send`), and the agent is told to call the send by its
name, which carries the line.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

from core.public_principal import workspace_principal_name
from modules.policy.ai_disclosure import is_html_body, send_body_key, send_notice, with_send_notice
from modules.tools.execution.agent_request import AGENT_REQUEST_HEADER

logger = logging.getLogger(__name__)

COMPOSIO_EXECUTE = "composio_execute"
EXPLICIT_KEYS = ("params", "parameters")  # composio_execute's own params object, in sent_params' order
NESTED_KEY = "params"                     # a per-action tool's nested params (execute_composio_tool)
ACTOR_TYPE = "actor_type"
USER_DIRECT = "user_direct"               # a person's own send (api/widget_email.py, policy gate locked #6)
TOP_LEVEL = None

RESOLVED_SEND_UNTOLD = (
    "Nothing was sent and no card was raised: '{asked}' runs as {action}, a send whose body must start with "
    "the line saying an AI agent wrote it. Call {action} by that name and ask again."
)

Holder = Tuple[Optional[str], str]   # (where the params object sits: None for the call's top level, its body key)


def sent_by_a_person(caller_context: Any) -> bool:
    """A person's own send, made directly through an API surface (no agent wrote it)."""
    return isinstance(caller_context, Mapping) and caller_context.get(ACTOR_TYPE) == USER_DIRECT


def told_call(db: Any, workspace_id: Any, tool_name: str, parameters: Any, slug: str, caller_context: Any) -> Any:
    """The call's ``parameters`` with the AI line at the top of the body ``slug`` sends.

    ``parameters`` itself (the same object) when the call is a person's own send, carries no body
    (an order, a payment) or already starts with the line; otherwise a new object. The public
    name is read only for a call that has a body.
    """
    if sent_by_a_person(caller_context) or not isinstance(parameters, dict):
        return parameters
    holder = _body_holder(tool_name, parameters, slug)
    if holder is None:
        return parameters
    return _with_the_line(parameters, holder, send_notice(workspace_principal_name(db, workspace_id)))


def told_if_an_agent_asked(headers: Any, db: Any, workspace_id: Any, action: str,
                           params: Dict[str, Any]) -> Dict[str, Any]:
    """A platform email route's params, with the AI line when an agent's ``http_request`` made the
    request (:data:`AGENT_REQUEST_HEADER`); a person's own send as it is."""
    if headers is None or not headers.get(AGENT_REQUEST_HEADER):
        return params
    key = send_body_key(action, params)
    if key is None:
        return params
    logger.info("[send_notice] %s through the email route was asked by an agent: the AI line goes on", action)
    return _with_the_line(params, (TOP_LEVEL, key), send_notice(workspace_principal_name(db, workspace_id)))


def untold_resolved_send(asked: Any, action: str, params: Any, caller_context: Any) -> Optional[str]:
    """Why a send the executor resolved from another name may not run: its body can't carry the
    AI line there. ``None`` for a person's own send, or a send with no body of its own."""
    if sent_by_a_person(caller_context) or not isinstance(params, dict):
        return None
    objects = (params, params.get(NESTED_KEY))
    if not any(send_body_key(action, held) for held in objects if isinstance(held, dict)):
        return None
    logger.info("[send_notice] '%s' resolved onto the send %s, whose body has no AI line: refused", asked, action)
    return RESOLVED_SEND_UNTOLD.format(asked=asked, action=action)


def _holders(tool_name: str, parameters: Dict[str, Any]) -> Tuple[Optional[str], ...]:
    """Where the executor reads the action's params, first to last."""
    if tool_name == COMPOSIO_EXECUTE:
        explicit = next((key for key in EXPLICIT_KEYS if isinstance(parameters.get(key), dict)), None)
        return (explicit, TOP_LEVEL) if explicit else (TOP_LEVEL,)
    nested = (NESTED_KEY,) if isinstance(parameters.get(NESTED_KEY), dict) else ()
    return (TOP_LEVEL, *nested)


def _body_holder(tool_name: str, parameters: Dict[str, Any], slug: str) -> Optional[Holder]:
    """Where the send's body is: the params object holding it and its key; ``None`` when it has none."""
    for where in _holders(tool_name, parameters):
        held = parameters if where is TOP_LEVEL else parameters[where]
        key = send_body_key(slug, held)
        if key is not None:
            return where, key
    return None


def _with_the_line(parameters: Dict[str, Any], holder: Holder, notice: str) -> Dict[str, Any]:
    """``parameters`` with ``notice`` at the top of the body ``holder`` names, as a new object;
    ``parameters`` itself when the body already starts with it."""
    where, key = holder
    held = parameters if where is TOP_LEVEL else parameters[where]
    body = held[key]
    told = with_send_notice(body, notice, is_html_body(body, held))
    if told == body:
        return parameters
    updated = {**held, key: told}
    return updated if where is TOP_LEVEL else {**parameters, where: updated}


__all__ = ["RESOLVED_SEND_UNTOLD", "USER_DIRECT", "sent_by_a_person", "told_call", "told_if_an_agent_asked",
           "untold_resolved_send"]
