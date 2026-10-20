"""PRD-264 EUAI-07 (Guidelines ¶36): an agent's email or message says at the top that an AI wrote it, and for whom.

Every Composio send or publish an agent starts (``owner_only.is_composio_send``) carries, as the
first line of its body, "Written and sent by an AI agent on behalf of {principal}." (or "... of
the sender." when the workspace has no public name, FR-15: ``core.public_principal``); an HTML
body gets it as its first paragraph. The rules and the body map live in
``modules.policy.ai_disclosure``; this module puts the line on the call, where the executor
reads the action's params:

* ``composio_execute``: its ``params`` object (else ``parameters``), then the stray top-level
  keys beside it (``composio_params.sent_params``: the explicit object wins);
* a per-action tool (GMAIL_SEND_EMAIL, composio_SLACK_SEND_MESSAGE): a ``params`` object nested
  in its params when it has one, else its own params (``exec_composio.execute_composio_tool``).

A send whose words are not in a text parameter (a Slack message of ``blocks`` or
``attachments``, a Discord message of ``embeds``) gets the line as its text parameter (the map's
body key, which those apps show beside the blocks or embeds). One that has no text parameter at
all, or holds something other than text there, can't carry the line: :func:`untold_send` says
why, and the gate refuses it before any card rather than sending it unlabelled.

``owner_only.asks_before_a_send`` calls :func:`told_call` before the owner's card is built, so
the card shows the body that goes out and the grant's params hash covers it; the click runs that
call again, and the line, already there, is not added twice. The owner's click never removes it
(D9). A person's own send (the ``user_direct`` actor of ``api/widget_email.py``) is unchanged,
unless an agent's ``http_request`` made it (:func:`told_if_an_agent_asked`).

A Composio call asked for under a name that is not a send, which the executor then resolves onto
one, can't have its body changed there (``core/composio/tool_executor.py`` resolves it after this
gate): any such send that delivers words is refused (:func:`untold_resolved_send`), and the agent
is told to call the send by its name, which carries the line.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

from core.public_principal import workspace_principal_name
from modules.policy.ai_disclosure import (
    is_html_body,
    send_action_key,
    send_body_key,
    send_notice,
    send_text_param,
    sends_words,
    starts_with_send_notice,
    with_send_notice,
)
from modules.tools.execution.agent_request import AGENT_REQUEST_HEADER
from modules.tools.execution.composio_params import sent_params

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
SEND_UNTOLD = (
    "Nothing was sent and no card was raised: this send can't carry the AI notice. What {action} sends must "
    "start with the line saying an AI agent wrote it, and this call has no text parameter to hold it. Put the "
    "message's words (for a message of blocks or embeds, the text shown beside them) in its text parameter "
    "and ask again."
)

Holder = Tuple[Optional[str], str]   # (where the params object sits: None for the call's top level, its body key)


def sent_by_a_person(caller_context: Any) -> bool:
    """A person's own send, made directly through an API surface (no agent wrote it)."""
    return isinstance(caller_context, Mapping) and caller_context.get(ACTOR_TYPE) == USER_DIRECT


def told_call(db: Any, workspace_id: Any, tool_name: str, parameters: Any, slug: str, caller_context: Any) -> Any:
    """The call's ``parameters`` with the AI line at the top of the body ``slug`` sends; with no
    text body, the line as the action's text parameter.

    ``parameters`` itself (the same object) when the call is a person's own send, sends no words
    (an order, a payment), already starts with the line, or has nowhere to hold it
    (:func:`untold_send` refuses that one); otherwise a new object. The public name is read only
    for a call the line goes on.
    """
    if not _an_agents_words(slug, parameters, caller_context):
        return parameters
    holder = _body_holder(tool_name, parameters, slug) or _text_holder(tool_name, parameters, slug)
    if holder is None:
        return parameters
    return _with_the_line(parameters, holder, send_notice(workspace_principal_name(db, workspace_id)))


def untold_send(tool_name: str, parameters: Any, slug: str, caller_context: Any) -> Optional[str]:
    """Why an agent's send, as :func:`told_call` left it, may not go out: what the executor sends
    for it does not start with the AI line. ``None`` for a person's own send, a send with no
    words of its own, or one whose body starts with the line."""
    if sent_by_a_person(caller_context) or not sends_words(slug):
        return None
    sent = _sent(tool_name, parameters)
    key = send_body_key(slug, sent)
    if key is not None and starts_with_send_notice(sent[key]):
        return None
    logger.warning("[send_notice] %s has no text parameter that carries the AI line: refused", slug)
    return SEND_UNTOLD.format(action=send_action_key(slug))


def told_if_an_agent_asked(headers: Any, db: Any, workspace_id: Any, action: str,
                           params: Dict[str, Any]) -> Dict[str, Any]:
    """A platform email route's params, with the AI line when an agent's ``http_request`` made the
    request (:data:`AGENT_REQUEST_HEADER`); a person's own send as it is. A body of only spaces is
    the line alone, as an agent's Composio send with no text body is."""
    if headers is None or not headers.get(AGENT_REQUEST_HEADER):
        return params
    holder = _body_holder(action, params, action) or _text_holder(action, params, action)
    if holder is None:
        return params
    logger.info("[send_notice] %s through the email route was asked by an agent: the AI line goes on", action)
    return _with_the_line(params, holder, send_notice(workspace_principal_name(db, workspace_id)))


def untold_resolved_send(asked: Any, action: str, caller_context: Any) -> Optional[str]:
    """Why a send the executor resolved from another name may not run: what it sends can't carry
    the AI line there, with a body or without one. ``None`` for a person's own send, or an action
    that sends no words of its own (an order, a payment)."""
    if sent_by_a_person(caller_context) or not sends_words(action):
        return None
    logger.info("[send_notice] '%s' resolved onto the send %s, which can't carry the AI line: refused", asked, action)
    return RESOLVED_SEND_UNTOLD.format(asked=asked, action=action)


def _an_agents_words(slug: str, parameters: Any, caller_context: Any) -> bool:
    """Whether the call is an agent's send of words of its own, in params the line can go on."""
    return not sent_by_a_person(caller_context) and isinstance(parameters, dict) and sends_words(slug)


def _holders(tool_name: str, parameters: Dict[str, Any]) -> Tuple[Optional[str], ...]:
    """Where the executor reads the action's params, first to last; none when a per-action tool's
    nested params are not an object (the executor sends them as they are)."""
    if tool_name == COMPOSIO_EXECUTE:
        explicit = next((key for key in EXPLICIT_KEYS if isinstance(parameters.get(key), dict)), None)
        return (explicit, TOP_LEVEL) if explicit else (TOP_LEVEL,)
    nested = parameters.get(NESTED_KEY)
    if nested is None:
        return (TOP_LEVEL,)
    return (NESTED_KEY,) if isinstance(nested, dict) else ()


def _held(parameters: Dict[str, Any], where: Optional[str]) -> Dict[str, Any]:
    return parameters if where is TOP_LEVEL else parameters[where]


def _sent(tool_name: str, parameters: Any) -> Dict[str, Any]:
    """The params the executor sends the action for this call (``{}`` when it can't read them)."""
    if not isinstance(parameters, dict):
        return {}
    if tool_name == COMPOSIO_EXECUTE:
        return sent_params(parameters)
    holders = _holders(tool_name, parameters)
    return _held(parameters, holders[0]) if holders else {}


def _body_holder(tool_name: str, parameters: Dict[str, Any], slug: str) -> Optional[Holder]:
    """Where the text body the executor sends is: the params object holding it and its key;
    ``None`` when what it sends has none."""
    key = send_body_key(slug, _sent(tool_name, parameters))
    if key is None:
        return None
    holders = _holders(tool_name, parameters)
    return next((where for where in holders if key in _held(parameters, where)), holders[0]), key


def _text_holder(tool_name: str, parameters: Dict[str, Any], slug: str) -> Optional[Holder]:
    """Where the line goes for a send with no text body (blocks, attachments, embeds): the action's
    text parameter, in the params object the executor reads first, when what it sends leaves that
    empty. ``None`` when the action documents no text parameter, or the call holds something else
    there (never overwritten: the send is refused instead)."""
    key = send_text_param(slug)
    holders = _holders(tool_name, parameters)
    if key is None or not holders:
        return None
    value = _sent(tool_name, parameters).get(key)
    blank = value is None or (isinstance(value, str) and not value.strip())
    return (holders[0], key) if blank else None


def _with_the_line(parameters: Dict[str, Any], holder: Holder, notice: str) -> Dict[str, Any]:
    """``parameters`` with ``notice`` at the top of the body ``holder`` names (or as that parameter,
    when it holds no text), as a new object; ``parameters`` itself when the body already starts with it."""
    where, key = holder
    held = _held(parameters, where)
    body = held.get(key)
    text = body if isinstance(body, str) else ""   # absent: _text_holder found it empty
    told = with_send_notice(text, notice, is_html_body(text, held))
    if told == body:
        return parameters
    updated = {**held, key: told}
    return updated if where is TOP_LEVEL else {**parameters, where: updated}


__all__ = ["RESOLVED_SEND_UNTOLD", "SEND_UNTOLD", "USER_DIRECT", "sent_by_a_person", "told_call",
           "told_if_an_agent_asked", "untold_resolved_send", "untold_send"]
