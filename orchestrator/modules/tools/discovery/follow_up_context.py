"""PRD-268 US-C11c (§6.3, K7, K10) — a follow-up's person and chat come from the server only.

``carries_the_follow_up`` wraps ``PlatformActionExecutor.execute``, beside
``carries_the_check_in``. It strips a caller-supplied ``_follow_up`` from every call, and
hands the three follow-up tools (``platform_schedule_follow_up``,
``platform_list_follow_ups``, ``platform_cancel_follow_up``) the chat's own facts from the
server-built ``caller_context`` (``consumers.chatbot.service.build_tool_caller_context``):

* ``user_id``: the person driving the turn (``core.security.driving_user.driving_user_id``;
  None for a widget visitor, a ticket, a Mission or a check-in);
* ``chat_id``: the turn's ``conversation_id``, the chat the follow-up comes back to;
* ``locale``: the turn's language when the context names one (else English).

PRD-268 US-C12a: ``platform_offer_replies``, the fourth chat-only tool, gets the same facts
(``CHAT_FACT_ACTIONS``) and reads only the language its refusals are written in.

With the executor's server-minted ``_agent_id`` (``exec_platform``) that is everything the
handlers trust: a chat id or user id the model sends is a parameter they never read, so a
follow-up can only be scheduled for the person whose turn it is, in their own chat
(``chat_messenger``'s rule for an origin chat).
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from core.security.driving_user import driving_user_id

__all__ = [
    "CANCEL_FOLLOW_UP", "CHAT_FACT_ACTIONS", "FOLLOW_UP_ACTIONS", "FOLLOW_UP_PARAM", "LIST_FOLLOW_UPS",
    "OFFER_REPLIES", "SCHEDULE_FOLLOW_UP", "carries_the_follow_up", "follow_up_caller", "follow_up_params",
]

SCHEDULE_FOLLOW_UP = "platform_schedule_follow_up"
LIST_FOLLOW_UPS = "platform_list_follow_ups"
CANCEL_FOLLOW_UP = "platform_cancel_follow_up"
FOLLOW_UP_ACTIONS = frozenset({SCHEDULE_FOLLOW_UP, LIST_FOLLOW_UPS, CANCEL_FOLLOW_UP})
# PRD-268 US-C12a: the reply chips get the chat's facts too (their refusals' language).
OFFER_REPLIES = "platform_offer_replies"
CHAT_FACT_ACTIONS = FOLLOW_UP_ACTIONS | {OFFER_REPLIES}
# The server-side parameter that hands the tools their caller; never the model's.
FOLLOW_UP_PARAM = "_follow_up"
CONVERSATION_KEY = "conversation_id"
LOCALE_KEY = "locale"

Execute = Callable[..., Awaitable[Dict[str, Any]]]


def _said(caller_context: Any, key: str) -> Optional[str]:
    """``caller_context[key]`` as text, or None when it names nothing."""
    value = caller_context.get(key) if isinstance(caller_context, Mapping) else None
    if value is None:
        return None
    return str(value).strip() or None


def follow_up_caller(caller_context: Any) -> Dict[str, Any]:
    """The follow-up tools' facts from the server-built ``caller_context``: ``user_id`` (their
    ``users.id``), ``chat_id`` and ``locale``, each None when the context names none. A
    check-in's call (a follow-up's own turn too) is made for nobody, whatever reaches the
    handler: the lane refuses it first (``check_in_lane.CHAT_ONLY_ACTIONS``), and this is
    the second wall."""
    from modules.tools.discovery.check_in_lane import in_a_check_in

    user_id = None if in_a_check_in(caller_context) else driving_user_id(caller_context)
    return {"user_id": user_id, "chat_id": _said(caller_context, CONVERSATION_KEY) if user_id is not None else None,
            "locale": _said(caller_context, LOCALE_KEY)}


def follow_up_params(action_name: str, params: Any, caller_context: Any) -> Any:
    """``params`` without a caller-supplied ``_follow_up``; a follow-up tool, and the reply
    chips, get the server's, from ``caller_context``. Only these tools' params are read as
    JSON text here."""
    from modules.tools.execution.params_text import params_object

    if action_name in CHAT_FACT_ACTIONS:
        params = params_object(params)
    if not isinstance(params, dict):
        return params
    kept = {key: value for key, value in params.items() if key != FOLLOW_UP_PARAM}
    if action_name not in CHAT_FACT_ACTIONS:
        return kept
    return {**kept, FOLLOW_UP_PARAM: follow_up_caller(caller_context)}


def carries_the_follow_up(execute: Execute) -> Execute:
    """Wrap ``PlatformActionExecutor.execute`` so a follow-up's person and chat reach the
    follow-up tools server-side, and never from the model."""
    @functools.wraps(execute)
    async def wrapped(self: Any, action_name: str, params: Any, caller_context: Any = None) -> Dict[str, Any]:
        return await execute(self, action_name, follow_up_params(action_name, params, caller_context), caller_context)
    return wrapped
