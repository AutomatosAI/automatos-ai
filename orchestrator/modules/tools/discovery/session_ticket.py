"""The ticket a session's platform call works, handed to the handlers that need it (PRD-255 US-012).

``render_preview`` writes the page it draws into the calling session's own folder,
``sessions/<ticket>/``, so its handler must know the ticket. The ticket is
server-side truth: ``session_tools`` puts it in the executor's caller context
(``session_task_id``, as for ``generate_document``, F341/F349) and this decorator
on ``PlatformActionExecutor.execute`` turns it into the ``_session_task_id``
parameter. Strip-then-inject, like the executor's other server-side keys: a
caller-supplied ``_session_task_id`` is removed from EVERY call, and only an
action in :data:`SESSION_TICKET_ACTIONS` gets the context's ticket. A call with no
session ticket (a chat turn, a board run, a heartbeat) carries none, and the
handler refuses.

PRD-255 US-014: the brand kit proposal card is filed on the caller's own ticket, and
the approved proposal is saved from it. A session's ticket is injected as above; an
API-runtime agent working a board card (the hosted edition's Brand designer) gets
that card's id from the run's server-built context (``board_task_id``) as
``_board_task_id``, stripped from every call in the same way.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, Optional

SESSION_TICKET_PARAM = "_session_task_id"
BOARD_CARD_PARAM = "_board_task_id"
SERVER_TICKET_PARAMS = (SESSION_TICKET_PARAM, BOARD_CARD_PARAM)
# The brand kit proposal's card is the caller's own ticket: a session's, or a board run's card.
BOARD_CARD_ACTIONS = frozenset({"platform_propose_brand_kit", "platform_save_approved_brand_kit"})
SESSION_TICKET_ACTIONS = frozenset({"platform_render_preview"}) | BOARD_CARD_ACTIONS

Execute = Callable[..., Awaitable[Dict[str, Any]]]


def _as_dict(params: Any) -> Any:
    """``params`` as a dict when it is one, or text of one (JSON, or a Python dict written out: F369); else as
    it came (``execute`` refuses it). The executor's one decoder, ``params_text.params_object``."""
    from modules.tools.execution.params_text import params_object

    return params_object(params)


def board_card(caller_context: Any) -> Optional[int]:
    """The board card a board run's call works (``board_task_id``, server-built), else ``None``."""
    from modules.tools.execution.generate_document_tool import BOARD_CARD_KEY, card_of

    raw = caller_context.get(BOARD_CARD_KEY) if isinstance(caller_context, dict) else None
    card = card_of({BOARD_CARD_KEY: raw}) if raw is not None else None
    return card if card is not None and card > 0 else None


def session_ticket_params(action_name: str, params: Any, caller_context: Any) -> Any:
    """``params`` without a caller-supplied ticket, plus the server's own for an action that works its ticket."""
    from modules.tools.execution.session_document_folder import session_ticket

    params = _as_dict(params)
    if not isinstance(params, dict):
        return params
    kept = {k: v for k, v in params.items() if k not in SERVER_TICKET_PARAMS}
    ticket = session_ticket(caller_context) if action_name in SESSION_TICKET_ACTIONS else None
    if ticket is not None:
        return {**kept, SESSION_TICKET_PARAM: ticket}
    card = board_card(caller_context) if action_name in BOARD_CARD_ACTIONS else None
    return {**kept, BOARD_CARD_PARAM: card} if card is not None else kept


def carries_the_session_ticket(execute: Execute) -> Execute:
    """Wrap ``PlatformActionExecutor.execute`` so the session's ticket reaches the handlers that write into its folder."""

    @functools.wraps(execute)
    async def wrapped(self: Any, action_name: str, params: Any, caller_context: Any = None) -> Dict[str, Any]:
        return await execute(self, action_name, session_ticket_params(action_name, params, caller_context), caller_context)

    return wrapped


__all__ = ["BOARD_CARD_ACTIONS", "BOARD_CARD_PARAM", "SESSION_TICKET_ACTIONS", "SESSION_TICKET_PARAM", "board_card",
           "carries_the_session_ticket", "session_ticket_params"]
