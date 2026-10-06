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
"""
from __future__ import annotations

import functools
import json
from typing import Any, Awaitable, Callable, Dict

SESSION_TICKET_PARAM = "_session_task_id"
SESSION_TICKET_ACTIONS = frozenset({"platform_render_preview"})

Execute = Callable[..., Awaitable[Dict[str, Any]]]


def _as_dict(params: Any) -> Any:
    """``params`` as a dict when it is one or a JSON object string; else as it came (``execute`` refuses it)."""
    if isinstance(params, str):
        try:
            loaded = json.loads(params)
        except (json.JSONDecodeError, TypeError):
            return params
        return loaded if isinstance(loaded, dict) else params
    return params


def session_ticket_params(action_name: str, params: Any, caller_context: Any) -> Any:
    """``params`` without a caller-supplied ticket, plus the session's own for an action that writes into its folder."""
    from modules.tools.execution.session_document_folder import session_ticket

    params = _as_dict(params)
    if not isinstance(params, dict):
        return params
    kept = {k: v for k, v in params.items() if k != SESSION_TICKET_PARAM}
    ticket = session_ticket(caller_context) if action_name in SESSION_TICKET_ACTIONS else None
    return {**kept, SESSION_TICKET_PARAM: ticket} if ticket is not None else kept


def carries_the_session_ticket(execute: Execute) -> Execute:
    """Wrap ``PlatformActionExecutor.execute`` so the session's ticket reaches the handlers that write into its folder."""

    @functools.wraps(execute)
    async def wrapped(self: Any, action_name: str, params: Any, caller_context: Any = None) -> Dict[str, Any]:
        return await execute(self, action_name, session_ticket_params(action_name, params, caller_context), caller_context)

    return wrapped


__all__ = ["SESSION_TICKET_ACTIONS", "SESSION_TICKET_PARAM", "carries_the_session_ticket", "session_ticket_params"]
