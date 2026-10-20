"""PRD-264 EUAI-07: a request an agent's ``http_request`` tool makes says so, in a header it can't drop.

The tool may call the platform's own API (``exec_shell._get_allowed_http_domains``), and so the
email routes a person's own Send uses (``api/widget_email.py``). Those routes run as a person's
direct send: their body carries no AI line. A request this tool makes always carries
:data:`AGENT_REQUEST_HEADER`, set by the server after the model's own headers (any copy of it the
model wrote, in any case, is dropped), so the route knows an agent wrote the email and puts the
line on (``send_notice.told_if_an_agent_asked``).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping, Tuple

logger = logging.getLogger(__name__)

AGENT_REQUEST_HEADER = "X-Automatos-Agent-Request"
AGENT_REQUEST_MARK = "1"
HEADERS = "headers"
HEADER_PAIR = 2

Execute = Callable[..., Awaitable[Dict[str, Any]]]


def _header_items(headers: Any) -> Iterable[Tuple[Any, Any]]:
    """The model's headers as ``(name, value)`` pairs: a mapping's items, or a list of pairs."""
    if isinstance(headers, Mapping):
        return headers.items()
    if isinstance(headers, (list, tuple)):
        return [tuple(pair) for pair in headers if isinstance(pair, (list, tuple)) and len(pair) == HEADER_PAIR]
    if headers:
        logger.warning("[http_request] headers of type %s dropped: not a mapping or a list of pairs",
                       type(headers).__name__)
    return []


def marked_request(parameters: Any) -> Any:
    """The tool's parameters with :data:`AGENT_REQUEST_HEADER` set, whatever the model wrote."""
    if not isinstance(parameters, dict):
        return parameters
    marker = AGENT_REQUEST_HEADER.lower()
    kept = {name: value for name, value in _header_items(parameters.get(HEADERS)) if str(name).lower() != marker}
    return {**parameters, HEADERS: {**kept, AGENT_REQUEST_HEADER: AGENT_REQUEST_MARK}}


def marks_an_agent_request(execute: Execute) -> Execute:
    """Wrap ``exec_shell.execute_http_request``: every request it makes carries the marker."""
    @functools.wraps(execute)
    async def wrapped(executor: Any, tool_name: str, parameters: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return await execute(executor, tool_name, marked_request(parameters), *args, **kwargs)
    return wrapped


__all__ = ["AGENT_REQUEST_HEADER", "AGENT_REQUEST_MARK", "marked_request", "marks_an_agent_request"]
