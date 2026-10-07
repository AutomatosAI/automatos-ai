"""Auto and the agents see a ticket by its board number only (Gerard, 7 Oct).

A ticket reference a person or an agent gives is the ticket with that board number
(``services.ticket_numbers.read_bare_refs``). The ticket tools answered with both the
ticket's ``number`` (#0892) and its database ``id`` (2147), and a model that copied the id
into its next call named another ticket whenever that id was also a board number: in a
workspace with one tenant, ids and numbers run close together.

So what a model reads of a tool's answer names each ticket by its number alone: wherever a
ticket's number stands, its integer ``id`` and ``task_id`` beside it are left out, and a bulk
answer's ``updated`` lists the numbers. A ticket with no number keeps its id, the one way to
name it. The answer itself is unchanged: the chat's cards, the frontend and the platform's
own code read the ids as before. The id fallback stays: an id no ticket has as its number
still names its ticket.
"""
from __future__ import annotations

import functools
import re
from typing import Any, Callable, Dict

# A ticket's number as the board shows it: "#0892", a mission step's "#0051.3".
TICKET_NUMBER = re.compile(r"^#\d+(?:\.\d+)?$")
ID_KEYS = ("id", "task_id")
PLATFORM_PREFIX = "platform_"
# Deep enough for a bulk answer's failures and a card inside frontend_data.
MAX_DEPTH = 6


def by_number_only(value: Any, depth: int = 0) -> Any:
    """``value`` (a tool's answer, or part of one) as a model reads it: a new copy in which each
    ticket is named by its board number alone."""
    if depth > MAX_DEPTH:
        return value
    if isinstance(value, list):
        return [by_number_only(item, depth + 1) for item in value]
    if not isinstance(value, dict):
        return value
    out = {key: by_number_only(item, depth + 1) for key, item in value.items()}
    if _is_number(out.get("number")):
        out = {key: item for key, item in out.items() if key not in ID_KEYS or not _is_row_id(item)}
    return _bulk_by_number(out)


def _bulk_by_number(out: Dict[str, Any]) -> Dict[str, Any]:
    """A bulk status answer's ``updated`` as the tickets' numbers (``updated_numbers``)."""
    numbers = out.get("updated_numbers")
    if not (isinstance(numbers, list) and isinstance(out.get("updated"), list)):
        return out
    rest = {key: item for key, item in out.items() if key != "updated_numbers"}
    return {**rest, "updated": [n if n else i for n, i in zip(numbers, out["updated"])]}


def _is_number(value: Any) -> bool:
    return isinstance(value, str) and bool(TICKET_NUMBER.match(value))


def _is_row_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def names_tickets_by_number(format_for_llm: Callable[..., str]) -> Callable[..., str]:
    """Wrap ``ToolResultFormatter.format_for_llm``: a platform tool's answer reaches the model
    with each ticket named by its board number alone."""
    @functools.wraps(format_for_llm)
    def wrapped(result: Dict[str, Any], tool_name: str, *args: Any, **kwargs: Any) -> str:
        seen = by_number_only(result) if str(tool_name or "").startswith(PLATFORM_PREFIX) else result
        return format_for_llm(seen, tool_name, *args, **kwargs)
    return wrapped


__all__ = ["by_number_only", "names_tickets_by_number"]
