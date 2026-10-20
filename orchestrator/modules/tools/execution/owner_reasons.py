"""P256-FIX-N17-10 (F449): why a call was refused, in words the owner reads.

Night 17 (455 asks): 141 refused receipts, and each said why in the text written for the model:
"'platform_board_snapshot' does not take ['status'], so nothing was done. Fix the call and send it
again…", "Unknown platform action: … Use one of: […]", a psycopg2 trace, "Validation failed" printing
the workspace's table list. The receipts and the lines above the reply are built after the stream,
past the filter that keeps the platform's names out of Auto's words, so the owner read them as written.

A refusal now carries the owner's reason beside the model's text: ``owner_reason`` on the result, set
where the refusal is made. The model's ``error`` and its ``llm_context`` are untouched (the router's
``select_failure_message`` reads only ``error``). A refusal that names the card it is about carries it as
``acts_on``: a later write to that card settles it (``receipts.refused_writes_left``).

``Refusal`` is a refusal's text (a str, so every caller that reads a refusal as text still does) that
carries its owner's reason and its card, for the rules and resolvers that return text, not a result.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

OWNER_REASON = "owner_reason"
ACTS_ON = "acts_on"

SENT_WRONG = "it was sent in the wrong form, so nothing ran"
NO_SUCH_ACTION = "the platform has no such action, so nothing ran"
DATA_FAILED = "the data query failed"
PLATFORM_FAILED = "it failed inside the platform"
INTENT_REFUSED = "{what} isn't allowed for this request"
AGENTS_SHARE_A_NAME = "{count} agents are called {name}, so it wasn't clear which one you meant"

# The data actions: a failure inside one is the query's (night 17: psycopg2 at asks 38, 100, 214, 436).
DATA_ACTIONS = frozenset({"query_data", "query_database", "smart_query_database"})
# A driver's or a validator's failure: its text names the platform's insides (a trace, the table list).
_FAILED_INSIDE = re.compile(r"\bpsycopg2?\b|\bsqlalchemy\b|^\s*Execution error:|^\s*Validation failed:", re.I)
_SLUG_SEP = "_"
PLATFORM_PREFIX = "platform_"


class Refusal(str):
    """A refusal's text for the model, carrying the owner's reason (``owner_reason``) and the card it
    is about (``card``, the board's number) when there are ones. Set once, when it is made."""

    owner_reason: Optional[str]
    card: Optional[str]

    def __new__(cls, text: str, owner_reason: Optional[str] = None, card: Optional[str] = None) -> "Refusal":
        made = super().__new__(cls, text)
        made.owner_reason = owner_reason
        made.card = card
        return made


def owner_fields(said: Any, reason: Optional[str] = None) -> Dict[str, str]:
    """The result's fields for a refusal: its owner's reason (``reason``, else the one ``said`` carries)
    and the card it acts on, each only when there is one."""
    fields = {OWNER_REASON: reason or getattr(said, "owner_reason", None), ACTS_ON: getattr(said, "card", None)}
    return {key: value for key, value in fields.items() if isinstance(value, str) and value}


def refused(said: Any, reason: Optional[str] = None) -> Dict[str, Any]:
    """A refused call's result: ``said`` for the model, with the owner's reason and the card when known."""
    return {"success": False, "error": str(said), **owner_fields(said, reason)}


def intent_refused(action_id: Any) -> str:
    """The intent gate's refusal of a Composio action: "the Gmail action isn't allowed for this request"."""
    app = str(action_id or "").strip().split(_SLUG_SEP)[0].capitalize()
    return INTENT_REFUSED.format(what=f"the {app} action" if app else "that action")


def agents_share_a_name(count: int, name: Any) -> str:
    """An agent-name clash: "2 agents are called BEANCOUNTER, so it wasn't clear which one you meant"."""
    return AGENTS_SHARE_A_NAME.format(count=count, name=str(name).strip())


def failed_inside(action: str, error: str) -> Optional[str]:
    """For a failure no one tagged: a driver's or validator's text is the data query's failure for a data
    action, else the platform's; None for any other text (it is said as it came)."""
    if not _FAILED_INSIDE.search(error or ""):
        return None
    return DATA_FAILED if str(action).removeprefix(PLATFORM_PREFIX) in DATA_ACTIONS else PLATFORM_FAILED


def said_in(answer: Dict[str, Any], key: str) -> Optional[str]:
    """``answer[key]`` when it is words, else None."""
    said = answer.get(key)
    return said.strip() if isinstance(said, str) and said.strip() else None


__all__ = ["ACTS_ON", "AGENTS_SHARE_A_NAME", "DATA_ACTIONS", "DATA_FAILED", "INTENT_REFUSED", "NO_SUCH_ACTION",
           "OWNER_REASON", "PLATFORM_FAILED", "Refusal", "SENT_WRONG", "agents_share_a_name", "failed_inside",
           "intent_refused", "owner_fields", "refused", "said_in"]
