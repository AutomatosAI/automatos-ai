"""P256-FIX-N15-9 (night 15, F429): the owner's new rule retires the one it replaces.

Night 15: the owner changed the cafe minimum from 12 kg to 10 kg. Auto stored the new rule and the
old one stayed in the memory store, so the Memory page listed both while the rules said 10. Now,
when ``platform_store_memory`` stores the owner's standing rule (their own words: ``memory_write``),
the memories of theirs it replaces are retired (``modules.memory.retired``: left out of every read,
nothing deleted). A memory is replaced when it is
- the standing rule of theirs Auto names in ``replaces``: its id (as platform_browse_memories lists
  it) or its words as the Standing Rules list shows them; or
- the same statement with another quantity: word for word the new rule's words with one or more
  quantities changed ('Cafe minimum order is 12 kg' → '... 10 kg', '£12' → '£14'), sharing at least
  ``MIN_SHARED_WORDS`` other words. A number is a quantity when it carries its unit or sign ('12kg',
  '£12', '12%') or the word after it is a unit (``QUANTITY_UNITS``); a number that names something
  ('Table 1', 'Q2', 'step 3') is not, so 'Table 2 is for VIPs' never retires 'Table 1 is for VIPs'.
Only that person's own memories in the workspace's namespace are weighed, and only for a write that
is their standing rule: what Auto words itself never retires anything.

P256-FIX-N17-4 (night 17, F447): 'forget the other posting rules, from November we post on
Wednesdays, replace the lot' named one old rule, so Auto deleted the other three. ``replaces`` now
takes a list (or JSON text of one): every rule it names is retired once the new one stands.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, FrozenSet, List, Tuple

from sqlalchemy.orm import Session

from modules.memory.agent_sources import memory_text
from modules.tools.discovery.rule_wording import words
from modules.tools.execution.params_text import json_container

logger = logging.getLogger(__name__)

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

REPLACES_PARAM = "replaces"
# 'Cafe minimum order is 12 kg' → '... 10 kg' shares four words besides the number; 'week 1' →
# 'week 2' shares one, too few to say the two are one statement.
MIN_SHARED_WORDS = 3
# The words after a number that make it an amount of something (the number is then a value the
# owner can change, not the name of a thing).
QUANTITY_UNITS = frozenset({
    "kg", "kgs", "kilo", "kilos", "g", "gram", "grams", "lb", "lbs", "oz", "tonne", "tonnes",
    "l", "litre", "litres", "liter", "liters", "ml",
    "bag", "bags", "box", "boxes", "case", "cases", "pack", "packs", "bottle", "bottles", "cup", "cups",
    "unit", "units", "item", "items", "piece", "pieces", "pcs", "order", "orders", "people", "covers",
    "time", "times", "minute", "minutes", "min", "mins", "hour", "hours", "hr", "hrs", "day", "days",
    "week", "weeks", "month", "months", "year", "years", "am", "pm",
    "percent", "pound", "pounds", "euro", "euros", "dollar", "dollars", "pence",
})
NOT_RETIRED_NOTE = ("The rule this one replaces is still stored: retiring it failed. Tell the owner both are "
                    "kept, and that they can delete the old one on the Memory page.")


def _numeric(word: str) -> bool:
    return any(c.isdigit() for c in word)


def _unit(word: str) -> str:
    """What a number is of ('kg' for '12kg', '£' for '£12'): its characters that are not digits."""
    return "".join(c for c in word if not (c.isdigit() or c in ".,"))


def _quantity(said: List[str], at: int) -> bool:
    """Whether the number at ``at`` is an amount: it carries its unit or sign ('12kg', '£12'), or the
    word after it is a unit ('12 kg'). A number after letters ('q2') names something."""
    word = said[at]
    if word[:1].isalpha():
        return False
    return bool(_unit(word)) or (at + 1 < len(said) and said[at + 1] in QUANTITY_UNITS)


def same_but_its_numbers(old: str, new: str) -> bool:
    """Whether ``new`` is ``old`` word for word with one or more of its quantities changed (each
    keeping its unit), sharing at least ``MIN_SHARED_WORDS`` words that are no number."""
    a, b = words(old), words(new)
    if len(a) != len(b) or a == b:
        return False
    for at, (x, y) in enumerate(zip(a, b)):
        changed = _numeric(x) and _numeric(y) and _unit(x) == _unit(y) and _quantity(a, at)
        if x != y and not changed:
            return False
    return sum(1 for x in a if not _numeric(x)) >= MIN_SHARED_WORDS


def _named(replaces: Any) -> Tuple[FrozenSet[str], FrozenSet[Tuple[str, ...]]]:
    """The ids and the words of the rules ``replaces`` names: one rule (an id or its words), a
    list of them, or JSON text of a list (N17-4); nothing for anything else."""
    decoded = json_container(replaces) if isinstance(replaces, str) else replaces
    named = decoded if isinstance(decoded, list) else [decoded]
    said = [str(item).strip() for item in named if isinstance(item, (str, int)) and not isinstance(item, bool)]
    said = [item for item in said if item]
    return frozenset(said), frozenset(tuple(words(item)) for item in said if words(item))


def replaced_rows(rows: List[Dict[str, Any]], new: str, replaces: Any) -> List[Dict[str, Any]]:
    """The rows (one person's memories) the rule ``new`` replaces: each standing rule ``replaces``
    names (by id or by its words), and any that is ``new`` with other quantities. Never ``new``."""
    from modules.memory.injection_filter import is_standing_rule

    said = words(new)
    named_ids, named_words = _named(replaces)
    out = []
    for row in rows:
        text = memory_text(row)
        if words(text) == said:
            continue
        by_name = is_standing_rule(row) and (str(row.get("id")) in named_ids or tuple(words(text)) in named_words)
        if by_name or same_but_its_numbers(text, new):
            out.append(row)
    return out


async def _retire_replaced(db: Session, workspace_id: Any, params: Dict[str, Any]) -> List[str]:
    """Retire what the stored rule replaces among its owner's memories; the words of each retired."""
    from modules.memory.retired import owners_rows, retire
    from modules.tools.discovery.handlers_workspace import _resolve_owner_subject

    owner = _resolve_owner_subject(db, params.get("_user_id"))
    if not owner:
        return []
    new = str(params.get("content") or "")
    rows = replaced_rows(await owners_rows(workspace_id, owner), new, params.get(REPLACES_PARAM))
    retired = set(await retire(workspace_id, rows, new))
    return [memory_text(row) for row in rows if str(row.get("id")) in retired]


def retires_what_it_replaces(handler: Handler) -> Handler:
    """Wrap ``store_memory``: once the owner's standing rule is stored, the memories of theirs it
    replaces are retired, and the answer lists them (``retired``). A retire that fails is logged
    and the answer says the old one is still stored."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        answer = await handler(db, workspace_id, params)
        if not (isinstance(answer, dict) and answer.get("success") and answer.get("standing_rule")):
            return answer
        try:
            retired = await _retire_replaced(db, workspace_id, params or {})
        except Exception:  # noqa: BLE001 — logged; the new rule stands, and the answer says the old one does too
            logger.exception("[N15-9] the rule a new standing rule replaces was not retired (workspace %s)",
                             workspace_id)
            return {**answer, "replaced_rule_kept": NOT_RETIRED_NOTE}
        return {**answer, "retired": retired} if retired else answer
    return wrapped


__all__ = ["MIN_SHARED_WORDS", "QUANTITY_UNITS", "REPLACES_PARAM", "replaced_rows", "retires_what_it_replaces",
           "same_but_its_numbers"]
