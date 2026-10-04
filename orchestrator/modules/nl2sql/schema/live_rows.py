"""Live rows by default: a closed account is not counted unless asked for (F323, night 9b).

"Which cafés are on 14-day terms?" got 6 from the Ops Manager, the Analyst and Auto all
night, and 4 from the Business Analyst. harbourline_shop has six wholesale_accounts with
``payment_terms_days = 14``; two of them (Lantern Bakehouse, Pier Bakehouse) have
``status = 'closed'``. The writer filtered on status only when the question said
"currently" or "active" (audit rows 671, 690, 696, 697 → 4); otherwise it did not (443,
679, 691, 704 → 6), and nobody said two were closed. The BA also took the closed "Quay
Bakehouse" for Quay Coffee House (#1970).

A column that says whether a row has ended is found on any table, by its values or its
name:

* a text column whose fixed set of values holds an ended word (closed, cancelled, ended,
  inactive, …) beside at least one other value: ``status`` ∈ {active, closed};
* a date column named for an end (``closed_on``, ``ended_at``, ``cancelled_on``,
  ``deleted_at``, ``end_date`` …): set means ended;
* a boolean named for being live (``is_active``, ``active``, ``enabled``) or for having
  ended (``is_deleted``, ``archived``, ``is_closed``).

The SQL writer is told, per table, which rows are live and to leave the ended ones out
unless the question asks for them; the agent's schema carries the same line; and an
answer whose query used such a table without looking at its marker says that ended rows
are in it.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, NamedTuple, Optional

from .grounding import Facts, column_kind, fact_key

ENDED_WORDS = frozenset({
    "closed", "cancelled", "canceled", "ended", "inactive", "terminated", "expired",
    "archived", "deleted", "churned", "lapsed", "deactivated", "disabled", "discontinued",
})
END_DATE_NAME = re.compile(
    r"^(?:end_date|ended|(?:closed|ended|cancell?ed|deleted|archived|terminated|churned|"
    r"expired|deactivated|discontinued)_(?:on|at|date))$",
    re.IGNORECASE,
)
# Not ended, not live either: B1 (F301) counted only active members; a paused one gets no box.
ON_HOLD_WORDS = frozenset({"paused", "suspended", "on hold", "on_hold", "frozen", "dormant"})
LIVE_FLAG_NAMES = frozenset({"active", "is_active", "enabled", "is_enabled", "live", "is_live", "is_open"})
ENDED_FLAG_NAMES = frozenset({"deleted", "is_deleted", "archived", "is_archived", "closed", "is_closed"})
WRITER_RULE = "LIVE ROWS: {live}. {ended}: leave those rows out unless the question asks about them.{held}"
AGENT_LINE = "live rows: {live} ({ended} left out unless asked{held_short})"
HELD_RULE = " {held} is on hold, not ended: count it only when the question would."
ENDED_ROWS_NOTE = (
    "{table} has rows that have ended ({ended}); this query does not look at {columns}, so "
    "they are included. If the question is about live ones only, those are {live}."
)


class LiveRule(NamedTuple):
    """How one column tells a live row from an ended one."""

    column: str
    live: str       # SQL-ish condition for a live row
    ended: str      # condition for an ended row
    held: str = ""  # condition for a row on hold (neither live nor ended), if any


def _quoted(values: List[str]) -> str:
    """``'a', 'b'``."""
    return ", ".join(f"'{v}'" for v in values)


def _value_rule(name: str, values: List[str]) -> Optional[LiveRule]:
    """``status`` ∈ {active, closed} → live ``status = 'active'``, ended ``status = 'closed'``."""
    ended = [v for v in values if v.strip().lower() in ENDED_WORDS]
    held = [v for v in values if v.strip().lower() in ON_HOLD_WORDS]
    live = [v for v in values if v not in ended and v not in held]
    if not ended or not live:
        return None
    return LiveRule(name, _condition(name, live), _condition(name, ended), _condition(name, held) if held else "")


def _condition(name: str, values: List[str]) -> str:
    """``status = 'active'`` or ``status IN ('active', 'open')``."""
    return f"{name} = '{values[0]}'" if len(values) == 1 else f"{name} IN ({_quoted(values)})"


def _named_rule(name: str, column_type: Any) -> Optional[LiveRule]:
    """A rule read from the column's name: an end date, or a live or ended flag."""
    lowered = name.lower()
    if column_kind(column_type) == "date" and END_DATE_NAME.match(lowered):
        return LiveRule(name, f"{name} IS NULL", f"{name} set")
    if "bool" not in str(column_type or "").lower():
        return None
    if lowered in LIVE_FLAG_NAMES:
        return LiveRule(name, f"{name} = TRUE", f"{name} = FALSE")
    if lowered in ENDED_FLAG_NAMES:
        return LiveRule(name, f"{name} = FALSE", f"{name} = TRUE")
    return None


def live_rules(table: Dict[str, Any], facts: Facts) -> List[LiveRule]:
    """Every column of ``table`` that marks a row as ended, by its values or its name."""
    rules: List[LiveRule] = []
    for column in table.get("columns") or []:
        name = str(column.get("name") or "")
        values = (facts.get(fact_key(str(table.get("name")), name)) or {}).get("values") or []
        rule = _value_rule(name, values) if values and column_kind(column.get("type")) == "category" else None
        rule = rule or (_named_rule(name, column.get("type")) if name else None)
        if rule:
            rules.append(rule)
    return rules


def _joined(rules: List[LiveRule]) -> Dict[str, str]:
    """The rules of one table as the texts above take them."""
    held = "; ".join(r.held for r in rules if r.held)
    return {
        "live": " and ".join(r.live for r in rules),
        "ended": "; ".join(r.ended for r in rules),
        "columns": ", ".join(r.column for r in rules),
        "held": HELD_RULE.format(held=held) if held else "",
        "held_short": f"; {held} on hold" if held else "",
    }


def writer_rule(table: Dict[str, Any], facts: Facts) -> Optional[str]:
    """The line the SQL writer gets for ``table`` (its description in the prompt)."""
    rules = live_rules(table, facts)
    return WRITER_RULE.format(**_joined(rules)) if rules else None


def agent_line(table: Dict[str, Any], facts: Facts) -> Optional[str]:
    """The same rule, short, for the agent's ``schema`` line of ``table``."""
    rules = live_rules(table, facts)
    return AGENT_LINE.format(**_joined(rules)) if rules else None


def annotate_live_rows(schema_metadata: Dict[str, Any], facts: Facts) -> None:
    """Write each table's live-row rule into its description, in place (the contract of
    the value-sampling call this rides on; ``_build_prompt`` prints a table's
    description above its columns)."""
    for table in schema_metadata.get("tables") or []:
        rule = writer_rule(table, facts)
        existing = str(table.get("description") or "").strip()
        if rule and rule not in existing:
            table["description"] = f"{existing} {rule}".strip()


def _mentions(sql: str, column: str) -> bool:
    """Whether the statement names ``column`` as a whole word."""
    return re.search(rf"\b{re.escape(column)}\b", sql, re.IGNORECASE) is not None


def ended_rows_notes(sql: str, tables: List[Dict[str, Any]], facts: Facts) -> List[str]:
    """A note for each table the query used that has ended rows when the query never
    looks at the column that marks them (F323: ``… FROM wholesale_accounts WHERE
    payment_terms_days = 14`` counted two closed cafés)."""
    notes: List[str] = []
    for table in tables:
        rules = live_rules(table, facts)
        if rules and not any(_mentions(sql or "", r.column) for r in rules):
            notes.append(ENDED_ROWS_NOTE.format(table=table["name"], **_joined(rules)))
    return notes
