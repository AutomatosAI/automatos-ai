"""A count for a date the data has not reached is not a count (F301 B1, build 14 retest).

"How many Harvest Club boxes go out on Monday 5 October?" On build 14 the Analyst's one
query (audit row 350, #1895) was ``… WHERE sp.name = 'Harvest Club' AND so.shipped_on =
'2026-10-05'``. ``subscription_orders.shipped_on`` ends at 2026-09-11, so the answer was a
single row, ``harvest_club_boxes_scheduled: 0``, with a note that October is not
recorded yet. The card led with "There are 0 Harvest Club boxes scheduled …" and the note
came second. Right: 63, the active Harvest Club subscribers (the 2 paused come back later).

So an empty or all-zero answer whose query filters a date column past its last recorded
value is never handed over as a count. The SQL writer is asked once more, told the date
is past the data and to count from the current state that decides it (tables of the
source with fixed-value columns such as a status, named from the schema), and that answer
comes back with a line saying how it was worked out. If the second answer fails or is
past the data too, the agent gets no rows and no count, only what to answer from.

General by construction: any table, any date column, any date — the ranges and value sets
are the source's facts (``schema.grounding``), never this shop's names.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Set

from .agent_answer import PastDate, dates_past_the_data, tables_in_sql
from .schema.grounding import Facts, fact_key

CANDIDATE_COLUMNS_MAX = 6     # fixed-value columns named per table in a redirect
REDIRECT = (
    "{dates}: rows for that date do not exist yet, so do not filter on that date and do not "
    "count them. Answer the question from the current state that decides it instead (the rows "
    "in force now, such as a status), keeping the question's other filters (which plan, which "
    "customer). Tables that hold such state: {candidates}."
)
WITHHELD = (
    "Not recorded yet, so there is no count to give: {dates}. Do not answer 0. Answer from the "
    "current state that decides it ({candidates}) and say that is how you got the number."
)
DERIVED = (
    "How this was worked out: {dates}, so counting rows for that date would give 0 only because "
    "it has not happened yet. The question was asked again from the current state instead "
    "(the SQL above). Say so in the answer."
)
DATE_PAST = "{table}.{column} has nothing recorded after {last}, and the question is about {asked}"
NO_CANDIDATES = "no table here holds a fixed set of values; say what is missing"


def _is_zero(value: Any) -> bool:
    """None, or a number that is 0 (a COUNT or SUM over no rows)."""
    if value is None:
        return True
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool) and value == 0


def is_empty_answer(result: Dict[str, Any]) -> bool:
    """No rows, or rows whose every value is 0 or empty. An answer that carries no rows
    at all (the analysis route's insight) is not one."""
    if "data" not in result:
        return False
    rows = result.get("data") or []
    return all(_is_zero(v) for row in rows if isinstance(row, dict) for v in row.values())


def past_the_data(result: Any, schema_metadata: Dict[str, Any], facts: Facts) -> List[PastDate]:
    """The dates past the data behind an empty or all-zero successful answer, or none."""
    if not isinstance(result, dict) or not result.get("success") or not is_empty_answer(result):
        return []
    return dates_past_the_data(str(result.get("sql") or ""), schema_metadata, facts)


def _value_columns(table: Dict[str, Any], facts: Facts) -> List[str]:
    """``status: active | paused`` for each column of ``table`` with a fixed value set."""
    named = []
    for column in table.get("columns") or []:
        values = (facts.get(fact_key(table["name"], str(column.get("name")))) or {}).get("values")
        if values:
            named.append(f"{column['name']}: {' | '.join(values)}")
    return named[:CANDIDATE_COLUMNS_MAX]


def _related(names: Set[str], schema_metadata: Dict[str, Any]) -> Set[str]:
    """``names`` and every table a relationship joins to one of them."""
    related = set(names)
    for rel in schema_metadata.get("relationships") or []:
        if rel.get("from_table") in names or rel.get("to_table") in names:
            related.update({rel.get("from_table"), rel.get("to_table")})
    return related


def candidates(past: List[PastDate], sql: str, schema_metadata: Dict[str, Any], facts: Facts) -> str:
    """The source's tables that can decide the answer now: joined to the tables the query
    used, not the ones whose dates ran out, holding fixed-value columns (a status, a plan).
    Falls back to every such table when none is joined."""
    used = {t["name"] for t in tables_in_sql(sql, schema_metadata)}
    stale = {p.table for p in past}
    tables = [t for t in schema_metadata.get("tables") or [] if t.get("name") and t["name"] not in stale]
    joined = _related(used, schema_metadata)
    for pool in ([t for t in tables if t["name"] in joined], tables):
        lines = [f"{t['name']} ({'; '.join(cols)})" for t in pool if (cols := _value_columns(t, facts))]
        if lines:
            return ", ".join(lines)
    return NO_CANDIDATES


def _dates(past: List[PastDate]) -> str:
    """Each date past the data, in a sentence."""
    return "; ".join(DATE_PAST.format(**p._asdict()) for p in past)


def redirect_instruction(past: List[PastDate], sql: str, schema_metadata: Dict[str, Any], facts: Facts) -> str:
    """What the SQL writer is told when it is asked again."""
    return REDIRECT.format(dates=_dates(past), candidates=candidates(past, sql, schema_metadata, facts))


def derived(result: Dict[str, Any], past: List[PastDate]) -> Dict[str, Any]:
    """The second answer, with the line saying how it was worked out."""
    return {**result, "derived": DERIVED.format(dates=_dates(past))}


def withheld(result: Dict[str, Any], past: List[PastDate], schema_metadata: Dict[str, Any], facts: Facts) -> Dict[str, Any]:
    """The first answer with its rows and count taken out, and what to answer from instead."""
    sql = str(result.get("sql") or "")
    kept = {k: v for k, v in result.items() if k not in ("data", "row_count", "columns", "answer")}
    return {
        "success": True,
        "not_recorded": True,
        "answer": WITHHELD.format(dates=_dates(past), candidates=candidates(past, sql, schema_metadata, facts)),
        **kept,
        "data": [],
    }
