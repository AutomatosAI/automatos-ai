"""F301 (night 9): a platform_query_data result says what its rows counted.

Night 9, on a shop database the TESTER checked:

* "How many Harvest Club members cancelled between April and September, and what's the most
  common reason?" went to platform_query_data in one call. The rows came back by reason, and
  Auto three times answered "4 cancelled ... most common reason 'too much coffee'" (L69, L101):
  the top reason's count read as the total. The database holds 11.
* "How many Harvest Club boxes go out on Monday 5 October?" came back 0 from rows with
  shipped_on = '2026-10-05', a date nothing has shipped on yet, and Auto said "0 Harvest Club
  boxes are scheduled to ship" (L103). The right answer is the 63 active members.

The rows were right for the SQL that ran; Auto read them as something else. The result now
carries ``counted``: one plain line beside the rows saying what they are. Grouped rows are
groups, and one group's figure is not the total. No rows, or a single count of zero, is
nothing recorded where the query looked, said with what was looked for, never a bare 0. The
building of the SQL itself is NL2SQL's (modules/nl2sql).
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Dict, List, Optional

ROWS_SHOWN = 50
CELL_CHARS = 50
HEADER_CELL_CHARS = 20
NO_ROWS = "Query returned no rows."

COUNTED_ROWS = "These rows are what the SQL selected: answer with their figures, as what they count."
COUNTED_GROUPS = (
    "These rows are groups (one per {by}): one group's figure is not the total. Answer a total "
    "only from a total; for one, ask platform_query_data for the total on its own."
)
COUNTED_NOTHING = (
    "Nothing is recorded where this query looked ({where}). Say that, with what was looked for, "
    "never a bare 0. Something that has not happened yet (a future ship or delivery date) has no "
    "rows by definition: ask platform_query_data for what decides it, such as who it goes to."
)
EVERYWHERE = "no filter"

_CLAUSE_END = r"(?=\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|\bOFFSET\b|;|\Z)"
_GROUP_BY = re.compile(r"\bGROUP\s+BY\s+(.+?)" + _CLAUSE_END, re.IGNORECASE | re.DOTALL)
_WHERE = re.compile(r"\bWHERE\s+(.+?)" + _CLAUSE_END, re.IGNORECASE | re.DOTALL)


def readable_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """platform_query_data's answer for a successful NL2SQL ``result``: the rows as a table
    (the first fifty), the SQL, and ``counted``, what the rows count. A new dict."""
    data = list(result.get("data") or [])
    columns = list(result.get("columns") or [])
    row_count = result.get("row_count")
    row_count = row_count if isinstance(row_count, int) else len(data)
    shown = data[:ROWS_SHOWN]
    sql = result.get("sql")
    return {
        "success": True,
        # First: the result reaches the model as JSON cut to a length, and the line must
        # survive the cut that a long table or row list would otherwise push it past.
        "counted": counted(sql, shown, row_count),
        "answer": _table(columns, shown, row_count) or NO_ROWS,
        "sql": sql,
        "row_count": row_count,
        "columns": columns,
        "data": shown,
        "explanation": result.get("explanation"),
        "confidence": result.get("confidence"),
    }


def counted(sql: Optional[str], rows: List[Dict[str, Any]], row_count: int) -> str:
    """One plain line saying what the rows are: nothing found, groups, or the rows themselves."""
    text = " ".join(str(sql or "").split())
    if row_count == 0 or _single_zero(rows):
        where = _WHERE.search(text)
        return COUNTED_NOTHING.format(where=where.group(1).strip() if where else EVERYWHERE)
    grouped = _GROUP_BY.search(text)
    if grouped:
        return COUNTED_GROUPS.format(by=grouped.group(1).strip())
    return COUNTED_ROWS


def _single_zero(rows: List[Dict[str, Any]]) -> bool:
    """One row whose every value is a number and zero: a COUNT or SUM that found nothing."""
    if len(rows) != 1 or not isinstance(rows[0], dict) or not rows[0]:
        return False
    values = list(rows[0].values())
    return all(isinstance(v, (int, float, Decimal)) and not isinstance(v, bool) and v == 0 for v in values)


def _table(columns: List[Any], rows: List[Dict[str, Any]], row_count: int) -> str:
    if not columns or not rows:
        return ""
    header = " | ".join(str(c) for c in columns)
    separator = "-+-".join("-" * min(len(str(c)), HEADER_CELL_CHARS) for c in columns)
    body = "\n".join(" | ".join(str(row.get(c, ""))[:CELL_CHARS] for c in columns) for row in rows)
    more = f"\n... ({row_count - ROWS_SHOWN} more rows)" if row_count > ROWS_SHOWN else ""
    return f"{header}\n{separator}\n{body}{more}"
