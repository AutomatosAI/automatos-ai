"""F460 (night 18): the same question about the database got 14, 30 and 17.

The query listed 30 wholesale accounts; its rows reached the model as indent=2 JSON cut at 2,000
characters part-way through a row, and the model counted what it could see. The SQL was cut at 300
characters, so its WHERE was hidden, and the answer's notes (F301) were dropped. Now the notes are
printed, the WHERE clause is shown, and rows that would be cut come one whole line each, then
"Rows shown k of 30", with Total Rows as the count.
"""
from __future__ import annotations

import re

from modules.tools.formatting.result_formatter import ToolResultFormatter

TOOL = "query_database"
COLUMNS = ["cafe_name", "town", "contact_email", "price_tier", "last_order_date", "payment_terms_days"]
SELECT = ", ".join(f"wholesale_accounts.{column} AS {column}" for column in COLUMNS * 3)
WHERE = "WHERE wholesale_accounts.status = 'active' AND wholesale_accounts.closed_at IS NULL"
SQL = f"SELECT {SELECT} FROM wholesale_accounts {WHERE} ORDER BY cafe_name LIMIT 100"
NOTES = ["Counts here are only the top groups: the total is Total Rows.",
         "Nothing is recorded after 21 September 2026: a later date has no rows yet."]
ROW = re.compile(r'^"Café \d{2}" \| ')


def _rows(count):
    return [{"cafe_name": f"Café {i:02d}", "town": "Taunton" if i % 7 == 0 else "Bristol",
             "contact_email": f"orders{i:02d}@example.com", "price_tier": "standard",
             "last_order_date": "2026-09-08", "payment_terms_days": 30} for i in range(count)]


def _answer(rows, **extra):
    return {"success": True, "sql": SQL, "data": rows, "columns": COLUMNS, "row_count": len(rows),
            "execution_time_ms": 41.0, "notes": NOTES, **extra}


def test_thirty_rows_reach_the_model_with_their_notes_and_their_count():
    said = ToolResultFormatter.format_for_llm(_answer(_rows(30)), TOOL)

    assert "Total Rows: 30" in said
    for note in NOTES:
        assert f"Note: {note}" in said
    shown = re.search(r"Rows shown (\d+) of 30\. The count is Total Rows, not the rows you can see\.", said)
    assert shown and 0 < int(shown.group(1)) <= 30


def test_no_row_is_cut_part_way():
    said = ToolResultFormatter.format_for_llm(_answer(_rows(30)), TOOL)
    shown = int(re.search(r"Rows shown (\d+) of 30", said).group(1))

    rows = [line for line in said.splitlines() if ROW.match(line)]
    assert len(rows) == shown and all(line.endswith(" | 30") for line in rows)
    assert f"Columns: {' | '.join(COLUMNS)}" in said


def test_the_where_clause_is_shown_past_a_long_select():
    said = ToolResultFormatter.format_for_llm(_answer(_rows(30)), TOOL)

    assert len(SQL) > 300 and WHERE in said


def test_rows_that_fit_are_the_json_they_were():
    said = ToolResultFormatter.format_for_llm(_answer(_rows(2)), TOOL)

    assert "Complete data (2 rows):" in said and '"payment_terms_days": 30' in said and "Rows shown" not in said


def test_the_results_shaped_answer_keeps_its_notes():
    shaped = ToolResultFormatter.format_database(_answer(_rows(3)))

    assert shaped["notes"] == NOTES
