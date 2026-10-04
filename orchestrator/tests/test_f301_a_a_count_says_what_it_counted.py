"""F301 (night 9) — platform_query_data's answer says what its rows counted.

Night 9, graded against the shop database:

* B4 "How many Harvest Club members cancelled between April and September, and what's the most
  common reason?" went to platform_query_data in one call. The rows came back one per reason, and
  Auto said "4 cancelled ... most common reason 'too much coffee'" three times (L69, L101): the top
  reason's count as the total. The database holds 11.
* B1 "How many Harvest Club boxes go out on Monday 5 October?" came back as a count of 0 over
  shipped_on = '2026-10-05', a date nothing has shipped on yet; Auto said "0 Harvest Club boxes are
  scheduled to ship" (L103). The right answer is the 63 active members.

The result now carries ``counted``, ahead of the rows, in the text the model reads. These drive
the real handler (platform_query_data → run_nl2sql → the resolver) with only the database reads
and the NL2SQL query faked, as test_f077_query_data does.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

from modules.nl2sql.service import DatabaseKnowledgeService
from modules.tools.discovery.handlers_scheduling import query_data
from modules.tools.formatting.result_formatter import ToolResultFormatter

WS = uuid4()
BY_REASON = (
    "SELECT s.cancel_reason, COUNT(*) AS cancellations FROM subscribers s "
    "JOIN subscription_plans p ON p.id = s.plan_id WHERE p.plan_code = 'CLUB' "
    "AND s.cancelled_on BETWEEN '2026-04-01' AND '2026-09-30' "
    "GROUP BY s.cancel_reason ORDER BY cancellations DESC LIMIT 1000"
)
REASONS = [
    {"cancel_reason": "too much coffee", "cancellations": 4},
    {"cancel_reason": "moving abroad", "cancellations": 3},
    {"cancel_reason": "price", "cancellations": 2},
    {"cancel_reason": "other", "cancellations": 2},
]
SHIPPED_ON = (
    "SELECT COUNT(*) AS boxes FROM subscription_orders o JOIN subscription_plans p ON p.id = o.plan_id "
    "WHERE p.plan_code = 'CLUB' AND o.shipped_on = '2026-10-05' LIMIT 1000"
)
TOTAL = (
    "SELECT COUNT(*) AS cancelled FROM subscribers s JOIN subscription_plans p ON p.id = s.plan_id "
    "WHERE p.plan_code = 'CLUB' AND s.cancelled_on BETWEEN '2026-04-01' AND '2026-09-30' LIMIT 1000"
)


class _Service(DatabaseKnowledgeService):
    """The real resolver; the source read and the NL2SQL query are the fakes."""

    def __init__(self, sql, rows, columns):
        self.sql, self.rows, self.columns = sql, rows, columns

    async def active_sources(self, workspace_id, db_session=None):
        return [(36, "harbourline_shop")]

    async def query_database(self, **kwargs):
        return {"success": True, "sql": self.sql, "data": self.rows, "columns": self.columns,
                "row_count": len(self.rows), "explanation": "as generated"}

    async def write_nl_audit(self, **kwargs):
        return None


class _Db:
    def rollback(self):
        raise AssertionError("a successful query never rolls the turn's session back")


def _ask(monkeypatch, sql, rows, columns, question="How many Harvest Club members cancelled?"):
    monkeypatch.setattr("modules.nl2sql.get_database_knowledge_service", lambda: _Service(sql, rows, columns))
    return asyncio.run(query_data(_Db(), WS, {"question": question, "_user_id": "7"}))


def test_rows_by_reason_say_they_are_groups_and_not_the_total(monkeypatch):
    result = _ask(monkeypatch, BY_REASON, REASONS, ["cancel_reason", "cancellations"])
    assert result["success"] is True and "too much coffee | 4" in result["answer"]
    assert "groups (one per s.cancel_reason)" in result["counted"]
    assert "one group's figure is not the total" in result["counted"]
    assert "ask platform_query_data for the total on its own" in result["counted"]


def test_a_count_of_zero_over_a_future_ship_date_is_nothing_recorded_not_no_boxes(monkeypatch):
    result = _ask(monkeypatch, SHIPPED_ON, [{"boxes": 0}], ["boxes"],
                  question="How many Harvest Club boxes go out on Monday 5 October?")
    counted = result["counted"]
    assert "Nothing is recorded where this query looked" in counted
    assert "o.shipped_on = '2026-10-05'" in counted and "never a bare 0" in counted
    assert "ask platform_query_data for what decides it, such as who it goes to" in counted


def test_no_rows_at_all_is_nothing_recorded_too(monkeypatch):
    result = _ask(monkeypatch, SHIPPED_ON.replace("COUNT(*) AS boxes", "o.id"), [], ["id"])
    assert result["answer"] == "Query returned no rows."
    assert result["counted"].startswith("Nothing is recorded where this query looked")


def test_a_total_is_reported_as_what_the_rows_count(monkeypatch):
    result = _ask(monkeypatch, TOTAL, [{"cancelled": 11}], ["cancelled"])
    assert "11" in result["answer"]
    assert result["counted"] == ("These rows are what the SQL selected: answer with their figures, "
                                 "as what they count.")


def test_the_line_reaches_the_model_ahead_of_rows_that_are_cut(monkeypatch):
    """The model reads the result as JSON cut to a length: the line comes before the rows."""
    many = [{"cancel_reason": f"reason {i}", "cancellations": 1} for i in range(400)]
    result = _ask(monkeypatch, BY_REASON, many, ["cancel_reason", "cancellations"])
    text = ToolResultFormatter.format_for_llm(result, "platform_query_data", max_chars=1500)
    assert "one group's figure is not the total" in text
    assert text.index('"counted"') < text.index('"answer"')
