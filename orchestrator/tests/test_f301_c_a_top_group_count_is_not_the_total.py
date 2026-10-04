"""F301 (night 9, query side, B4) — the top group's count is not the total.

"How many Harvest Club members cancelled between April and September, and what's the
most common reason?" Twice (audit rows 279 and 336) the SQL writer answered both halves
with one query: ``GROUP BY s.cancel_reason ORDER BY count DESC LIMIT 1``. One row came
back, "too much coffee: 4", and Auto gave 4 as the number who cancelled (chat d42de780).
11 cancelled (the Business Analyst's grouped query without the LIMIT, row 182, #1857).

An answer whose grouped query was cut to its top few groups, and came back full, now
says each count is that group's own, not the total.
"""
from __future__ import annotations

from tests import helpers_shop_database as shop

TOP_REASON = (
    "SELECT s.cancel_reason, COUNT(s.subscriber_id) AS num_cancellations FROM subscribers AS s "
    "JOIN subscription_plans AS sp ON s.plan_code = sp.plan_code WHERE sp.name = 'Harvest Club' "
    "AND s.cancelled_on BETWEEN '2026-04-01' AND '2026-09-30' GROUP BY s.cancel_reason "
    "ORDER BY num_cancellations DESC LIMIT 1"
)


def _answer(sql, rows):
    return {"success": True, "sql": sql, "columns": ["cancel_reason", "num_cancellations"],
            "data": rows, "row_count": len(rows)}


def test_the_most_common_reasons_count_comes_back_marked_as_not_the_total(monkeypatch):
    shop.use_shop(monkeypatch, _answer(TOP_REASON, [{"cancel_reason": "too much coffee", "num_cancellations": 4}]),
                  facts=shop.SHOP_FACTS)

    answer = shop.ask_like_a_board_agent(
        "How many Harvest Club members cancelled between April and September, and what's the most common reason?")

    assert answer["notes"] == [
        "This query keeps only the top 1 group(s) (LIMIT 1), so each count here is that group's own, "
        "not the total across all groups. Ask again for the total if the question needs one."
    ]


def test_every_group_returned_needs_no_note(monkeypatch):
    every_reason = TOP_REASON.replace("LIMIT 1", "LIMIT 1000")
    rows = [{"cancel_reason": reason, "num_cancellations": n}
            for reason, n in (("too much coffee", 4), ("switched to supermarket", 3), ("gift ended", 1),
                              ("money is tight", 1), ("prefer to buy one-off", 1), ("price", 1))]
    shop.use_shop(monkeypatch, _answer(every_reason, rows), facts=shop.SHOP_FACTS)

    answer = shop.ask_like_a_board_agent("Harvest Club cancellations April to September by reason")

    assert "notes" not in answer


def test_a_top_three_that_came_back_short_needs_no_note():
    from modules.nl2sql.agent_answer import top_groups_notes

    sql = "SELECT cafe_name, SUM(kg) FROM wholesale_orders GROUP BY cafe_name ORDER BY 2 DESC LIMIT 3"
    assert top_groups_notes(sql, 2) == []
    assert top_groups_notes(sql, 3) and top_groups_notes("SELECT COUNT(*) FROM subscribers LIMIT 1", 1) == []
