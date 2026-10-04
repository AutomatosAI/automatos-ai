"""F300 (night 9, tool side) — a failed query says which columns do exist.

Mission step #1888 (the top three cafés by kilos) stopped with "the query failed because
``wo.order_date`` and ``wo.quantity_kg`` are not recognized as valid columns … I will try
to get the schema … Since there is no direct tool to get the schema". The error the agent
got named only what was wrong. wholesale_orders has ``ordered_on`` and ``kg``.

A failed answer now names the real columns of the tables the query used (or every table,
when it used none that exist) and tells the agent to use them rather than ask the owner.
"""
from __future__ import annotations

import asyncio

from tests import helpers_shop_database as shop

GUESSED = (
    "SELECT wa.cafe_name, SUM(wo.quantity_kg) FROM wholesale_orders AS wo JOIN wholesale_accounts AS wa "
    "ON wa.account_id = wo.account_id WHERE wo.order_date BETWEEN '2026-06-01' AND '2026-08-31' "
    "GROUP BY wa.cafe_name LIMIT 3"
)


def _failed(sql, error):
    return {"success": False, "sql": sql, "error": error, "data": [], "row_count": 0}


def test_a_guessed_column_comes_back_with_the_real_ones(monkeypatch):
    shop.use_shop(monkeypatch, _failed(GUESSED, 'Execution error: column wo.order_date does not exist'),
                  facts=shop.SHOP_FACTS)

    answer = shop.ask_like_a_board_agent("Top three cafés by kilos, June to August 2026")

    assert answer["success"] is False
    assert answer["error"].startswith("Execution error: column wo.order_date does not exist The columns that exist: ")
    assert "wholesale_orders has order_id, account_id, ordered_on, kg, amount_gbp, delivered_on, paid_on" \
        in answer["error"]
    assert "wholesale_accounts has account_id, cafe_name" in answer["error"]
    assert "the owner does not know the database's names" in answer["error"]
    assert answer["schema"]  # and the whole schema beside it


def test_a_guessed_table_comes_back_with_the_tables_that_exist(monkeypatch):
    shop.use_shop(monkeypatch, _failed("SELECT SUM(kg) FROM cafe_orders LIMIT 1",
                                       'Validation failed: table "cafe_orders" is not allowed'), facts=None)

    answer = shop.ask_like_a_board_agent("Total kilos ordered by cafés")

    assert ("The tables that exist: retail_orders, subscription_orders, subscription_plans, "
            "wholesale_accounts, wholesale_orders, subscribers.") in answer["error"]


def test_platform_query_data_hands_the_schema_on_in_a_failure_and_a_success(monkeypatch):
    from modules.tools.discovery.handlers_scheduling import query_data

    shop.use_shop(monkeypatch, _failed(GUESSED, "Execution error: column wo.order_date does not exist"),
                  facts=shop.SHOP_FACTS)
    failed = asyncio.run(query_data(db=None, workspace_id=shop.WORKSPACE, params={"question": "top cafés by kg"}))

    shop.use_shop(monkeypatch, {"success": True, "sql": "SELECT 1", "columns": ["n"], "data": [{"n": 1}],
                                "row_count": 1}, facts=shop.SHOP_FACTS)
    answered = asyncio.run(query_data(db=None, workspace_id=shop.WORKSPACE, params={"question": "one"}))

    assert "ordered_on, kg" in failed["error"] and failed["schema"] == answered["schema"]
    assert any(line.startswith("subscription_orders: ") for line in answered["schema"])
