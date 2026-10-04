"""F300 (night 9, tool side) — the database's names come with every answer.

Board agents asked the shop owner for the database's names: "provide the table names and
column names" (#1856, asks #1458/#1459/#1461), "I need the exact plan_code" (#1886), "the
subscription_orders schema" (#1891). They spent their five queries per turn asking the
SQL writer to list tables (audit rows 207, 247, 293, 307, 318, 345), and the writer
answered from imagination: row 207 listed green_lots with ``country_of_origin, region,
farm, total_kg_original`` — none of which exist. #1891 ran out of queries mid-word.

The platform already holds the schema (``schema_metadata`` of source 47). Every answer
now carries it — tables, columns, types, the complete value set of a small column
(plan_code: CLUB, REGULAR, TASTER) and each date column's first and last recorded date —
read from the owner's database once per source and cached.
"""
from __future__ import annotations

import json

from sqlalchemy import create_engine, text

from tests import helpers_shop_database as shop


def test_the_first_answer_lists_every_table_with_its_columns_values_and_dates(monkeypatch):
    shop.use_shop(monkeypatch, {"success": True, "sql": "SELECT COUNT(*) AS n FROM subscription_plans LIMIT 1",
                                "columns": ["n"], "data": [{"n": 3}], "row_count": 1}, facts=shop.SHOP_FACTS)

    answer = shop.ask_like_a_board_agent("How many subscription plans are there?")

    schema = json.loads(json.dumps(answer))["schema"]
    assert schema[1] == (
        "subscription_orders: order_id integer, subscriber_id integer, box_month date (2024-06-01 to 2026-09-01), "
        "plan_code text (one of CLUB | REGULAR | TASTER), amount_gbp numeric, "
        "shipped_on date (2024-06-03 to 2026-09-11), shipped_late boolean (one of False | True)"
    )
    assert "subscription_plans: plan_code text (one of CLUB | REGULAR | TASTER), " \
           "name text (one of Harvest Club | Regular | Taster)" in schema
    assert any(line.startswith("wholesale_orders: order_id integer, account_id integer, ordered_on date")
               for line in schema)
    assert schema[-1] == ("joins: subscription_orders.plan_code -> subscription_plans.plan_code, "
                          "wholesale_orders.account_id -> wholesale_accounts.account_id, "
                          "subscribers.plan_code -> subscription_plans.plan_code, "
                          "subscription_orders.subscriber_id -> subscribers.subscriber_id")


def test_names_and_types_still_come_when_the_values_could_not_be_read(monkeypatch):
    shop.use_shop(monkeypatch, {"success": True, "sql": "SELECT 1", "data": [], "row_count": 0}, facts=None)

    answer = shop.ask_like_a_board_agent("Which tables are there?")

    assert "wholesale_orders: order_id integer, account_id integer, ordered_on date, kg numeric, " \
           "amount_gbp numeric, delivered_on date, paid_on date" in answer["schema"]


def _night_9_shop():
    """A small copy of the shop's subscription_orders and subscribers, in SQLite."""
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE subscription_orders (order_id INTEGER, plan_code TEXT, "
                          "shipped_on DATE, amount_gbp NUMERIC)"))
        conn.execute(text("CREATE TABLE subscribers (subscriber_id INTEGER, email TEXT, status TEXT)"))
        for i, (plan, shipped) in enumerate([("CLUB", "2026-09-08"), ("REGULAR", "2026-09-11"),
                                             ("TASTER", "2024-06-03"), ("CLUB", "2026-09-07")]):
            conn.execute(text("INSERT INTO subscription_orders VALUES (:i, :p, :s, 12.0)"),
                         {"i": i, "p": plan, "s": shipped})
        for i in range(14):
            conn.execute(text("INSERT INTO subscribers VALUES (:i, :e, :s)"),
                         {"i": i, "e": f"member{i}@example.com", "s": ("active", "paused", "cancelled")[i % 3]})
    return engine


def test_the_facts_are_read_from_the_owners_database():
    from modules.nl2sql.schema.grounding import probe_facts
    from modules.nl2sql.service import DatabaseKnowledgeService

    schema = {"tables": [
        {"name": "subscription_orders", "columns": [
            {"name": "order_id", "type": "integer"}, {"name": "plan_code", "type": "text"},
            {"name": "shipped_on", "type": "date"}, {"name": "amount_gbp", "type": "numeric"}]},
        {"name": "subscribers", "columns": [
            {"name": "email", "type": "text"}, {"name": "status", "type": "text"}]},
    ]}
    with _night_9_shop().connect() as conn:
        facts = probe_facts(conn, "sqlite", schema, DatabaseKnowledgeService._quote_ident)

    assert facts == {
        "subscription_orders.plan_code": {"values": ["CLUB", "REGULAR", "TASTER"]},
        "subscription_orders.shipped_on": {"range": ["2024-06-03", "2026-09-11"]},
        "subscribers.status": {"values": ["active", "cancelled", "paused"]},
    }  # 14 emails are no list to choose from; numbers and ids are not probed


def test_the_facts_are_read_once_per_source_and_kept(monkeypatch):
    from modules.nl2sql import agent_answer

    reads = []
    service = shop.use_shop(monkeypatch, {"success": True, "sql": "SELECT 1", "data": [], "row_count": 0}, facts=None)
    monkeypatch.setattr(service, "_decrypt_source_credentials", lambda source: {}, raising=False)
    monkeypatch.setattr(agent_answer, "read_facts",
                        lambda service, source, credentials: reads.append(source.id) or shop.SHOP_FACTS)

    first = shop.ask_like_a_board_agent("How many club members?")
    shop.ask_like_a_board_agent("How many regular members?")

    assert reads == [47]
    assert any("plan_code text (one of CLUB | REGULAR | TASTER)" in line for line in first["schema"])
