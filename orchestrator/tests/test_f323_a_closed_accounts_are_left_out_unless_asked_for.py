"""F323 (night 9b) — a closed account is left out unless the question asks for it.

"Which cafés are on 14-day terms?" came back 4 from the Business Analyst and 6 from the
Ops Manager, the Analyst and Auto all night, and nobody said why. harbourline_shop has six
wholesale_accounts with ``payment_terms_days = 14``; two (Lantern Bakehouse, Pier
Bakehouse) have ``status = 'closed'``. The SQL writer filtered on status only when the
question said "currently" or "active" (audit rows 671, 690, 696, 697 → 4); otherwise not
(443, 679, 691, 704 → 6). The BA also took the closed "Quay Bakehouse" for Quay Coffee
House (#1970).

The writer now reads, for each table with a column that marks ended rows, which rows are
live and that ended ones are left out unless asked for; the agent's schema says the same;
and an answer whose query never looked at that column says ended rows are in it.
"""
from __future__ import annotations

from types import SimpleNamespace

from tests import helpers_shop_database as shop

ACCOUNTS = {
    "tables": [
        {"name": "wholesale_accounts", "columns": [
            {"name": "account_id", "type": "integer"}, {"name": "cafe_name", "type": "text"},
            {"name": "payment_terms_days", "type": "integer"}, {"name": "status", "type": "text"}]},
        {"name": "wholesale_orders", "columns": [
            {"name": "order_id", "type": "integer"}, {"name": "account_id", "type": "integer"},
            {"name": "kg", "type": "numeric"}]},
    ],
    "relationships": [{"from_table": "wholesale_orders", "from_column": "account_id",
                       "to_table": "wholesale_accounts", "to_column": "account_id"}],
}
FACTS = {"wholesale_accounts.status": {"values": ["active", "closed"]}}
ALL_SIX = "SELECT cafe_name FROM wholesale_accounts WHERE payment_terms_days = 14 LIMIT 1000"  # audit row 443
LIVE_FOUR = ALL_SIX.replace("= 14", "= 14 AND status = 'active'")                              # audit row 696
CAFES = ["Gull Espresso Bar", "Lantern Bakehouse", "Marsh Espresso Bar", "Pier Bakehouse",
         "Quay Coffee House", "Tide Espresso Bar"]


def _answer(sql, names):
    return {"success": True, "sql": sql, "columns": ["cafe_name"], "data": [{"cafe_name": n} for n in names],
            "row_count": len(names)}


def test_the_sql_writer_is_told_closed_accounts_are_left_out_unless_asked():
    from modules.nl2sql.query.nl2sql_service import NaturalLanguageToSQLService
    from modules.nl2sql.schema import grounding
    from modules.nl2sql.service import DatabaseKnowledgeService

    grounding.remember_facts("47", FACTS)
    try:
        schema = {**ACCOUNTS, "tables": [dict(t, columns=[dict(c) for c in t["columns"]]) for t in ACCOUNTS["tables"]]}
        service = DatabaseKnowledgeService.__new__(DatabaseKnowledgeService)
        service._augment_schema_with_samples(SimpleNamespace(id=47, dialect="sqlite"), {}, schema)
        prompt = NaturalLanguageToSQLService(llm_provider=None)._build_prompt(
            question="Which cafés are on 14-day payment terms?", schema_metadata=schema,
            semantic_layer=None, dialect="postgresql", examples=None)
    finally:
        grounding._FACTS.pop("47", None)

    assert ("Table: wholesale_accounts\nDescription: LIVE ROWS: status = 'active'. status = 'closed': "
            "leave those rows out unless the question asks about them.") in prompt


def test_the_agents_schema_says_which_accounts_are_live(monkeypatch):
    shop.use_shop(monkeypatch, _answer(LIVE_FOUR, [c for c in CAFES if "Bakehouse" not in c]),
                  schema=ACCOUNTS, facts=FACTS)

    answer = shop.ask_like_a_board_agent("Which cafés are on 14-day terms?")

    assert answer["schema"][0] == (
        "wholesale_accounts: account_id integer, cafe_name text, payment_terms_days integer, "
        "status text (one of active | closed) — live rows: status = 'active' (status = 'closed' left out unless asked)"
    )
    assert "notes" not in answer  # the query looked at status: nothing to add


def test_a_count_that_let_closed_accounts_in_says_so(monkeypatch):
    shop.use_shop(monkeypatch, _answer(ALL_SIX, CAFES), schema=ACCOUNTS, facts=FACTS)

    answer = shop.ask_like_a_board_agent("Which cafés are on 14-day terms?")

    assert answer["notes"] == [
        "wholesale_accounts has rows that have ended (status = 'closed'); this query does not look at status, "
        "so they are included. If the question is about live ones only, those are status = 'active'."
    ]


def test_a_member_on_hold_is_neither_live_nor_ended():
    """Subscribers: cancelled ones have ended; a paused one is on hold (F301 B1 counted 63
    active Harvest Club members, not the 2 paused)."""
    from modules.nl2sql.schema.live_rows import writer_rule

    subscribers = next(t for t in shop.SHOP_SCHEMA["tables"] if t["name"] == "subscribers")

    assert writer_rule(subscribers, shop.SHOP_FACTS) == (
        "LIVE ROWS: status = 'active' and cancelled_on IS NULL. status = 'cancelled'; cancelled_on set: "
        "leave those rows out unless the question asks about them. status = 'paused' is on hold, not ended: "
        "count it only when the question would."
    )


def test_any_table_by_its_values_or_its_names():
    from modules.nl2sql.schema.live_rows import live_rules

    table = {"name": "contracts", "columns": [
        {"name": "state", "type": "character varying"}, {"name": "closed_on", "type": "date"},
        {"name": "is_active", "type": "boolean"}, {"name": "is_deleted", "type": "boolean"},
        {"name": "payment", "type": "text"}, {"name": "shipped_late", "type": "boolean"},
        {"name": "paid_on", "type": "date"}]}
    facts = {"contracts.state": {"values": ["open", "suspended", "terminated"]},
             "contracts.payment": {"values": ["paid", "unpaid"]}}

    assert [tuple(r) for r in live_rules(table, facts)] == [
        ("state", "state = 'open'", "state = 'terminated'", "state = 'suspended'"),
        ("closed_on", "closed_on IS NULL", "closed_on set", ""),
        ("is_active", "is_active = TRUE", "is_active = FALSE", ""),
        ("is_deleted", "is_deleted = FALSE", "is_deleted = TRUE", ""),
    ]  # paid/unpaid, shipped_late and paid_on say nothing about a row having ended
