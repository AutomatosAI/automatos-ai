"""F301 (night 9, query side, B1) — a date after the data is "not recorded yet", not 0.

"How many Harvest Club boxes go out on Monday 5 October?" The SQL writer wrote
``so.shipped_on = '2026-10-05'`` every time (audit rows 185, 202, 206, 335, 349) and the
answer was 0 — from the Analyst on #1891 and from Auto (chat 669bac98). In harbourline_shop
``subscription_orders`` holds boxes already sent: shipped_on runs to 2026-09-11 and
box_month to 2026-09-01; October has no rows yet. The right reading is the 63 active
Harvest Club members (subscribers: plan_code CLUB, status active), which the Business
Analyst gave on #1853.

Two things now carry that to whoever reads the answer:
* the SQL writer sees each date column's first and last recorded date in its schema
  (``grounding.grounded`` on the value-sampling step);
* an answer whose query compares a date column with a date after its last recorded
  value carries a note saying so.
"""
from __future__ import annotations

from types import SimpleNamespace

from tests import helpers_shop_database as shop

FIVE_OCTOBER = (
    "SELECT COUNT(so.order_id) AS harvest_club_boxes FROM subscription_orders AS so JOIN subscription_plans AS sp "
    "ON so.plan_code = sp.plan_code WHERE sp.name = 'Harvest Club' AND so.shipped_on = '2026-10-05' LIMIT 1000"
)


def _counted(sql, n):
    return {"success": True, "sql": sql, "columns": ["harvest_club_boxes"], "data": [{"harvest_club_boxes": n}],
            "row_count": 1}


def test_a_count_of_boxes_on_a_date_after_the_last_shipment_says_it_is_not_recorded_yet(monkeypatch):
    shop.use_shop(monkeypatch, _counted(FIVE_OCTOBER, 0), facts=shop.SHOP_FACTS)

    answer = shop.ask_like_a_board_agent("How many Harvest Club boxes go out on Monday 5 October 2026?")

    assert answer["notes"] == [
        "subscription_orders.shipped_on has nothing recorded after 2026-09-11, and this query asks about "
        "2026-10-05. Rows for that date do not exist yet, so 0 or no rows here means 'not recorded yet', "
        "not 'none'. Answer from what is recorded now (for example, who is active) and say that is what you did."
    ]


def test_a_date_inside_the_recorded_data_has_no_note(monkeypatch):
    inside = FIVE_OCTOBER.replace("so.shipped_on = '2026-10-05'",
                                  "so.shipped_on BETWEEN '2026-09-01' AND '2026-09-30'")
    shop.use_shop(monkeypatch, _counted(inside, 68), facts=shop.SHOP_FACTS)

    answer = shop.ask_like_a_board_agent("How many Harvest Club boxes went out in September 2026?")

    assert "notes" not in answer


def test_a_range_that_starts_after_the_data_is_called_out_too():
    from modules.nl2sql.agent_answer import future_date_notes

    sql = ("SELECT COUNT(*) FROM subscription_orders WHERE box_month >= DATE '2026-10-01' "
           "AND shipped_on BETWEEN '2026-10-05' AND '2026-10-09'")
    notes = future_date_notes(sql, shop.SHOP_SCHEMA, shop.SHOP_FACTS)

    assert [n.split(",")[0] for n in notes] == [
        "subscription_orders.box_month has nothing recorded after 2026-09-01",
        "subscription_orders.shipped_on has nothing recorded after 2026-09-11",
    ]


def test_the_sql_writer_sees_every_value_and_the_last_recorded_date():
    """The writer's prompt printed three sample values per column and no dates. Each
    column now says what it holds."""
    from modules.nl2sql.query.nl2sql_service import NaturalLanguageToSQLService
    from modules.nl2sql.schema import grounding
    from modules.nl2sql.service import DatabaseKnowledgeService

    grounding.remember_facts("47", shop.SHOP_FACTS)
    try:
        schema = {"tables": [dict(t, columns=[dict(c) for c in t["columns"]]) for t in shop.SHOP_SCHEMA["tables"]]}
        service = DatabaseKnowledgeService.__new__(DatabaseKnowledgeService)
        # An unsupported dialect: the value sampling itself returns before opening anything.
        service._augment_schema_with_samples(SimpleNamespace(id=47, dialect="sqlite"), {}, schema)
        prompt = NaturalLanguageToSQLService(llm_provider=None)._build_prompt(
            question="How many Harvest Club boxes go out on Monday 5 October?", schema_metadata=schema,
            semantic_layer=None, dialect="postgresql", examples=None)
    finally:
        grounding._FACTS.pop("47", None)

    assert "  - plan_code (text) -- one of: CLUB, REGULAR, TASTER (every value it holds)" in prompt
    assert ("  - shipped_on (date) -- recorded from 2024-06-03 to 2026-09-11; "
            "nothing is recorded after 2026-09-11 yet") in prompt
    assert "  - name (text) -- one of: Harvest Club, Regular, Taster (every value it holds)" in prompt
