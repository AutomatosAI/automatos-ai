"""F301 B1 (build 14 retest) — a date past the data never comes back as a count of 0.

#1895 (Analyst, build 14): "How many Harvest Club boxes go out on Monday 5 October?"
Its one query (audit row 350) counted ``subscription_orders`` rows with ``shipped_on =
'2026-10-05'``; shipped_on ends at 2026-09-11. The tool answered one row,
``harvest_club_boxes_scheduled: 0``, plus a note that October is not recorded yet. The
card led with "There are 0 Harvest Club boxes scheduled …". Right: 63 active CLUB
subscribers.

Now the tool asks the SQL writer once more — the date is past the data, count from the
state that decides it, here are the tables that hold such state — and returns that
answer with a line saying how it was worked out. If the second answer is no better,
the agent gets no rows and no count, only what to answer from. A date inside the data
whose answer really is 0 stays 0.
"""
from __future__ import annotations

import asyncio
import json

from tests import helpers_shop_database as shop

PAST_THE_DATA = (
    "SELECT COUNT(so.order_id) AS harvest_club_boxes_scheduled FROM subscription_orders AS so "
    "JOIN subscription_plans AS sp ON so.plan_code = sp.plan_code "
    "WHERE sp.name = 'Harvest Club' AND so.shipped_on = '2026-10-05' LIMIT 1000"
)
FROM_THE_STATE = (
    "SELECT COUNT(s.subscriber_id) AS harvest_club_boxes FROM subscribers AS s JOIN subscription_plans AS sp "
    "ON s.plan_code = sp.plan_code WHERE sp.name = 'Harvest Club' AND s.status = 'active' LIMIT 1000"
)
QUESTION = "How many Harvest Club boxes are scheduled to go out on Monday, October 5th, 2026?"


def _counted(sql, column, n):
    return {"success": True, "sql": sql, "columns": [column], "data": [{column: n}], "row_count": 1}


ZERO = _counted(PAST_THE_DATA, "harvest_club_boxes_scheduled", 0)


def test_the_writer_is_asked_again_from_the_state_that_decides_it(monkeypatch):
    service = shop.use_shop(monkeypatch, [ZERO, _counted(FROM_THE_STATE, "harvest_club_boxes", 63)],
                            facts=shop.SHOP_FACTS)

    answer = shop.ask_like_a_board_agent(QUESTION)

    assert answer["data"] == [{"harvest_club_boxes": 63}] and answer["sql"] == FROM_THE_STATE
    assert answer["derived"].startswith(
        "How this was worked out: subscription_orders.shipped_on has nothing recorded after 2026-09-11, "
        "and the question is about 2026-10-05, so counting rows for that date would give 0 only because "
        "it has not happened yet.")
    again = service.asked[1]["text"]
    assert again.startswith(QUESTION + "\n\nsubscription_orders.shipped_on has nothing recorded after 2026-09-11")
    assert "do not filter on that date" in again and "keeping the question's other filters" in again
    assert "subscribers (plan_code: CLUB | REGULAR | TASTER; status: active | cancelled | paused)" in again
    assert "subscription_orders (" not in again  # the table whose dates ran out is no candidate


def test_the_owners_words_carry_the_redirect_too(monkeypatch):
    service = shop.use_shop(monkeypatch, [ZERO, _counted(FROM_THE_STATE, "harvest_club_boxes", 63)],
                            facts=shop.SHOP_FACTS)

    shop.ask_like_a_board_agent(QUESTION, caller_context={"user_query": "How many club boxes go out on Monday?"})

    first, second = service.asked
    assert first["owner_question"] == "How many club boxes go out on Monday?"
    assert second["owner_question"].startswith("How many club boxes go out on Monday?\n\nsubscription_orders.shipped_on")


def test_when_the_second_answer_is_no_better_there_is_no_count_to_lead_with(monkeypatch):
    shop.use_shop(monkeypatch, [ZERO, {"success": False, "sql": None, "error": "Execution error", "data": []}],
                  facts=shop.SHOP_FACTS)

    answer = shop.ask_like_a_board_agent(QUESTION)

    written = json.dumps(answer)
    assert answer["not_recorded"] is True and answer["data"] == [] and "row_count" not in answer
    assert '"harvest_club_boxes_scheduled": 0' not in written  # the 0 row is gone, not just explained
    assert answer["answer"].startswith(
        "Not recorded yet, so there is no count to give: subscription_orders.shipped_on has nothing "
        "recorded after 2026-09-11, and the question is about 2026-10-05. Do not answer 0.")
    assert "subscribers (plan_code: CLUB | REGULAR | TASTER; status: active | cancelled | paused)" in answer["answer"]


def test_a_second_answer_past_the_data_too_is_withheld_as_well(monkeypatch):
    service = shop.use_shop(monkeypatch, [ZERO, ZERO], facts=shop.SHOP_FACTS)

    answer = shop.ask_like_a_board_agent(QUESTION)

    assert len(service.asked) == 2 and answer["not_recorded"] is True and answer["data"] == []


def test_a_date_inside_the_data_whose_answer_is_0_is_still_0(monkeypatch):
    inside = PAST_THE_DATA.replace("2026-10-05", "2026-09-06")  # a Sunday: nothing shipped
    service = shop.use_shop(monkeypatch, _counted(inside, "harvest_club_boxes_scheduled", 0),
                            facts=shop.SHOP_FACTS)

    answer = shop.ask_like_a_board_agent("How many Harvest Club boxes went out on Sunday 6 September?")

    assert len(service.asked) == 1
    assert answer["data"] == [{"harvest_club_boxes_scheduled": 0}] and answer["row_count"] == 1
    assert "not_recorded" not in answer and "derived" not in answer and "notes" not in answer


def test_rows_with_something_in_them_are_an_answer_even_past_the_data(monkeypatch):
    """A non-zero value means the data does reach that far for some rows: never re-asked."""
    service = shop.use_shop(monkeypatch, _counted(PAST_THE_DATA, "harvest_club_boxes_scheduled", 2),
                            facts=shop.SHOP_FACTS)

    answer = shop.ask_like_a_board_agent(QUESTION)

    assert len(service.asked) == 1 and answer["data"] == [{"harvest_club_boxes_scheduled": 2}]


def test_platform_query_data_gives_no_count_either(monkeypatch):
    from modules.tools.discovery.handlers_scheduling import query_data

    shop.use_shop(monkeypatch, [ZERO, ZERO], facts=shop.SHOP_FACTS)

    answer = asyncio.run(query_data(db=None, workspace_id=shop.WORKSPACE, params={"question": QUESTION}))

    assert answer["row_count"] is None and answer["data"] == [] and answer["not_recorded"] is True
    assert answer["answer"].startswith("Not recorded yet, so there is no count to give")


def test_any_table_and_any_date_column_not_just_this_shop():
    from modules.nl2sql import not_recorded

    schema = {
        "tables": [
            {"name": "bookings", "columns": [{"name": "room_id", "type": "integer"},
                                             {"name": "night_of", "type": "timestamp without time zone"}]},
            {"name": "rooms", "columns": [{"name": "room_id", "type": "integer"},
                                          {"name": "state", "type": "character varying"}]},
        ],
        "relationships": [{"from_table": "bookings", "from_column": "room_id",
                           "to_table": "rooms", "to_column": "room_id"}],
    }
    facts = {"bookings.night_of": {"range": ["2025-01-01 00:00:00", "2026-03-31 00:00:00"]},
             "rooms.state": {"values": ["free", "held", "let"]}}
    result = {"success": True, "sql": "SELECT COUNT(*) FROM bookings WHERE night_of >= '2026-12-24'",
              "data": [], "row_count": 0}

    past = not_recorded.past_the_data(result, schema, facts)

    assert [(p.table, p.column, p.last, p.asked) for p in past] == [("bookings", "night_of", "2026-03-31", "2026-12-24")]
    assert "rooms (state: free | held | let)" in not_recorded.redirect_instruction(past, result["sql"], schema, facts)
