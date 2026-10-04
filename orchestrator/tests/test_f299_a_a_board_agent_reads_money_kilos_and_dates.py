"""F299 (night 9) — a board agent reads the shop's money, kilos and dates.

On 4 Oct the board agents' ``smart_query_database`` answers failed with "Object of type
Decimal is not JSON serializable" (#1858 twice, #1856, #1869, #1879, #1880, #1881; the
backend log holds 72 Decimal and 18 date failures). The query ran; the agent lane then
wrote the answer with a bare ``json.dumps(raw)`` (agent_factory's tool callback) and the
``Decimal`` amounts, kilos and ``date`` values in the rows raised. September's takings
(B5) and the top cafés by kilos (B3) were never answered by a board agent, while Auto's
chat, which writes rows with ``json.dumps(..., default=str)``, read the same tables.

These tests send the night's rows through the real tool path and serialise the answer
exactly as the agent lane does.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import UUID

from tests import helpers_shop_database as shop

TAKINGS = {
    "success": True,
    "sql": "SELECT channel, SUM(amount_gbp) AS takings FROM retail_orders "
           "WHERE ordered_on BETWEEN '2026-09-01' AND '2026-09-30' GROUP BY channel LIMIT 1000",
    "columns": ["channel", "takings"],
    "data": [{"channel": "market stall", "takings": Decimal("936.00")},
             {"channel": "online shop", "takings": Decimal("519.00")}],
    "row_count": 2,
}
TOP_CAFES = {
    "success": True,
    "sql": "SELECT wa.cafe_name, SUM(wo.kg) AS kg FROM wholesale_orders wo JOIN wholesale_accounts wa "
           "ON wa.account_id = wo.account_id GROUP BY wa.cafe_name ORDER BY kg DESC LIMIT 3",
    "columns": ["cafe_name", "kg", "last_order", "logged_at", "ref"],
    "data": [{"cafe_name": "Crane Café", "kg": Decimal("214.0"), "last_order": date(2026, 8, 27),
              "logged_at": datetime(2026, 8, 27, 9, 30, tzinfo=timezone.utc),
              "ref": UUID("febae41b-374b-4580-a5ef-f698bdd382e4")}],
    "row_count": 1,
}


def test_september_takings_reach_the_agent_as_plain_json(monkeypatch):
    shop.use_shop(monkeypatch, TAKINGS)

    answer = shop.ask_like_a_board_agent("What did the shop take in retail orders in September 2026?")

    as_the_agent_reads_it = json.dumps(answer)  # agent_factory's tool callback, unchanged
    assert answer["success"] is True
    assert '"takings": "936.00"' in as_the_agent_reads_it
    assert '"takings": "519.00"' in as_the_agent_reads_it


def test_kilos_dates_times_and_ids_reach_the_agent_written_as_autos_chat_writes_them(monkeypatch):
    shop.use_shop(monkeypatch, TOP_CAFES)

    answer = shop.ask_like_a_board_agent("Which three cafés ordered the most coffee by weight over June to August?")

    row = json.loads(json.dumps(answer))["data"][0]
    assert row == json.loads(json.dumps(TOP_CAFES["data"][0], default=str))  # Auto's chat form
    assert row["kg"] == "214.0" and row["last_order"] == "2026-08-27"
    assert row["logged_at"] == "2026-08-27 09:30:00+00:00"
    assert row["ref"] == "febae41b-374b-4580-a5ef-f698bdd382e4"


def test_platform_query_data_through_platform_execute_reads_them_too(monkeypatch):
    """28 of the night's Decimal failures were ``platform_execute`` → platform_query_data."""
    from modules.tools.discovery.handlers_scheduling import query_data

    shop.use_shop(monkeypatch, TAKINGS)

    answer = asyncio.run(query_data(db=None, workspace_id=shop.WORKSPACE,
                                    params={"question": "September retail takings by channel", "_agent_id": 343}))

    as_the_agent_reads_it = json.dumps(answer)
    assert answer["success"] is True and "market stall | 936.00" in answer["answer"]
    assert '"takings": "936.00"' in as_the_agent_reads_it
