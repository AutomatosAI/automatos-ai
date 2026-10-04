"""F329 (D): what a data tool hands back is the action's answer, readable and whole.

* Everything the action returns travels: the rows, the SQL, and whatever context
  other fixes add beside them (the schema digest and notes of F300/F301, the edge
  direction of F312), under the action's own key names.
* Money, kilos and dates (``Decimal``, ``date``) arrive as plain text (F299's shape).
* An answer longer than the bridge's cap loses whole rows from its longest list,
  with a note saying so, instead of being cut mid-row by the bridge.
* A failed question keeps its context: the schema shows under the error, so the
  session can ask again with the real names.
"""
from __future__ import annotations

import asyncio
import datetime
import json
from decimal import Decimal

from services import session_data_tools as sdt
from services import session_tools as st
from services import session_tools_rpc as rpc

CTX = st.SessionContext(task_id=1995, agent_id=268, agent_name="Business Analyst", workspace_id="ws-shop")
SCHEMA = ["orders(id integer, status text (one of active | cancelled), placed_on date (2026-01-02 to 2026-10-03))"]


def _rendered(name, result):
    """The text the session reads for ``result`` from tool ``name``."""
    projected = st._projected(st.get_tool(name), result)
    return rpc.render_result(projected)


def test_the_rows_and_the_actions_own_context_arrive_as_plain_json():
    answer = {
        "success": True, "answer": "status | n\nactive | 383", "sql": "SELECT status, COUNT(*) …",
        "row_count": 1, "columns": ["status", "revenue", "placed_on"],
        "data": [{"status": "active", "revenue": Decimal("1234.50"), "placed_on": datetime.date(2026, 10, 3)}],
        "schema": SCHEMA, "notes": ["0 here means not recorded yet"],
    }
    out = _rendered("query_database", answer)
    assert out["isError"] is False
    body = json.loads(out["content"][0]["text"])
    assert body["data"] == [{"status": "active", "revenue": "1234.50", "placed_on": "2026-10-03"}]
    assert body["schema"] == SCHEMA and body["notes"] == ["0 here means not recorded yet"]
    assert body["sql"].startswith("SELECT") and "cut" not in body


def test_a_long_answer_loses_whole_rows_with_a_note_not_its_tail():
    rows = [{"order_id": i, "customer": f"customer-{i:05d}", "note": "x" * 60} for i in range(3000)]
    out = _rendered("query_database", {"success": True, "answer": "…", "row_count": 3000,
                                       "columns": ["order_id", "customer", "note"], "data": rows,
                                       "schema": SCHEMA})
    text = out["content"][0]["text"]
    assert len(text) <= st.MAX_TOOL_RESULT_CHARS and not text.endswith("(truncated)")
    body = json.loads(text)                                   # still one whole JSON answer
    assert 0 < len(body["data"]) < 3000
    assert body["data"] == rows[: len(body["data"])]          # whole rows, in order
    assert body["schema"] == SCHEMA and body["row_count"] == 3000
    assert "'data'" in body["cut"] and "narrower question" in body["cut"]


def test_a_failed_question_shows_the_schema_under_the_error():
    failed = {"success": False, "error": "column orders.state does not exist; the columns are id, status",
              "sql": "SELECT state FROM orders", "schema": SCHEMA}
    out = _rendered("query_database", failed)
    text = out["content"][0]["text"]
    assert out["isError"] is True
    assert text.startswith("column orders.state does not exist")
    assert "status text (one of active | cancelled)" in text and "SELECT state FROM orders" in text


def test_a_failure_with_nothing_beside_it_is_unchanged():
    out = _rendered("query_graph", {"success": False, "error": "No knowledge graph built for this workspace yet."})
    assert out["isError"] is True
    assert out["content"][0]["text"] == "No knowledge graph built for this workspace yet."


def test_graph_links_keep_their_direction_and_fit():
    link = {"neighbor": "Supplier A", "relation": "supplies", "direction": "incoming",
            "reads": "Supplier A supplies House Blend", "neighbor_attrs": {"summary": "y" * 300}}
    neighbours = [{**link, "neighbor": f"Supplier {i}"} for i in range(400)]
    out = _rendered("query_graph", {"success": True, "node": "House Blend", "neighbor_count": 400,
                                    "neighbors": neighbours})
    body = json.loads(out["content"][0]["text"])
    assert body["neighbor_count"] == 400 and 0 < len(body["neighbors"]) < 400
    assert body["neighbors"] == neighbours[: len(body["neighbors"])]   # whole links, F312's keys and all
    assert body["neighbors"][0]["direction"] == "incoming"
    assert "'neighbors'" in body["cut"]


def test_through_the_runner_the_graph_answer_is_fitted_too(monkeypatch):
    from modules.tools.execution import unified_executor

    class _Executor:
        def __init__(self, db):
            pass

        async def execute_tool(self, **kwargs):
            return {"success": True, "neighbors": [{"neighbor": str(i), "pad": "z" * 200} for i in range(1000)]}

    monkeypatch.setattr(unified_executor, "UnifiedToolExecutor", _Executor)
    tool = st.get_tool("query_graph")
    result = asyncio.run(st.call_tool(None, tool, st.resolve_parameters(tool, {"concept": "c"}, CTX), CTX))
    assert "cut" in result and 0 < len(result["neighbors"]) < 1000
    assert len(json.dumps(result, indent=2)) <= st.MAX_TOOL_RESULT_CHARS - sdt.RESULT_FRAME_CHARS + len(result["cut"]) + 32
