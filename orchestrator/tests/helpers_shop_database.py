"""The night-9 shop database (harbourline_shop, source 47) as the F299–F301 tests see it.

Table and column names, types, value sets and date ranges are the ones the platform
stored and read for source 47 on 4 Oct 2026 (``database_knowledge_sources.schema_metadata``
and read-only SELECTs on harbourline_shop). Only the database's answer is faked: the
tool path from the agent's call to the JSON the agent reads is the real one.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

SOURCE_ID = "47"
WORKSPACE = "febae41b-374b-4580-a5ef-f698bdd382e4"

SHOP_SCHEMA: Dict[str, Any] = {
    "tables": [
        {"name": "retail_orders", "columns": [
            {"name": "order_id", "type": "integer"}, {"name": "ordered_on", "type": "date"},
            {"name": "channel", "type": "text"}, {"name": "amount_gbp", "type": "numeric"}]},
        {"name": "subscription_orders", "columns": [
            {"name": "order_id", "type": "integer"}, {"name": "subscriber_id", "type": "integer"},
            {"name": "box_month", "type": "date"}, {"name": "plan_code", "type": "text"},
            {"name": "amount_gbp", "type": "numeric"}, {"name": "shipped_on", "type": "date"},
            {"name": "shipped_late", "type": "boolean"}]},
        {"name": "subscription_plans", "columns": [
            {"name": "plan_code", "type": "text"}, {"name": "name", "type": "text"}]},
        {"name": "wholesale_accounts", "columns": [
            {"name": "account_id", "type": "integer"}, {"name": "cafe_name", "type": "text"}]},
        {"name": "wholesale_orders", "columns": [
            {"name": "order_id", "type": "integer"}, {"name": "account_id", "type": "integer"},
            {"name": "ordered_on", "type": "date"}, {"name": "kg", "type": "numeric"},
            {"name": "amount_gbp", "type": "numeric"}, {"name": "delivered_on", "type": "date"},
            {"name": "paid_on", "type": "date"}]},
        {"name": "subscribers", "columns": [
            {"name": "subscriber_id", "type": "integer"}, {"name": "plan_code", "type": "text"},
            {"name": "status", "type": "text"}, {"name": "started_on", "type": "date"},
            {"name": "cancelled_on", "type": "date"}, {"name": "paused_until", "type": "date"}]},
    ],
    "relationships": [
        {"from_table": "subscription_orders", "from_column": "plan_code",
         "to_table": "subscription_plans", "to_column": "plan_code"},
        {"from_table": "wholesale_orders", "from_column": "account_id",
         "to_table": "wholesale_accounts", "to_column": "account_id"},
        {"from_table": "subscribers", "from_column": "plan_code",
         "to_table": "subscription_plans", "to_column": "plan_code"},
        {"from_table": "subscription_orders", "from_column": "subscriber_id",
         "to_table": "subscribers", "to_column": "subscriber_id"},
    ],
}

SHOP_FACTS: Dict[str, Dict[str, List[str]]] = {
    "retail_orders.channel": {"values": ["market stall", "online shop"]},
    "retail_orders.ordered_on": {"range": ["2024-06-01", "2026-09-30"]},
    "subscription_orders.box_month": {"range": ["2024-06-01", "2026-09-01"]},
    "subscription_orders.plan_code": {"values": ["CLUB", "REGULAR", "TASTER"]},
    "subscription_orders.shipped_on": {"range": ["2024-06-03", "2026-09-11"]},
    "subscription_orders.shipped_late": {"values": ["False", "True"]},
    "subscription_plans.plan_code": {"values": ["CLUB", "REGULAR", "TASTER"]},
    "subscription_plans.name": {"values": ["Harvest Club", "Regular", "Taster"]},
    "wholesale_orders.ordered_on": {"range": ["2024-06-02", "2026-09-30"]},
    "subscribers.plan_code": {"values": ["CLUB", "REGULAR", "TASTER"]},
    "subscribers.status": {"values": ["active", "cancelled", "paused"]},
    "subscribers.started_on": {"range": ["2024-04-05", "2026-08-31"]},
    "subscribers.cancelled_on": {"range": ["2025-02-27", "2026-09-20"]},
    "subscribers.paused_until": {"range": ["2026-10-05", "2026-11-05"]},
}


class FakeShopService:
    """The workspace's one source resolves; the query answer is what the test gives.
    A list of answers is given out in turn (the last one repeats)."""

    def __init__(self, answer: Any):
        self.answers = list(answer) if isinstance(answer, list) else [answer]
        self.asked: List[Dict[str, Any]] = []

    async def resolve_source_id(self, workspace_id, database_name=None, db_session=None):
        return SOURCE_ID

    def _next(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        self.asked.append(kwargs)
        return self.answers[min(len(self.asked), len(self.answers)) - 1]

    async def smart_query(self, **kwargs):
        return self._next(kwargs)

    async def query_database(self, **kwargs):
        return self._next(kwargs)

    async def write_nl_audit(self, **kwargs):
        return None


def use_shop(monkeypatch, answer: Any, *, schema: Optional[Dict[str, Any]] = SHOP_SCHEMA,
             facts: Optional[Dict[str, Any]] = None) -> FakeShopService:
    """Wire the fake service, the source row (``schema``; None = not readable) and the
    facts already read for source 47 (``facts``; None = none read)."""
    from modules.nl2sql import agent_answer
    from modules.nl2sql.schema import grounding

    service = FakeShopService(answer)
    monkeypatch.setattr("modules.nl2sql.get_database_knowledge_service", lambda: service)
    monkeypatch.setattr(grounding, "_FACTS", {})
    if facts is not None:
        grounding.remember_facts(SOURCE_ID, facts)
    row = SimpleNamespace(id=int(SOURCE_ID), dialect="postgresql", schema_metadata=schema) if schema else None
    monkeypatch.setattr(agent_answer, "load_source", lambda source_id, workspace_id: row)
    return service


def ask_like_a_board_agent(question: str, caller_context: Optional[Dict[str, Any]] = None,
                           db: Any = None) -> Dict[str, Any]:
    """``smart_query_database`` as a board agent's run calls it."""
    from modules.tools.execution import exec_research

    return asyncio.run(exec_research.execute_smart_database_tool(
        executor=SimpleNamespace(db=db), tool_name="smart_query_database",
        parameters={"query": question}, agent_id=343, workspace_id=WORKSPACE,
        caller_context=caller_context or {"board_task_id": None},
    ))
