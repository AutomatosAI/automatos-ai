"""F302 (night 9) — the owner's own words reach the SQL writer through platform_query_data.

F077 (A) sends what the person typed to NL2SQL beside Auto's restatement ("not counting cancelled
ones" had become "active": 383 for 400). It rode the chat's caller context into
smart_query_database. Auto's chat now asks the database through platform_query_data, whose
handler builds a context with the user id alone, so the words stopped at the platform action.
While a platform action runs, the turn's words are held for it. Driven through the real handler
and resolver with only the source read and the NL2SQL query faked, as test_f077_query_data does.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import uuid4

from modules.nl2sql.service import DatabaseKnowledgeService
from modules.tools.discovery.handlers_scheduling import query_data
from modules.tools.execution import exec_platform
from modules.tools.execution.exec_research import owner_words
from modules.tools.execution.turn_owner_words import held_owner_words, owner_words_held

WS = uuid4()
TYPED = "How many Harvest Club members do we have, not counting cancelled ones?"
RESTATED = "How many active Harvest Club members are there?"


class _Service(DatabaseKnowledgeService):
    """The real resolver; the source read and the NL2SQL query are the fakes."""

    def __init__(self):
        self.queries = []

    async def active_sources(self, workspace_id, db_session=None):
        return [(36, "harbourline_shop")]

    async def query_database(self, **kwargs):
        self.queries.append(kwargs)
        return {"success": True, "sql": "SELECT count(*) AS members FROM subscribers", "data": [{"members": 65}],
                "columns": ["members"], "row_count": 1}

    async def write_nl_audit(self, **kwargs):
        return None


class _Db:
    def rollback(self):
        raise AssertionError("a successful query never rolls the turn's session back")


def _ask(monkeypatch):
    service = _Service()
    monkeypatch.setattr("modules.nl2sql.get_database_knowledge_service", lambda: service)
    result = asyncio.run(query_data(_Db(), WS, {"question": RESTATED, "_user_id": "7"}))
    assert result["success"] is True
    [call] = service.queries
    return call


def test_the_words_the_turn_holds_reach_the_sql_writer(monkeypatch):
    with owner_words_held({"user_query": TYPED, "user_id": "7"}):
        call = _ask(monkeypatch)
    assert call["owner_question"] == TYPED
    assert call["natural_language_query"] == RESTATED


def test_a_lane_nobody_typed_into_sends_no_words(monkeypatch):
    with owner_words_held({"user_id": "7"}):             # a heartbeat or a card's run: nothing typed
        call = _ask(monkeypatch)
    assert call["owner_question"] is None
    assert held_owner_words() is None and owner_words({}) is None


def test_a_platform_action_holds_its_turns_words_while_it_runs(monkeypatch):
    seen = []

    class _Executor:
        def __init__(self, db, workspace_id):
            pass

        async def execute(self, action, params, caller_context=None):
            seen.append(owner_words({"user_id": "7"}))
            return {"success": True}

    monkeypatch.setattr("modules.tools.discovery.platform_executor.PlatformActionExecutor", _Executor)
    result = asyncio.run(exec_platform.execute_platform_action(
        NS(db=None), "platform_query_data", {"question": RESTATED}, workspace_id=WS,
        caller_context={"user_query": TYPED, "user_id": "7"},
    ))
    assert result == {"success": True} and seen == [TYPED]
    assert held_owner_words() is None                     # released when the action returns
