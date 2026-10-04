"""F329 (C): neither data tool can change the owner's database or the graph.

Three locks, each checked here on what it actually does:
* the tools reach only read-permission platform actions;
* a session sends a question, never SQL: no field it sends can carry a statement;
* the SQL the database question becomes must pass ``SQLValidator``, which refuses
  every write (PRD-160 S2) before it reaches the owner's database.
"""
from __future__ import annotations

import pytest

from modules.nl2sql.query.validator import SQLValidationError, SQLValidator
from services import session_tools as st

CTX = st.SessionContext(task_id=1995, agent_id=268, agent_name="Business Analyst", workspace_id="ws-shop")
DATA_TOOLS = ("query_database", "query_graph")
SHOP_SCHEMA = {"tables": [{"name": "orders", "columns": [{"name": "id"}, {"name": "status"}]}]}


def _actions(tool):
    return (tool.action,) + tuple(a for a in tool.also_runs if a.startswith("platform_"))


def test_every_action_they_reach_only_reads():
    from modules.tools.discovery import get_action_registry

    registry = get_action_registry()
    for name in DATA_TOOLS:
        tool = st.get_tool(name)
        assert tool.reads_only is True, name
        for action in _actions(tool):
            definition = registry.get(action)
            assert definition is not None, action
            assert definition.permission_level == "read", (name, action, definition.permission_level)


def test_the_graph_tool_can_only_ever_name_its_three_read_actions():
    tool = st.get_tool("query_graph")
    reached = set()
    for arguments in ({"question": "q"}, {"concept": "c"}, {"from": "a", "to": "b"},
                      {"concept": "c", "action": "platform_delete_document"}):
        reached.add(st.resolve_parameters(tool, arguments, CTX)["action"])
    assert reached == set(_actions(tool))


def test_sql_sent_by_a_session_never_reaches_the_action():
    tool = st.get_tool("query_database")
    scoped = st.resolve_parameters(tool, {"question": "how many orders?", "sql": "DELETE FROM orders",
                                          "statement": "DROP TABLE orders"}, CTX)
    assert scoped == {"question": "how many orders?"}
    from modules.tools.discovery import get_action_registry

    takes = set(get_action_registry().get("platform_query_data").parameters["properties"])
    assert takes == {"question", "database_id"}          # the action itself has no field for SQL


@pytest.mark.parametrize("statement", [
    "DELETE FROM orders",
    "UPDATE orders SET status = 'cancelled'",
    "DROP TABLE orders",
    "INSERT INTO orders (id) VALUES (1)",
    "SELECT 1; DELETE FROM orders",
])
def test_the_query_a_question_becomes_is_refused_if_it_writes(statement):
    with pytest.raises(SQLValidationError):
        SQLValidator().validate_and_rewrite(sql=statement, schema_metadata=SHOP_SCHEMA)


def test_a_read_still_passes_with_a_limit():
    safe, _warnings = SQLValidator().validate_and_rewrite(
        sql="SELECT status, COUNT(*) FROM orders GROUP BY status", schema_metadata=SHOP_SCHEMA)
    assert "LIMIT" in safe.upper()
