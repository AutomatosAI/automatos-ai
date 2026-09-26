"""F077 (C) — a question about numbers goes to the database, and a refused call is Auto's to fix.

Retest of refresh 4, question #21 ("What is my monthly subscription revenue at
today's prices, from active subscribers?", expected £6323.00). Auto never called
smart_query_database. It called platform_query_graph, whose description said the
graph holds "metrics" and gave "what metrics track growth?" as an example, and it
left out the required ``question``. The refusal showed the exact call. Auto then
asked the owner to "confirm the exact question you'd like me to ask the knowledge
graph".

Now the graph's description says it holds no live figures and sends counts,
money, totals, averages and rankings to smart_query_database. The refusal says
the call is Auto's to make, never a question for the user.
"""
from __future__ import annotations


def _graph():
    from modules.tools.discovery import get_action_registry

    return get_action_registry().get("platform_query_graph")


def test_the_graph_says_it_holds_no_figures_and_where_they_are():
    graph = _graph()
    assert "It holds no live figures" in graph.description
    assert "counts, money, totals, averages and rankings" in graph.description
    assert "smart_query_database" in graph.description
    assert "what metrics track growth" not in graph.parameters["properties"]["question"]["description"]


def test_a_refused_call_is_autos_to_fix_never_the_users():
    """#21's own call: platform_query_graph with no question."""
    from modules.tools.execution.unified_executor import missing_params_error

    graph = _graph()
    error = missing_params_error("platform_query_graph", graph.parameters, ["question"], {})
    assert "a refused call is never a question for the user" in error
    lines = error.splitlines()
    assert lines[1].startswith('Call it exactly like this: {"action": "platform_query_graph"')  # the call first
    assert lines[2].startswith("Make this call yourself now")


def test_a_value_only_the_user_can_give_is_still_theirs_to_give():
    """Review LOW: the line must not push Auto to invent a value it was never given.
    A document id the user never named is theirs to give, never made up."""
    from modules.tools.discovery import get_action_registry
    from modules.tools.execution.unified_executor import missing_params_error

    read = get_action_registry().get("platform_read_document")
    error = missing_params_error("platform_read_document", read.parameters, ["document_id"], {})
    assert "Ask the user only for a value only they can give, and never invent one" in error
    assert "Make this call yourself now with the values you have." in error
