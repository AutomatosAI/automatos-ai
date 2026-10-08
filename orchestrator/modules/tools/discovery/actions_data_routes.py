"""The two data routes Auto holds first-class: platform_query_data and platform_query_graph.

Night 9 (F301, F302, F312): Auto now holds both on every chat turn in a workspace with a
database and a Knowledge Graph (modules/tools/data_routes.py), so their descriptions are what
the model reads, and they now say how each is asked:

* platform_query_data: the business's own records. It returns the database's schema (FIXER's
  F300), so the owner is never asked for table, column or schema names (Auto asked "the exact
  names of the fields?" twice, L69, L101). One figure per call, in the owner's words: "how many
  cancelled, and the most common reason?" in one call came back by reason, and the top
  reason's 4 was read as the total (11).
* platform_query_graph: questions across documents and about how things relate (which
  customers an item, supplier or late delivery affects; who supplies or buys what). It was never
  called all night, and "Brazil Cerrado is late: which cafés?" missed a café (L8, L26, L108).
  Its figures go to platform_query_data, which Auto holds, not smart_query_database.
* platform_graph_neighbors and platform_graph_impact named relations the graph never holds
  (implements, constrained_by, semantically_similar_to, conceptually_related_to): extraction
  snaps every edge to ``GRAPH_RELATIONS``. The neighbors filter now takes exactly those, and the
  impact walk follows the ones that carry an effect (handlers_graph).

They moved here from actions_analytics.py and actions_graph.py; those registrars register them
in the same places as before through ``registers_with``, but for platform_graph_impact, which
now follows platform_graph_stats.
"""

from __future__ import annotations

import functools
from typing import Callable, Iterable

from .action_registry import ActionDefinition, ActionRegistry

Registrar = Callable[[ActionRegistry], None]

# The relations a Knowledge Graph edge carries: extraction snaps every edge to one of these
# (modules/knowledge/graph_relations.py on FIXER's fix/f312-graph-relations, 667a359cc, which
# adds supplies, buys, responsible_for and substitutes_for). A literal: the actions files load
# without the app (scripts/generate_utterance_corpus.py); a test holds it to the vocabulary.
GRAPH_RELATIONS = (
    "uses", "part_of", "member_of", "depends_on", "produces", "supplies", "buys", "responsible_for",
    "substitutes_for", "causes", "enables", "blocks", "mitigates", "measures", "governed_by", "precedes",
    "triggers", "has_property", "references", "related_to",
)
# The ones an effect travels along (a late supply reaches the blend it is part of, then whoever buys it).
IMPACT_RELATIONS = (
    "part_of", "uses", "depends_on", "produces", "supplies", "buys", "causes", "enables", "blocks",
    "triggers", "precedes", "governed_by", "measures",
)

QUERY_DATA_PARAMETERS = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": (
                "Natural language question about business data "
                "(e.g. 'How many active users this month?', "
                "'Top 10 customers by revenue'). Sent as 'query', it is read as the question."
            ),
        },
        "database_id": {
            "type": "string",
            "description": (
                "The database source to query: its name (e.g. 'sales_db') "
                "or its numeric id. Omit it when the workspace has one "
                "database — that one is used. With several, name one; "
                "the error lists them."
            ),
        },
    },
    "required": ["question"],
}

QUERY_GRAPH_PARAMETERS = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": (
                "Natural-language question to answer from the knowledge graph. "
                "Examples: 'how does pricing connect to retention?', "
                "'what processes depend on the API?', 'which rules govern refunds?'"
            ),
        },
        "mode": {
            "type": "string",
            "enum": ["bfs", "dfs"],
            "description": (
                "Traversal strategy. 'bfs' (default) explores broadly — best for "
                "'what is connected to X?'. 'dfs' follows one path deep — best for "
                "'how does X reach Y?' or tracing dependency chains."
            ),
        },
        "depth": {
            "type": "integer",
            "description": "Maximum traversal depth in hops (default 3). Higher = more context but slower.",
        },
        "token_budget": {
            "type": "integer",
            "description": "Maximum tokens for the returned context window (default 2000).",
        },
    },
    "required": ["question"],
}


def register_query_data_action(registry: ActionRegistry) -> None:
    """platform_query_data: the NL2SQL query over the workspace's connected database."""
    registry.register(ActionDefinition(
        name="platform_query_data",
        description=(
            "Answer a question about the business's own records (orders, subscriptions, "
            "members, stock, sales, customers) from a connected database. It turns the "
            "question into SQL, runs it and returns the rows and the SQL with the database's "
            "schema (its tables, columns and their values): read the schema it returns, and "
            "never ask the user for table, column or schema names. Ask in the user's words, "
            "with every qualifier they gave (active, cancelled, a plan, a date range), and "
            "ask one figure per call: a total and a breakdown (how many cancelled, and the "
            "most common reason) are two calls. Report the figure the rows show and say "
            "what it counts; a group's count is never the total, and its notes say when a "
            "count is only the top groups' or a date has nothing recorded yet. With one "
            "database connected, pass only the question: that one is used. Name a database "
            "(database_id) only when several are connected."
        ),
        category="database",
        parameters=QUERY_DATA_PARAMETERS,
        permission_level="read",
        promoted=True,  # PRD-256 US-006: a first-class tool, pinned
        requires_confirmation=False,
        tags=["database", "query", "analytics", "metrics", "nl2sql", "data"],
        examples=[
            "how many active users do we have",
            "what's our current MRR",
            "show revenue trend for last 6 months",
            "top 5 products by sales",
            "how many users signed up last week",
            "query the database for average order value",
        ],
    ))


def register_query_graph_action(registry: ActionRegistry) -> None:
    """platform_query_graph: a traversal of the workspace's Knowledge Graph for a question."""
    registry.register(ActionDefinition(
        name="platform_query_graph",
        description=(
            "Query the business knowledge graph to find connections between concepts, "
            "trace dependencies, and discover relationships across documents. The graph "
            "contains the entities, processes, named metrics and rules the workspace's "
            "documents describe, and how they connect. Use it for a question that spans "
            "documents or asks how things relate: which customers, orders or products are "
            "affected if something is late or changes, who supplies or buys what, what goes "
            "into what; and search_knowledge for what one document says. It holds no live "
            "figures: counts, money, totals, averages and rankings come from the workspace's "
            "databases, so use platform_query_data for those (and for the live orders a "
            "graph answer points to). Returns a traversal-based answer with source nodes and "
            "edges. Use 'bfs' mode (default) for broad context or 'dfs' to trace a specific chain."
        ),
        category="graph",
        parameters=QUERY_GRAPH_PARAMETERS,
        permission_level="read",
        promoted=True,
        tags=["graph", "knowledge", "query", "search", "relationships", "dependencies"],
        examples=[
            "query the knowledge graph about our pricing strategy",
            "what does the graph say about customer onboarding?",
            "search graph for marketing dependencies",
            "how are authentication and user management connected?",
            "what processes depend on the payment system?",
            "which customers are affected if a supplier's delivery is late?",
            "who supplies the products in this gift box?",
            "which of our products use this ingredient?",
        ],
    ))


GRAPH_NEIGHBORS_PARAMETERS = {
    "type": "object",
    "properties": {
        "concept": {
            "type": "string",
            "description": (
                "Name or label of the concept node to look up. "
                "Case-insensitive, supports partial matches. "
                "Examples: 'pricing', 'authentication', 'customer retention'."
            ),
        },
        "relation_filter": {
            "type": "string",
            "enum": list(GRAPH_RELATIONS),
            "description": "Only return edges with this relation type, one of the graph's: " + ", ".join(GRAPH_RELATIONS) + ".",
        },
    },
    "required": ["concept"],
}

GRAPH_IMPACT_PARAMETERS = {
    "type": "object",
    "properties": {
        "concept": {
            "type": "string",
            "description": (
                "Name of the concept to analyze impact for. "
                "Examples: 'pricing model', 'API gateway', 'user authentication'."
            ),
        },
        "max_depth": {
            "type": "integer",
            "description": (
                "How many hops to follow (default 3). "
                "Higher = finds more distant impacts but may include noise."
            ),
        },
    },
    "required": ["concept"],
}


def register_graph_neighbors_action(registry: ActionRegistry) -> None:
    """platform_graph_neighbors: a concept's direct links, optionally of one relation."""
    registry.register(ActionDefinition(
        name="platform_graph_neighbors",
        description=(
            "Get the direct connections (neighbors) of a specific concept in the "
            "knowledge graph. Returns every node directly linked to the concept with "
            "relation types (part_of, supplies, buys, uses, depends_on, etc.) and confidence "
            "scores. Use to answer 'what is X connected to?' or to explore a concept's "
            "immediate context before doing a deeper traversal with platform_query_graph."
        ),
        category="graph",
        parameters=GRAPH_NEIGHBORS_PARAMETERS,
        permission_level="read",
        promoted=True,
        tags=["graph", "neighbors", "connections", "explore", "relationships"],
        examples=[
            "what is connected to the pricing concept?",
            "show neighbors of customer onboarding",
            "get connections for revenue model",
            "what depends on the authentication module?",
        ],
    ))


def register_graph_impact_action(registry: ActionRegistry) -> None:
    """platform_graph_impact: what a change to, or a delay of, one concept reaches."""
    registry.register(ActionDefinition(
        name="platform_graph_impact",
        description=(
            "Analyze the downstream impact of changing, removing or delaying a concept. "
            "Performs a BFS along the edges that carry an effect ("
            + ", ".join(IMPACT_RELATIONS)
            + ") to find all affected nodes grouped by distance: a late supply reaches the "
            "blend it is part of, then the customers who buy it. Answers 'what breaks if we "
            "change X?', 'which customers are affected if X is late?' and 'how far does this "
            "change ripple?'"
        ),
        category="graph",
        parameters=GRAPH_IMPACT_PARAMETERS,
        permission_level="read",
        tags=["graph", "impact", "analysis", "dependencies", "risk"],
        examples=[
            "what happens if we change pricing?",
            "analyze impact of removing the referral program",
            "show downstream effects of changing the API",
            "what would break if we removed the notification system?",
            "which customers are affected if this delivery is late?",
        ],
    ))


def registers_with(before: Iterable[Registrar] = (), after: Iterable[Registrar] = ()) -> Callable[[Registrar], Registrar]:
    """Decorate a registrar so ``before``'s actions are registered ahead of its own and
    ``after``'s behind them: the moved definitions keep their place in the registry's order."""
    first, last = tuple(before), tuple(after)

    def decorate(register: Registrar) -> Registrar:
        @functools.wraps(register)
        def register_all(registry: ActionRegistry) -> None:
            for builder in first:
                builder(registry)
            register(registry)
            for builder in last:
                builder(registry)

        return register_all

    return decorate
