"""Graph ActionDefinitions — knowledge graph query, traversal, and analytics."""

from .action_registry import ActionDefinition, ActionRegistry


def register_graph_actions(registry: ActionRegistry) -> None:
    """Register knowledge-graph platform actions."""

    registry.register(ActionDefinition(
        name="platform_query_graph",
        description=(
            "Query the business knowledge graph to find connections between concepts, "
            "trace dependencies, and discover relationships across documents. The graph "
            "contains entities, processes, metrics, and rules extracted from all workspace "
            "documents. Returns a traversal-based answer with source nodes and edges. "
            "Use 'bfs' mode (default) for broad context or 'dfs' to trace a specific chain."
        ),
        category="graph",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Natural-language question to answer from the knowledge graph. "
                        "Examples: 'how does pricing connect to retention?', "
                        "'what processes depend on the API?', 'what metrics track growth?'"
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
        },
        permission_level="read",
        promoted=True,
        tags=["graph", "knowledge", "query", "search", "relationships", "dependencies"],
        examples=[
            "query the knowledge graph about our pricing strategy",
            "what does the graph say about customer onboarding?",
            "search graph for marketing dependencies",
            "how are authentication and user management connected?",
            "what processes depend on the payment system?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_graph_neighbors",
        description=(
            "Get the direct connections (neighbors) of a specific concept in the "
            "knowledge graph. Returns every node directly linked to the concept with "
            "relation types (depends_on, implements, triggers, etc.) and confidence "
            "scores. Use to answer 'what is X connected to?' or to explore a concept's "
            "immediate context before doing a deeper traversal with platform_query_graph."
        ),
        category="graph",
        parameters={
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
                    "description": (
                        "Only return edges with this relation type. "
                        "Common types: depends_on, implements, triggers, measures, "
                        "constrained_by, semantically_similar_to, conceptually_related_to."
                    ),
                },
            },
            "required": ["concept"],
        },
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

    registry.register(ActionDefinition(
        name="platform_graph_communities",
        description=(
            "List the auto-detected business-domain communities (clusters) in the "
            "knowledge graph. Communities group tightly connected concepts — e.g. "
            "'Authentication & Access Control', 'Revenue & Pricing', 'Data Pipeline'. "
            "Use to understand the high-level domain structure of the workspace's "
            "knowledge base. Pass community_id for full member list of one cluster."
        ),
        category="graph",
        parameters={
            "type": "object",
            "properties": {
                "community_id": {
                    "type": "integer",
                    "description": (
                        "ID of a specific community to get detailed members for. "
                        "Omit to get a summary of all communities with member counts."
                    ),
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["graph", "communities", "domains", "clusters", "overview"],
        examples=[
            "list graph communities",
            "what business domains exist in the graph?",
            "show community details for cluster 3",
            "how is our knowledge organized?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_graph_impact",
        description=(
            "Analyze the downstream impact of changing or removing a concept. "
            "Performs a directed BFS following dependency edges (depends_on, implements, "
            "triggers, constrained_by, measures) to find all affected nodes grouped by "
            "distance. Essential for risk assessment before strategic changes — answers "
            "'what breaks if we change X?' and 'how far does this change ripple?'"
        ),
        category="graph",
        parameters={
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
        },
        permission_level="read",
        tags=["graph", "impact", "analysis", "dependencies", "risk"],
        examples=[
            "what happens if we change pricing?",
            "analyze impact of removing the referral program",
            "show downstream effects of changing the API",
            "what would break if we removed the notification system?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_graph_stats",
        description=(
            "Get health metrics for the knowledge graph: total nodes, edges, "
            "community count, god nodes (highest-connected concepts), and when "
            "the graph was last built. Use to check coverage, identify if the "
            "graph needs rebuilding, or report on knowledge base health."
        ),
        category="graph",
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level="read",
        tags=["graph", "stats", "health", "metrics", "coverage"],
        examples=[
            "how big is the knowledge graph?",
            "graph health check",
            "show graph statistics",
            "when was the knowledge graph last updated?",
            "what are the most connected concepts?",
        ],
    ))
