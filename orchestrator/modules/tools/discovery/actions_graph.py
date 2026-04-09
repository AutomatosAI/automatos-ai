"""Graph ActionDefinitions — knowledge graph query, traversal, and analytics."""

from .action_registry import ActionDefinition, ActionRegistry


def register_graph_actions(registry: ActionRegistry) -> None:
    """Register knowledge-graph platform actions."""

    registry.register(ActionDefinition(
        name="platform_query_graph",
        description=(
            "Query the business knowledge graph with a natural-language question. "
            "Returns a subgraph context window with relevant nodes and edges. "
            "Use mode to control traversal strategy (bfs/dfs), depth to limit hops, "
            "and token_budget to cap response size."
        ),
        category="graph",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural-language question to answer from the graph.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["bfs", "dfs"],
                    "description": "Traversal strategy. Defaults to bfs.",
                },
                "depth": {
                    "type": "integer",
                    "description": "Maximum traversal depth in hops. Defaults to 3.",
                },
                "token_budget": {
                    "type": "integer",
                    "description": "Maximum tokens for the returned context window. Defaults to 2000.",
                },
            },
            "required": ["question"],
        },
        permission_level="read",
        tags=["graph", "knowledge", "query", "search"],
        examples=[
            "query the knowledge graph about our pricing strategy",
            "what does the graph say about customer onboarding?",
            "search graph for marketing dependencies",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_graph_neighbors",
        description=(
            "Get the direct connections (neighbors) of a concept in the knowledge "
            "graph. Optionally filter by relation type. Use to explore what a "
            "concept is linked to."
        ),
        category="graph",
        parameters={
            "type": "object",
            "properties": {
                "concept": {
                    "type": "string",
                    "description": "Name of the concept node to look up.",
                },
                "relation_filter": {
                    "type": "string",
                    "description": "Optional relation type to filter edges (e.g. 'depends_on').",
                },
            },
            "required": ["concept"],
        },
        permission_level="read",
        tags=["graph", "neighbors", "connections", "explore"],
        examples=[
            "what is connected to the pricing concept?",
            "show neighbors of customer onboarding",
            "get connections for revenue model",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_graph_communities",
        description=(
            "List detected business-domain communities in the knowledge graph. "
            "Communities are clusters of tightly connected concepts. Pass a "
            "community_id to get details for a specific community."
        ),
        category="graph",
        parameters={
            "type": "object",
            "properties": {
                "community_id": {
                    "type": "integer",
                    "description": "Optional ID of a specific community to retrieve.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["graph", "communities", "domains", "clusters"],
        examples=[
            "list graph communities",
            "what business domains exist in the graph?",
            "show community details",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_graph_impact",
        description=(
            "Analyze the downstream impact of changing a concept in the knowledge "
            "graph. Returns affected nodes and paths up to max_depth hops away. "
            "Use before making strategic changes to understand ripple effects."
        ),
        category="graph",
        parameters={
            "type": "object",
            "properties": {
                "concept": {
                    "type": "string",
                    "description": "Name of the concept to analyze impact for.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth for impact propagation. Defaults to 4.",
                },
            },
            "required": ["concept"],
        },
        permission_level="read",
        tags=["graph", "impact", "analysis", "dependencies"],
        examples=[
            "what happens if we change pricing?",
            "analyze impact of removing the referral program",
            "show downstream effects of changing the API",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_graph_stats",
        description=(
            "Get health metrics for the knowledge graph: node count, edge count, "
            "density, community count, and staleness indicators. Use to monitor "
            "graph quality and coverage."
        ),
        category="graph",
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level="read",
        tags=["graph", "stats", "health", "metrics"],
        examples=[
            "how big is the knowledge graph?",
            "graph health check",
            "show graph statistics",
        ],
    ))
