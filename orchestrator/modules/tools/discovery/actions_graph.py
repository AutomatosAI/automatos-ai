"""Graph ActionDefinitions — knowledge graph query, traversal, and analytics."""

from .action_registry import ActionDefinition, ActionRegistry
from .actions_data_routes import (
    register_graph_impact_action,
    register_graph_neighbors_action,
    register_query_graph_action,
    registers_with,
)


# F312: the graph route, and the two tools whose relation names now match the graph's, moved there.
@registers_with(before=[register_query_graph_action, register_graph_neighbors_action],
                after=[register_graph_impact_action])
def register_graph_actions(registry: ActionRegistry) -> None:
    """Register knowledge-graph platform actions."""

    registry.register(ActionDefinition(
        name="platform_graph_path",
        description=(
            "Find the shortest path between two concepts in the knowledge graph. "
            "Returns the ordered chain of nodes connecting the source to the target "
            "plus the hop count. Use to answer 'how is X related to Y?' or "
            "'what connects X and Y?'."
        ),
        category="graph",
        parameters={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": (
                        "Start concept name or label. Case-insensitive, supports "
                        "partial matches. Example: 'pricing'."
                    ),
                },
                "target": {
                    "type": "string",
                    "description": (
                        "End concept name or label. Case-insensitive, supports "
                        "partial matches. Example: 'customer churn'."
                    ),
                },
            },
            "required": ["source", "target"],
        },
        permission_level="read",
        promoted=True,
        tags=["graph", "path", "connection", "relationship", "explore"],
        examples=[
            "how is pricing connected to customer churn?",
            "what links authentication and billing?",
            "find the path between onboarding and retention",
            "shortest path from product to revenue",
        ],
        accepts=("from", "to"),
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
