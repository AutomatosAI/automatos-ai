"""Shopify sync + freshness ActionDefinitions (PRD-183 S3, F088).

Promotes catalog sync and graph-freshness to first-class platform tools so
Auto reaches parity with the manual ``/sync/products/start`` route: it can
refresh the commerce graph and report what changed, and check when the graph
last synced — all through its own tool surface.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_shopify_actions(registry: ActionRegistry) -> None:
    """Register Shopify sync + freshness executors as platform actions."""

    registry.register(ActionDefinition(
        name="platform_shopify_sync_catalog",
        description=(
            "Refresh this store's product catalog into the knowledge graph, then "
            "report what changed (product/collection/vendor node counts and edge "
            "counts). Use when the merchant asks to re-sync the catalog or when "
            "catalog answers look stale. Runs against THIS workspace's connected "
            "Shopify store."
        ),
        category="shopify",
        parameters={"type": "object", "properties": {}},
        permission_level="write",
        promoted=True,
        tags=["shopify", "catalog", "sync", "graph", "commerce"],
        examples=[
            "refresh the catalog and tell me what changed",
            "re-sync the product graph",
            "update the store catalog in the knowledge graph",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_shopify_sync_status",
        description=(
            "Report catalog-graph freshness for this store: the last sync status, "
            "when it last synced, and how many nodes/edges it produced. Answers "
            "'when did the catalog last sync?' and 'is the product graph fresh?'. "
            "Returns never_synced if no sync has run."
        ),
        category="shopify",
        parameters={"type": "object", "properties": {}},
        permission_level="read",
        promoted=True,
        tags=["shopify", "catalog", "freshness", "status", "sync"],
        examples=[
            "when did the catalog last sync?",
            "is the product graph up to date?",
            "how fresh is the store catalog?",
        ],
    ))
