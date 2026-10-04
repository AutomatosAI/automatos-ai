"""F302 / F312 (night 9): Auto's ways into the workspace's own data, on every chat turn.

Night 9 ran on a workspace with a connected shop database (harbourline_shop) and a Knowledge
Graph built from its 12 documents. The prompt told Auto to "call smart_query_database" for
numbers (F078), but SmartToolRouter kept smart_query_database only on a turn whose words
matched its data patterns ("how many", "total", "count") or AutoBrain's hints. "Were any
Harvest Club boxes late going out in September?", "Have we got enough Guji for October's club
boxes?", "How much Guji have we got right now?" and "which cafés' orders are at risk?" matched
none: Auto had no data tool, and improvised. It called platform_field_query (the mission field)
four times, platform_shopify_sync_catalog, an invented platform_smart_query_database, asked the
owner "the exact names of the fields" or "which system holds your live order data", or said it
had no access. The same question went two ways in two fresh chats. platform_query_data, the
action that answered whenever Auto found it, sat only in the dispatcher's enum. And
platform_query_graph was never called all night: it too was only an enum entry, behind a
catalog that ranks a dozen actions a turn.

Now Auto's chat surface carries both routes first-class whenever the workspace has them:
``platform_query_data`` while the workspace has an active database, ``platform_query_graph``
while its Knowledge Graph holds anything. Each clears the gates its dispatcher entry clears: a
registered action that can run here, outside the workspace's hidden categories, never an
admin-only one, and inside the plan tier's families. With the data route attached,
``smart_query_database`` and ``query_database`` leave the chat surface: one door to the
database, with one description (smart_query_database's says it "asks follow-ups", and Auto
asked the owner for field names). A route that cannot be confirmed is left off and logged; the
surface is then what it was.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DATA_ROUTE = "platform_query_data"
GRAPH_ROUTE = "platform_query_graph"
# The other NL2SQL tools: the same service behind another name and another description.
OTHER_DATA_TOOLS = frozenset({"smart_query_database", "query_database"})
TRACE = "data-routes"


async def with_data_routes(tools: List[Dict[str, Any]], workspace_id: Any, db: Any) -> List[Dict[str, Any]]:
    """Auto's chat tool surface with the workspace's data routes attached first-class.

    Returns a new list; no schema is mutated. An empty surface (a turn that ships no tools)
    stays empty.
    """
    if not tools or not workspace_id or db is None:
        return list(tools or [])
    held = {_name(tool) for tool in tools}
    wanted = [name for name in await routes_for(workspace_id, db) if name not in held]
    added = [schema for schema in (route_schema(name, workspace_id, db) for name in wanted) if schema]
    has_data_route = DATA_ROUTE in held or any(_name(schema) == DATA_ROUTE for schema in added)
    kept = [tool for tool in tools if not (has_data_route and _name(tool) in OTHER_DATA_TOOLS)]
    return kept + added


async def routes_for(workspace_id: Any, db: Any) -> List[str]:
    """The routes this workspace has: the data route with an active database, the graph
    route with a Knowledge Graph that holds anything."""
    routes = []
    if _has_database(workspace_id, db):
        routes.append(DATA_ROUTE)
    if await _has_graph(workspace_id):
        routes.append(GRAPH_ROUTE)
    return routes


def route_schema(name: str, workspace_id: Any, db: Any) -> Optional[Dict[str, Any]]:
    """The action's first-class schema when it clears the gates its enum entry clears, else None."""
    from modules.tools.discovery.action_registry import action_is_available, get_action_registry
    from modules.tools.discovery.hidden_categories import hidden_categories_for_workspace
    from modules.tools.tool_router import _apply_tier_exposure

    action = get_action_registry().get(name)
    if action is None or not action_is_available(action) or action.admin_only or action.super_admin_only:
        return None
    if action.category in hidden_categories_for_workspace(workspace_id, db):
        return None
    schema = action.to_openai_schema()
    return schema if _apply_tier_exposure(db, workspace_id, [schema], TRACE) else None


def _has_database(workspace_id: Any, db: Any) -> bool:
    from modules.context.sections.documents_inventory import connected_databases

    try:
        return bool(connected_databases(db, workspace_id))
    except Exception:  # noqa: BLE001 — an unconfirmed route is left off; the surface stays as it was
        logger.exception("[data-routes] workspace %s's databases could not be read", workspace_id)
        return False


async def _has_graph(workspace_id: Any) -> bool:
    from modules.knowledge.graph_service import get_graph_service

    try:
        graph = await get_graph_service().load_graph(str(workspace_id))
    except Exception:  # noqa: BLE001 — an unconfirmed route is left off; the surface stays as it was
        logger.exception("[data-routes] workspace %s's Knowledge Graph could not be read", workspace_id)
        return False
    return graph is not None and graph.number_of_nodes() > 0


def _name(tool: Any) -> str:
    return str(((tool or {}).get("function") or {}).get("name", "")) if isinstance(tool, dict) else ""
