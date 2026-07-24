"""
Knowledge Graph API — workspace graph management (PRD-126).

The PRD-21 entity-explorer endpoints (``/entities/*``, ``/stats/entities``) were
removed in PRD-165 (graph consolidation): the per-workspace knowledge graph is
the single canonical surface (rendered by the Knowledge Graph panel), and agents
query it through the ``platform_graph_*`` tools
(modules/tools/discovery/handlers_graph.py), not REST. What remains here is the
workspace-graph lifecycle: import a graphify graph.json, rebuild, and delete.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Query, Body
import json
import logging

from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["🧠 Knowledge Graph"])


# ============================================================================
# WORKSPACE GRAPH MANAGEMENT (PRD-126)
# ============================================================================

_MAX_GRAPH_IMPORT_SIZE = 50 * 1024 * 1024  # 50MB


@router.post("/graph/import", dependencies=[Depends(require_workspace_permission("knowledge:create"))])
async def import_graph(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    file: UploadFile = File(...),
    merge: bool = Form(False),
):
    """Import a graphify graph.json into the workspace's knowledge graph.

    Accepts a graph.json file (NetworkX node_link_data format) as produced by
    the graphify CLI or the /graphify skill. Re-runs clustering, analysis, and
    exports all derived artefacts (HTML viz, communities, meta).

    Set merge=true to add imported nodes/edges into an existing graph rather
    than replacing it.
    """
    if not file.filename or not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="File must be a .json file")

    content = await file.read()
    if len(content) > _MAX_GRAPH_IMPORT_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    try:
        graph_data = json.loads(content)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    # Validate it looks like node_link_data
    if not isinstance(graph_data, dict) or "nodes" not in graph_data:
        raise HTTPException(
            status_code=400,
            detail="Invalid graph format — expected NetworkX node_link_data with 'nodes' and 'links' keys",
        )

    try:
        from modules.knowledge.graph_service import get_graph_service
        service = get_graph_service()
        meta = await service.import_graph(
            str(ctx.workspace_id), graph_data, merge=merge,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Graph import failed for workspace %s", ctx.workspace_id)
        raise HTTPException(status_code=500, detail=f"Import failed: {exc}")

    return {
        "success": True,
        "message": f"Graph imported — {meta.get('node_count', 0)} nodes, {meta.get('edge_count', 0)} edges, {meta.get('community_count', 0)} communities",
        "meta": meta,
    }


@router.delete("/graph", dependencies=[Depends(require_workspace_permission("knowledge:delete"))])
async def delete_graph(
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Delete the workspace knowledge graph (all artefacts)."""
    from core.graph_storage import DbWorkspaceClient
    ws = DbWorkspaceClient(str(ctx.workspace_id))
    for path in ["graph/graph.json", "graph/meta.json", "graph/communities.json", "graph/graph.html"]:
        try:
            await ws.delete_file(path)
        except Exception:
            pass
    from modules.knowledge.graph_service import get_graph_service
    get_graph_service()._cache.pop(str(ctx.workspace_id), None)
    return {"success": True, "message": "Graph deleted"}


@router.post("/graph/build", dependencies=[Depends(require_workspace_permission("knowledge:create"))])
async def trigger_graph_build(
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Manually trigger a full knowledge graph rebuild for the workspace."""
    try:
        from modules.knowledge.graph_service import get_graph_service
        service = get_graph_service()
        meta = await service.build_graph(str(ctx.workspace_id))
    except Exception as exc:
        logger.exception("Graph build failed for workspace %s", ctx.workspace_id)
        raise HTTPException(status_code=500, detail=f"Build failed: {exc}")

    return {
        "success": True,
        "message": f"Graph built — {meta.get('node_count', 0)} nodes, {meta.get('edge_count', 0)} edges",
        "meta": meta,
    }


# ============================================================================
# CLUSTER-FIRST DRILL-IN (PRD-165 S2)
# Server-side subgraph queries so the browser never downloads the full
# graph.json (Q28, LightRAG pattern): communities -> community subgraph ->
# expand a node -> path between two nodes -> search-to-focus.
# ============================================================================


async def _load_ws_graph(ctx: RequestContext):
    """Load the workspace knowledge graph or 404 if none is built."""
    from modules.knowledge.graph_service import get_graph_service
    graph = await get_graph_service().load_graph(str(ctx.workspace_id))
    if graph is None:
        raise HTTPException(
            status_code=404,
            detail="No knowledge graph built for this workspace yet.",
        )
    return graph


async def _read_communities(ctx: RequestContext) -> list:
    """Read communities.json or 404."""
    from core.graph_storage import DbWorkspaceClient
    ws = DbWorkspaceClient(str(ctx.workspace_id))
    result = await ws.read_file("graph/communities.json")
    if not result.get("success") or not result.get("content"):
        raise HTTPException(
            status_code=404,
            detail="No communities — build the knowledge graph first.",
        )
    try:
        return json.loads(result["content"])
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=500, detail="Corrupt communities data.")


@router.get("/graph/communities")
async def list_graph_communities(
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Community overview for cluster-first drill-in: id, size, and (once the
    graph is enriched in PRD-165 S3) a title/summary per community. This is the
    entry point — the client lists clusters first, then drills into one."""
    communities = await _read_communities(ctx)
    overview = [
        {
            "community_id": c.get("community_id"),
            "member_count": c.get("member_count", len(c.get("members", []))),
            "title": c.get("title"),
            "summary": c.get("summary"),
        }
        for c in communities
        if c.get("member_count", len(c.get("members", []))) > 0
    ]
    overview.sort(key=lambda c: c["member_count"], reverse=True)
    return {"success": True, "community_count": len(overview), "communities": overview}


@router.get("/graph/community/{community_id}")
async def get_community_subgraph(
    community_id: int,
    max_nodes: int = Query(300, ge=1, le=2000),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """The induced subgraph for one community's members (nodes + internal
    edges), capped server-side — the drill-in payload."""
    communities = await _read_communities(ctx)
    match = next(
        (c for c in communities if c.get("community_id") == community_id), None
    )
    if match is None:
        raise HTTPException(status_code=404, detail=f"Community {community_id} not found.")

    graph = await _load_ws_graph(ctx)
    from modules.knowledge.graph_service import get_graph_service
    data = await get_graph_service().community_subgraph(
        graph, match.get("members", []), max_nodes=max_nodes,
    )
    return {
        "success": True,
        "community_id": community_id,
        "title": match.get("title"),
        "summary": match.get("summary"),
        **data,
    }


@router.get("/graph/node/{node_id}/neighbors")
async def expand_node(
    node_id: str,
    max_nodes: int = Query(150, ge=1, le=1000),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """The node + its 1-hop neighbourhood ('expand from here')."""
    graph = await _load_ws_graph(ctx)
    from modules.knowledge.graph_service import get_graph_service
    data = await get_graph_service().node_neighbors_subgraph(graph, node_id, max_nodes=max_nodes)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found in the graph.")
    return {"success": True, "node_id": node_id, **data}


@router.get("/graph/path")
async def graph_path(
    source: str = Query(..., min_length=1),
    target: str = Query(..., min_length=1),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Shortest path between two node ids for the path-finding UI."""
    graph = await _load_ws_graph(ctx)
    from modules.knowledge.graph_service import get_graph_service
    result = await get_graph_service().shortest_path(graph, source, target)
    return {"success": bool(result.get("found")), **result}


@router.get("/graph/search")
async def search_graph_nodes(
    q: str = Query(..., min_length=1),
    limit: int = Query(25, ge=1, le=100),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Label search for search-to-focus."""
    graph = await _load_ws_graph(ctx)
    from modules.knowledge.graph_service import get_graph_service
    matches = await get_graph_service().search_nodes(graph, q, limit=limit)
    return {"success": True, "query": q, "match_count": len(matches), "matches": matches}


@router.patch("/graph/community/{community_id}/label", dependencies=[Depends(require_workspace_permission("knowledge:update"))])
async def set_community_label(
    community_id: int,
    payload: dict = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Rename a community / edit its summary (PRD-165 S3 — editable labels)."""
    title = (payload.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    summary = payload.get("summary")

    from modules.knowledge.graph_service import get_graph_service
    ok = await get_graph_service().set_community_label(
        str(ctx.workspace_id), community_id, title, summary,
    )
    if not ok:
        raise HTTPException(status_code=404, detail=f"Community {community_id} not found.")
    return {"success": True, "community_id": community_id, "title": title}
