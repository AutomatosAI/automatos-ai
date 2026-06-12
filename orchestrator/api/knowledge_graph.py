"""
Knowledge Graph API — workspace graph management (PRD-126).

The PRD-21 entity-explorer endpoints (``/entities/*``, ``/stats/entities``) were
removed in PRD-165 (graph consolidation): the per-workspace knowledge graph is
the single canonical surface (rendered by the Knowledge Graph panel), and agents
query it through the ``platform_graph_*`` tools
(modules/tools/discovery/handlers_graph.py), not REST. What remains here is the
workspace-graph lifecycle: import a graphify graph.json, rebuild, and delete.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
import json
import logging

from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["🧠 Knowledge Graph"])


# ============================================================================
# WORKSPACE GRAPH MANAGEMENT (PRD-126)
# ============================================================================

_MAX_GRAPH_IMPORT_SIZE = 50 * 1024 * 1024  # 50MB


@router.post("/graph/import")
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


@router.delete("/graph")
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


@router.post("/graph/build")
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
