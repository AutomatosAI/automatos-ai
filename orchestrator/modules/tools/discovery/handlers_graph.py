"""Graph knowledge handlers — query, neighbors, communities, impact, stats.

PRD-126 US-005: Platform tool handlers for the knowledge graph.
All handlers follow the standard (db, workspace_id, params) signature
used by PlatformActionExecutor.
"""

import json
import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Directional edge types for impact analysis
_DIRECTIONAL_RELATIONS: Set[str] = {
    "depends_on",
    "implements",
    "constrained_by",
    "triggers",
    "measures",
}
_BIDIRECTIONAL_RELATIONS: Set[str] = {
    "semantically_similar_to",
    "conflicts_with",
}
_IMPACT_RELATIONS: Set[str] = _DIRECTIONAL_RELATIONS | _BIDIRECTIONAL_RELATIONS


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_service():
    """Lazy import to avoid circular dependency at module load."""
    from modules.knowledge.graph_service import get_graph_service

    return get_graph_service()


def _resolve_agent_team(db: Session, agent_id: Optional[int]) -> Optional[str]:
    """Look up the agent's team from the DB. Returns None if no team set.

    PRD-124: agent with team=NULL sees all nodes (no filtering).
    """
    if not agent_id:
        return None
    try:
        from core.models.core import Agent
        agent = db.query(Agent.team).filter(Agent.id == agent_id).first()
        if agent and agent.team:
            from core.team_access import normalize_team
            return normalize_team(agent.team)
        return None
    except Exception:
        logger.debug("_resolve_agent_team: failed for agent_id=%s", agent_id)
        return None


def _get_filtered_graph(graph, agent_team: Optional[str]):
    """Apply PRD-124 team filtering to the graph."""
    from modules.knowledge.graph_service import team_filtered_view
    return team_filtered_view(graph, agent_team)


def _find_node_by_label(graph, label: str) -> Optional[str]:
    """Find a node ID by case-insensitive label match, falling back to ID match."""
    label_lower = label.lower()

    # Exact ID match first
    if label in graph:
        return label

    # Case-insensitive label search
    for node_id, attrs in graph.nodes(data=True):
        node_label = attrs.get("label", str(node_id))
        if node_label.lower() == label_lower:
            return node_id

    # Substring match as last resort
    for node_id, attrs in graph.nodes(data=True):
        node_label = attrs.get("label", str(node_id))
        if label_lower in node_label.lower():
            return node_id

    return None


# ------------------------------------------------------------------
# 1. handle_query_graph
# ------------------------------------------------------------------


async def handle_query_graph(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Query the knowledge graph with a natural-language question.

    Loads the workspace graph, scores nodes against the query using
    graphify's ``_score_nodes``, traverses from the top-scored node
    via BFS or DFS, and returns a text summary within token_budget.

    Params:
        question (str): Natural-language query.
        mode (str): "bfs" (default) or "dfs".
        depth (int): Traversal depth (default 2).
        token_budget (int): Max tokens for the text summary (default 1500).
    """
    question = (params.get("question") or "").strip()
    if not question:
        return {"success": False, "error": "question is required"}

    mode = (params.get("mode") or "bfs").lower()
    depth = int(params.get("depth", 2))
    token_budget = int(params.get("token_budget", 1500))

    try:
        svc = _get_service()
        graph = await svc.load_graph(str(workspace_id))
        if graph is None:
            return {
                "success": False,
                "error": "No knowledge graph built for this workspace yet.",
            }

        # PRD-124: filter graph to team-visible nodes
        agent_team = _resolve_agent_team(db, params.get("_agent_id"))
        graph = _get_filtered_graph(graph, agent_team)

        # Score nodes against the question terms
        terms = question.split()
        scored = await svc.score_nodes(graph, terms)
        if not scored:
            return {
                "success": True,
                "answer": "Graph has no scorable nodes.",
                "node_count": graph.number_of_nodes(),
            }

        # Pick the best-scoring start node
        top_node = scored[0]
        start_id = top_node.get("id", "")
        if start_id not in graph:
            # Fall back to first graph node
            start_id = next(iter(graph.nodes()))

        # Traverse — returns {"nodes": set, "edges": list}
        if mode == "dfs":
            result = await svc.dfs(graph, start_id, depth)
        else:
            result = await svc.bfs(graph, start_id, depth)

        traversed_nodes = result["nodes"]
        traversed_edges = result["edges"]

        # Convert to text, respecting budget
        text = await svc.subgraph_to_text(
            graph, traversed_nodes, traversed_edges, token_budget
        )

        return {
            "success": True,
            "answer": text,
            "start_node": start_id,
            "mode": mode,
            "nodes_traversed": len(traversed_nodes),
            "edges_traversed": len(traversed_edges),
        }

    except Exception as e:
        logger.error("handle_query_graph failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------
# 2. handle_graph_neighbors
# ------------------------------------------------------------------


async def handle_graph_neighbors(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Find all neighbors of a given node.

    Params:
        concept (str): Node label or ID to look up.
        relation_filter (str, optional): Only return edges with this relation type.
    """
    concept_label = (params.get("concept") or "").strip()
    if not concept_label:
        return {"success": False, "error": "concept is required"}

    relation_filter = (params.get("relation_filter") or "").strip().lower() or None

    try:
        svc = _get_service()
        graph = await svc.load_graph(str(workspace_id))
        if graph is None:
            return {
                "success": False,
                "error": "No knowledge graph built for this workspace yet.",
            }

        # PRD-124: filter graph to team-visible nodes
        agent_team = _resolve_agent_team(db, params.get("_agent_id"))
        graph = _get_filtered_graph(graph, agent_team)

        node_id = _find_node_by_label(graph, concept_label)
        if node_id is None:
            return {
                "success": False,
                "error": f"Node '{concept_label}' not found in the graph.",
            }

        neighbors: List[Dict[str, Any]] = []
        for u, v, edge_data in graph.edges(node_id, data=True):
            relation = edge_data.get("relation", "related_to")
            if relation_filter and relation.lower() != relation_filter:
                continue
            target = v if u == node_id else u
            target_attrs = dict(graph.nodes.get(target, {}))
            edge_attrs = dict(edge_data.get("attrs") or {})
            neighbors.append(
                {
                    "target": str(target),
                    "target_label": target_attrs.get("label", str(target)),
                    "target_attrs": target_attrs,
                    "relation": relation,
                    "confidence": edge_data.get("confidence", edge_data.get("weight", 1.0)),
                    "weight": edge_data.get("weight"),
                    "edge_attrs": edge_attrs,
                }
            )

        node_attrs = dict(graph.nodes.get(node_id, {}))
        return {
            "success": True,
            "node": str(node_id),
            "node_label": node_attrs.get("label", str(node_id)),
            "node_attrs": node_attrs,
            "neighbor_count": len(neighbors),
            "neighbors": neighbors,
        }

    except Exception as e:
        logger.error("handle_graph_neighbors failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------
# 3. handle_graph_communities
# ------------------------------------------------------------------


async def handle_graph_communities(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """List communities or get details for a specific community.

    Reads /graph/communities.json from the workspace.

    Params:
        community_id (int, optional): Specific community to return.
    """
    community_id = params.get("community_id")
    # PRD-124: resolve agent team for member filtering
    agent_team = _resolve_agent_team(db, params.get("_agent_id"))

    try:
        # Read from the SAME store _export_graph wrote to — Postgres
        # workspace_graphs via DbWorkspaceClient. The file-backed
        # WorkspaceClient returns nothing for a DB-backed workspace.
        from core.graph_storage import DbWorkspaceClient

        ws = DbWorkspaceClient(str(workspace_id))
        result = await ws.read_file("graph/communities.json")

        if not result.get("success"):
            return {
                "success": False,
                "error": "No communities data found. Build the knowledge graph first.",
            }

        content = result.get("content", "")
        if not content:
            return {"success": False, "error": "Communities file is empty."}

        try:
            communities = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error(
                "handle_graph_communities: corrupt communities.json for workspace %s: %s",
                workspace_id,
                exc,
            )
            return {"success": False, "error": "Corrupt communities data."}

        # PRD-124: filter community members by team visibility
        if agent_team is not None:
            svc = _get_service()
            graph = await svc.load_graph(str(workspace_id))
            if graph is not None:
                from modules.knowledge.graph_service import node_is_visible
                for c in communities:
                    members = c.get("members", [])
                    c["members"] = [
                        m for m in members
                        if node_is_visible(graph, m, agent_team)
                    ]
                    c["member_count"] = len(c["members"])

        # Filter to specific community if requested
        if community_id is not None:
            cid = int(community_id)
            matched = [c for c in communities if c.get("community_id") == cid]
            if not matched:
                return {
                    "success": False,
                    "error": f"Community {cid} not found.",
                }
            return {
                "success": True,
                "community": matched[0],
            }

        # Return summary of all communities (exclude empty after filtering)
        summary = [
            {
                "community_id": c.get("community_id"),
                "member_count": c.get("member_count", len(c.get("members", []))),
            }
            for c in communities
            if c.get("member_count", len(c.get("members", []))) > 0
        ]

        return {
            "success": True,
            "community_count": len(summary),
            "communities": summary,
        }

    except Exception as e:
        logger.error("handle_graph_communities failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------
# 4. handle_graph_impact
# ------------------------------------------------------------------


async def handle_graph_impact(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """BFS impact analysis from a concept node.

    Follows directional edges (depends_on, implements, constrained_by,
    triggers, measures) and bidirectional edges (semantically_similar_to,
    conflicts_with). Returns affected nodes grouped by BFS depth.

    Params:
        concept (str): Starting concept label or ID.
        max_depth (int): Maximum BFS depth (default 3).
    """
    concept = (params.get("concept") or "").strip()
    if not concept:
        return {"success": False, "error": "concept is required"}

    max_depth = int(params.get("max_depth", 3))

    try:
        svc = _get_service()
        graph = await svc.load_graph(str(workspace_id))
        if graph is None:
            return {
                "success": False,
                "error": "No knowledge graph built for this workspace yet.",
            }

        # PRD-124: filter graph to team-visible nodes
        agent_team = _resolve_agent_team(db, params.get("_agent_id"))
        graph = _get_filtered_graph(graph, agent_team)

        start = _find_node_by_label(graph, concept)
        if start is None:
            return {
                "success": False,
                "error": f"Concept '{concept}' not found in the graph.",
            }

        # BFS with relation filtering
        visited: Set[str] = {start}
        queue: deque = deque([(start, 0)])
        depth_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        while queue:
            current, current_depth = queue.popleft()
            if current_depth >= max_depth:
                continue

            for u, v, edge_data in graph.edges(current, data=True):
                neighbor = v if u == current else u
                relation = edge_data.get("relation", "")

                if relation not in _IMPACT_RELATIONS:
                    continue
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                neighbor_attrs = graph.nodes.get(neighbor, {})
                depth_groups[current_depth + 1].append(
                    {
                        "node": str(neighbor),
                        "label": neighbor_attrs.get("label", str(neighbor)),
                        "relation": relation,
                        "from_node": str(current),
                    }
                )
                queue.append((neighbor, current_depth + 1))

        # Format output
        impact_layers = [
            {"depth": d, "affected_count": len(nodes), "nodes": nodes}
            for d, nodes in sorted(depth_groups.items())
        ]

        start_attrs = graph.nodes.get(start, {})
        return {
            "success": True,
            "concept": str(start),
            "concept_label": start_attrs.get("label", str(start)),
            "total_affected": sum(len(layer["nodes"]) for layer in impact_layers),
            "max_depth_reached": max(depth_groups.keys()) if depth_groups else 0,
            "impact_layers": impact_layers,
        }

    except Exception as e:
        logger.error("handle_graph_impact failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------
# 5. handle_graph_stats
# ------------------------------------------------------------------


async def handle_graph_stats(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Return high-level knowledge graph statistics.

    Reads /graph/meta.json via GraphifyService.get_meta().

    Params: (none required)
    """
    try:
        svc = _get_service()
        meta = await svc.get_meta(str(workspace_id))
        if meta is None:
            return {
                "success": False,
                "error": "No knowledge graph built for this workspace yet.",
            }

        return {
            "success": True,
            "node_count": meta.get("node_count", 0),
            "edge_count": meta.get("edge_count", 0),
            "community_count": meta.get("community_count", 0),
            "god_nodes": meta.get("god_nodes", []),
            "last_built": meta.get("last_built"),
        }

    except Exception as e:
        logger.error("handle_graph_stats failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------
# 6. handle_graph_path (PRD-165 S2)
# ------------------------------------------------------------------


async def handle_graph_path(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Find the shortest path between two nodes in the knowledge graph.

    Params:
        source (str): Start node label or ID.
        target (str): End node label or ID.
    """
    source_label = (params.get("source") or params.get("from") or "").strip()
    target_label = (params.get("target") or params.get("to") or "").strip()
    if not source_label or not target_label:
        return {"success": False, "error": "source and target are required"}

    try:
        svc = _get_service()
        graph = await svc.load_graph(str(workspace_id))
        if graph is None:
            return {
                "success": False,
                "error": "No knowledge graph built for this workspace yet.",
            }

        # PRD-124: filter graph to team-visible nodes
        agent_team = _resolve_agent_team(db, params.get("_agent_id"))
        graph = _get_filtered_graph(graph, agent_team)

        source_id = _find_node_by_label(graph, source_label)
        if source_id is None:
            return {"success": False, "error": f"Node '{source_label}' not found in the graph."}
        target_id = _find_node_by_label(graph, target_label)
        if target_id is None:
            return {"success": False, "error": f"Node '{target_label}' not found in the graph."}

        result = await svc.shortest_path(graph, source_id, target_id)
        if not result.get("found"):
            return {"success": False, "error": result.get("error", "No path found.")}

        # Compact, agent-friendly: the ordered label trail + the hop count,
        # with the full node/edge payloads for any UI that wants to render it.
        trail = [n.get("label", n.get("id")) for n in result.get("path", [])]
        return {
            "success": True,
            "source": str(source_id),
            "target": str(target_id),
            "hops": result.get("length", max(0, len(trail) - 1)),
            "path": trail,
            "path_nodes": result.get("path", []),
            "edges": result.get("links", []),
        }

    except Exception as e:
        logger.error("handle_graph_path failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}
