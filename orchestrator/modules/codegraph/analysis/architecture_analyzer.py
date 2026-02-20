"""
Architecture analysis — module detection, coupling metrics, hotspot identification.

Inspired by Emerge's Louvain clustering and heatmap approach.
Uses NetworkX for graph analysis on the CodeGraph relationship data.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureReport:
    """Complete architecture analysis result."""
    project_id: int
    project_name: str
    communities: List[Dict[str, Any]]  # Louvain-detected module clusters
    metrics: Dict[str, Dict[int, float]]  # Per-node metrics
    hotspots: List[Dict[str, Any]]  # High-risk nodes
    cycles: List[List[str]]  # Circular dependencies
    total_nodes: int
    total_edges: int
    modularity_score: float = 0.0


class ArchitectureAnalyzer:
    """Analyze codebase architecture from the code graph."""

    def __init__(self, db_session):
        self.db = db_session

    async def analyze(
        self,
        project_id: int,
        workspace_id: Optional[str] = None,
    ) -> ArchitectureReport:
        """Run full architecture analysis on a CodeGraph project."""
        from sqlalchemy import text

        # Fetch project info
        project = self.db.execute(
            text("SELECT id, name FROM codegraph_projects WHERE id = :pid AND workspace_id = :wid"),
            {"pid": project_id, "wid": workspace_id}
        ).fetchone()

        if not project:
            raise ValueError(f"Project {project_id} not found")

        # Build the graph
        G = await self._build_graph(project_id, workspace_id)

        if len(G.nodes()) == 0:
            return ArchitectureReport(
                project_id=project_id,
                project_name=project.name,
                communities=[], metrics={}, hotspots=[], cycles=[],
                total_nodes=0, total_edges=0,
            )

        # Run analyses
        communities, modularity = self._detect_communities(G)
        metrics = self._compute_metrics(G)
        hotspots = self._identify_hotspots(G, metrics)
        cycles = self._detect_cycles(G)

        return ArchitectureReport(
            project_id=project_id,
            project_name=project.name,
            communities=communities,
            metrics=self._serialize_metrics(metrics),
            hotspots=hotspots,
            cycles=cycles,
            total_nodes=len(G.nodes()),
            total_edges=len(G.edges()),
            modularity_score=modularity,
        )

    async def _build_graph(self, project_id: int, workspace_id: Optional[str]) -> nx.DiGraph:
        """Build a NetworkX DiGraph from codegraph_relationships."""
        from sqlalchemy import text

        G = nx.DiGraph()

        # Fetch symbols as nodes
        symbols = self.db.execute(
            text("""
                SELECT id, name, qualified_name, symbol_type, file_path
                FROM codegraph_symbols
                WHERE project_id = :pid AND workspace_id = :wid
            """),
            {"pid": project_id, "wid": workspace_id}
        ).fetchall()

        for s in symbols:
            G.add_node(s.id, name=s.name, qualified_name=s.qualified_name,
                       symbol_type=s.symbol_type, file_path=s.file_path)

        # Fetch relationships as edges
        rels = self.db.execute(
            text("""
                SELECT from_symbol_id, to_symbol_id, relationship_type
                FROM codegraph_relationships
                WHERE project_id = :pid AND workspace_id = :wid
                AND relationship_type != 'external_reference'
                AND from_symbol_id IS NOT NULL AND to_symbol_id IS NOT NULL
            """),
            {"pid": project_id, "wid": workspace_id}
        ).fetchall()

        for r in rels:
            if r.from_symbol_id in G and r.to_symbol_id in G:
                G.add_edge(r.from_symbol_id, r.to_symbol_id,
                           relationship_type=r.relationship_type)

        return G

    def _detect_communities(self, G: nx.DiGraph) -> Tuple[List[Dict[str, Any]], float]:
        """Detect module clusters using Louvain community detection."""
        # Louvain works on undirected graphs
        UG = G.to_undirected()

        try:
            communities_gen = nx.community.louvain_communities(UG, seed=42)
            community_list = list(communities_gen)
        except Exception as e:
            logger.warning(f"Louvain community detection failed: {e}")
            return [], 0.0

        # Calculate modularity
        try:
            modularity = nx.community.modularity(UG, community_list)
        except Exception:
            modularity = 0.0

        # Format communities with metadata
        result = []
        for i, community_nodes in enumerate(community_list):
            members = []
            file_set: Set[str] = set()
            for node_id in community_nodes:
                data = G.nodes.get(node_id, {})
                members.append({
                    "id": node_id,
                    "name": data.get("name", "?"),
                    "type": data.get("symbol_type", "?"),
                    "file": data.get("file_path", "?"),
                })
                file_set.add(data.get("file_path", "?"))

            result.append({
                "cluster_id": i,
                "size": len(community_nodes),
                "files": sorted(file_set),
                "members": members[:50],  # Limit for response size
                "member_count": len(members),
            })

        # Sort by size descending
        result.sort(key=lambda c: c["size"], reverse=True)
        return result, modularity

    def _compute_metrics(self, G: nx.DiGraph) -> Dict[str, Dict[int, float]]:
        """Compute per-node graph metrics."""
        metrics = {}

        try:
            metrics['fan_in'] = dict(G.in_degree())
            metrics['fan_out'] = dict(G.out_degree())
        except Exception:
            metrics['fan_in'] = {}
            metrics['fan_out'] = {}

        try:
            metrics['betweenness'] = nx.betweenness_centrality(G)
        except Exception:
            metrics['betweenness'] = {}

        try:
            metrics['pagerank'] = nx.pagerank(G, alpha=0.85)
        except Exception:
            metrics['pagerank'] = {}

        return metrics

    def _identify_hotspots(
        self, G: nx.DiGraph, metrics: Dict[str, Dict[int, float]]
    ) -> List[Dict[str, Any]]:
        """Identify high-risk nodes (high fan_out AND high betweenness)."""
        fan_out = metrics.get('fan_out', {})
        betweenness = metrics.get('betweenness', {})

        if not fan_out or not betweenness:
            return []

        # Thresholds: top 10% for each metric
        fo_values = sorted(fan_out.values(), reverse=True)
        bt_values = sorted(betweenness.values(), reverse=True)

        fo_threshold = fo_values[max(len(fo_values) // 10, 1) - 1] if fo_values else 0
        bt_threshold = bt_values[max(len(bt_values) // 10, 1) - 1] if bt_values else 0

        hotspots = []
        for node_id in G.nodes():
            fo = fan_out.get(node_id, 0)
            bt = betweenness.get(node_id, 0)
            if fo >= fo_threshold and bt >= bt_threshold and (fo > 0 or bt > 0):
                data = G.nodes[node_id]
                hotspots.append({
                    "id": node_id,
                    "name": data.get("name", "?"),
                    "file": data.get("file_path", "?"),
                    "type": data.get("symbol_type", "?"),
                    "fan_out": fo,
                    "betweenness": round(bt, 4),
                    "risk_reason": "High fan-out and high betweenness centrality",
                })

        hotspots.sort(key=lambda h: h["betweenness"], reverse=True)
        return hotspots[:20]  # Limit

    def _detect_cycles(self, G: nx.DiGraph) -> List[List[str]]:
        """Detect circular dependencies."""
        try:
            raw_cycles = list(nx.simple_cycles(G))
        except Exception:
            return []

        # Convert node IDs to readable names, limit to 20
        cycles = []
        for cycle in raw_cycles[:20]:
            named = []
            for node_id in cycle:
                data = G.nodes.get(node_id, {})
                named.append(data.get("qualified_name", str(node_id)))
            cycles.append(named)

        return cycles

    def _serialize_metrics(self, metrics: Dict[str, Dict[int, float]]) -> Dict[str, Any]:
        """Serialize metrics for JSON response (convert node IDs to strings)."""
        serialized = {}
        for metric_name, values in metrics.items():
            # Summarize instead of sending all values
            vals = list(values.values())
            if vals:
                serialized[metric_name] = {
                    "mean": round(sum(vals) / len(vals), 4),
                    "max": round(max(vals), 4),
                    "min": round(min(vals), 4),
                    "count": len(vals),
                }
            else:
                serialized[metric_name] = {"mean": 0, "max": 0, "min": 0, "count": 0}
        return serialized
