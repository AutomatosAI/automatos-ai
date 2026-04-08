"""
GraphifyService — Knowledge-graph build, cache, and export
===========================================================

Single point of contact for all graphify library calls. Other modules
import from here instead of touching graphify directly, so upstream API
changes only require edits in this file.

Pipeline:
  collect sources → extract → merge → build_from_json →
  cluster → god_nodes → surprising_connections → export → cache →
  snapshot + diff → build report → prune history

Exports to workspace files under ``/graph/``:
  - graph.json   — NetworkX node_link_data
  - meta.json    — summary stats
  - communities.json — community labels + members
  - graph.html   — interactive vis.js visualization (US-011)
  - latest_diff.json — diff against previous build (US-010)
  - history/{date}_graph.json — daily snapshots (max 30)
  - reports/{date}_build.md  — human-readable build report

Source: PRD-126 US-001, US-010, US-011
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date
from functools import partial
from typing import Any, Dict, List, Optional

import networkx as nx
from cachetools import LRUCache

# ---------------------------------------------------------------------------
# graphify imports — ALL graphify usage funnelled through this module
# ---------------------------------------------------------------------------
from graphify.analyze import god_nodes, graph_diff, surprising_connections
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.export import to_html, to_json
from graphify.serve import (
    _bfs,
    _dfs,
    _load_graph,
    _score_nodes,
    _subgraph_to_text,
)

from config import config
from core.workspace_client import WorkspaceClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GRAPH_DIR = "/graph"
_GRAPH_JSON_PATH = f"{_GRAPH_DIR}/graph.json"
_META_JSON_PATH = f"{_GRAPH_DIR}/meta.json"
_COMMUNITIES_JSON_PATH = f"{_GRAPH_DIR}/communities.json"
_GRAPH_HTML_PATH = f"{_GRAPH_DIR}/graph.html"

_HISTORY_DIR = f"{_GRAPH_DIR}/history"
_REPORTS_DIR = f"{_GRAPH_DIR}/reports"
_LATEST_DIFF_PATH = f"{_GRAPH_DIR}/latest_diff.json"
_MAX_HISTORY_SNAPSHOTS = 30

_CACHE_MAX_SIZE = 20
_DEBOUNCE_SECONDS = 60


# ---------------------------------------------------------------------------
# GraphifyService
# ---------------------------------------------------------------------------


class GraphifyService:
    """Manages knowledge-graph lifecycle per workspace.

    - Builds NetworkX graphs via graphify from extracted sources
    - Caches loaded graphs in an LRU cache (max 20 workspaces)
    - Exports artefacts to workspace files
    - Provides debounced incremental rebuild scheduling
    """

    def __init__(self) -> None:
        self._cache: LRUCache[int, nx.Graph] = LRUCache(maxsize=_CACHE_MAX_SIZE)
        self._debounce_handles: Dict[int, asyncio.TimerHandle] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_graph(self, workspace_id: int) -> Dict[str, Any]:
        """Full graph build pipeline for a workspace.

        Returns a meta dict on success, or raises on failure.
        """
        logger.info("build_graph: starting for workspace %s", workspace_id)
        ws = WorkspaceClient(str(workspace_id))

        # 1. Collect sources from the extraction module
        sources = await self._collect_sources(workspace_id)
        if not sources:
            logger.warning("build_graph: no sources found for workspace %s", workspace_id)
            return {"node_count": 0, "edge_count": 0, "community_count": 0}

        # 2. Extract and merge — delegates to graph_extraction (future module)
        extractions = await self._extract_all(workspace_id, sources)
        merged = self._merge_extractions(extractions)

        # 3. Build graph (sync graphify call → executor)
        loop = asyncio.get_event_loop()
        graph: nx.Graph = await loop.run_in_executor(
            None, partial(build_from_json, merged)
        )
        logger.info(
            "build_graph: built graph with %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

        # 4. Cluster
        graph = await loop.run_in_executor(None, partial(cluster, graph))
        communities = await loop.run_in_executor(None, partial(score_all, graph))

        # 5. Analyze
        top_gods = await loop.run_in_executor(None, partial(god_nodes, graph))
        surprises = await loop.run_in_executor(
            None, partial(surprising_connections, graph)
        )

        # 6. Export artefacts
        await self._export_graph(ws, graph, communities, top_gods)

        # 7. Build and save meta
        meta = self._build_meta(graph, communities, top_gods)
        await self._write_json(ws, _META_JSON_PATH, meta)

        # 8. Cache the built graph
        self._cache[workspace_id] = graph

        # 9. Save history snapshot + compute diff (US-010)
        graph_data = await loop.run_in_executor(None, partial(to_json, graph))
        diff_result = await self._snapshot_and_diff(ws, graph, graph_data, loop)
        if diff_result is not None:
            meta["diff_summary"] = diff_result.get("summary", "")

        # 10. Generate build report
        today = date.today().isoformat()
        await self._write_build_report(ws, today, meta, diff_result)

        # 11. Prune old history snapshots
        await self._prune_history(ws)

        logger.info("build_graph: completed for workspace %s", workspace_id)
        return meta

    async def load_graph(self, workspace_id: int) -> Optional[nx.Graph]:
        """Load a workspace graph from LRU cache or workspace files.

        Returns None if no graph has been built yet.
        """
        # Check cache first
        cached = self._cache.get(workspace_id)
        if cached is not None:
            logger.debug("load_graph: cache hit for workspace %s", workspace_id)
            return cached

        # Try loading from workspace files
        ws = WorkspaceClient(str(workspace_id))
        result = await ws.read_file(_GRAPH_JSON_PATH)
        if not result.get("success"):
            logger.debug("load_graph: no graph file for workspace %s", workspace_id)
            return None

        content = result.get("content", "")
        if not content:
            return None

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("load_graph: corrupt graph.json for workspace %s: %s", workspace_id, exc)
            return None

        loop = asyncio.get_event_loop()
        graph: nx.Graph = await loop.run_in_executor(
            None, partial(nx.node_link_graph, data)
        )

        # Warm cache
        self._cache[workspace_id] = graph
        logger.info(
            "load_graph: loaded from file for workspace %s (%d nodes)",
            workspace_id,
            graph.number_of_nodes(),
        )
        return graph

    def invalidate_cache(self, workspace_id: int) -> None:
        """Remove a workspace graph from the LRU cache."""
        self._cache.pop(workspace_id, None)
        logger.debug("invalidate_cache: cleared workspace %s", workspace_id)

    def schedule_incremental_update(
        self, workspace_id: int, changed_sources: List[Dict[str, Any]]
    ) -> None:
        """Schedule a debounced rebuild.

        Multiple calls within ``_DEBOUNCE_SECONDS`` reset the timer so only
        one rebuild fires per window.
        """
        # Cancel existing timer if present
        existing = self._debounce_handles.pop(workspace_id, None)
        if existing is not None:
            existing.cancel()
            logger.debug(
                "schedule_incremental_update: reset debounce timer for workspace %s",
                workspace_id,
            )

        loop = asyncio.get_event_loop()
        handle = loop.call_later(
            _DEBOUNCE_SECONDS,
            lambda: asyncio.ensure_future(self._debounced_rebuild(workspace_id)),
        )
        self._debounce_handles[workspace_id] = handle
        logger.info(
            "schedule_incremental_update: rebuild scheduled in %ds for workspace %s "
            "(%d changed sources)",
            _DEBOUNCE_SECONDS,
            workspace_id,
            len(changed_sources),
        )

    async def get_meta(self, workspace_id: int) -> Optional[Dict[str, Any]]:
        """Read ``/graph/meta.json`` without loading the full graph."""
        ws = WorkspaceClient(str(workspace_id))
        result = await ws.read_file(_META_JSON_PATH)
        if not result.get("success"):
            return None

        content = result.get("content", "")
        if not content:
            return None

        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("get_meta: corrupt meta.json for workspace %s: %s", workspace_id, exc)
            return None

    # ------------------------------------------------------------------
    # Graph query helpers (thin wrappers around graphify.serve)
    # ------------------------------------------------------------------

    async def score_nodes(
        self, graph: nx.Graph, terms: List[str]
    ) -> List[Dict[str, Any]]:
        """Return scored node list via graphify.

        Args:
            graph: NetworkX graph to score.
            terms: Search terms to score nodes against.

        Returns list of dicts with ``score``, ``id``, and ``label`` keys,
        sorted highest-score first.
        """
        loop = asyncio.get_event_loop()
        # _score_nodes returns list[tuple[float, str]]
        raw: List = await loop.run_in_executor(
            None, partial(_score_nodes, graph, terms)
        )
        return [
            {
                "score": score,
                "id": node_id,
                "label": graph.nodes[node_id].get("label", node_id)
                if node_id in graph
                else node_id,
            }
            for score, node_id in raw
        ]

    async def bfs(
        self, graph: nx.Graph, start: str, depth: int = 2
    ) -> Dict[str, Any]:
        """BFS traversal from a starting node.

        Returns dict with ``nodes`` (set[str]) and ``edges`` (list[tuple]).
        """
        loop = asyncio.get_event_loop()
        # _bfs expects start_nodes as a list
        nodes, edges = await loop.run_in_executor(
            None, partial(_bfs, graph, [start], depth)
        )
        return {"nodes": nodes, "edges": edges}

    async def dfs(
        self, graph: nx.Graph, start: str, depth: int = 2
    ) -> Dict[str, Any]:
        """DFS traversal from a starting node.

        Returns dict with ``nodes`` (set[str]) and ``edges`` (list[tuple]).
        """
        loop = asyncio.get_event_loop()
        # _dfs expects start_nodes as a list
        nodes, edges = await loop.run_in_executor(
            None, partial(_dfs, graph, [start], depth)
        )
        return {"nodes": nodes, "edges": edges}

    async def subgraph_to_text(
        self,
        graph: nx.Graph,
        nodes: set,
        edges: list,
        token_budget: int = 2000,
    ) -> str:
        """Convert a subgraph (nodes + edges) to a text summary."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, partial(_subgraph_to_text, graph, nodes, edges, token_budget)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _collect_sources(self, workspace_id: int) -> List[Dict[str, Any]]:
        """Collect all indexable sources for a workspace.

        This is a seam for future integration with the document system,
        cloud sync, and other source providers. Currently returns an empty
        list — callers (build_graph) handle the empty case gracefully.
        """
        # TODO: PRD-126 US-002+ will wire this to document/cloud source providers
        logger.debug("_collect_sources: stub for workspace %s", workspace_id)
        return []

    async def _extract_all(
        self, workspace_id: int, sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run entity/relation extraction on each source.

        Delegates to the graph_extraction module (future US).
        Returns a list of extraction dicts ready for merging.
        """
        # TODO: PRD-126 — wire to graph_extraction module
        logger.debug(
            "_extract_all: stub for workspace %s (%d sources)",
            workspace_id,
            len(sources),
        )
        return sources

    @staticmethod
    def _merge_extractions(extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple extraction results into a single graph-ready dict.

        Combines nodes and edges from all extractions, deduplicating by id.
        """
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        for extraction in extractions:
            for node in extraction.get("nodes", []):
                node_id = node.get("id", "")
                if node_id:
                    # Later extraction wins on conflict — last-write-wins merge
                    nodes[node_id] = node
            edges.extend(extraction.get("edges", []))

        return {"nodes": list(nodes.values()), "edges": edges}

    async def _export_graph(
        self,
        ws: WorkspaceClient,
        graph: nx.Graph,
        communities: Any,
        god_node_list: List[Any],
    ) -> None:
        """Export graph artefacts to workspace files."""
        loop = asyncio.get_event_loop()

        # graph.json — node_link_data format
        graph_data = await loop.run_in_executor(
            None, partial(to_json, graph)
        )
        await self._write_json(ws, _GRAPH_JSON_PATH, graph_data)

        # communities.json
        community_data = self._format_communities(graph, communities)
        await self._write_json(ws, _COMMUNITIES_JSON_PATH, community_data)

        # graph.html — interactive visualization
        html_content: str = await loop.run_in_executor(
            None, partial(to_html, graph)
        )
        await ws.write_file(_GRAPH_HTML_PATH, html_content)

    @staticmethod
    def _format_communities(
        graph: nx.Graph, communities: Any
    ) -> List[Dict[str, Any]]:
        """Format community data for export.

        Extracts community labels from node attributes and groups members.
        """
        community_map: Dict[int, List[str]] = {}
        for node_id, attrs in graph.nodes(data=True):
            comm_id = attrs.get("community", -1)
            if comm_id not in community_map:
                community_map[comm_id] = []
            community_map[comm_id].append(str(node_id))

        return [
            {"community_id": cid, "member_count": len(members), "members": members}
            for cid, members in sorted(community_map.items())
        ]

    @staticmethod
    def _build_meta(
        graph: nx.Graph,
        communities: Any,
        god_node_list: List[Any],
    ) -> Dict[str, Any]:
        """Build the meta.json payload."""
        # Count unique communities from node attributes
        community_ids = {
            attrs.get("community", -1)
            for _, attrs in graph.nodes(data=True)
        }

        return {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "community_count": len(community_ids),
            "last_built": time.time(),
            "god_nodes": [
                str(g) if not isinstance(g, dict) else g
                for g in (god_node_list or [])
            ],
        }

    async def _snapshot_and_diff(
        self,
        ws: WorkspaceClient,
        graph: nx.Graph,
        graph_data: Any,
        loop: asyncio.AbstractEventLoop,
    ) -> Optional[Dict[str, Any]]:
        """Save today's graph snapshot and compute diff against previous.

        Returns the diff dict, or None on first build / error.
        """
        today = date.today().isoformat()
        history_path = f"{_HISTORY_DIR}/{today}_graph.json"

        # Save today's snapshot
        await self._write_json(ws, history_path, graph_data)

        # Find most recent previous snapshot (not today's)
        prev_data = await self._load_previous_snapshot(ws, today)
        if prev_data is None:
            logger.info("_snapshot_and_diff: first build — no previous snapshot")
            return None

        # Reconstruct previous graph and compute diff
        try:
            prev_graph: nx.Graph = await loop.run_in_executor(
                None, partial(nx.node_link_graph, prev_data)
            )
            diff: Dict[str, Any] = await loop.run_in_executor(
                None, partial(graph_diff, prev_graph, graph)
            )
        except Exception:
            logger.exception("_snapshot_and_diff: diff computation failed")
            return None

        await self._write_json(ws, _LATEST_DIFF_PATH, diff)
        logger.info(
            "_snapshot_and_diff: diff saved — %s",
            diff.get("summary", "no summary"),
        )
        return diff

    async def _load_previous_snapshot(
        self, ws: WorkspaceClient, exclude_date: str
    ) -> Optional[Dict[str, Any]]:
        """Find and load the most recent history snapshot before *exclude_date*."""
        dir_result = await ws.list_dir(_HISTORY_DIR)
        entries = dir_result.get("entries", [])
        if not entries:
            return None

        # Filter to *_graph.json files, excluding today's
        snapshot_names = sorted(
            [
                e.get("name", "") if isinstance(e, dict) else str(e)
                for e in entries
                if (
                    (e.get("name", "") if isinstance(e, dict) else str(e))
                    .endswith("_graph.json")
                    and not (e.get("name", "") if isinstance(e, dict) else str(e))
                    .startswith(exclude_date)
                )
            ],
            reverse=True,
        )

        if not snapshot_names:
            return None

        # Load the most recent
        prev_path = f"{_HISTORY_DIR}/{snapshot_names[0]}"
        result = await ws.read_file(prev_path)
        if not result.get("success"):
            return None

        content = result.get("content", "")
        if not content:
            return None

        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            logger.error("_load_previous_snapshot: corrupt snapshot %s", prev_path)
            return None

    async def _write_build_report(
        self,
        ws: WorkspaceClient,
        today: str,
        meta: Dict[str, Any],
        diff: Optional[Dict[str, Any]],
    ) -> None:
        """Write a human-readable markdown build report."""
        report_path = f"{_REPORTS_DIR}/{today}_build.md"

        lines = [
            "# Graph Build Report",
            f"**Date:** {today}",
            "",
            "## Graph Stats",
            f"- Nodes: {meta.get('node_count', 0)}",
            f"- Edges: {meta.get('edge_count', 0)}",
            f"- Communities: {meta.get('community_count', 0)}",
            "",
        ]

        if diff is not None:
            new_nodes = diff.get("new_nodes", [])
            removed_nodes = diff.get("removed_nodes", [])
            new_edges = diff.get("new_edges", [])
            removed_edges = diff.get("removed_edges", [])
            lines.extend([
                "## Changes",
                f"- {len(new_nodes)} new node(s)",
                f"- {len(removed_nodes)} removed node(s)",
                f"- {len(new_edges)} new edge(s)",
                f"- {len(removed_edges)} removed edge(s)",
                "",
            ])
            summary = diff.get("summary", "")
            if summary:
                lines.extend(["## Summary", summary, ""])
        else:
            lines.extend(["## Changes", "First build — no previous snapshot to compare.", ""])

        await ws.write_file(report_path, "\n".join(lines))

    async def _prune_history(self, ws: WorkspaceClient) -> None:
        """Keep at most ``_MAX_HISTORY_SNAPSHOTS`` in /graph/history/.

        Deletes oldest snapshots first.  If WorkspaceClient lacks a
        delete_file method the excess files are logged but not removed.
        """
        dir_result = await ws.list_dir(_HISTORY_DIR)
        entries = dir_result.get("entries", [])
        if not entries:
            return

        snapshot_names = sorted(
            e.get("name", "") if isinstance(e, dict) else str(e)
            for e in entries
            if (e.get("name", "") if isinstance(e, dict) else str(e)).endswith("_graph.json")
        )

        excess_count = len(snapshot_names) - _MAX_HISTORY_SNAPSHOTS
        if excess_count <= 0:
            return

        to_delete = snapshot_names[:excess_count]

        if hasattr(ws, "delete_file"):
            for name in to_delete:
                await ws.delete_file(f"{_HISTORY_DIR}/{name}")  # type: ignore[attr-defined]
            logger.info("_prune_history: deleted %d old snapshot(s)", len(to_delete))
        else:
            logger.warning(
                "_prune_history: %d snapshot(s) exceed limit but "
                "WorkspaceClient.delete_file() is not available: %s",
                len(to_delete),
                to_delete,
            )

    async def _debounced_rebuild(self, workspace_id: int) -> None:
        """Execute the deferred rebuild after debounce window expires."""
        self._debounce_handles.pop(workspace_id, None)
        logger.info("_debounced_rebuild: executing for workspace %s", workspace_id)
        try:
            self.invalidate_cache(workspace_id)
            await self.build_graph(workspace_id)
        except Exception:
            logger.exception(
                "_debounced_rebuild: failed for workspace %s", workspace_id
            )

    @staticmethod
    async def _write_json(
        ws: WorkspaceClient, path: str, data: Any
    ) -> None:
        """Serialize data to JSON and write to workspace."""
        content = json.dumps(data, default=str, indent=2)
        result = await ws.write_file(path, content)
        if not result.get("success"):
            logger.error(
                "_write_json: failed to write %s: %s",
                path,
                result.get("error", "unknown"),
            )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_service: Optional[GraphifyService] = None


def get_graph_service() -> GraphifyService:
    """Return the module-level GraphifyService singleton."""
    global _service
    if _service is None:
        _service = GraphifyService()
    return _service
