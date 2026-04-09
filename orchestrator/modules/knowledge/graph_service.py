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
import os
import tempfile
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

_GRAPH_DIR = "graph"
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
# Team scoping (PRD-124 integration)
# ---------------------------------------------------------------------------


def node_is_visible(graph: nx.Graph, node_id: str, agent_team: Optional[str]) -> bool:
    """Check if a node is visible to an agent based on team_access.

    PRD-124 filtering rule:
      - team_access == []  → visible to all agents
      - agent_team is None → agent sees everything (e.g. AUTO/CTO)
      - agent_team in team_access → visible
      - otherwise → hidden

    Returns True if the node should be visible.
    """
    if agent_team is None:
        return True
    attrs = graph.nodes.get(node_id, {})
    team_access = attrs.get("team_access", [])
    if not team_access:
        return True
    return agent_team in team_access


def team_filtered_view(graph: nx.Graph, agent_team: Optional[str]) -> nx.Graph:
    """Return a subgraph containing only nodes visible to *agent_team*.

    If agent_team is None (no team restriction), returns the original graph
    unchanged. Otherwise creates a filtered view that excludes team-blocked
    nodes and all edges touching them.
    """
    if agent_team is None:
        return graph
    visible = [
        n for n in graph.nodes
        if node_is_visible(graph, n, agent_team)
    ]
    return graph.subgraph(visible).copy()


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
        self._cache: LRUCache[str, nx.Graph] = LRUCache(maxsize=_CACHE_MAX_SIZE)
        self._debounce_handles: Dict[str, asyncio.TimerHandle] = {}
        # Accumulate changed source IDs across debounce window
        self._pending_sources: Dict[str, List[Dict[str, Any]]] = {}
        # Per-workspace build lock — prevents concurrent builds clobbering each other
        self._build_locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, workspace_id: str) -> asyncio.Lock:
        """Return (or create) the per-workspace build lock."""
        if workspace_id not in self._build_locks:
            self._build_locks[workspace_id] = asyncio.Lock()
        return self._build_locks[workspace_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_graph(self, workspace_id: str) -> Dict[str, Any]:
        """Full graph build pipeline for a workspace.

        Returns a meta dict on success, or raises on failure.
        """
        async with self._lock_for(workspace_id):
            return await self._build_graph_unlocked(workspace_id)

    async def _build_graph_unlocked(self, workspace_id: str) -> Dict[str, Any]:
        """Full graph build (caller must hold the workspace lock)."""
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

        # 4. Cluster — returns {community_id: [node_ids]} (does NOT mutate graph)
        communities = await loop.run_in_executor(None, partial(cluster, graph))
        community_scores = await loop.run_in_executor(
            None, partial(score_all, graph, communities)
        )

        # 5. Analyze
        top_gods = await loop.run_in_executor(None, partial(god_nodes, graph))
        surprises = await loop.run_in_executor(
            None, partial(surprising_connections, graph, communities)
        )

        # 6. Export artefacts
        await self._export_graph(ws, graph, communities, top_gods)

        # 7. Build and save meta
        meta = self._build_meta(graph, communities, top_gods)
        await self._write_json(ws, _META_JSON_PATH, meta)

        # 8. Cache the built graph
        self._cache[workspace_id] = graph

        # 9. Save history snapshot + compute diff (US-010)
        graph_data = await loop.run_in_executor(
            None, partial(nx.node_link_data, graph)
        )
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

    async def load_graph(self, workspace_id: str) -> Optional[nx.Graph]:
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

    def invalidate_cache(self, workspace_id: str) -> None:
        """Remove a workspace graph from the LRU cache."""
        self._cache.pop(workspace_id, None)
        logger.debug("invalidate_cache: cleared workspace %s", workspace_id)

    async def import_graph(
        self, workspace_id: str, graph_data: Dict[str, Any], *, merge: bool = False
    ) -> Dict[str, Any]:
        """Import a graphify graph.json into a workspace.

        Accepts NetworkX node_link_data format (same as graphify's graph.json).
        Re-runs clustering, analysis, and exports all derived artefacts.

        Args:
            workspace_id: Target workspace.
            graph_data: Parsed JSON from a graphify graph.json file.
            merge: If True, merge into existing workspace graph instead of replacing.

        Returns:
            Meta dict with node_count, edge_count, community_count.
        """
        async with self._lock_for(workspace_id):
            return await self._import_graph_unlocked(
                workspace_id, graph_data, merge=merge
            )

    async def _import_graph_unlocked(
        self, workspace_id: str, graph_data: Dict[str, Any], *, merge: bool = False
    ) -> Dict[str, Any]:
        """Import graph (caller must hold the workspace lock)."""
        logger.info(
            "import_graph: importing for workspace %s (merge=%s, nodes=%d)",
            workspace_id, merge, len(graph_data.get("nodes", [])),
        )
        ws = WorkspaceClient(str(workspace_id))
        loop = asyncio.get_event_loop()

        # Parse the imported graph
        imported: nx.Graph = await loop.run_in_executor(
            None, partial(nx.node_link_graph, graph_data)
        )

        if imported.number_of_nodes() == 0:
            raise ValueError("Imported graph.json contains no nodes")

        # Merge with existing graph if requested
        if merge:
            existing = await self.load_graph(workspace_id)
            if existing is not None:
                # Add new nodes/edges into the existing graph
                for node, attrs in imported.nodes(data=True):
                    if node not in existing:
                        existing.add_node(node, **attrs)
                for u, v, attrs in imported.edges(data=True):
                    if not existing.has_edge(u, v):
                        existing.add_edge(u, v, **attrs)
                graph = existing
                logger.info(
                    "import_graph: merged — %d nodes, %d edges",
                    graph.number_of_nodes(), graph.number_of_edges(),
                )
            else:
                graph = imported
        else:
            graph = imported

        # Re-run the same pipeline as build_graph (steps 4-11)
        communities = await loop.run_in_executor(None, partial(cluster, graph))
        await loop.run_in_executor(None, partial(score_all, graph, communities))
        top_gods = await loop.run_in_executor(None, partial(god_nodes, graph))

        await self._export_graph(ws, graph, communities, top_gods)

        meta = self._build_meta(graph, communities, top_gods)
        meta["source"] = "import"
        await self._write_json(ws, _META_JSON_PATH, meta)

        self._cache[workspace_id] = graph

        # Snapshot + diff
        exported_data = await loop.run_in_executor(
            None, partial(nx.node_link_data, graph)
        )
        diff_result = await self._snapshot_and_diff(ws, graph, exported_data, loop)
        if diff_result is not None:
            meta["diff_summary"] = diff_result.get("summary", "")

        today = date.today().isoformat()
        await self._write_build_report(ws, today, meta, diff_result)
        await self._prune_history(ws)

        logger.info("import_graph: completed for workspace %s", workspace_id)
        return meta

    def schedule_incremental_update(
        self, workspace_id: str, changed_sources: List[Dict[str, Any]]
    ) -> None:
        """Schedule a debounced rebuild.

        Multiple calls within ``_DEBOUNCE_SECONDS`` reset the timer so only
        one rebuild fires per window.  Changed source metadata is accumulated
        so the rebuild can run incrementally instead of re-extracting every
        document.
        """
        # Accumulate changed sources across the debounce window
        if workspace_id not in self._pending_sources:
            self._pending_sources[workspace_id] = []
        self._pending_sources[workspace_id].extend(changed_sources)

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

    async def get_meta(self, workspace_id: str) -> Optional[Dict[str, Any]]:
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

    async def _collect_sources(
        self,
        workspace_id: str,
        doc_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Collect indexable sources for a workspace.

        Args:
            workspace_id: Target workspace.
            doc_ids: If provided, only collect these document IDs
                (incremental mode).  ``None`` means collect everything
                (full rebuild).

        Source types returned:
          - ``document`` — text content assembled from document_chunks
          - ``agents``   — agent roster rows for deterministic mapping
        """
        from sqlalchemy import text as sa_text

        from core.database.database import get_db_session
        from core.models.core import Agent, Document

        _MAX_DOC_CHARS = 8000  # cap text sent to LLM extraction

        sources: List[Dict[str, Any]] = []

        try:
            with get_db_session() as db:
                # --- Documents with chunk content -------------------------
                query = db.query(Document).filter(
                    Document.workspace_id == workspace_id,
                    Document.status.in_(["completed", "processed"]),
                )
                if doc_ids is not None:
                    query = query.filter(Document.id.in_(doc_ids))
                docs = query.all()

                for doc in docs:
                    # Assemble full text from chunks (ordered by chunk_index)
                    rows = db.execute(
                        sa_text(
                            "SELECT content FROM document_chunks "
                            "WHERE document_id = :doc_id "
                            "ORDER BY chunk_index"
                        ),
                        {"doc_id": doc.id},
                    ).fetchall()
                    full_text = "\n\n".join(r[0] for r in rows if r[0])
                    if not full_text.strip():
                        logger.debug(
                            "_collect_sources: skipping doc %s (no chunk text)",
                            doc.id,
                        )
                        continue

                    # Cap text to avoid LLM truncation on large docs
                    if len(full_text) > _MAX_DOC_CHARS:
                        logger.debug(
                            "_collect_sources: capping doc %s from %d to %d chars",
                            doc.id,
                            len(full_text),
                            _MAX_DOC_CHARS,
                        )
                        full_text = full_text[:_MAX_DOC_CHARS]

                    sources.append({
                        "type": "document",
                        "id": doc.id,
                        "path": doc.original_filename or doc.filename or f"doc_{doc.id}",
                        "text": full_text,
                        "team_access": list(doc.team_access or []),
                    })

                # --- Agent roster (only on full rebuild) ------------------
                if doc_ids is None:
                    agents = (
                        db.query(Agent)
                        .filter(
                            Agent.workspace_id == workspace_id,
                            Agent.status == "active",
                        )
                        .all()
                    )
                    if agents:
                        agent_rows = [
                            {
                                "id": a.id,
                                "name": a.name,
                                "role": a.agent_type,
                                "reports_to": (a.configuration or {}).get("reports_to"),
                            }
                            for a in agents
                        ]
                        sources.append({"type": "agents", "rows": agent_rows})

        except Exception:
            logger.exception(
                "_collect_sources: failed to query DB for workspace %s",
                workspace_id,
            )
            return []

        logger.info(
            "_collect_sources: found %d sources for workspace %s "
            "(docs=%d, agent_roster=%s)",
            len(sources),
            workspace_id,
            sum(1 for s in sources if s["type"] == "document"),
            "yes" if any(s["type"] == "agents" for s in sources) else "no",
        )
        return sources

    async def _extract_all(
        self, workspace_id: str, sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run entity/relation extraction on each source.

        Dispatches each source to the appropriate extractor:
          - ``document`` → LLM-based ``extract_from_document``
          - ``agents``   → deterministic ``map_agent_roster``

        Returns a list of extraction dicts (each with nodes/edges keys).
        """
        from modules.knowledge.graph_extraction import (
            extract_from_document,
            map_agent_roster,
        )

        extractions: List[Dict[str, Any]] = []

        for source in sources:
            src_type = source.get("type", "")
            try:
                if src_type == "document":
                    extraction = await extract_from_document(
                        doc_text=source["text"],
                        doc_path=source["path"],
                        workspace_id=int(workspace_id) if workspace_id.isdigit() else 0,
                        team_access=source.get("team_access"),
                    )
                    extractions.append(extraction)
                    logger.debug(
                        "_extract_all: doc '%s' → %d nodes, %d edges",
                        source["path"],
                        len(extraction.get("nodes", [])),
                        len(extraction.get("edges", [])),
                    )

                elif src_type == "agents":
                    extraction = map_agent_roster(source["rows"])
                    extractions.append(extraction)
                    logger.debug(
                        "_extract_all: agent roster → %d nodes, %d edges",
                        len(extraction.get("nodes", [])),
                        len(extraction.get("edges", [])),
                    )

                else:
                    logger.warning(
                        "_extract_all: unknown source type '%s', skipping",
                        src_type,
                    )
            except Exception:
                logger.exception(
                    "_extract_all: extraction failed for source type '%s'",
                    src_type,
                )

        logger.info(
            "_extract_all: completed %d/%d extractions for workspace %s",
            len(extractions),
            len(sources),
            workspace_id,
        )
        return extractions

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
        communities: Dict[int, List[str]],
        god_node_list: List[Any],
    ) -> None:
        """Export graph artefacts to workspace files."""
        loop = asyncio.get_event_loop()

        # graph.json — NetworkX node_link_data (in-memory, not graphify's to_json)
        graph_data = await loop.run_in_executor(
            None, partial(nx.node_link_data, graph)
        )
        await self._write_json(ws, _GRAPH_JSON_PATH, graph_data)

        # communities.json
        community_data = self._format_communities(communities)
        await self._write_json(ws, _COMMUNITIES_JSON_PATH, community_data)

        # graph.html — use graphify's to_html (writes to temp file, then upload)
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "graph.html")
            await loop.run_in_executor(
                None, partial(to_html, graph, communities, html_path)
            )
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        await ws.write_file(_GRAPH_HTML_PATH, html_content)

    @staticmethod
    def _format_communities(
        communities: Dict[int, List[str]],
    ) -> List[Dict[str, Any]]:
        """Format community data for export.

        Takes the communities dict from ``cluster()`` (maps community_id → member list).
        """
        return [
            {"community_id": cid, "member_count": len(members), "members": members}
            for cid, members in sorted(communities.items())
        ]

    @staticmethod
    def _build_meta(
        graph: nx.Graph,
        communities: Dict[int, List[str]],
        god_node_list: List[Any],
    ) -> Dict[str, Any]:
        """Build the meta.json payload."""
        return {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "community_count": len(communities),
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

    async def _debounced_rebuild(self, workspace_id: str) -> None:
        """Execute the deferred rebuild after debounce window expires.

        Acquires the per-workspace build lock so concurrent rebuilds cannot
        clobber each other.  If a graph already exists, runs incremental;
        otherwise falls back to a full rebuild.
        """
        self._debounce_handles.pop(workspace_id, None)
        pending = self._pending_sources.pop(workspace_id, [])
        logger.info(
            "_debounced_rebuild: executing for workspace %s (%d pending sources)",
            workspace_id,
            len(pending),
        )
        try:
            async with self._lock_for(workspace_id):
                # Invalidate cache and reload from file so we read the
                # latest version written by a previous build.
                self.invalidate_cache(workspace_id)
                existing_graph = await self.load_graph(workspace_id)
                if existing_graph is not None and pending:
                    await self._incremental_build(workspace_id, existing_graph, pending)
                else:
                    self.invalidate_cache(workspace_id)
                    await self._build_graph_unlocked(workspace_id)
        except Exception:
            logger.exception(
                "_debounced_rebuild: failed for workspace %s", workspace_id
            )

    async def _incremental_build(
        self,
        workspace_id: str,
        existing_graph: nx.Graph,
        pending: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Incremental graph update — only extracts changed documents.

        1. Collect ONLY the documents referenced in *pending*
        2. Extract entities/relations from those docs
        3. Merge new nodes/edges into the existing graph
        4. Re-cluster, re-export, update cache
        """
        logger.info(
            "_incremental_build: %d changed sources for workspace %s",
            len(pending),
            workspace_id,
        )
        ws = WorkspaceClient(str(workspace_id))

        # Collect only the changed document IDs
        changed_doc_ids = {
            s.get("id") for s in pending
            if s.get("type") == "document" and s.get("id")
        }

        if not changed_doc_ids:
            logger.info("_incremental_build: no document IDs in pending, skipping")
            return {"node_count": existing_graph.number_of_nodes(),
                    "edge_count": existing_graph.number_of_edges()}

        # Collect only those docs from DB
        sources = await self._collect_sources(
            workspace_id, doc_ids=changed_doc_ids
        )
        if not sources:
            logger.info("_incremental_build: changed docs not found in DB, skipping")
            return {"node_count": existing_graph.number_of_nodes(),
                    "edge_count": existing_graph.number_of_edges()}

        # Extract and merge new nodes/edges
        extractions = await self._extract_all(workspace_id, sources)
        merged = self._merge_extractions(extractions)

        # Add new nodes and edges to existing graph
        for node in merged.get("nodes", []):
            node_id = node.get("id", "")
            if node_id:
                existing_graph.add_node(node_id, **node)

        for edge in merged.get("edges", []):
            src = edge.get("source", edge.get("_src", ""))
            tgt = edge.get("target", edge.get("_tgt", ""))
            if src and tgt and src in existing_graph and tgt in existing_graph:
                existing_graph.add_edge(src, tgt, **edge)

        logger.info(
            "_incremental_build: graph now has %d nodes, %d edges",
            existing_graph.number_of_nodes(),
            existing_graph.number_of_edges(),
        )

        # Re-cluster and re-export (same as full build steps 4-11)
        loop = asyncio.get_event_loop()
        communities = await loop.run_in_executor(
            None, partial(cluster, existing_graph)
        )
        top_gods = await loop.run_in_executor(
            None, partial(god_nodes, existing_graph)
        )

        await self._export_graph(ws, existing_graph, communities, top_gods)

        meta = self._build_meta(existing_graph, communities, top_gods)
        await self._write_json(ws, _META_JSON_PATH, meta)

        self._cache[workspace_id] = existing_graph

        graph_data = await loop.run_in_executor(
            None, partial(nx.node_link_data, existing_graph)
        )
        diff_result = await self._snapshot_and_diff(
            ws, existing_graph, graph_data, loop
        )
        if diff_result is not None:
            meta["diff_summary"] = diff_result.get("summary", "")

        today = date.today().isoformat()
        await self._write_build_report(ws, today, meta, diff_result)
        await self._prune_history(ws)

        logger.info(
            "_incremental_build: completed for workspace %s "
            "(+%d sources, %d total nodes)",
            workspace_id,
            len(sources),
            existing_graph.number_of_nodes(),
        )
        return meta

    @staticmethod
    async def _write_json(
        ws: WorkspaceClient, path: str, data: Any
    ) -> None:
        """Serialize data to JSON and write to workspace.

        Raises RuntimeError if the workspace worker rejects the write so
        callers know the export failed instead of silently losing data.
        """
        content = json.dumps(data, default=str, indent=2)
        result = await ws.write_file(path, content)
        if not result.get("success"):
            error_msg = result.get("error", "unknown")
            logger.error("_write_json: failed to write %s: %s", path, error_msg)
            raise RuntimeError(f"Failed to write {path}: {error_msg}")


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
