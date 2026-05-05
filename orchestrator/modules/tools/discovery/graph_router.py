"""
GraphRouter (PRD-139 US-004)
=============================

Ranks tool chains using the tool routing graph built by the edge builder (US-003).
Wraps ActionSemanticIndex.rank_actions() for entry node selection -- does NOT
reimplement embedding search. Falls back to pure embedding ranking when the graph
is empty or unavailable.

Cache: traversal results cached in CacheService for 5 minutes keyed on
(query_embedding_hash, agent_id, top_k).
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import List, Optional, Tuple

from .action_semantic_index import get_action_semantic_index

logger = logging.getLogger(__name__)

# Hard limits per the PRD
_MAX_DEPTH = 2  # single hop from entry node
_MAX_EXPANDED_NODES = 50
_ENTRY_TOP_K = 5
_CACHE_TTL_SECONDS = 300  # 5 minutes


class GraphRouter:
    """Ranks tool chains by combining cosine similarity with graph edges/affinities."""

    def __init__(self) -> None:
        self._semantic_index = get_action_semantic_index()

    # ------------------------------------------------------------------
    # Config accessors -- safe defaults until US-005 adds them to config.py
    # ------------------------------------------------------------------

    @staticmethod
    def _min_confidence() -> float:
        try:
            from config import config
            return getattr(config, "TOOL_ROUTING_GRAPH_MIN_CONFIDENCE", 0.6)
        except Exception:
            return 0.6

    @staticmethod
    def _agent_sample_floor() -> int:
        try:
            from config import config
            return getattr(config, "TOOL_ROUTING_GRAPH_AGENT_SAMPLE_FLOOR", 50)
        except Exception:
            return 50

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(query: str, agent_id: Optional[int], top_k: int) -> str:
        raw = json.dumps({"q": query, "a": agent_id, "k": top_k}, sort_keys=True)
        h = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"cache:graph_router:{h}"

    def _get_cache(self):
        """Lazy-fetch CacheService (avoids import-time side effects)."""
        try:
            from core.cache.service import get_cache_service
            return get_cache_service()
        except Exception:
            return None

    def _read_cache(self, key: str) -> Optional[List[Tuple[str, float, List[str]]]]:
        cache = self._get_cache()
        if cache is None:
            return None
        try:
            raw = cache.redis.get(key)
            if raw is None:
                return None
            data = json.loads(raw)
            return [(r[0], r[1], r[2]) for r in data]
        except Exception:
            return None

    def _write_cache(
        self,
        key: str,
        result: List[Tuple[str, float, List[str]]],
    ) -> None:
        cache = self._get_cache()
        if cache is None:
            return
        try:
            cache.redis.setex(key, _CACHE_TTL_SECONDS, json.dumps(result))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rank_chains(
        self,
        query: str,
        agent_id: Optional[int] = None,
        top_k: int = 15,
        exclude_admin: bool = True,
        exclude_promoted: bool = True,
    ) -> List[Tuple[str, float, List[str]]]:
        """Rank tool chains by combining embedding similarity with graph edges.

        Returns:
            List of (primary_action, score, chain_actions) sorted descending by score.
        """
        cache_key = self._cache_key(query, agent_id, top_k)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        # Step 1: entry nodes from ActionSemanticIndex
        entry_nodes = await self._semantic_index.rank_actions(
            query,
            top_k=_ENTRY_TOP_K,
            exclude_admin=exclude_admin,
            exclude_promoted=exclude_promoted,
        )
        if not entry_nodes:
            return []

        # Step 2: expand through graph edges + affinities
        try:
            chains = self._expand_with_graph(entry_nodes, agent_id)
        except Exception as e:
            logger.warning("GraphRouter: graph expansion failed, falling back to embedding-only: %s", e)
            chains = self._to_single_chains(entry_nodes)

        # Step 3: deduplicate by action set, keep highest
        chains = self._deduplicate(chains)

        # Step 4: sort + truncate
        chains.sort(key=lambda x: x[1], reverse=True)
        result = chains[:top_k]

        self._write_cache(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Graph expansion
    # ------------------------------------------------------------------

    def _expand_with_graph(
        self,
        entry_nodes: List[Tuple[str, float]],
        agent_id: Optional[int],
    ) -> List[Tuple[str, float, List[str]]]:
        """Query edge + affinity tables and build scored chains."""
        from core.database.database import get_db_session

        min_conf = self._min_confidence()
        sample_floor = self._agent_sample_floor()

        chains: List[Tuple[str, float, List[str]]] = []

        # Always include single-action chains from entry nodes
        chains.extend(self._to_single_chains(entry_nodes))

        entry_action_names = [name for name, _ in entry_nodes]
        cosine_by_name = {name: score for name, score in entry_nodes}

        with get_db_session() as db:
            # Determine whether to use agent-specific edges
            use_agent_scope = False
            if agent_id is not None:
                agent_sample_count = self._agent_total_samples(db, agent_id)
                use_agent_scope = agent_sample_count >= sample_floor

            # Batch-query edges for all entry nodes
            edges = self._query_edges(
                db, entry_action_names, min_conf,
                agent_id if use_agent_scope else None,
            )

            # Batch-query affinities for entry + expansion targets
            all_action_names = set(entry_action_names)
            for edge in edges:
                all_action_names.add(edge["to_action"])
            affinities = self._query_affinities(
                db, list(all_action_names),
                agent_id if use_agent_scope else None,
            )

        # Build chains from edges (depth 1 only -- _MAX_DEPTH = 2 means
        # chains of length 2: [entry, next])
        expanded = 0
        for edge in edges:
            if expanded >= _MAX_EXPANDED_NODES:
                break
            from_action = edge["from_action"]
            to_action = edge["to_action"]
            cosine = cosine_by_name.get(from_action, 0.0)
            edge_confidence = edge["confidence"]

            # Affinity boosts for the chain actions
            boost = 0.0
            for action in (from_action, to_action):
                boost += affinities.get(action, 0.0)

            score = cosine * edge_confidence + boost
            chains.append((from_action, score, [from_action, to_action]))
            expanded += 1

        return chains

    # ------------------------------------------------------------------
    # DB queries
    # ------------------------------------------------------------------

    @staticmethod
    def _agent_total_samples(db, agent_id: int) -> int:
        """Count total edge samples for this agent to decide scope."""
        from sqlalchemy import func
        from core.models.tool_routing import ToolRoutingEdge

        result = (
            db.query(func.coalesce(func.sum(ToolRoutingEdge.sample_count), 0))
            .filter(ToolRoutingEdge.agent_id == agent_id)
            .scalar()
        )
        return int(result)

    @staticmethod
    def _query_edges(
        db,
        from_actions: List[str],
        min_confidence: float,
        agent_id: Optional[int],
    ) -> List[dict]:
        """Query tool_routing_edges for used_after edges from entry nodes."""
        from sqlalchemy import and_, or_
        from core.models.tool_routing import ToolRoutingEdge

        if not from_actions:
            return []

        filters = [
            ToolRoutingEdge.from_action.in_(from_actions),
            ToolRoutingEdge.edge_type == "used_after",
            ToolRoutingEdge.confidence >= min_confidence,
        ]

        if agent_id is not None:
            # Agent-scoped: include edges for this agent OR global (agent_id IS NULL)
            filters.append(
                or_(
                    ToolRoutingEdge.agent_id == agent_id,
                    ToolRoutingEdge.agent_id.is_(None),
                )
            )
        else:
            # Global only
            filters.append(ToolRoutingEdge.agent_id.is_(None))

        rows = (
            db.query(ToolRoutingEdge)
            .filter(and_(*filters))
            .order_by(ToolRoutingEdge.confidence.desc())
            .limit(_MAX_EXPANDED_NODES)
            .all()
        )

        return [
            {
                "from_action": r.from_action,
                "to_action": r.to_action,
                "confidence": r.confidence,
                "weight": r.weight,
                "agent_id": r.agent_id,
            }
            for r in rows
        ]

    @staticmethod
    def _query_affinities(
        db,
        action_names: List[str],
        agent_id: Optional[int],
    ) -> dict:
        """Query tool_routing_affinities and return action_name -> total boost."""
        from sqlalchemy import and_, or_
        from core.models.tool_routing import ToolRoutingAffinity

        if not action_names:
            return {}

        filters = [
            ToolRoutingAffinity.action_name.in_(action_names),
        ]

        if agent_id is not None:
            filters.append(
                or_(
                    ToolRoutingAffinity.agent_id == agent_id,
                    ToolRoutingAffinity.agent_id.is_(None),
                )
            )
        else:
            filters.append(ToolRoutingAffinity.agent_id.is_(None))

        rows = (
            db.query(ToolRoutingAffinity)
            .filter(and_(*filters))
            .all()
        )

        boosts: dict = {}
        for r in rows:
            # succeeds_for_intent adds positive boost; fails subtracts
            if r.affinity_type == "succeeds_for_intent":
                boosts[r.action_name] = boosts.get(r.action_name, 0.0) + r.weight * r.confidence
            elif r.affinity_type == "fails_for_intent":
                boosts[r.action_name] = boosts.get(r.action_name, 0.0) - r.weight * r.confidence
            elif r.affinity_type == "agent_prefers":
                boosts[r.action_name] = boosts.get(r.action_name, 0.0) + r.weight * r.confidence

        return boosts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_single_chains(
        entry_nodes: List[Tuple[str, float]],
    ) -> List[Tuple[str, float, List[str]]]:
        """Convert embedding-only results to single-action chains."""
        return [(name, score, [name]) for name, score in entry_nodes]

    @staticmethod
    def _deduplicate(
        chains: List[Tuple[str, float, List[str]]],
    ) -> List[Tuple[str, float, List[str]]]:
        """Deduplicate by action set, keeping the highest-scored version."""
        seen: dict = {}
        for primary, score, actions in chains:
            key = frozenset(actions)
            if key not in seen or score > seen[key][1]:
                seen[key] = (primary, score, actions)
        return list(seen.values())


# ======================================================================
# Singleton factory
# ======================================================================

_instance_lock = threading.Lock()
_instance: Optional[GraphRouter] = None


def get_graph_router() -> GraphRouter:
    """Process-singleton factory (matches get_action_semantic_index pattern)."""
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is None:
            _instance = GraphRouter()
    return _instance
