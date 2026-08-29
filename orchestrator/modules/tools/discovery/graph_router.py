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
import math
import threading
from typing import Dict, List, Optional, Tuple

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

    @staticmethod
    def _cluster_match_threshold() -> float:
        # PRD-232 US-010: cosine floor for assigning a query to an intent cluster.
        try:
            from config import config
            return float(getattr(config, "TOOL_ROUTING_GRAPH_CLUSTER_MATCH_THRESHOLD", 0.6))
        except Exception:
            return 0.6

    @staticmethod
    def _global_affinity_discount() -> float:
        # PRD-232 US-010: weight for cluster-blind (intent_cluster_id IS NULL)
        # affinity rows when a cluster matched — a weak global prior.
        try:
            from config import config
            return float(getattr(config, "TOOL_ROUTING_GRAPH_GLOBAL_AFFINITY_DISCOUNT", 0.5))
        except Exception:
            return 0.5

    @staticmethod
    def _failed_after_penalty_weight() -> float:
        # PRD-232 US-010c: scale for the failed_after de-ranking penalty.
        try:
            from config import config
            return float(getattr(config, "TOOL_ROUTING_GRAPH_FAILED_AFTER_PENALTY", 1.0))
        except Exception:
            return 1.0

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(
        query: str,
        agent_id: Optional[int],
        top_k: int,
        include_super_admin: bool = False,
        workspace_id: Optional[str] = None,
    ) -> str:
        # PRD-143: the su flag is part of the key — a super-admin result
        # cached under an operator key (or vice versa) would cross the tier.
        # PRD-177 S5: workspace_id is part of the key — a per-tenant graph result
        # cached under one workspace must never be served to another (moat leak).
        raw = json.dumps(
            {
                "q": query,
                "a": agent_id,
                "k": top_k,
                "su": include_super_admin,
                "ws": str(workspace_id) if workspace_id else None,
            },
            sort_keys=True,
        )
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
        *,
        workspace_id: Optional[str],
        agent_id: Optional[int] = None,
        top_k: int = 15,
        exclude_admin: bool = True,
        exclude_promoted: bool = True,
        include_super_admin: bool = False,
    ) -> List[Tuple[str, float, List[str]]]:
        """Rank tool chains by combining embedding similarity with graph edges.

        PRD-177 S5: ``workspace_id`` is a REQUIRED keyword. The learned operating
        graph is per-tenant (owner decision) — edge/affinity reads are filtered
        to this workspace, and there is no unfiltered global-read fallback that
        would bleed one tenant's edges into another's routing. Pass the caller's
        workspace id, or ``None`` explicitly for a genuinely unscoped read (the
        offline eval harness); the keyword is required so no caller can silently
        reintroduce a global read.

        PRD-143: fail-closed — super_admin_only actions are excluded from
        entry nodes AND from edge-expansion targets unless
        include_super_admin=True is passed explicitly.

        Returns:
            List of (primary_action, score, chain_actions) sorted descending by score.
        """
        cache_key = self._cache_key(query, agent_id, top_k, include_super_admin, workspace_id)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        # Step 1: entry nodes from ActionSemanticIndex. PRD-232 US-003: pass
        # workspace_id so this entry rank reuses the turn's shared cosine
        # ranking (the graph slices its top-5 from the one computation the
        # dispatcher narrowing / catalog already ran).
        entry_nodes = await self._semantic_index.rank_actions(
            query,
            top_k=_ENTRY_TOP_K,
            exclude_admin=exclude_admin,
            exclude_promoted=exclude_promoted,
            include_super_admin=include_super_admin,
            workspace_id=workspace_id,
        )
        if not entry_nodes:
            return []

        # Step 1.5 (PRD-232 US-010): resolve the query vector for intent-cluster
        # matching. Reuses the semantic index's bounded/cached embed (a Redis hit
        # on the vector the entry-node ranking just computed). None = no vector →
        # _expand_with_graph skips cluster matching (embedding floor only).
        query_vec, model_key = await self._match_query_vector(query)

        # Step 2: expand through graph edges + affinities (workspace-scoped),
        # cluster-aware when a query vector is available.
        try:
            chains = self._expand_with_graph(
                entry_nodes, agent_id, workspace_id, query_vec, model_key
            )
        except Exception as e:
            logger.warning("GraphRouter: graph expansion failed, falling back to embedding-only: %s", e)
            chains = self._to_single_chains(entry_nodes)

        # Step 2.5 (PRD-143 + PRD-232 US-010 P232-RVW-4): the final role net over
        # the WHOLE expanded surface. Entry nodes are already role-ranked, but two
        # downstream paths add un-gated names — edge-expansion targets (edges learned
        # from privileged usage can point AT a gated action) and the cluster
        # action_names_hot US-010 merges into the entry candidates — so drop any chain
        # touching an action this caller may not see. The graph layer stays fail-closed
        # on admin AND su on its own, independent of any downstream re-gate (the
        # TOOL_ROUTING_GRAPH flip is what this PRD builds toward).
        chains = self._drop_ineligible_chains(chains, exclude_admin, include_super_admin)

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
        workspace_id: Optional[str],
        query_vec: Optional[List[float]] = None,
        model_key: Optional[str] = None,
    ) -> List[Tuple[str, float, List[str]]]:
        """Query edge + affinity tables and build scored chains.

        PRD-177 S5: all edge/affinity reads are scoped to ``workspace_id`` — the
        learned graph is per-tenant.

        PRD-232 US-010: when ``query_vec`` is provided the live query is matched to
        the nearest intent cluster — its ``action_names_hot`` join the entry
        candidates, and its id scopes the affinity read so ``succeeds/fails_for_intent``
        apply PER-INTENT (not summed across every intent). A missed / absent cluster
        leaves routing at the embedding floor exactly as before.
        """
        from core.database.database import get_db_session

        min_conf = self._min_confidence()
        sample_floor = self._agent_sample_floor()

        chains: List[Tuple[str, float, List[str]]] = []

        with get_db_session() as db:
            # PRD-232 US-010(a): match the query to an intent cluster (if we have a
            # vector). A hit adds its hot actions as entry candidates and its id
            # scopes the affinity read below.
            intent_cluster_id: Optional[int] = None
            if query_vec is not None and model_key is not None:
                try:
                    cluster = self._match_intent_cluster(
                        db, query_vec, model_key, self._cluster_match_threshold()
                    )
                except Exception as e:
                    logger.warning(
                        "GraphRouter: intent-cluster match failed, embedding floor only: %s", e
                    )
                    cluster = None
                if cluster is not None:
                    intent_cluster_id = cluster[0]
                    entry_nodes = self._merge_cluster_hot_actions(entry_nodes, cluster)

            # Always include single-action chains from entry nodes (now including
            # any merged cluster hot actions)
            chains.extend(self._to_single_chains(entry_nodes))

            entry_action_names = [name for name, _ in entry_nodes]
            cosine_by_name = {name: score for name, score in entry_nodes}

            # Determine whether to use agent-specific edges
            use_agent_scope = False
            if agent_id is not None:
                agent_sample_count = self._agent_total_samples(db, agent_id)
                use_agent_scope = agent_sample_count >= sample_floor
            scoped_agent = agent_id if use_agent_scope else None

            # Batch-query edges for all entry nodes (workspace-scoped)
            edges = self._query_edges(db, entry_action_names, min_conf, scoped_agent, workspace_id)

            # Batch-query affinities for entry + expansion targets (workspace-scoped)
            all_action_names = set(entry_action_names)
            for edge in edges:
                all_action_names.add(edge["to_action"])
            # PRD-232 US-010(b): pass intent_cluster_id ONLY when a cluster matched,
            # so the no-cluster path keeps the exact legacy 4-arg call signature.
            if intent_cluster_id is not None:
                positive_boosts, negative_penalties = self._query_affinities(
                    db, list(all_action_names), scoped_agent, workspace_id, intent_cluster_id
                )
            else:
                positive_boosts, negative_penalties = self._query_affinities(
                    db, list(all_action_names), scoped_agent, workspace_id
                )

            # PRD-232 US-010(c): failed_after edges read as a de-ranking penalty.
            # Best-effort — a failure to read the (optional) failure signal must
            # never sink the whole expansion, which is already the embedding floor's
            # refinement, not its foundation.
            try:
                failed_penalties = self._query_failed_after(
                    db, list(all_action_names), min_conf, scoped_agent, workspace_id
                )
            except Exception as e:
                logger.warning(
                    "GraphRouter: failed_after read failed, no de-ranking applied: %s", e
                )
                failed_penalties = {}

        failed_weight = self._failed_after_penalty_weight()

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

            # Affinity boosts (succeeds/prefers) lift the chain; negative
            # penalties (fails_for_intent) lower it — PRD-141 US-017.
            boost = 0.0
            penalty = 0.0
            for action in (from_action, to_action):
                boost += positive_boosts.get(action, 0.0)
                penalty += negative_penalties.get(action, 0.0)

            # A learned failed_after transition de-ranks this chain (US-010c).
            failed_pen = failed_penalties.get((from_action, to_action), 0.0) * failed_weight

            score = cosine * edge_confidence + boost - penalty - failed_pen
            chains.append((from_action, score, [from_action, to_action]))
            expanded += 1

        return chains

    # ------------------------------------------------------------------
    # Intent-cluster matching (PRD-232 US-010a)
    # ------------------------------------------------------------------

    async def _match_query_vector(
        self, query: str
    ) -> Tuple[Optional[List[float]], Optional[str]]:
        """Resolve ``(query_vector, model_key)`` via the semantic index's embed.

        Returns ``(None, None)`` when the index cannot embed (unit fakes expose
        only ``rank_actions``; a degraded/timed-out embed) — the caller then skips
        cluster matching and stays at the embedding floor. Never raises.
        """
        embed = getattr(self._semantic_index, "embed_query", None)
        if embed is None:
            return None, None
        try:
            return await embed(query)
        except Exception as e:
            logger.debug("GraphRouter: query embed for cluster match failed: %s", e)
            return None, None

    @staticmethod
    def _match_intent_cluster(
        db,
        query_vec: List[float],
        model_key: str,
        threshold: float,
    ) -> Optional[Tuple[int, List[str], float]]:
        """Nearest ToolRoutingIntentCluster by centroid cosine, over ``threshold``.

        Only clusters embedded under the SAME canonical ``model_key`` are
        candidates — a centroid from another embedding model is not comparable.
        Returns ``(cluster_id, action_names_hot, similarity)`` or ``None`` (a miss:
        the query belongs to no learned intent, so routing stays at the embedding
        floor rather than forcing it into an ill-fitting cluster).
        """
        from core.models.tool_routing import ToolRoutingIntentCluster

        rows = (
            db.query(ToolRoutingIntentCluster)
            .filter(ToolRoutingIntentCluster.embedding_model_key == model_key)
            .all()
        )
        if not rows:
            return None

        q_norm = math.sqrt(sum(v * v for v in query_vec))
        if q_norm == 0.0:
            return None

        best: Optional[Tuple[int, List[str], float]] = None
        for r in rows:
            centroid = r.centroid_embedding or []
            if len(centroid) != len(query_vec):
                continue  # dimension mismatch — not comparable
            c_norm = math.sqrt(sum(v * v for v in centroid))
            if c_norm == 0.0:
                continue
            dot = sum(a * b for a, b in zip(query_vec, centroid))
            similarity = dot / (q_norm * c_norm)
            if best is None or similarity > best[2]:
                best = (r.id, list(r.action_names_hot or []), similarity)

        if best is None or best[2] < threshold:
            return None
        return best

    @staticmethod
    def _merge_cluster_hot_actions(
        entry_nodes: List[Tuple[str, float]],
        cluster: Tuple[int, List[str], float],
    ) -> List[Tuple[str, float]]:
        """A matched cluster's ``action_names_hot`` join the entry candidates.

        Existing cosine entries keep their score and order; a hot action not
        already present is appended at the cluster-similarity score so it enters
        the surface — its final rank is then decided by the per-intent affinity,
        not by this seed score. Pure; order-preserving (organic entries first).
        """
        _cluster_id, hot_actions, similarity = cluster
        present = {name for name, _ in entry_nodes}
        merged = list(entry_nodes)
        for name in hot_actions:
            if name not in present:
                merged.append((name, float(similarity)))
                present.add(name)
        return merged

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
        workspace_id: Optional[str],
    ) -> List[dict]:
        """Query tool_routing_edges for used_after edges from entry nodes.

        PRD-177 S5 + PRD-232 US-004: reconcile the global bootstrap seeds with
        the per-tenant lock. The learned graph is per-tenant, so a read for
        workspace A returns workspace A's edges EXACTLY — plus the genuinely
        unscoped (``workspace_id IS NULL``) ``meta_sibling`` cold-start seeds
        PRD-143's metadata_graph_seed writes globally, so a zero-telemetry tenant
        is still graph-reachable. An unscoped ``used_after`` row is NEVER admitted
        for a tenant (that would bleed one tenant's learned co-occurrence into
        another's routing). A None caller (system/global read) sees only the
        unscoped meta_sibling seeds. There is no cross-tenant global-read fallback.
        """
        from sqlalchemy import and_, or_
        from core.models.tool_routing import ToolRoutingEdge

        if not from_actions:
            return []

        # US-004: NULL-workspace rows are admissible ONLY for meta_sibling (the
        # global cold-start seeds). Every other edge type requires an exact
        # workspace match.
        meta_global = and_(
            ToolRoutingEdge.workspace_id.is_(None),
            ToolRoutingEdge.edge_type == "meta_sibling",
        )
        if workspace_id is not None:
            workspace_filter = or_(ToolRoutingEdge.workspace_id == workspace_id, meta_global)
        else:
            workspace_filter = meta_global

        filters = [
            ToolRoutingEdge.from_action.in_(from_actions),
            # used_after = learned co-occurrence (telemetry); meta_sibling =
            # metadata cold-start (PRD-143 metadata_graph_seed) so zero-telemetry
            # tools are still graph-reachable. Both are confidence-filtered, so
            # real usage (higher Wilson confidence) outranks metadata edges.
            ToolRoutingEdge.edge_type.in_(("used_after", "meta_sibling")),
            ToolRoutingEdge.confidence >= min_confidence,
            workspace_filter,
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
        workspace_id: Optional[str],
        intent_cluster_id: Optional[int] = None,
    ) -> Tuple[dict, dict]:
        """Query tool_routing_affinities, returning (positive_boosts, negative_penalties).

        PRD-141 US-017: positive and negative signals are kept in separate dicts
        rather than netted into one, so the caller can apply them explicitly as
        ``score = cosine * edge_confidence + boost - penalty``.

        * ``succeeds_for_intent`` / ``agent_prefers`` -> positive_boosts[action]
          += weight*confidence.
        * ``fails_for_intent`` -> negative_penalties[action] += weight*confidence,
          recorded as a POSITIVE magnitude (the caller subtracts it).

        PRD-177 S5: filtered to ``workspace_id`` — affinities are per-tenant, so a
        succeeds/fails-for-intent signal learned in one workspace never boosts or
        penalizes another's routing.

        PRD-232 US-010(b): when ``intent_cluster_id`` is given (a live query matched
        a cluster), read PER-INTENT rows (``intent_cluster_id ==`` the match) at full
        weight PLUS cluster-blind rows (``intent_cluster_id IS NULL``) as a weak
        global prior, discounted by ``_global_affinity_discount()``. This is the fix
        for C4: previously affinities were summed across EVERY intent, so an action
        that fails for intent X but succeeds for intent Y looked neutral. When no
        cluster matched, only the cluster-blind rows apply (exact legacy behaviour).
        """
        from sqlalchemy import and_, or_
        from core.models.tool_routing import ToolRoutingAffinity

        if not action_names:
            return {}, {}

        filters = [
            ToolRoutingAffinity.action_name.in_(action_names),
            # Per-tenant isolation (moat): scope to this workspace exactly.
            # None reads only the unscoped rows (IS NULL), never a tenant's.
            ToolRoutingAffinity.workspace_id == workspace_id
            if workspace_id is not None
            else ToolRoutingAffinity.workspace_id.is_(None),
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

        # PRD-232 US-010(b): per-intent scoping. A matched cluster admits its own
        # rows AND the cluster-blind global prior; a miss admits only the global
        # prior (intent_cluster_id IS NULL), never another intent's rows.
        if intent_cluster_id is not None:
            filters.append(
                or_(
                    ToolRoutingAffinity.intent_cluster_id == intent_cluster_id,
                    ToolRoutingAffinity.intent_cluster_id.is_(None),
                )
            )
        else:
            filters.append(ToolRoutingAffinity.intent_cluster_id.is_(None))

        rows = (
            db.query(ToolRoutingAffinity)
            .filter(and_(*filters))
            .all()
        )

        discount = GraphRouter._global_affinity_discount() if intent_cluster_id is not None else 1.0

        positive_boosts: dict = {}
        negative_penalties: dict = {}
        for r in rows:
            magnitude = r.weight * r.confidence
            # A cluster-blind row is a WEAK global prior when a cluster matched;
            # a per-intent row applies at full weight. (When no cluster matched,
            # discount is 1.0 and every admitted row is cluster-blind anyway.)
            if intent_cluster_id is not None and getattr(r, "intent_cluster_id", None) is None:
                magnitude *= discount
            if r.affinity_type in ("succeeds_for_intent", "agent_prefers"):
                positive_boosts[r.action_name] = positive_boosts.get(r.action_name, 0.0) + magnitude
            elif r.affinity_type == "fails_for_intent":
                negative_penalties[r.action_name] = negative_penalties.get(r.action_name, 0.0) + magnitude

        return positive_boosts, negative_penalties

    @staticmethod
    def _query_failed_after(
        db,
        from_actions: List[str],
        min_confidence: float,
        agent_id: Optional[int],
        workspace_id: Optional[str],
    ) -> Dict[Tuple[str, str], float]:
        """Read ``failed_after`` edges as a de-ranking penalty (PRD-232 US-010c).

        Returns ``{(from_action, to_action): confidence}`` — the failure signal for
        each risky transition, where ``confidence`` is the Wilson lower bound of the
        failure rate (edge_builder). The caller multiplies by the configured penalty
        weight and subtracts it from that chain's score, so a chain whose transition
        reliably fails is suppressed. Turns the previously write-only ``failed_after``
        rows into a live signal — no write-only tables (US-010c).

        Scoping mirrors ``_query_edges`` for tenancy, but ``failed_after`` is
        telemetry-derived (like ``used_after``), so NULL-workspace rows are NOT
        admitted for a tenant read (only ``meta_sibling`` cold-start seeds cross
        tenants, per US-004). Defence-in-depth: the ``edge_type`` is re-checked in
        Python so a permissive fake/DB layer can never mistake a ``used_after`` row
        for a failure.
        """
        from sqlalchemy import and_, or_
        from core.models.tool_routing import ToolRoutingEdge

        if not from_actions:
            return {}

        filters = [
            ToolRoutingEdge.from_action.in_(from_actions),
            ToolRoutingEdge.edge_type == "failed_after",
            ToolRoutingEdge.confidence >= min_confidence,
            ToolRoutingEdge.workspace_id == workspace_id
            if workspace_id is not None
            else ToolRoutingEdge.workspace_id.is_(None),
        ]

        if agent_id is not None:
            filters.append(
                or_(
                    ToolRoutingEdge.agent_id == agent_id,
                    ToolRoutingEdge.agent_id.is_(None),
                )
            )
        else:
            filters.append(ToolRoutingEdge.agent_id.is_(None))

        rows = (
            db.query(ToolRoutingEdge)
            .filter(and_(*filters))
            .limit(_MAX_EXPANDED_NODES)
            .all()
        )

        penalties: Dict[Tuple[str, str], float] = {}
        for r in rows:
            if getattr(r, "edge_type", None) != "failed_after":
                continue  # defence-in-depth against a permissive filter layer
            key = (r.from_action, r.to_action)
            # Keep the strongest failure signal if a pair recurs across scopes.
            penalties[key] = max(penalties.get(key, 0.0), float(r.confidence))
        return penalties

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_single_chains(
        entry_nodes: List[Tuple[str, float]],
    ) -> List[Tuple[str, float, List[str]]]:
        """Convert embedding-only results to single-action chains."""
        return [(name, score, [name]) for name, score in entry_nodes]

    def _drop_ineligible_chains(
        self,
        chains: List[Tuple[str, float, List[str]]],
        exclude_admin: bool,
        include_super_admin: bool,
    ) -> List[Tuple[str, float, List[str]]]:
        """Drop every chain touching an action the caller is not entitled to
        (PRD-143 fail-closed; PRD-232 US-010 P232-RVW-4 extends it to admin_only).

        The final role net over the whole expanded surface. Entry nodes are already
        role-ranked by rank_actions, but two paths add un-gated names downstream:
        edge-expansion targets (edges learned from privileged usage can point AT a
        gated action) and the cluster ``action_names_hot`` US-010 merges into the
        entry candidates. Enforce the SAME eligibility here as organic entry nodes:
          * super_admin_only chains drop unless include_super_admin=True;
          * admin_only chains drop when exclude_admin=True (a non-admin caller).
        So rank_chains' exclude_admin / include_super_admin contract holds for EVERY
        action it returns, not just the ranked entry nodes.

        The registry is resolved via the semantic index's reference (always present
        in production; keeps unit fakes lightweight) with the canonical singleton
        as fallback.
        """
        registry = getattr(self._semantic_index, "_registry", None)
        if registry is None:
            from .action_registry import get_action_registry
            registry = get_action_registry()
        blocked = set()
        for a in registry.get_all():
            if getattr(a, "super_admin_only", False) and not include_super_admin:
                blocked.add(a.name)
            elif getattr(a, "admin_only", False) and exclude_admin:
                blocked.add(a.name)
        if not blocked:
            return chains
        return [c for c in chains if not blocked.intersection(c[2])]

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
