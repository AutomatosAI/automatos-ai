"""
ActionSemanticIndex (PRD-138 US-003)
====================================

Embeds platform ActionDefinition records and ranks them by cosine similarity
to a query. Reuses EmbeddingManager (provider-agnostic) and CacheService
(per-text Redis cache). No custom blob storage, no hardcoded models.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from contextvars import ContextVar
from typing import Dict, List, Optional, Tuple

import numpy as np

from .action_registry import ActionDefinition, get_action_registry

logger = logging.getLogger(__name__)


# PRD-232 US-003: per-request scope for the shared rank_actions computation.
# When a turn opens this scope (ContextService.build_context wraps its section
# render + tool load in it), every rank_actions call in the turn — dispatcher
# narrowing, the shadow surface, the prompt catalog, the graph entry nodes —
# computes the su-gated cosine ranking ONCE and slices its own view from that
# one result. Outside a scope the ContextVar is None and rank_actions computes
# per call exactly as before (so nothing but real turn assembly opts in, and
# every existing caller/test is byte-for-byte unchanged). The scope dict is
# created fresh per turn, so it is also the request/tenant isolation boundary —
# a completed ranking never survives the turn that produced it.
_rank_scope: ContextVar[Optional[Dict[tuple, List[Tuple[str, float]]]]] = ContextVar(
    "action_semantic_rank_scope", default=None
)


@contextlib.contextmanager
def rank_actions_scope():
    """Open a per-request rank_actions memo for the duration of the block.

    Sync context manager (setting a ContextVar needs no await) that wraps an
    async body: ``with rank_actions_scope(): ... await build ...``. Nesting is
    safe — each entry installs a fresh dict and the token restores the previous
    on exit. asyncio.gather child tasks copy the context, so sections rendered
    concurrently share the one dict."""
    token = _rank_scope.set({})
    try:
        yield
    finally:
        _rank_scope.reset(token)


def _relevance_floor_config() -> Tuple[float, float]:
    """(absolute_floor, ratio_floor) from the canonical config singleton.

    Both default 0 (floor off). Read lazily so pure unit tests need no config.
    """
    try:
        from config import config

        return (
            float(getattr(config, "SEMANTIC_TOOL_ROUTING_FLOOR", 0) or 0),
            float(getattr(config, "SEMANTIC_TOOL_ROUTING_FLOOR_RATIO", 0) or 0),
        )
    except Exception:
        return 0.0, 0.0


def _apply_relevance_floor(
    scored: List[Tuple[str, float]],
    floor: float,
    ratio: float,
) -> List[Tuple[str, float]]:
    """Drop candidates below max(floor, best*ratio). Pure; order-preserving.

    ``scored`` must be sorted best-first. With both dials 0 this is the
    identity — the legacy blind top-K. A ratio floor only bites when the
    best score is positive (cosine can be negative on a hostile query;
    a negative cutoff would keep everything and mean nothing).
    """
    if not scored or (floor <= 0 and ratio <= 0):
        return scored
    cutoffs = []
    if floor > 0:
        cutoffs.append(floor)
    if ratio > 0 and scored[0][1] > 0:
        cutoffs.append(scored[0][1] * ratio)
    if not cutoffs:
        # Only the ratio dial is set and the best score is non-positive —
        # there is no meaningful cutoff; dropping everything here would turn
        # a hostile query into an empty surface by accident.
        return scored
    cutoff = max(cutoffs)
    return [(n, s) for n, s in scored if s >= cutoff]


class ActionSemanticIndex:
    """Per-process semantic index over platform ActionDefinitions."""

    def __init__(self) -> None:
        from core.cache.service import get_cache_service
        from core.llm import create_embedding_manager

        self._embedding_manager = create_embedding_manager()
        self._cache = get_cache_service()
        self._registry = get_action_registry()
        self._action_embeddings: Dict[str, List[float]] = {}
        self._indexed: bool = False
        # PRD-232 US-006: corpus content hash the per-process dict was built
        # under. When the seeded utterance corpus changes (a deploy swaps the
        # YAMLs), ensure_indexed drops the name-keyed dict so stale vectors don't
        # persist; Redis re-embeds only the texts that actually changed.
        self._corpus_hash: Optional[str] = None
        self._lock: Optional[asyncio.Lock] = None
        # In-flight live query embeds, keyed by (loop id, model_key, query):
        # concurrent same-query callers (enum narrowing + the prompt catalog
        # rank the same turn text) share ONE upstream embed instead of racing
        # duplicates. Loop id in the key because a task is only awaitable on
        # the loop that created it (the sync bridge runs on its own loop).
        self._inflight: Dict[tuple, asyncio.Task] = {}

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @staticmethod
    def _collect_enum_values(parameters: Optional[Dict]) -> List[str]:
        """Flatten every parameter enum value (e.g. the task-status enum
        inbox/assigned/in_progress/review/done). These are the vocabulary a user
        speaks ("mark it done", "anything in review?") that name+description never
        surface — PRD-232 C6. Order-preserving, deduped."""
        values: List[str] = []
        props = (parameters or {}).get("properties", {}) or {}
        for spec in props.values():
            if not isinstance(spec, dict):
                continue
            enum = spec.get("enum")
            if isinstance(enum, list):
                values.extend(str(v) for v in enum)
            items = spec.get("items")  # array-of-enum params (items.enum)
            if isinstance(items, dict) and isinstance(items.get("enum"), list):
                values.extend(str(v) for v in items["enum"])
        seen: set = set()
        return [v for v in values if not (v in seen or seen.add(v))]

    @staticmethod
    def _build_embedding_text(action: ActionDefinition) -> str:
        # PRD-232 US-006: fold the seeded utterance corpus (US-005) and the
        # parameter enum values into the embedded text, so a query phrased like
        # any seeded utterance — or naming an enum value — lands near the action.
        # Local import keeps the module import cheap and lets tests monkeypatch.
        from .utterance_corpus import utterances_for

        tags = ", ".join(action.tags) if action.tags else ""
        examples = "; ".join(action.examples) if action.examples else ""
        options = ", ".join(ActionSemanticIndex._collect_enum_values(getattr(action, "parameters", None)))
        utterances = "; ".join(utterances_for(action.name))
        return (
            f"{action.name}: {action.description} | Tags: {tags} | "
            f"Examples: {examples} | Category: {action.category} | "
            f"Options: {options} | Utterances: {utterances}"
        )

    def _cache_model_key(self) -> str:
        # Force provider init before reading provider info — otherwise
        # get_provider_info() returns {"provider": "none", "status": "not_initialized"}
        # and the key collapses to "none:None:{dim}". Multiple runs writing under
        # that broken key produce vector mixes (real qwen + deterministic
        # fallback) that destroy ranking quality. Discovered post-PRD-138 build
        # when filtered in-set rate dropped from 97.9% → 57.4% across all models.
        ensure = getattr(self._embedding_manager, "_ensure_provider", None)
        if callable(ensure):
            ensure()
        info = self._embedding_manager.get_provider_info()
        provider = info.get("provider") or self._embedding_manager.__class__.__name__
        model = info.get("model")
        if model is None:
            cfg = getattr(self._embedding_manager.provider, "config", None)
            model = getattr(cfg, "model", None)
        dimension = info.get("dimension") or self._embedding_manager.get_dimension()
        return f"{provider}:{model}:{dimension}"

    def _eligible_actions(
        self,
        exclude_admin: bool,
        exclude_promoted: bool,
        include_super_admin: bool = False,
    ) -> List[ActionDefinition]:
        # PRD-143: fail-closed — super_admin_only actions are eligible ONLY
        # when include_super_admin=True is passed explicitly.
        return [
            a for a in self._registry.get_all()
            if not (exclude_admin and a.admin_only)
            and not (exclude_promoted and a.promoted)
            and (include_super_admin or not a.super_admin_only)
        ]

    async def ensure_indexed(
        self,
        exclude_admin: bool = True,
        exclude_promoted: bool = True,
        include_super_admin: bool = False,
    ) -> None:
        async with self._get_lock():
            # PRD-232 US-006 — corpus-hash guard. The per-process dict is keyed by
            # action NAME (not text), so a corpus change would otherwise leave
            # stale vectors indexed forever. If the hash moved, drop the dict and
            # re-embed; the Redis layer is text-addressed so unchanged texts stay
            # cache hits and only the changed actions embed upstream.
            from .utterance_corpus import corpus_hash

            current_hash = corpus_hash()
            prev_hash = getattr(self, "_corpus_hash", None)
            if prev_hash is not None and prev_hash != current_hash:
                self._action_embeddings = {}
                self._indexed = False
            self._corpus_hash = current_hash

            actions = self._eligible_actions(exclude_admin, exclude_promoted, include_super_admin)
            missing = [a for a in actions if a.name not in self._action_embeddings]
            if not missing:
                return
            model_key = self._cache_model_key()
            text_by_name = {a.name: self._build_embedding_text(a) for a in missing}
            texts = list(text_by_name.values())
            cached = self._cache.get_embeddings_batch(texts, model=model_key)
            misses_by_text: Dict[str, str] = {
                text: name for name, text in text_by_name.items() if not cached.get(text)
            }
            new_embeddings: Dict[str, List[float]] = {}
            if misses_by_text:
                miss_texts = list(misses_by_text.keys())
                vectors = await self._embedding_manager.generate_embeddings_batch(miss_texts)
                if len(vectors) != len(miss_texts):
                    raise RuntimeError(
                        f"Embedding manager returned {len(vectors)} vectors for "
                        f"{len(miss_texts)} texts; index would silently drop entries"
                    )
                new_embeddings = dict(zip(miss_texts, vectors))
                self._cache.set_embeddings_batch(new_embeddings, model=model_key)
            for name, text in text_by_name.items():
                vec = cached.get(text) or new_embeddings.get(text)
                if vec is not None:
                    self._action_embeddings[name] = vec
            if not self._indexed:
                logger.info("ActionSemanticIndex: indexed %d actions", len(self._action_embeddings))
                self._indexed = True

    def _embed_timeout_s(self) -> Optional[float]:
        """Budget for a LIVE query embed, from the canonical config singleton.

        <= 0 disables the bound. Defaults to 2.5s when config is unavailable
        (pure unit-test environments)."""
        try:
            from config import config
            timeout = float(getattr(config, "SEMANTIC_TOOL_ROUTING_EMBED_TIMEOUT_S", 2.5))
        except Exception:
            timeout = 2.5
        return timeout if timeout > 0 else None

    async def _embed_query_bounded(
        self, query: str, model_key: str, timeout_s: Optional[float]
    ) -> Tuple[Optional[List[float]], bool, bool]:
        """Resolve the query vector: Redis cache first, then a time-bounded
        live embed.

        Returns (vector-or-None, cache_hit, timed_out). On timeout the embed
        task is NOT cancelled — it finishes in the background and writes the
        Redis cache so the next identical query narrows instantly. Rationale:
        narrowing is an optimization; it must never cost more than it saves
        (observed 37–67s/call when the OpenRouter embedding upstream degrades).
        """
        try:
            cached = self._cache.get_embeddings_batch([query], model=model_key).get(query)
        except Exception:
            cached = None
        if cached:
            return cached, True, False

        # One live embed per (loop, model, query) — concurrent callers share
        # it. The finalize callback owns cleanup + the cache write, so the
        # vector lands in Redis whether the winner was awaited or timed out.
        # Lazy-init like _get_lock(): tests construct the index via __new__,
        # so instance state must not assume __init__ ran.
        inflight = getattr(self, "_inflight", None)
        if inflight is None:
            inflight = self._inflight = {}
        key = (id(asyncio.get_running_loop()), model_key, query)
        task = inflight.get(key)
        if task is None:
            task = asyncio.ensure_future(self._embedding_manager.generate_embedding(query))
            inflight[key] = task

            def _finalize(t: "asyncio.Task", _key: tuple = key) -> None:
                inflight.pop(_key, None)
                try:
                    self._cache.set_embeddings_batch({query: t.result()}, model=model_key)
                except Exception:
                    logger.debug("query-embed background cache write failed", exc_info=True)

            task.add_done_callback(_finalize)

        done, pending = await asyncio.wait({task}, timeout=timeout_s)
        if pending:
            logger.warning(
                "ActionSemanticIndex: query embed exceeded %.1fs — narrowing "
                "falls back to the full enum; embed continues in background "
                "to warm the cache (query=%r)",
                timeout_s,
                query[:80],
            )
            return None, False, True

        vec = task.result()  # raises if the embed failed — caller's fallback handles it
        try:
            self._cache.set_embeddings_batch({query: vec}, model=model_key)
        except Exception:
            logger.debug("query-embed cache write failed", exc_info=True)
        return vec, False, False

    async def embed_query(
        self, query: str, embed_timeout_s: Optional[float] = None
    ) -> Tuple[Optional[List[float]], str]:
        """Resolve the query embedding + canonical model key (PRD-232 US-010).

        The GraphRouter uses this to match a live query to the nearest intent
        cluster centroid. It reuses the SAME bounded/cached embed as
        ``rank_actions`` (``_embed_query_bounded``), so within a turn — where the
        entry-node ranking has already embedded this exact query text — this is a
        Redis cache hit, not a second upstream embed. Returns ``(vector, model_key)``
        with ``vector`` None on timeout/failure, so the caller falls back to the
        embedding floor (no cluster) rather than guessing.
        """
        model_key = self._cache_model_key()
        if embed_timeout_s is None:
            embed_timeout_s = self._embed_timeout_s()
        elif embed_timeout_s <= 0:
            embed_timeout_s = None
        vec, _cache_hit, _timed_out = await self._embed_query_bounded(
            query, model_key=model_key, timeout_s=embed_timeout_s
        )
        return vec, model_key

    async def rank_actions(
        self,
        query: str,
        top_k: int = 15,
        exclude_admin: bool = True,
        exclude_promoted: bool = True,
        include_super_admin: bool = False,
        embed_timeout_s: Optional[float] = None,
        workspace_id: Optional[str] = None,
        agent_id: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Rank eligible actions by cosine similarity to ``query``.

        PRD-232 US-003: the expensive work — indexing, the query embed, and the
        cosine pass over the su-gated candidate set — is computed ONCE per turn
        (``_shared_full_ranking``) and every surface slices its own view here.
        This method is now the cheap Phase 2: filter the shared ranking by this
        caller's (exclude_admin, exclude_promoted) gate, apply the relevance
        floor, and cut to ``top_k``. The shared ranking spans the widest
        su-gated set (admin + promoted included), so narrowing (exclude_promoted)
        and the shadow surface (keep promoted) both slice from ONE computation
        instead of ranking the same query four times.

        ``workspace_id`` scopes the shared-ranking memo (US-003 AC3 — no
        cross-tenant reuse); ``agent_id`` is accepted for call-site symmetry but
        is NOT in the memo key: the ranking is over GLOBAL platform actions and
        is agent-independent, and keying on it would defeat the per-turn dedup
        (the catalog and narrowing surfaces source agent_id differently).

        PR-B (tool-surface review): results below the configured relevance
        floor — max(SEMANTIC_TOOL_ROUTING_FLOOR, best*FLOOR_RATIO) — are
        dropped BEFORE the top_k cut, so a greeting can legitimately rank
        zero actions instead of the 15 least-dissimilar. Both dials default
        to 0 (floor off, exact legacy behavior).
        """
        full = await self._shared_full_ranking(
            query,
            include_super_admin=include_super_admin,
            embed_timeout_s=embed_timeout_s,
            workspace_id=workspace_id,
        )
        if not full:
            return []
        eligible = {
            a.name
            for a in self._eligible_actions(exclude_admin, exclude_promoted, include_super_admin)
        }
        scored = [(n, s) for (n, s) in full if n in eligible]
        scored = _apply_relevance_floor(scored, *_relevance_floor_config())
        return scored[:top_k]

    async def _shared_full_ranking(
        self,
        query: str,
        include_super_admin: bool,
        embed_timeout_s: Optional[float],
        workspace_id: Optional[str],
    ) -> List[Tuple[str, float]]:
        """The once-per-turn su-gated cosine ranking (PRD-232 US-003).

        Returns the FULL su-gated ranking (admin + promoted included; su actions
        in ONLY when ``include_super_admin``) sorted best-first, so every caller
        can slice its own eligibility view. Layered like ``_embed_query_bounded``:

        1. request-scoped completed-result cache (``_rank_scope`` dict) — a
           second sequential caller in the same turn returns instantly;
        2. an in-flight future keyed per (loop, query, model, su, workspace) —
           concurrently-rendered sections (``asyncio.gather``) share ONE
           computation instead of racing duplicates.

        A timeout/empty result is NOT cached (a degraded embed never poisons the
        turn). Outside a scope, only the in-flight dedup applies, so sequential
        callers recompute exactly as before US-003."""
        model_key = self._cache_model_key()
        cache_key = (
            query,
            model_key,
            bool(include_super_admin),
            str(workspace_id) if workspace_id is not None else None,
        )
        scope = _rank_scope.get()
        if scope is not None:
            hit = scope.get(cache_key)
            if hit is not None:
                return hit

        inflight = getattr(self, "_rank_inflight", None)
        if inflight is None:
            inflight = self._rank_inflight = {}
        loop_key = (id(asyncio.get_running_loop()),) + cache_key
        task = inflight.get(loop_key)
        if task is None:
            task = asyncio.ensure_future(
                self._compute_full_ranking(query, include_super_admin, embed_timeout_s, model_key)
            )
            inflight[loop_key] = task
            task.add_done_callback(lambda t, _k=loop_key: inflight.pop(_k, None))

        result = await task
        # Only memoize a real ranking; a timeout/empty must not stick for the turn.
        if scope is not None and result:
            scope[cache_key] = result
        return result

    async def _compute_full_ranking(
        self,
        query: str,
        include_super_admin: bool,
        embed_timeout_s: Optional[float],
        model_key: str,
    ) -> List[Tuple[str, float]]:
        """Embed ``query`` and cosine-rank the FULL su-gated candidate set.

        The candidate set is the widest su-gated view (exclude_admin=False,
        exclude_promoted=False) so a single computation serves every per-call
        slice. Ranks the full eligible set (≤ ~110 actions, sub-ms) — no
        pre-scoring truncation, matching the PRD-138 Appendix A baselines."""
        _t0 = time.monotonic()
        await self.ensure_indexed(
            exclude_admin=False,
            exclude_promoted=False,
            include_super_admin=include_super_admin,
        )
        _t1 = time.monotonic()
        candidate_names = [
            a.name
            for a in self._eligible_actions(False, False, include_super_admin)
            if a.name in self._action_embeddings
        ]
        if not candidate_names:
            return []
        if embed_timeout_s is None:
            embed_timeout_s = self._embed_timeout_s()
        elif embed_timeout_s <= 0:
            embed_timeout_s = None
        raw_vec, cache_hit, timed_out = await self._embed_query_bounded(
            query,
            model_key=model_key,
            timeout_s=embed_timeout_s,
        )
        _t2 = time.monotonic()
        if raw_vec is None:
            logger.info(
                "[perf] rank_actions: ensure_indexed=%.0fms query_embed=%.0fms (TIMED OUT) n_candidates=%d",
                (_t1 - _t0) * 1000,
                (_t2 - _t1) * 1000,
                len(candidate_names),
            )
            return []
        query_vec = np.asarray(raw_vec, dtype=float)
        q_norm = float(np.linalg.norm(query_vec))
        if q_norm == 0.0:
            return []
        scored: List[Tuple[str, float]] = []
        for name in candidate_names:
            vec = np.asarray(self._action_embeddings[name], dtype=float)
            v_norm = float(np.linalg.norm(vec))
            if v_norm == 0.0:
                continue
            scored.append((name, float(np.dot(query_vec, vec) / (q_norm * v_norm))))
        scored.sort(key=lambda x: x[1], reverse=True)
        logger.info(
            "[perf] rank_actions: ensure_indexed=%.0fms query_embed=%.0fms cosine=%.0fms n_candidates=%d cache_hit=%d",
            (_t1 - _t0) * 1000,
            (_t2 - _t1) * 1000,
            (time.monotonic() - _t2) * 1000,
            len(candidate_names),
            int(cache_hit),
        )
        return scored


_index_lock = threading.Lock()
_index_instance: Optional[ActionSemanticIndex] = None


def get_action_semantic_index() -> ActionSemanticIndex:
    """Process-singleton factory (matches get_action_registry pattern)."""
    global _index_instance
    if _index_instance is not None:
        return _index_instance
    with _index_lock:
        if _index_instance is None:
            _index_instance = ActionSemanticIndex()
    return _index_instance
