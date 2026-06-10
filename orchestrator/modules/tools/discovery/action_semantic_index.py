"""
ActionSemanticIndex (PRD-138 US-003)
====================================

Embeds platform ActionDefinition records and ranks them by cosine similarity
to a query. Reuses EmbeddingManager (provider-agnostic) and CacheService
(per-text Redis cache). No custom blob storage, no hardcoded models.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from .action_registry import ActionDefinition, get_action_registry

logger = logging.getLogger(__name__)


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
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @staticmethod
    def _build_embedding_text(action: ActionDefinition) -> str:
        tags = ", ".join(action.tags) if action.tags else ""
        examples = "; ".join(action.examples) if action.examples else ""
        return (
            f"{action.name}: {action.description} | Tags: {tags} | "
            f"Examples: {examples} | Category: {action.category}"
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

    async def rank_actions(
        self,
        query: str,
        top_k: int = 15,
        exclude_admin: bool = True,
        exclude_promoted: bool = True,
        include_super_admin: bool = False,
    ) -> List[Tuple[str, float]]:
        await self.ensure_indexed(
            exclude_admin=exclude_admin,
            exclude_promoted=exclude_promoted,
            include_super_admin=include_super_admin,
        )
        # Pre-filter eligibility (admin/promoted), then score every remaining
        # action and let cosine similarity decide ranking. Earlier revisions
        # truncated `candidate_names` at 50 in registration order BEFORE
        # scoring, which silently dropped any action registered after the 50th
        # spot from ever being surfaced — the parity check against the
        # PRD-138 Appendix A baselines (US-005) caught this. The PRD allows
        # a wider-candidate-set heuristic only AFTER ranking; we keep things
        # simple by ranking the full eligible set (≤ ~110 actions, sub-ms).
        eligible_names = [
            a.name
            for a in self._eligible_actions(exclude_admin, exclude_promoted, include_super_admin)
        ]
        candidate_names = [n for n in eligible_names if n in self._action_embeddings]
        if not candidate_names:
            return []
        query_vec = np.asarray(await self._embedding_manager.generate_embedding(query), dtype=float)
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
        return scored[:top_k]


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
