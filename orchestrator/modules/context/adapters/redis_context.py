"""
Redis Shared Context — PRD-107 (Message-Passing Baseline)
=========================================================

Deliberately simple shared context using Redis lists and hashes.
BASELINE for A/B comparison against VectorFieldSharedContext (PRD-108).
Keyword matching only — no embeddings, no decay, no reinforcement,
no resonance. Exists to be beaten by the vector field.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import redis.asyncio as aioredis

from config import config
from core.ports.context import SharedContextPort

logger = logging.getLogger(__name__)

_PREFIX = "field"


def _meta_key(context_id: str) -> str:
    return f"{_PREFIX}:{context_id}:meta"


def _patterns_key(context_id: str) -> str:
    return f"{_PREFIX}:{context_id}:patterns"


async def _get_async_redis() -> aioredis.Redis:
    """Build an async Redis connection from centralised config."""
    url = config.REDIS_URL
    if url:
        return aioredis.from_url(url, decode_responses=True)

    host = config.REDIS_HOST or "127.0.0.1"
    port = int(config.REDIS_PORT or 6379)
    password = config.REDIS_PASSWORD or None
    return aioredis.Redis(
        host=host, port=port, password=password, db=0, decode_responses=True
    )


def _keyword_score(query: str, text: str) -> float:
    """Compute a naive keyword-overlap score normalised to 0-1.

    Counts how many unique query tokens appear in the target text.
    This is intentionally primitive — no stemming, no TF-IDF.
    """
    query_words = set(query.lower().split())
    if not query_words:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for w in query_words if w in text_lower)
    return hits / len(query_words)


class RedisSharedContext(SharedContextPort):
    """PRD-107: message-passing baseline. No embeddings, no decay,
    no resonance. Control group for A/B against PRD-108 vector field."""

    async def create_context(
        self,
        team_agent_ids: list[int],
        initial_data: Optional[dict[str, Any]] = None,
        provenance: Optional[dict[str, Any]] = None,
    ) -> str:
        context_id = str(uuid.uuid4())
        r = await _get_async_redis()
        try:
            meta = {
                "context_id": context_id,
                "agent_ids": json.dumps(team_agent_ids),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "backend": "redis_baseline",
            }
            if initial_data:
                meta["initial_data"] = json.dumps(initial_data)
            if provenance:
                meta["provenance"] = json.dumps(provenance)

            await r.hset(_meta_key(context_id), mapping=meta)
            logger.info(
                "redis_context: created context %s for agents %s",
                context_id,
                team_agent_ids,
            )
            return context_id
        finally:
            await r.aclose()

    async def inject(
        self,
        context_id: str,
        key: str,
        value: str,
        agent_id: int,
        strength: float = 1.0,
        provenance: Optional[dict[str, Any]] = None,
    ) -> None:
        r = await _get_async_redis()
        try:
            current_len = await r.llen(_patterns_key(context_id))
            pattern = json.dumps(
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "key": key,
                    "value": value,
                    "strength": strength,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "injected_order": current_len,
                    "provenance": provenance or {},
                }
            )
            await r.rpush(_patterns_key(context_id), pattern)
            logger.info(
                "redis_context: injected pattern key=%s from agent %d into %s",
                key,
                agent_id,
                context_id,
            )
        finally:
            await r.aclose()

    async def query(
        self,
        context_id: str,
        query: str,
        agent_id: int,
        top_k: int = 10,
        record_access: bool = True,
    ) -> list[dict[str, Any]]:
        # PRD-178 S2: ``record_access`` is part of the port contract. The Redis
        # baseline reads via lrange and never reinforces, so it is already
        # read-only — the flag is accepted for interface parity, no-op here.
        r = await _get_async_redis()
        try:
            raw_patterns = await r.lrange(_patterns_key(context_id), 0, -1)
            if not raw_patterns:
                logger.debug(
                    "redis_context: query on %s returned 0 patterns", context_id
                )
                return []

            scored: list[dict[str, Any]] = []
            for raw in raw_patterns:
                p = json.loads(raw)
                searchable = f"{p['key']} {p['value']}"
                score = _keyword_score(query, searchable)
                scored.append(
                    {
                        "id": p["id"],
                        "key": p["key"],
                        "value": p["value"],
                        "score": score,
                        "agent_id": p["agent_id"],
                        "decayed_strength": p["strength"],
                        "cosine_similarity": score,
                    }
                )

            # Sort descending by score, then by injection order (earlier first)
            scored.sort(key=lambda x: (-x["score"],))
            result = scored[:top_k]
            logger.info(
                "redis_context: query on %s matched %d/%d patterns (top_k=%d) for agent %d",
                context_id,
                len(result),
                len(scored),
                top_k,
                agent_id,
            )
            return result
        finally:
            await r.aclose()

    async def destroy_context(self, context_id: str) -> None:
        r = await _get_async_redis()
        try:
            cursor = b"0"
            prefix = f"{_PREFIX}:{context_id}:*"
            deleted = 0
            while True:
                cursor, keys = await r.scan(
                    cursor=cursor, match=prefix, count=100
                )
                if keys:
                    await r.delete(*keys)
                    deleted += len(keys)
                if cursor == 0 or cursor == b"0":
                    break

            logger.info(
                "redis_context: destroyed context %s (%d keys removed)",
                context_id,
                deleted,
            )
        finally:
            await r.aclose()
