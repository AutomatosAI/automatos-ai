"""Knowledge → Memory's status tile: whether durable memory works here, and if not, why.

Durable (L3) memory needs a memory store (Qdrant, ``QDRANT_URL``) and a real embeddings
provider. The local edition's default stack runs Qdrant and points the backend at it,
so the usual gap there is the provider: with no key the embedding manager falls back to
its degraded stand-in, which refuses to embed, and nothing can be stored or found. The
tile said only "Unavailable", or "Degraded" with nothing stored, and never why; the
``reason`` here says what to do. ``GET /api/v1/memory/health`` (api/memory_stats.py)
answers with ``memory_status`` and ``search_effectiveness``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config import config

logger = logging.getLogger(__name__)

HEALTHY = "healthy"
DEGRADED = "degraded"  # works, nothing stored yet
UNAVAILABLE = "unavailable"

NO_STORE = "Memory is off on this install: no memory store is configured (QDRANT_URL is empty)."
STORE_DOWN = "The memory store is not answering, so nothing can be remembered or recalled right now."
NO_EMBEDDINGS = (
    "Memory needs an embeddings key. Add the key for your embedding provider (OpenRouter, unless "
    "Settings → System Settings → Embeddings names another) in Settings → API Keys, then restart "
    "the backend so memory picks it up."
)

_SEARCHES_SQL = text(
    "SELECT COUNT(*) AS total_searches, SUM(CASE WHEN had_results THEN 1 ELSE 0 END) AS hits "
    "FROM memory_access_log WHERE workspace_id = :ws_id"
)


async def memory_status(service: Optional[Any], reachable: bool, total: int) -> Dict[str, Optional[str]]:
    """``health_status`` and ``reason`` for the memory health report.

    ``service`` is the UnifiedMemoryService, None when it is not configured or did not start;
    ``reachable`` says whether listing the workspace's memories answered; ``total`` is how many
    it found. The embeddings check asks the durable store's own embedder, the one its adds and
    searches use, and reads settings and the key store, so it runs in a thread.
    """
    if not config.QDRANT_URL:
        return {"health_status": UNAVAILABLE, "reason": NO_STORE}
    if service is None or not reachable:
        return {"health_status": UNAVAILABLE, "reason": STORE_DOWN}
    if not await asyncio.to_thread(service._durable.embeddings_ready):
        return {"health_status": UNAVAILABLE, "reason": NO_EMBEDDINGS}
    return {"health_status": HEALTHY if total > 0 else DEGRADED, "reason": None}


def search_effectiveness(db: Session, workspace_id: Any) -> Dict[str, Any]:
    """How the workspace's memory searches went (memory_access_log): searches, hits, hit rate."""
    stats: Dict[str, Any] = {"total_searches": 0, "hits": 0, "hit_rate": 0}
    try:
        row = db.execute(_SEARCHES_SQL, {"ws_id": str(workspace_id)}).fetchone()
    except SQLAlchemyError as exc:  # the tile shows zeros; the status stands on its own
        logger.debug("memory_access_log query failed: %s", exc)
        return stats
    if row and row.total_searches:
        stats = {
            "total_searches": row.total_searches,
            "hits": row.hits or 0,
            "hit_rate": round((row.hits or 0) / max(row.total_searches, 1), 2),
        }
    return stats
