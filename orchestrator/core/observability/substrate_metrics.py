"""PRD-197 S4 — the substrate telemetry seam.

``record_substrate_search`` is the one call every retrieval seam makes
after a search: it writes a ``substrate_metric_events`` row (the always-on
DB sink the Command Center tile aggregates) and, when a query string is
provided, also emits through the PRD-185 tracer plane
(``fire_retrieval_score``, guarded, default-OFF Langfuse).

Contract with the hot path:
- **Never raises.** A telemetry failure is logged at WARNING (not DEBUG —
  the tool-telemetry type-poison hid a severed learning plane behind a
  DEBUG swallow for two months; we do not repeat that) and retrieval
  proceeds untouched.
- **Never blocks.** Call sites fire-and-forget via
  ``record_substrate_search_nowait``; the insert runs in the default
  executor off the event loop.

Seam vocabulary is fixed here; status vocabulary is the tracer's
(hit / empty / error).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from core.observability.tracer import (
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_HIT,
    fire_retrieval_score,
)

logger = logging.getLogger(__name__)

SEAM_DOCUMENTS = "documents"
SEAM_MEMORY = "memory"
SEAM_FIELD = "field"

__all__ = [
    "SEAM_DOCUMENTS",
    "SEAM_MEMORY",
    "SEAM_FIELD",
    "STATUS_HIT",
    "STATUS_EMPTY",
    "STATUS_ERROR",
    "record_substrate_search",
    "record_substrate_search_nowait",
]


def _insert_event_sync(
    seam: str,
    workspace_id: Optional[str],
    status: str,
    candidates: int,
    latency_ms: float,
) -> None:
    from core.database.database import SessionLocal
    from core.models.substrate_metrics import SubstrateMetricEvent

    db = SessionLocal()
    try:
        db.add(
            SubstrateMetricEvent(
                seam=seam,
                workspace_id=str(workspace_id) if workspace_id else None,
                status=status,
                candidates=int(candidates),
                latency_ms=float(latency_ms),
            )
        )
        db.commit()
    finally:
        db.close()


async def record_substrate_search(
    *,
    seam: str,
    workspace_id: Any = None,
    candidates: int,
    latency_ms: float,
    status: str,
    query: Optional[str] = None,
    top_score: float = 0.0,
) -> None:
    """Record one substrate search. Guarded end-to-end — never raises.

    ``query`` is optional: pass it on seams that don't already emit a
    retrieval score at a higher level (memory/field); the documents seam
    passes None because ``RAGService.retrieve`` fires the query-level
    score itself.
    """
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            _insert_event_sync,
            seam,
            str(workspace_id) if workspace_id else None,
            status,
            candidates,
            latency_ms,
        )
    except Exception:
        logger.warning(
            "[substrate-metrics] %s sink write failed", seam, exc_info=True
        )
    if query is not None:
        fire_retrieval_score(
            query=query,
            num_docs=candidates,
            top_score=top_score,
            status=status,
            workspace_id=workspace_id,
            metadata={"seam": seam, "latency_ms": latency_ms},
        )


def record_substrate_search_nowait(**kwargs: Any) -> None:
    """Fire-and-forget wrapper for hot paths: schedules the record as a
    task so the retrieval that produced it never waits on telemetry."""
    try:
        asyncio.get_running_loop().create_task(record_substrate_search(**kwargs))
    except RuntimeError:
        # No running loop (sync bridge caller) — record inline, still guarded.
        try:
            _insert_event_sync(
                kwargs["seam"],
                str(kwargs.get("workspace_id")) if kwargs.get("workspace_id") else None,
                kwargs["status"],
                kwargs.get("candidates", 0),
                kwargs.get("latency_ms", 0.0),
            )
        except Exception:
            logger.warning(
                "[substrate-metrics] %s sync sink write failed",
                kwargs.get("seam"),
                exc_info=True,
            )
