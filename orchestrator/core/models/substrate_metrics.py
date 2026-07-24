"""PRD-197 S4 — per-seam retrieval substrate metrics.

One row per substrate search on each retrieval seam (documents / memory /
field): candidates returned, latency, and the hit/empty/error status from
the PRD-185 tracer vocabulary. The Command Center substrate-health tile and
``services/substrate_health.py`` aggregate these — the point is that the
next dark plane trips a tile, not a user complaint.

Writes are fire-and-forget through
``core/observability/substrate_metrics.record_substrate_search`` and must
never block or fail a retrieval. Rows are pruned to
``SUBSTRATE_METRICS_RETENTION_DAYS`` by the memory-jobs sweep (the
heartbeat_results 148k-row lesson: no unbounded telemetry tables).
"""

from sqlalchemy import BigInteger, Column, DateTime, Float, Index, Integer, String
from sqlalchemy.sql import func

from core.database.base import Base


class SubstrateMetricEvent(Base):
    __tablename__ = "substrate_metric_events"
    __table_args__ = (
        Index("idx_substrate_metrics_seam_created", "seam", "created_at"),
        Index("idx_substrate_metrics_ws_created", "workspace_id", "created_at"),
        {"extend_existing": True},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Which retrieval seam: "documents" | "memory" | "field"
    # (core/observability/substrate_metrics.py owns the vocabulary).
    seam = Column(String(16), nullable=False)

    # Str-cast workspace id. Nullable: the context-scoped field query path
    # has no workspace in hand; those rows still count toward seam health.
    workspace_id = Column(String(64), nullable=True)

    # "hit" | "empty" | "error" — the PRD-185 tracer status vocabulary.
    status = Column(String(8), nullable=False)

    candidates = Column(Integer, nullable=False, default=0)
    latency_ms = Column(Float, nullable=False, default=0.0)

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
