"""Instrumentation wrapper for SharedContextPort A/B experiments."""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from core.ports.context import SharedContextPort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldMetricEvent:
    """Single metric event — immutable."""

    timestamp: str  # ISO-8601
    operation: str  # "inject" | "query" | "create" | "destroy"
    context_id: str
    agent_id: int
    latency_ms: float
    # inject-specific
    pattern_key: Optional[str] = None
    # query-specific
    query_text: Optional[str] = None
    results_returned: Optional[int] = None
    top_score: Optional[float] = None


@dataclass
class ExperimentMetrics:
    """Accumulated metrics for one context/mission run."""

    context_id: str
    backend: str  # "vector_field" or "redis"
    events: list[FieldMetricEvent] = field(default_factory=list)

    @property
    def total_injections(self) -> int:
        return sum(1 for e in self.events if e.operation == "inject")

    @property
    def total_queries(self) -> int:
        return sum(1 for e in self.events if e.operation == "query")

    @property
    def avg_query_latency_ms(self) -> float:
        latencies = [e.latency_ms for e in self.events if e.operation == "query"]
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)

    @property
    def avg_results_per_query(self) -> float:
        counts = [
            e.results_returned
            for e in self.events
            if e.operation == "query" and e.results_returned is not None
        ]
        if not counts:
            return 0.0
        return sum(counts) / len(counts)

    def injections_by_agent(self) -> dict[int, int]:
        counter: Counter[int] = Counter()
        for e in self.events:
            if e.operation == "inject":
                counter[e.agent_id] += 1
        return dict(counter)

    def queries_by_agent(self) -> dict[int, int]:
        counter: Counter[int] = Counter()
        for e in self.events:
            if e.operation == "query":
                counter[e.agent_id] += 1
        return dict(counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "backend": self.backend,
            "total_injections": self.total_injections,
            "total_queries": self.total_queries,
            "avg_query_latency_ms": round(self.avg_query_latency_ms, 2),
            "avg_results_per_query": round(self.avg_results_per_query, 2),
            "injections_by_agent": self.injections_by_agent(),
            "queries_by_agent": self.queries_by_agent(),
            "event_count": len(self.events),
        }


class InstrumentedSharedContext(SharedContextPort):
    """Decorator that wraps any SharedContextPort and captures metrics."""

    def __init__(self, inner: SharedContextPort, backend_name: str) -> None:
        self._inner = inner
        self._backend_name = backend_name
        self._metrics: dict[str, ExperimentMetrics] = {}

    def _ensure_metrics(self, context_id: str) -> ExperimentMetrics:
        if context_id not in self._metrics:
            self._metrics[context_id] = ExperimentMetrics(
                context_id=context_id,
                backend=self._backend_name,
            )
        return self._metrics[context_id]

    def _record(self, event: FieldMetricEvent) -> None:
        metrics = self._ensure_metrics(event.context_id)
        metrics.events.append(event)
        logger.debug(
            "instrumentation %s context=%s agent=%d latency=%.1fms",
            event.operation,
            event.context_id,
            event.agent_id,
            event.latency_ms,
        )

    async def create_context(
        self,
        team_agent_ids: list[int],
        initial_data: Optional[dict[str, Any]] = None,
        provenance: Optional[dict[str, Any]] = None,
    ) -> str:
        start = time.monotonic()
        context_id = await self._inner.create_context(team_agent_ids, initial_data, provenance)
        elapsed_ms = (time.monotonic() - start) * 1000

        self._record(
            FieldMetricEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                operation="create",
                context_id=context_id,
                agent_id=0,
                latency_ms=elapsed_ms,
            )
        )
        return context_id

    async def inject(
        self,
        context_id: str,
        key: str,
        value: str,
        agent_id: int,
        strength: float = 1.0,
        provenance: Optional[dict[str, Any]] = None,
    ) -> None:
        start = time.monotonic()
        await self._inner.inject(context_id, key, value, agent_id, strength, provenance)
        elapsed_ms = (time.monotonic() - start) * 1000

        self._record(
            FieldMetricEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                operation="inject",
                context_id=context_id,
                agent_id=agent_id,
                latency_ms=elapsed_ms,
                pattern_key=key,
            )
        )

    async def query(
        self,
        context_id: str,
        query: str,
        agent_id: int,
        top_k: int = 10,
        record_access: bool = True,
    ) -> list[dict[str, Any]]:
        start = time.monotonic()
        results = await self._inner.query(
            context_id, query, agent_id, top_k, record_access=record_access,
        )
        elapsed_ms = (time.monotonic() - start) * 1000

        top_score = results[0].get("score") if results else None

        self._record(
            FieldMetricEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                # PRD-178 S2: distinguish a read-only trace from a live query in
                # telemetry (a trace neither reinforces nor should skew query KPIs).
                operation="query" if record_access else "trace",
                context_id=context_id,
                agent_id=agent_id,
                latency_ms=elapsed_ms,
                query_text=query,
                results_returned=len(results),
                top_score=top_score,
            )
        )
        return results

    async def context_exists(self, context_id: str) -> bool:
        """Pass through to inner adapter's existence check (no metric)."""
        if hasattr(self._inner, "context_exists"):
            return await self._inner.context_exists(context_id)
        return True  # Backends without exists check assume valid

    async def destroy_context(self, context_id: str) -> None:
        start = time.monotonic()
        await self._inner.destroy_context(context_id)
        elapsed_ms = (time.monotonic() - start) * 1000

        self._record(
            FieldMetricEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                operation="destroy",
                context_id=context_id,
                agent_id=0,
                latency_ms=elapsed_ms,
            )
        )

    def get_metrics(self, context_id: str) -> Optional[ExperimentMetrics]:
        return self._metrics.get(context_id)

    def get_all_metrics(self) -> dict[str, ExperimentMetrics]:
        return dict(self._metrics)
