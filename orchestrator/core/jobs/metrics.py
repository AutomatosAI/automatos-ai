"""The queue's metrics (PRD-276 §10-7), over the PRD-256 OpenTelemetry meter.

Two gauges, read from the table each time the meter collects (once a minute):

* ``automatos.jobs.count`` ({job}): jobs per ``job.kind`` and ``job.status`` (queued, running, dead),
  so a backlog or a growing pile of dead jobs shows before anyone asks;
* ``automatos.jobs.oldest_due_wait`` (s): per ``job.kind``, how long the oldest due, unclaimed job
  has waited, the first sign that workers can't keep up.

Their attributes are the kind and the status only, never a workspace. Only a process with the
``worker`` role registers them (one per worker, not one per API process), and only when
OpenTelemetry is on; off, nothing is imported or read. A read that fails is logged and reports
nothing for that collection.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Tuple

from sqlalchemy import text

logger = logging.getLogger(__name__)

METER_NAME = "automatos.jobs"
COUNT = "automatos.jobs.count"
OLDEST_DUE_WAIT = "automatos.jobs.oldest_due_wait"
REPORTED_STATUSES = ("queued", "running", "dead")  # done rows are pruned history, not load

_COUNTS = text("SELECT kind, status, count(*) FROM jobs.jobs WHERE status IN ('queued', 'running', 'dead') "
               "GROUP BY kind, status")
_WAITS = text("SELECT kind, extract(epoch FROM clock_timestamp() - min(run_at)) FROM jobs.jobs "
              "WHERE status = 'queued' AND run_at <= clock_timestamp() GROUP BY kind")


def read_counts(db: Any) -> Dict[Tuple[str, str], int]:
    """(kind, status) to how many jobs, for the statuses that are load."""
    return {(kind, status): int(n) for kind, status, n in db.execute(_COUNTS).all()}


def read_waits(db: Any) -> Dict[str, float]:
    """Kind to the seconds its oldest due, unclaimed job has waited; kinds with none due are absent."""
    return {kind: max(0.0, float(seconds)) for kind, seconds in db.execute(_WAITS).all()}


def _observed(read: Any, to_points: Any) -> List[Any]:
    """One collection: ``read`` on its own session, as observations; nothing when it fails."""
    from core.jobs.worker import on_own_session

    try:
        return to_points(on_own_session(read))
    except Exception:  # noqa: BLE001 — logged; this collection reports nothing, the next tries again
        logger.exception("[jobs] queue metrics not read")
        return []


def _count_points(counts: Dict[Tuple[str, str], int]) -> List[Any]:
    from opentelemetry.metrics import Observation

    return [Observation(n, {"job.kind": kind, "job.status": status}) for (kind, status), n in counts.items()]


def _wait_points(waits: Dict[str, float]) -> List[Any]:
    from opentelemetry.metrics import Observation

    return [Observation(seconds, {"job.kind": kind}) for kind, seconds in waits.items()]


def observe_queue() -> bool:
    """Registers the two gauges on this process's meter, when OpenTelemetry is on; True when it did."""
    from core.observability.otel import tracing_enabled

    if not tracing_enabled():
        return False
    try:
        from opentelemetry import metrics

        meter = metrics.get_meter(METER_NAME)
        meter.create_observable_gauge(COUNT, callbacks=[_counts_callback], unit="{job}",
                                      description="Jobs per kind and status")
        meter.create_observable_gauge(OLDEST_DUE_WAIT, callbacks=[_waits_callback], unit="s",
                                      description="How long each kind's oldest due job has waited")
    except Exception:  # noqa: BLE001 — logged; the queue runs without its metrics
        logger.exception("[jobs] queue metrics not registered")
        return False
    return True


def _counts_callback(options: Any) -> Iterable[Any]:
    return _observed(read_counts, _count_points)


def _waits_callback(options: Any) -> Iterable[Any]:
    return _observed(read_waits, _wait_points)
