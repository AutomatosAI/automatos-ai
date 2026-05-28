"""
ToolSignalRecorder (PRD-141 US-019)
====================================

Batched, incremental tool-execution signal recorder for the tool routing graph.

When a tool runs (success or failure), the ToolRouter enqueues a lightweight
signal onto an in-process ``asyncio.Queue``. A SINGLE background drain task
batches these and applies incremental upserts to ``tool_routing_edges`` /
``tool_routing_affinities`` using exactly ONE DB session per flush.

Why batched (NOT fire-and-forget)
----------------------------------
The original PRD-141 draft opened a DB session via ``asyncio.ensure_future`` on
*every* tool call, which exhausts the connection pool under load — the exact
failure mode Phase 1 fixes. This recorder NEVER opens a DB session per call and
NEVER creates a task per call: the drain task is a process singleton spawned
once, and ``record()`` only does a non-blocking ``put_nowait``.

Division of labour with the nightly edge_builder
-------------------------------------------------
* This recorder gives intra-day freshness. It ACCUMULATES evidence
  (``sample_count``, and edge ``weight`` as a raw count) in real time and sets a
  conservative PROVISIONAL confidence on brand-new rows.
* ``core/services/edge_builder.py`` is authoritative: nightly it RECOMPUTES
  weight + Wilson confidence from ``tool_execution_logs`` and SETs absolute
  values. So the recorder never overwrites a row's confidence on update (nightly
  owns it), and never inflates an ``agent_prefers`` weight (a normalized
  frequency the nightly edge_builder owns) — on update of an affinity it only bumps
  ``sample_count``.

Null-safe upsert
----------------
``uq_tre_full_key`` / ``uq_tra_full_key`` include nullable columns
(``workspace_id``, ``agent_id``, ``intent_cluster_id``). Postgres treats NULLs
as distinct, so a plain ``ON CONFLICT`` would NOT match a row whose scope column
is NULL and would insert a duplicate. The recorder always writes a NULL
``intent_cluster_id`` for affinities, so it uses ``UPDATE ... WHERE col IS NOT
DISTINCT FROM :col`` (NULL-safe equality) and only ``INSERT``s when no row
matched — guaranteeing "increment, no duplicate rows".

Leaf-loadable: module-top imports are stdlib-only; config / DB / wilson are
imported lazily inside methods so this module can be unit-tested under a
synthetic package without the DB-backed executor chain (matches graph_router.py).
"""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSignal:
    """One tool-execution outcome to fold into the routing graph."""

    action_name: str
    success: bool
    agent_id: Optional[int] = None
    workspace_id: Optional[str] = None
    prior_action: Optional[str] = None


def _wilson(successes: int, total: int) -> float:
    """Provisional confidence for brand-new rows (nightly edge_builder recomputes).

    Reuses the canonical Wilson lower bound rather than reimplementing it.
    """
    from core.services.edge_builder import wilson_lower_bound

    return wilson_lower_bound(successes, total)


class ToolSignalRecorder:
    """Process-singleton batched recorder. See module docstring."""

    def __init__(self) -> None:
        self._queue: Optional[asyncio.Queue] = None
        self._drain_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Config accessors (lazy — keep this module leaf-loadable)
    # ------------------------------------------------------------------

    @staticmethod
    def _enabled() -> bool:
        try:
            from config import config

            return bool(getattr(config, "TOOL_SIGNAL_RECORDER_ENABLED", False))
        except Exception:
            return False

    @staticmethod
    def _batch_size() -> int:
        try:
            from config import config

            return int(getattr(config, "TOOL_SIGNAL_FLUSH_BATCH_SIZE", 50))
        except Exception:
            return 50

    @staticmethod
    def _interval_seconds() -> float:
        try:
            from config import config

            return float(getattr(config, "TOOL_SIGNAL_FLUSH_INTERVAL_SECONDS", 5.0))
        except Exception:
            return 5.0

    @staticmethod
    def _queue_maxsize() -> int:
        try:
            from config import config

            return int(getattr(config, "TOOL_SIGNAL_QUEUE_MAXSIZE", 10000))
        except Exception:
            return 10000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, signal: ToolSignal) -> None:
        """NON-BLOCKING enqueue from the tool hot path.

        No DB, no per-call task. Drops the signal silently if the recorder is
        disabled, if there is no running event loop, or if the bounded queue is
        full — telemetry is best-effort and must never block or fail a tool call.
        """
        if not self._enabled():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # not on an event loop -> skip
        self._ensure_started(loop)
        try:
            self._queue.put_nowait(signal)
        except asyncio.QueueFull:
            logger.debug(
                "ToolSignalRecorder: queue full, dropping signal for %s",
                signal.action_name,
            )

    def _ensure_started(self, loop: asyncio.AbstractEventLoop) -> None:
        """Create the queue and the SINGLE drain task, once.

        The drain task is spawned exactly once per loop (guarded), NOT per tool
        call — this is the whole point of the batched design.
        """
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self._queue_maxsize())
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = loop.create_task(self._drain_loop())

    # ------------------------------------------------------------------
    # Background drain
    # ------------------------------------------------------------------

    async def _drain_loop(self) -> None:
        while True:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._flush(batch)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # never let the drain loop die
                self._record_flush_error(e)
                await asyncio.sleep(1)

    async def _collect_batch(self) -> List[ToolSignal]:
        """Block for the first signal, then drain up to batch_size or until
        interval seconds elapse — whichever comes first."""
        batch_size = self._batch_size()
        interval = self._interval_seconds()

        first = await self._queue.get()
        batch: List[ToolSignal] = [first]

        loop = asyncio.get_running_loop()
        deadline = loop.time() + interval
        while len(batch) < batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break
        return batch

    # ------------------------------------------------------------------
    # Aggregation (pure) + flush (one session)
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(
        batch: List[ToolSignal],
    ) -> Tuple[Dict[tuple, int], Dict[tuple, int]]:
        """Collapse a batch into incremental upsert counts (pure, no DB).

        Returns ``(edge_counts, affinity_counts)``:
            edge_counts[(from_action, to_action, edge_type, agent_id, ws)] = inc
            affinity_counts[(action_name, affinity_type, agent_id, ws)] = inc

        success -> used_after edge (prior->action) + agent_prefers affinity
        failure -> failed_after edge (prior->action) + fails_for_intent affinity

        An edge is only produced when ``prior_action`` is present (an edge needs
        two endpoints) and is not a self-transition (matches edge_builder); the
        single-action affinity is always produced.
        """
        edge_counts: Dict[tuple, int] = {}
        aff_counts: Dict[tuple, int] = {}
        for s in batch:
            if s.success:
                edge_type, affinity_type = "used_after", "agent_prefers"
            else:
                edge_type, affinity_type = "failed_after", "fails_for_intent"

            ak = (s.action_name, affinity_type, s.agent_id, s.workspace_id)
            aff_counts[ak] = aff_counts.get(ak, 0) + 1

            if s.prior_action and s.prior_action != s.action_name:
                ek = (s.prior_action, s.action_name, edge_type, s.agent_id, s.workspace_id)
                edge_counts[ek] = edge_counts.get(ek, 0) + 1

        return edge_counts, aff_counts

    async def _flush(self, batch: List[ToolSignal]) -> None:
        """Apply one batch with exactly ONE DB session."""
        edge_counts, aff_counts = self._aggregate(batch)
        if not edge_counts and not aff_counts:
            return

        now = datetime.utcnow()
        try:
            from core.database.database import get_db_session

            with get_db_session() as db:  # exactly ONE session for the whole batch
                for (from_action, to_action, edge_type, agent_id, ws), inc in edge_counts.items():
                    self._upsert_edge(db, from_action, to_action, edge_type, ws, agent_id, inc, now)
                for (action_name, affinity_type, agent_id, ws), inc in aff_counts.items():
                    self._upsert_affinity(db, action_name, affinity_type, ws, agent_id, inc, now)
                db.flush()
        except Exception as e:
            self._record_flush_error(e)

    # ------------------------------------------------------------------
    # Null-safe incremental upserts
    # ------------------------------------------------------------------

    @staticmethod
    def _upsert_edge(
        db,
        from_action: str,
        to_action: str,
        edge_type: str,
        workspace_id: Optional[str],
        agent_id: Optional[int],
        inc: int,
        now: datetime,
    ) -> None:
        """Increment an existing edge or insert a new one. weight is a raw count
        (matches edge_builder); confidence is left to the nightly recompute."""
        from sqlalchemy import text

        params = {
            "from_action": from_action,
            "to_action": to_action,
            "edge_type": edge_type,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "inc": inc,
            "now": now,
        }
        update_stmt = text("""
            UPDATE tool_routing_edges
            SET sample_count = tool_routing_edges.sample_count + :inc,
                weight = tool_routing_edges.weight + :inc,
                last_updated = :now
            WHERE from_action = :from_action
              AND to_action = :to_action
              AND edge_type = :edge_type
              AND workspace_id IS NOT DISTINCT FROM :workspace_id
              AND agent_id IS NOT DISTINCT FROM :agent_id
        """)
        result = db.execute(update_stmt, params)
        if (getattr(result, "rowcount", 0) or 0) > 0:
            return

        insert_stmt = text("""
            INSERT INTO tool_routing_edges
                (from_action, to_action, edge_type, workspace_id, agent_id,
                 weight, confidence, sample_count, last_updated)
            VALUES
                (:from_action, :to_action, :edge_type, :workspace_id, :agent_id,
                 :inc, :confidence, :inc, :now)
        """)
        db.execute(insert_stmt, {**params, "confidence": _wilson(inc, inc)})

    @staticmethod
    def _upsert_affinity(
        db,
        action_name: str,
        affinity_type: str,
        workspace_id: Optional[str],
        agent_id: Optional[int],
        inc: int,
        now: datetime,
    ) -> None:
        """Increment an existing affinity's sample_count or insert a new one.

        On update, weight + confidence are left untouched (the nightly edge_builder owns
        them — agent_prefers is a normalized frequency that must not be inflated
        by a raw real-time count). The recorder only ever writes a NULL
        intent_cluster_id, so the match pins ``intent_cluster_id IS NULL``.
        """
        from sqlalchemy import text

        params = {
            "action_name": action_name,
            "affinity_type": affinity_type,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "inc": inc,
            "now": now,
        }
        update_stmt = text("""
            UPDATE tool_routing_affinities
            SET sample_count = tool_routing_affinities.sample_count + :inc,
                last_updated = :now
            WHERE action_name = :action_name
              AND affinity_type = :affinity_type
              AND workspace_id IS NOT DISTINCT FROM :workspace_id
              AND agent_id IS NOT DISTINCT FROM :agent_id
              AND intent_cluster_id IS NULL
        """)
        result = db.execute(update_stmt, params)
        if (getattr(result, "rowcount", 0) or 0) > 0:
            return

        provisional = _wilson(inc, inc)
        insert_stmt = text("""
            INSERT INTO tool_routing_affinities
                (action_name, affinity_type, workspace_id, agent_id,
                 intent_cluster_id, weight, confidence, sample_count, last_updated)
            VALUES
                (:action_name, :affinity_type, :workspace_id, :agent_id,
                 NULL, :provisional, :provisional, :inc, :now)
        """)
        db.execute(insert_stmt, {**params, "provisional": provisional})

    @staticmethod
    def _record_flush_error(error: Exception) -> None:
        try:
            from core.utils.exception_telemetry import record_error

            record_error(subsystem="routing", operation="tool_signal_flush", error=error)
        except Exception:
            logger.warning("ToolSignalRecorder flush failed: %s", error)


# ======================================================================
# Singleton factory
# ======================================================================

_instance_lock = threading.Lock()
_instance: Optional[ToolSignalRecorder] = None


def get_tool_signal_recorder() -> ToolSignalRecorder:
    """Process-singleton factory (matches get_graph_router pattern)."""
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is None:
            _instance = ToolSignalRecorder()
    return _instance
