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

Clean shutdown & the bounded-loss window (PRD-232 US-009, PRD-142 W4-S9)
-----------------------------------------------------------------------
``stop()`` is the clean-shutdown path: it halts the drain loop and flushes every
signal still queued (the in-flight batch AND anything left in the queue) in one
final session, so a graceful stop loses NOTHING. The honest bound: the queue is
an in-process ``asyncio.Queue`` — a HARD crash (SIGKILL / OOM), not a clean
stop(), drops whatever it holds. That is acceptable BY DESIGN and is not a fake
durability claim: ``tool_execution_logs`` is the durable ground truth, and the
nightly ``edge_builder`` RECOMPUTES authoritative edges/affinities from it, so a
lost intra-day batch only delays freshness — it never corrupts the learned graph.

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

# Sentinel enqueued by stop() to wake a drain loop blocked on an empty queue so
# it can flush its in-flight batch and exit (never persisted).
_STOP_SENTINEL = object()

# PRD-232 US-011b: the synthetic action name for a persisted surfaced-set
# observation. WIRE PROTOCOL — must match core.services.edge_builder._TOOL_SHOWN_ACTION
# (the nightly reader) and telemetry.TOOL_GAP_ACTION's sibling. A drift-guard test
# asserts they agree. Kept as a local literal so this module stays leaf-loadable
# (stdlib-only top imports; edge_builder pulls numpy).
_TOOL_SHOWN_ACTION = "__tool_shown__"
# telemetry_source for the shown row — NOT 'production' (would pollute the
# success-rate SLO / silence canary). Must match telemetry.SYNTHETIC_SIGNAL_SOURCE.
_SYNTHETIC_SIGNAL_SOURCE = "synthetic_signal"


@dataclass(frozen=True)
class ToolSignal:
    """One tool-execution outcome to fold into the routing graph."""

    action_name: str
    success: bool
    agent_id: Optional[int] = None
    workspace_id: Optional[str] = None
    prior_action: Optional[str] = None


@dataclass(frozen=True)
class SelectionSignal:
    """PRD-232 US-011b: one surfaced-set observation to persist durably.

    Enqueued by ``record_selection`` alongside the in-memory stash, drained by the
    SAME batched recorder (one DB session per flush — the PRD-141 US-019 contract),
    and written as a ``__tool_shown__`` row on tool_execution_logs. The nightly
    edge_builder reads these for the shown-vs-used decay. ``shown_actions`` is a
    tuple (hashable/frozen); ``query`` is what clusters the observation.
    """

    query: str
    shown_actions: tuple
    workspace_id: Optional[str] = None
    agent_id: Optional[int] = None


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
        # PRD-232 US-009: set by stop() so the drain loop exits after flushing
        # its in-flight batch (clean-shutdown flush, no lost queued signals).
        self._stopping: bool = False
        # Process-lifetime observability counters for the self-learning tile.
        # Restart note: these and the queue are in-memory, so a restart loses any
        # *queued* signals — safe by design: the nightly edge_builder RECOMPUTES
        # authoritative edges/affinities from the durable tool_execution_logs, so
        # no authoritative learning is lost on restart (the recorder only provides
        # intra-day freshness).
        self._stats: Dict[str, int] = {
            "recorded": 0, "dropped": 0, "flushes": 0, "flush_errors": 0,
            "selection_narrowed": 0, "selection_fallback": 0,
        }
        # PRD-143 S14: last selection outcome per (workspace_id, agent_id) —
        # written by get_tools_for_agent when it builds the dispatcher schema,
        # peeked by the platform_execute dispatch so the universal telemetry
        # hook can persist hit/fallback per execution row. Pure in-memory
        # (no DB, no event loop), bounded FIFO, last-write-wins per key.
        # Best-effort by design: a dispatch with no recorded surface (other
        # process, restart) simply carries no selection outcome.
        self._last_selection: Dict[tuple, Dict[str, object]] = {}

    def stats(self) -> Dict[str, int]:
        """Observability snapshot: signals recorded vs dropped, flush successes
        vs failures, selection outcomes (narrowed vs fallback, PRD-143 S14),
        and live queue depth — the numbers the self-learning tile
        (W4-S16) reads to answer 'is tool-routing learning healthy?'."""
        depth = self._queue.qsize() if self._queue is not None else 0
        return {**self._stats, "queue_depth": depth}

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

    @staticmethod
    def _selection_stash_maxsize() -> int:
        try:
            from config import config

            return int(getattr(config, "TOOL_SELECTION_STASH_MAXSIZE", 512))
        except Exception:
            return 512

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
            self._stats["dropped"] += 1
            return  # not on an event loop -> skip
        self._ensure_started(loop)
        try:
            self._queue.put_nowait(signal)
            self._stats["recorded"] += 1
        except asyncio.QueueFull:
            self._stats["dropped"] += 1
            logger.debug(
                "ToolSignalRecorder: queue full, dropping signal for %s",
                signal.action_name,
            )

    # ------------------------------------------------------------------
    # PRD-143 S14: per-selection outcome (narrowed vs fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _selection_key(workspace_id, agent_id) -> tuple:
        """Normalize stash keys across callers (UUID vs str workspace,
        0/None agent)."""
        ws = str(workspace_id) if workspace_id else None
        aid = int(agent_id) if agent_id else None
        return (ws, aid)

    def record_selection(
        self,
        *,
        workspace_id=None,
        agent_id=None,
        narrowed: bool,
        reason: Optional[str] = None,
        allowed_names: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> None:
        """Record one dispatcher-selection outcome (PRD-143 S14 + PRD-232 US-011b).

        Called by ``get_tools_for_agent`` where the tool-trace log used to be
        the only record of narrowed-vs-not-narrowed. Bumps the process-lifetime
        counters and stashes the outcome so the next ``platform_execute``
        dispatch for this (workspace, agent) can attach hit/fallback telemetry.

        PRD-232 US-011b: when the surface was NARROWED to a specific set for a
        ``query``, ALSO persist that surfaced set durably — enqueued to the same
        batched recorder as a ``SelectionSignal`` (one DB session per flush) so the
        nightly can compute shown-vs-used and decay never-used affinities. Only the
        targeted (narrowed) surface is persisted: the full non-narrowed catalog is
        not a meaningful 'shown' signal and would decay everything.

        Pure in-memory for the stash, best-effort for the durable enqueue; never
        raises — selection telemetry must never break the surface build.
        """
        try:
            self._stats["selection_narrowed" if narrowed else "selection_fallback"] += 1
            stash = self._last_selection
            while len(stash) >= self._selection_stash_maxsize():
                stash.pop(next(iter(stash)))  # FIFO eviction
            entry: Dict[str, object] = {
                "narrowed": bool(narrowed),
                "reason": reason,
                "allowed": frozenset(allowed_names or ()),
                "enum_size": len(allowed_names) if narrowed and allowed_names else None,
            }
            key = self._selection_key(workspace_id, agent_id)
            stash.pop(key, None)  # re-insert so FIFO order tracks recency
            stash[key] = entry
        except Exception:  # pragma: no cover - defensive, never blocks the hot path
            logger.debug("ToolSignalRecorder: record_selection failed", exc_info=True)

        # Durable shown-set persistence (US-011b): best-effort, batched.
        if not (query and allowed_names and self._enabled()):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # off-loop (sync context) — skip the durable half, keep the stash
        try:
            self._ensure_started(loop)
            self._queue.put_nowait(SelectionSignal(
                query=query,
                shown_actions=tuple(allowed_names),
                workspace_id=str(workspace_id) if workspace_id else None,
                agent_id=int(agent_id) if agent_id else None,
            ))
        except asyncio.QueueFull:
            self._stats["dropped"] += 1
        except Exception:  # pragma: no cover - defensive, never blocks the hot path
            logger.debug("ToolSignalRecorder: shown-set enqueue failed", exc_info=True)

    def peek_selection(self, *, workspace_id=None, agent_id=None) -> Optional[Dict[str, object]]:
        """Return the last recorded selection outcome for this
        (workspace, agent), or None when no surface was recorded in-process.
        Peek (not consume): one surface may serve several dispatches in a
        single tool-loop turn."""
        try:
            return self._last_selection.get(self._selection_key(workspace_id, agent_id))
        except Exception:  # pragma: no cover - defensive
            return None

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
        while not self._stopping:
            try:
                batch = await self._collect_batch()
                # A stop() sentinel may ride in the batch — flush the real
                # signals beside it, then the while-guard exits the loop.
                batch = [s for s in batch if s is not _STOP_SENTINEL]
                if batch:
                    await self._flush(batch)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # never let the drain loop die
                self._record_flush_error(e)
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Clean-shutdown flush (US-009): stop the drain loop and flush every
        still-queued signal — the in-flight batch and the queue remainder — in
        one final session, so a graceful stop loses nothing.

        Idempotent and safe if the recorder never started (no queue/task). See
        the module docstring for the honest bounded-loss window (a HARD crash,
        not stop(), drops the in-process queue)."""
        self._stopping = True
        task = self._drain_task
        self._drain_task = None
        if task is not None and not task.done():
            # Wake _collect_batch if it is blocked on an empty queue so it can
            # flush its in-flight batch and see the stop guard.
            if self._queue is not None:
                try:
                    self._queue.put_nowait(_STOP_SENTINEL)
                except asyncio.QueueFull:
                    pass
            try:
                await task
            except asyncio.CancelledError:
                pass
        # Flush anything the loop left queued (drain to empty, ONE session).
        if self._queue is not None:
            remaining: List[ToolSignal] = []
            while True:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is not _STOP_SENTINEL:
                    remaining.append(item)
            if remaining:
                await self._flush(remaining)
        self._stopping = False

    async def _collect_batch(self) -> List[ToolSignal]:
        """Block for the first signal, then drain up to batch_size or until
        interval seconds elapse — whichever comes first."""
        batch_size = self._batch_size()
        interval = self._interval_seconds()

        first = await self._queue.get()
        # PRD-232 US-009: stop()'s sentinel returns the loop to its guard at once
        # (as first, or mid-collect) instead of waiting out the flush interval.
        if first is _STOP_SENTINEL:
            return [first]
        batch: List[ToolSignal] = [first]

        loop = asyncio.get_running_loop()
        deadline = loop.time() + interval
        while len(batch) < batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if item is _STOP_SENTINEL:
                break  # flush the real batch now; don't wait for more
            batch.append(item)
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

    async def _flush(self, batch: List) -> None:
        """Apply one batch with exactly ONE DB session.

        A batch may mix ``ToolSignal`` (→ edges/affinities) and ``SelectionSignal``
        (→ ``__tool_shown__`` rows, PRD-232 US-011b). Both are written in the SAME
        session so the one-session-per-flush contract (PRD-141 US-019) holds.
        """
        tool_signals = [s for s in batch if isinstance(s, ToolSignal)]
        selection_signals = [s for s in batch if isinstance(s, SelectionSignal)]
        edge_counts, aff_counts = self._aggregate(tool_signals)
        if not edge_counts and not aff_counts and not selection_signals:
            return

        now = datetime.utcnow()
        try:
            from core.database.database import get_db_session

            with get_db_session() as db:  # exactly ONE session for the whole batch
                for (from_action, to_action, edge_type, agent_id, ws), inc in edge_counts.items():
                    self._upsert_edge(db, from_action, to_action, edge_type, ws, agent_id, inc, now)
                for (action_name, affinity_type, agent_id, ws), inc in aff_counts.items():
                    self._upsert_affinity(db, action_name, affinity_type, ws, agent_id, inc, now)
                for sel in selection_signals:
                    self._insert_shown_row(db, sel, now)
                db.flush()
            self._stats["flushes"] += 1
        except Exception as e:
            self._stats["flush_errors"] += 1
            self._record_flush_error(e)

    @staticmethod
    def _insert_shown_row(db, sel: "SelectionSignal", now: datetime) -> None:
        """Persist one surfaced-set observation as a __tool_shown__ telemetry row
        (PRD-232 US-011b) — router_decision.candidates carries the shown action
        names, user_query clusters it. status='shown' keeps it out of edges and
        normal affinities; the nightly shown-not-used decay is its only reader."""
        import json
        from sqlalchemy import text

        db.execute(
            text("""
                INSERT INTO tool_execution_logs
                    (agent_id, app_name, action_name, workspace_id, user_query,
                     status, router_decision, telemetry_source, executed_at)
                VALUES
                    (:agent_id, 'PLATFORM', :action, :workspace_id, :user_query,
                     'shown', CAST(:router AS JSONB), :source, :now)
            """),
            {
                "agent_id": sel.agent_id,
                "action": _TOOL_SHOWN_ACTION,
                "workspace_id": sel.workspace_id,
                "user_query": sel.query,
                "router": json.dumps({"candidates": list(sel.shown_actions or ())}),
                # NOT 'production' — a shown row is a signal, not a tool execution;
                # keep it out of the success-rate SLO / silence canary. Must match
                # telemetry.SYNTHETIC_SIGNAL_SOURCE (leaf-load: kept as a literal).
                "source": _SYNTHETIC_SIGNAL_SOURCE,
                "now": now,
            },
        )

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
