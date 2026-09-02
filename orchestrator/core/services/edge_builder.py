"""
PRD-139 US-003: Edge + affinity builder service (nightly job).

Turns tool execution telemetry into routing graph edges and affinities.
Reads from ToolExecutionLog, writes to ToolRoutingEdge / ToolRoutingAffinity /
ToolRoutingIntentCluster.

Key properties:
- Idempotent: running twice on same data produces same results (upserts)
- Deterministic: K-means uses random_state=42
- Recency: controlled by window parameter only, no continuous time-decay
- Confidence: Wilson lower bound (95% CI), not raw frequency

PRD-232 §6.5 — the TWO-LAYER graph (RVW-2, Gerard's ruling):
- used_after edges are written per-tenant (keyed by each log's own workspace_id).
- On a FULL (all-workspaces) recompute, a TEXT-FREE GLOBAL layer is ALSO written:
  one aggregated used_after edge per (from,to), summed across tenants, at
  workspace_id=NULL. GraphRouter reads it as a reduced-weight cross-tenant prior a
  zero-telemetry tenant rides. A SCOPED (single --workspace-id) run writes NO global
  rows (a one-tenant aggregate would just be that tenant's data relabeled).
- PRIVACY / PRD-181 erasure: the global layer is text-free by construction — global
  edges/affinities carry only action names + counts, and organic intent clusters
  (which are global — no workspace_id column) redact sample_query to an action-name
  label (see _redacted_cluster_label). So GDPR erasure scope stays PER-TENANT rows
  only; nothing in the global layer identifies a user.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import and_, text
from sqlalchemy.orm import Session

from core.database.database import get_db_session
from core.llm.embedding_manager import get_embedding_manager
from core.models.composio_cache import ToolExecutionLog
from core.models.tool_routing import (
    ToolRoutingAffinity,
    ToolRoutingEdge,
    ToolRoutingIntentCluster,
)
from core.services.intent_clustering import compute_intent_clusters

logger = logging.getLogger(__name__)

# Floor for computing affinities: ignore agents/clusters with fewer observations
_SAMPLE_FLOOR = 3

# PRD-232 §6.5: a "global" prior edge is a CROSS-TENANT aggregate — it must draw on
# at least this many distinct workspaces before it is written, so a pair unique to a
# single tenant never becomes everyone's prior (the full-run analogue of the
# scoped-run leak the write guard already prevents).
_GLOBAL_MIN_TENANTS = 2

# PRD-232 US-011: synthetic telemetry markers on the EXISTING tool_execution_logs
# table (no new table). A tool_gap row records that the model hunted for a
# capability (platform_find_tools) or a tool-requiring turn ran zero platform
# tools; a tool_shown row records the surfaced action set (shown-vs-used decay).
# These are NOT tool executions, so they are excluded from used_after/failed_after
# edges and from the normal succeeds/fails/agent_prefers affinities — they are
# consumed ONLY by the gap-resolution join and the shown-not-used decay. Their
# QUERIES still cluster (the gap's intent must have a cluster to attribute to);
# their non-'success' status keeps them out of any cluster's action_names_hot.
_TOOL_GAP_ACTION = "__tool_gap__"
_TOOL_SHOWN_ACTION = "__tool_shown__"
_SYNTHETIC_ACTIONS = frozenset({_TOOL_GAP_ACTION, _TOOL_SHOWN_ACTION})

# Defaults (build_edges overrides from config — never hardcoded at the call site).
_GAP_RESOLUTION_WINDOW = timedelta(hours=24)  # gap → resolution look-ahead
_SHOWN_DECAY_FACTOR = 0.9   # geometric erosion per shown-not-used excess
_AFFINITY_WEIGHT_FLOOR = 0.5  # decay never drives an affinity below this


@dataclass
class EdgeBuildSummary:
    """Summary returned after an edge-build run."""

    edges_built: int = 0
    global_edges_built: int = 0  # PRD-232 §6.5: text-free cross-tenant used_after prior
    failed_edges_built: int = 0
    affinities_built: int = 0
    intent_clusters: int = 0
    gap_resolutions_built: int = 0  # PRD-232 US-011: gap→resolution affinities
    affinities_decayed: int = 0     # PRD-232 US-011: shown-not-used decays applied
    logs_processed: int = 0
    duration_ms: int = 0


def _gap_resolution_window() -> timedelta:
    """gap→resolution look-ahead window from config (PRD-232 US-011)."""
    try:
        from config import config
        return timedelta(hours=float(getattr(config, "TOOL_ROUTING_GAP_RESOLUTION_HOURS", 24)))
    except Exception:
        return _GAP_RESOLUTION_WINDOW


def _shown_decay_factor() -> float:
    """Geometric shown-not-used decay factor from config (PRD-232 US-011)."""
    try:
        from config import config
        return float(getattr(config, "TOOL_ROUTING_SHOWN_DECAY_FACTOR", _SHOWN_DECAY_FACTOR))
    except Exception:
        return _SHOWN_DECAY_FACTOR


def _affinity_weight_floor() -> float:
    """Floor the shown-not-used decay never drops below, from config (US-011)."""
    try:
        from config import config
        return float(getattr(config, "TOOL_ROUTING_AFFINITY_WEIGHT_FLOOR", _AFFINITY_WEIGHT_FLOOR))
    except Exception:
        return _AFFINITY_WEIGHT_FLOOR


def derive_embedding_model_key(embedding_manager) -> str:
    """Canonical ``provider:model:dimension`` key for the active embedding model.

    The intent-cluster centroid is only comparable to a query embedded under the
    SAME model, so every ToolRoutingIntentCluster records this key and the reader
    (GraphRouter._match_intent_cluster) filters on it. Shared by the nightly
    recompute and the human-applied corpus seed (PRD-232 US-007) so both stamp an
    identical key — a seeded centroid and an organic one for the same model are
    matched against the same live queries.
    """
    ensure = getattr(embedding_manager, "_ensure_provider", None)
    if callable(ensure):
        ensure()
    info = embedding_manager.get_provider_info()
    provider = info.get("provider") or embedding_manager.__class__.__name__
    model = info.get("model")
    if model is None:
        cfg = getattr(embedding_manager.provider, "config", None)
        model = getattr(cfg, "model", None)
    dimension = info.get("dimension") or embedding_manager.get_dimension()
    return f"{provider}:{model}:{dimension}"


def _redacted_cluster_label(action_names_hot: List[str]) -> str:
    """PRD-232 §6.5 (RVW-2, privacy hard rule): organic intent clusters are GLOBAL
    (the ToolRoutingIntentCluster table has no workspace_id) and cross tenants as a
    text-free prior — so they must NOT store a raw user query in ``sample_query``.

    Redact to a NON-identifying, action-name label (action names are explicitly
    allowed in the global layer): the cluster's top hot action, else ``(organic)``.
    ``sample_query`` is display/debug metadata only — routing matches on the centroid
    vector + ``action_names_hot``, never on this text — so redaction changes nothing
    live while keeping the global layer text-free by construction (PRD-181 erasure
    then never needs to touch a global cluster row). Seeded clusters keep their
    SYNTHETIC utterance (authored, not user text) — see seed_tool_routing_graph.py.
    """
    if action_names_hot:
        return f"(organic:{action_names_hot[0]})"
    return "(organic)"


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Wilson lower bound at 95% confidence."""
    if total == 0:
        return 0.0
    p = successes / total
    denominator = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    return (centre - spread) / denominator


async def build_edges(
    window: timedelta = timedelta(days=30),
    workspace_id: Optional[str] = None,
) -> EdgeBuildSummary:
    """Main entry point: read telemetry, compute edges + affinities, upsert.

    Args:
        window: How far back to look in tool_execution_logs.
        workspace_id: Optional workspace UUID string — restrict the recompute
            to one workspace's logs (PRD-143 S12 seed backfill scoping).

    Returns:
        EdgeBuildSummary with counts of what was built.
    """
    import time

    start = time.monotonic()
    summary = EdgeBuildSummary()

    with get_db_session() as db:
        # 1. Load execution logs within window
        cutoff = datetime.utcnow() - window
        logs = _load_logs(db, cutoff, workspace_id=workspace_id)
        summary.logs_processed = len(logs)

        if not logs:
            logger.info("EdgeBuilder: no logs in window, nothing to build")
            return summary

        # PRD-232 US-011: synthetic gap/shown rows are telemetry markers, not tool
        # executions — exclude them from co-occurrence edges + normal affinities.
        # (Clustering still sees the FULL list so a gap's intent gets a cluster;
        # _compute_affinities skips them internally to stay index-aligned.)
        real_logs = [l for l in logs if l.get("action_name") not in _SYNTHETIC_ACTIONS]

        # 2. Compute used_after edges from sequences (keyed by each log's own
        #    workspace_id — the per-tenant layer of PRD-232 §6.5's two-layer graph).
        edge_data = _compute_used_after_edges(real_logs)
        summary.edges_built = _upsert_edges(db, edge_data)

        # 2a. PRD-232 §6.5 WRITE path: on a FULL (all-workspaces) recompute, also write
        #     a TEXT-FREE GLOBAL used_after edge per (from,to) — the per-tenant counts
        #     summed across every workspace. This is the cross-tenant PRIOR a
        #     zero-telemetry tenant rides (GraphRouter admits it at reduced weight). A
        #     SCOPED run (a single --workspace-id) writes NO global rows: a one-tenant
        #     "aggregate" would just be that tenant's data relabeled global (a leak).
        #     `not workspace_id` matches _load_logs' own scope check (a falsy id — None
        #     or "" — is the full, all-workspaces recompute).
        if not workspace_id:
            summary.global_edges_built = _upsert_global_edges(db, edge_data)

        # 2b. Compute failed_after edges (PRD-141 US-018): A succeeded then a
        #     tool within 2 steps errored. Same table, distinct edge_type;
        #     GraphRouter._query_edges reads these as a de-ranking penalty
        #     (PRD-232 US-010c) but never expands them into chains.
        failed_data = _compute_failed_after_edges(real_logs)
        summary.failed_edges_built = _upsert_failed_after_edges(db, failed_data)

        # 3. Compute intent clusters from query embeddings (ALL logs — a tool_gap
        #    row's query must cluster so the resolution join can attribute to it)
        cluster_map = await _compute_and_upsert_clusters(db, logs)
        summary.intent_clusters = len(cluster_map)

        # 4. Compute affinities (succeeds/fails for intent, agent_prefers)
        affinities = _compute_affinities(logs, cluster_map)

        # 4b. PRD-232 US-011(c): gap→resolution join — a tool_gap answered later in
        #     the same conversation becomes a succeeds_for_intent for the resolving
        #     action, merged into (not colliding with) the organic affinities.
        gap_affinities = _compute_gap_resolution_affinities(
            logs, cluster_map, _gap_resolution_window()
        )
        summary.gap_resolutions_built = len(gap_affinities)
        affinities = _merge_affinities(affinities, gap_affinities)

        # 4c. PRD-232 US-011(b): shown-not-used decay — an action surfaced in an
        #     intent cluster far more than it was used loses boost (never below
        #     the floor).
        affinities, summary.affinities_decayed = _apply_shown_not_used_decay(
            affinities, logs, cluster_map,
            _shown_decay_factor(), _affinity_weight_floor(),
        )
        summary.affinities_built = _upsert_affinities(db, affinities)

    elapsed_ms = int((time.monotonic() - start) * 1000)
    summary.duration_ms = elapsed_ms

    logger.info(
        f"EdgeBuilder: built {summary.edges_built} used_after edges "
        f"(+{summary.global_edges_built} global-prior), "
        f"{summary.failed_edges_built} failed_after edges, "
        f"{summary.affinities_built} affinities across "
        f"{summary.intent_clusters} intent clusters "
        f"({summary.gap_resolutions_built} gap→resolution, "
        f"{summary.affinities_decayed} shown-not-used decayed)"
    )
    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_logs(
    db: Session,
    cutoff: datetime,
    workspace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load tool_execution_logs rows from cutoff onwards, return as dicts."""
    query = db.query(ToolExecutionLog).filter(ToolExecutionLog.executed_at >= cutoff)
    if workspace_id:
        query = query.filter(ToolExecutionLog.workspace_id == workspace_id)
    rows = query.order_by(ToolExecutionLog.executed_at.asc()).all()
    results = []
    for row in rows:
        # Extract turn_id from router_decision JSONB
        router = row.router_decision or {}
        # PRD-232 US-011: the surfaced action set (router_decision.candidates) is
        # the 'shown' half of shown-vs-used decay. Present on execution rows and
        # on the synthetic __tool_shown__ rows record_selection persists.
        shown = router.get("candidates")
        results.append({
            "id": row.id,
            "agent_id": row.agent_id,
            "workspace_id": str(row.workspace_id) if row.workspace_id else None,
            "action_name": row.action_name,
            "app_name": row.app_name,
            "status": row.status,
            "user_query": row.user_query,
            "executed_at": row.executed_at,
            "turn_id": router.get("turn_id"),
            "conversation_id": router.get("conversation_id"),
            "shown_actions": list(shown) if isinstance(shown, (list, tuple)) else [],
        })
    return results


def _derive_session_key(log: Dict[str, Any]) -> str:
    """Derive a grouping key to identify tool call sequences.

    Priority: turn_id > conversation_id > (agent_id + workspace_id + 5min window)
    """
    if log.get("turn_id"):
        return f"turn:{log['turn_id']}"
    if log.get("conversation_id"):
        return f"conv:{log['conversation_id']}"
    # Fallback: group by agent + workspace (sequences within will be time-bucketed)
    return f"agent:{log.get('agent_id')}:ws:{log.get('workspace_id')}"


def _compute_used_after_edges(
    logs: List[Dict[str, Any]],
) -> Dict[Tuple[str, str, Optional[str], Optional[int]], int]:
    """Compute used_after(A, B) edge counts.

    Groups logs by session (turn_id / conversation_id) and counts sequential pairs.
    For fallback grouping (no turn/conv ID), uses a 5-minute time window.

    Returns:
        Dict mapping (from_action, to_action, workspace_id, agent_id) -> count
    """
    # Group by session key
    sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for log in logs:
        key = _derive_session_key(log)
        sessions[key].append(log)

    edge_counts: Dict[Tuple[str, str, Optional[str], Optional[int]], int] = defaultdict(int)

    for session_key, session_logs in sessions.items():
        # Within a session, logs are already ordered by executed_at
        # For fallback keys (no turn/conv), split into 5-min windows
        if session_key.startswith("agent:"):
            windows = _split_by_time_window(session_logs, window_seconds=300)
        else:
            windows = [session_logs]

        for window_logs in windows:
            for i in range(len(window_logs) - 1):
                a = window_logs[i]
                b = window_logs[i + 1]
                if a["action_name"] == b["action_name"]:
                    continue  # Skip self-edges
                edge_key = (
                    a["action_name"],
                    b["action_name"],
                    a.get("workspace_id"),
                    a.get("agent_id"),
                )
                edge_counts[edge_key] += 1

    return dict(edge_counts)


def _compute_failed_after_edges(
    logs: List[Dict[str, Any]],
) -> Dict[Tuple[str, str, Optional[str], Optional[int]], Tuple[int, int]]:
    """Compute failed_after(A, B) edges.

    Within a session, when tool A SUCCEEDS and a later tool B (within the next
    2 steps) ERRORS, that is evidence the A->B transition is risky. We track
    BOTH the failure count and the total number of (A-succeeded, B-within-2)
    co-occurrences, so the edge confidence can be the Wilson lower bound of the
    failure RATE (failed / total) rather than a raw count -- a pair that fails
    3-of-100 times should not look as dangerous as one that fails 30-of-40.

    Same grouping/windowing as used_after. failed_after edges are written to the
    same table under a distinct edge_type; GraphRouter._query_edges only follows
    'used_after', so these never become recommended chains.

    Returns:
        Dict mapping (from_action, to_action, workspace_id, agent_id)
        -> (failed_count, total_count), only for pairs with >=1 failure.
    """
    sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for log in logs:
        key = _derive_session_key(log)
        sessions[key].append(log)

    failed_counts: Dict[Tuple[str, str, Optional[str], Optional[int]], int] = defaultdict(int)
    total_counts: Dict[Tuple[str, str, Optional[str], Optional[int]], int] = defaultdict(int)

    for session_key, session_logs in sessions.items():
        if session_key.startswith("agent:"):
            windows = _split_by_time_window(session_logs, window_seconds=300)
        else:
            windows = [session_logs]

        for window_logs in windows:
            for i in range(len(window_logs)):
                a = window_logs[i]
                if a.get("status") != "success":
                    continue  # A must have succeeded to originate a failed_after edge
                # Look ahead up to 2 steps within the same window
                for j in range(i + 1, min(i + 3, len(window_logs))):
                    b = window_logs[j]
                    if a["action_name"] == b["action_name"]:
                        continue  # Skip self-edges (chains never loop)
                    edge_key = (
                        a["action_name"],
                        b["action_name"],
                        a.get("workspace_id"),
                        a.get("agent_id"),
                    )
                    total_counts[edge_key] += 1
                    if b.get("status") != "success":
                        failed_counts[edge_key] += 1

    return {
        key: (failed_counts[key], total)
        for key, total in total_counts.items()
        if failed_counts.get(key, 0) > 0
    }


def _split_by_time_window(
    logs: List[Dict[str, Any]], window_seconds: int = 300
) -> List[List[Dict[str, Any]]]:
    """Split a list of time-ordered logs into sub-windows."""
    if not logs:
        return []
    windows: List[List[Dict[str, Any]]] = [[logs[0]]]
    for log in logs[1:]:
        prev = windows[-1][-1]
        delta = (log["executed_at"] - prev["executed_at"]).total_seconds()
        if delta <= window_seconds:
            windows[-1].append(log)
        else:
            windows.append([log])
    return windows


def _upsert_edge_row(
    db: Session,
    from_action: str,
    to_action: str,
    edge_type: str,
    workspace_id: Optional[str],
    agent_id: Optional[int],
    weight: float,
    confidence: float,
    sample_count: int,
    now: datetime,
) -> None:
    """Upsert a single routing edge (any edge_type) using ON CONFLICT UPDATE.

    The unique key uq_tre_full_key includes edge_type, so used_after and
    failed_after rows for the same (from, to, scope) coexist without clobbering.
    """
    stmt = text("""
        INSERT INTO tool_routing_edges
            (from_action, to_action, edge_type, workspace_id, agent_id,
             weight, confidence, sample_count, last_updated)
        VALUES
            (:from_action, :to_action, :edge_type, :workspace_id, :agent_id,
             :weight, :confidence, :sample_count, :last_updated)
        ON CONFLICT ON CONSTRAINT uq_tre_full_key
        DO UPDATE SET
            weight = :weight,
            confidence = :confidence,
            sample_count = :sample_count,
            last_updated = :last_updated
    """)
    db.execute(stmt, {
        "from_action": from_action,
        "to_action": to_action,
        "edge_type": edge_type,
        "workspace_id": workspace_id,
        "agent_id": agent_id,
        "weight": weight,
        "confidence": confidence,
        "sample_count": sample_count,
        "last_updated": now,
    })


def _upsert_edges(
    db: Session,
    edge_data: Dict[Tuple[str, str, Optional[str], Optional[int]], int],
) -> int:
    """Upsert used_after edges into tool_routing_edges."""
    count = 0
    now = datetime.utcnow()

    for (from_action, to_action, workspace_id, agent_id), sample_count in edge_data.items():
        if sample_count < _SAMPLE_FLOOR:
            continue

        weight = float(sample_count)
        confidence = wilson_lower_bound(sample_count, sample_count)
        _upsert_edge_row(
            db, from_action, to_action, "used_after",
            workspace_id, agent_id, weight, confidence, sample_count, now,
        )
        count += 1

    db.flush()
    return count


def _upsert_global_edges(
    db: Session,
    edge_data: Dict[Tuple[str, str, Optional[str], Optional[int]], int],
) -> int:
    """PRD-232 §6.5 (RVW-2): rebuild the TEXT-FREE GLOBAL used_after layer.

    Sums the per-tenant ``used_after`` counts into one aggregate per ``(from, to)``
    pair (across every workspace AND agent) and writes it as a global row
    (``workspace_id = NULL, agent_id = NULL``). A pair that clears the sample floor
    only in aggregate (sub-floor in each single tenant) still earns a global edge —
    the point of the cross-tenant cold-start prior — but ONLY if at least
    ``_GLOBAL_MIN_TENANTS`` distinct workspaces contributed to it, so a pattern unique
    to one tenant is never relabeled everyone's prior. Global rows carry only action
    names + counts (no user text): the global layer is text-free by construction.

    IDEMPOTENCY: a plain ON CONFLICT upsert CANNOT work here — ``uq_tre_full_key`` is a
    normal unique constraint and Postgres treats NULLs as DISTINCT, so a global row's
    (workspace_id NULL, agent_id NULL) key never matches an existing one and every
    nightly run would INSERT a duplicate. So this is a DELETE-then-INSERT rebuild of
    the whole global used_after layer (edges have no inbound FK; meta_sibling globals
    are a different edge_type, untouched). A pair that later drops below the floor or
    the tenant threshold correctly disappears from the global layer, never goes stale.
    """
    totals: Dict[Tuple[str, str], int] = defaultdict(int)
    tenants: Dict[Tuple[str, str], set] = defaultdict(set)
    for (from_action, to_action, ws, _agent), count in edge_data.items():
        pair = (from_action, to_action)
        totals[pair] += count
        if ws is not None:
            tenants[pair].add(ws)

    # Idempotent rebuild: clear the existing global used_after layer first (the NULL
    # key defeats ON CONFLICT — see the docstring), then insert the fresh aggregate.
    db.query(ToolRoutingEdge).filter(
        ToolRoutingEdge.workspace_id.is_(None),
        ToolRoutingEdge.agent_id.is_(None),
        ToolRoutingEdge.edge_type == "used_after",
    ).delete(synchronize_session=False)

    written = 0
    now = datetime.utcnow()
    for (from_action, to_action), sample_count in totals.items():
        if sample_count < _SAMPLE_FLOOR:
            continue
        if len(tenants[(from_action, to_action)]) < _GLOBAL_MIN_TENANTS:
            continue  # not genuinely cross-tenant — do not leak one tenant's pattern
        weight = float(sample_count)
        confidence = wilson_lower_bound(sample_count, sample_count)
        _upsert_edge_row(
            db, from_action, to_action, "used_after",
            None, None, weight, confidence, sample_count, now,
        )
        written += 1

    db.flush()
    return written


def _upsert_failed_after_edges(
    db: Session,
    failed_data: Dict[Tuple[str, str, Optional[str], Optional[int]], Tuple[int, int]],
) -> int:
    """Upsert failed_after edges into tool_routing_edges.

    weight = failure count; sample_count = total co-occurrences; confidence =
    Wilson lower bound of the failure rate (failed / total). The _SAMPLE_FLOOR
    is applied to the total co-occurrences, matching used_after's floor.
    """
    count = 0
    now = datetime.utcnow()

    for (from_action, to_action, workspace_id, agent_id), (failed, total) in failed_data.items():
        if total < _SAMPLE_FLOOR:
            continue

        weight = float(failed)
        confidence = wilson_lower_bound(failed, total)
        _upsert_edge_row(
            db, from_action, to_action, "failed_after",
            workspace_id, agent_id, weight, confidence, total, now,
        )
        count += 1

    db.flush()
    return count


async def _compute_and_upsert_clusters(
    db: Session,
    logs: List[Dict[str, Any]],
) -> Dict[int, int]:
    """Compute intent clusters and upsert to DB.

    Returns:
        Mapping of log index -> cluster_id (DB primary key) for logs with queries.
    """
    # Filter logs with user queries
    query_logs = [(i, log) for i, log in enumerate(logs) if log.get("user_query")]
    if not query_logs:
        return {}

    # Generate embeddings for queries
    embedding_manager = get_embedding_manager()
    queries = [log["user_query"] for _, log in query_logs]
    action_names = [log["action_name"] for _, log in query_logs]
    statuses = [log["status"] for _, log in query_logs]

    embeddings = await embedding_manager.generate_embeddings_batch(queries)
    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Run clustering
    cluster_result = compute_intent_clusters(
        embeddings=embeddings_array,
        queries=queries,
        action_names=action_names,
        statuses=statuses,
    )

    if not cluster_result.centroids:
        return {}

    embedding_model_key = derive_embedding_model_key(embedding_manager)

    # Upsert clusters - delete old ORGANIC ones and insert fresh (idempotent
    # rebuild). PRD-232 US-007: the delete is provenance-scoped — 'seeded'
    # cold-start clusters (and their affinities) are NEVER touched by the nightly,
    # so the synthetic-utterance seed survives 03:00 UTC and the graph routes
    # day-one. Affinities referencing the doomed organic clusters go FIRST: the FK
    # (ToolRoutingAffinity.intent_cluster_id → tool_routing_intent_clusters.id, the
    # ONLY live FK to this table) has no cascade, so without this a re-run would
    # either FK-error on the cluster delete or strand intent affinities under dead
    # cluster ids — re-runs must converge (PRD-143 S12).
    existing_clusters = (
        db.query(ToolRoutingIntentCluster)
        .filter(ToolRoutingIntentCluster.embedding_model_key == embedding_model_key)
        .filter(ToolRoutingIntentCluster.provenance == "organic")
        .all()
    )
    if existing_clusters:
        doomed_ids = [c.id for c in existing_clusters]
        db.query(ToolRoutingAffinity).filter(
            ToolRoutingAffinity.intent_cluster_id.in_(doomed_ids)
        ).delete(synchronize_session="fetch")
        db.query(ToolRoutingIntentCluster).filter(
            ToolRoutingIntentCluster.id.in_(doomed_ids)
        ).delete(synchronize_session="fetch")
    db.flush()

    now = datetime.utcnow()
    cluster_db_ids: List[int] = []

    for idx in range(len(cluster_result.centroids)):
        cluster = ToolRoutingIntentCluster(
            centroid_embedding=cluster_result.centroids[idx],
            embedding_model_key=embedding_model_key,
            # §6.5 privacy: the global cluster layer is text-free — redact the raw
            # user query to an action-name label (routing uses centroid + hot only).
            sample_query=_redacted_cluster_label(cluster_result.action_names_hot[idx]),
            action_names_hot=cluster_result.action_names_hot[idx],
            sample_count=cluster_result.sample_counts[idx],
            provenance="organic",  # US-007: nightly-built rows are organic
            last_updated=now,
        )
        db.add(cluster)
        db.flush()
        cluster_db_ids.append(cluster.id)

    # Build mapping: original log index -> cluster DB id
    log_to_cluster: Dict[int, int] = {}
    for label_idx, (orig_idx, _) in enumerate(query_logs):
        cluster_label = cluster_result.labels[label_idx]
        log_to_cluster[orig_idx] = cluster_db_ids[cluster_label]

    return log_to_cluster


@dataclass
class _AffinityKey:
    """Hashable affinity key for deduplication."""

    action_name: str
    affinity_type: str
    workspace_id: Optional[str]
    agent_id: Optional[int]
    intent_cluster_id: Optional[int]

    def __hash__(self):
        return hash((
            self.action_name, self.affinity_type,
            self.workspace_id, self.agent_id, self.intent_cluster_id,
        ))

    def __eq__(self, other):
        if not isinstance(other, _AffinityKey):
            return False
        return (
            self.action_name == other.action_name
            and self.affinity_type == other.affinity_type
            and self.workspace_id == other.workspace_id
            and self.agent_id == other.agent_id
            and self.intent_cluster_id == other.intent_cluster_id
        )


@dataclass
class _AffinityAccumulator:
    """Accumulates success/total counts for one affinity."""

    success_count: int = 0
    total_count: int = 0


def _compute_affinities(
    logs: List[Dict[str, Any]],
    cluster_map: Dict[int, int],
) -> List[Dict[str, Any]]:
    """Compute all affinity types from logs and cluster assignments.

    Types:
    - succeeds_for_intent: action succeeded for this intent cluster
    - fails_for_intent: action failed for this intent cluster
    - agent_prefers: action frequency for this agent (normalized)
    """
    # Accumulators keyed by _AffinityKey
    intent_accum: Dict[_AffinityKey, _AffinityAccumulator] = defaultdict(_AffinityAccumulator)
    agent_accum: Dict[_AffinityKey, _AffinityAccumulator] = defaultdict(_AffinityAccumulator)

    for idx, log in enumerate(logs):
        action_name = log["action_name"]
        # PRD-232 US-011: synthetic gap/shown rows are not tool outcomes — they
        # never earn succeeds/fails/agent_prefers. The index is still consumed so
        # it stays aligned with cluster_map (built over the SAME full log list).
        if action_name in _SYNTHETIC_ACTIONS:
            continue
        agent_id = log.get("agent_id")
        workspace_id = log.get("workspace_id")
        is_success = log["status"] == "success"
        cluster_id = cluster_map.get(idx)

        # Intent-based affinities (only for logs with a cluster assignment)
        if cluster_id is not None:
            if is_success:
                key = _AffinityKey(
                    action_name=action_name,
                    affinity_type="succeeds_for_intent",
                    workspace_id=workspace_id,
                    agent_id=None,
                    intent_cluster_id=cluster_id,
                )
                intent_accum[key].success_count += 1
                intent_accum[key].total_count += 1
            else:
                key = _AffinityKey(
                    action_name=action_name,
                    affinity_type="fails_for_intent",
                    workspace_id=workspace_id,
                    agent_id=None,
                    intent_cluster_id=cluster_id,
                )
                intent_accum[key].success_count += 1  # "success" at failing
                intent_accum[key].total_count += 1

        # Agent preference affinities
        if agent_id is not None:
            key = _AffinityKey(
                action_name=action_name,
                affinity_type="agent_prefers",
                workspace_id=workspace_id,
                agent_id=agent_id,
                intent_cluster_id=None,
            )
            agent_accum[key].success_count += 1
            agent_accum[key].total_count += 1

    # Normalize agent_prefers: weight is frequency relative to agent's total calls
    # First compute total calls per agent
    agent_totals: Dict[Tuple[Optional[int], Optional[str]], int] = defaultdict(int)
    for key, acc in agent_accum.items():
        agent_totals[(key.agent_id, key.workspace_id)] += acc.total_count

    # Build affinity records
    results: List[Dict[str, Any]] = []

    # Intent affinities
    for key, acc in intent_accum.items():
        if acc.total_count < _SAMPLE_FLOOR:
            continue
        results.append({
            "action_name": key.action_name,
            "affinity_type": key.affinity_type,
            "workspace_id": key.workspace_id,
            "agent_id": key.agent_id,
            "intent_cluster_id": key.intent_cluster_id,
            "weight": float(acc.total_count),
            "confidence": wilson_lower_bound(acc.success_count, acc.total_count),
            "sample_count": acc.total_count,
        })

    # Agent preference affinities
    for key, acc in agent_accum.items():
        if acc.total_count < _SAMPLE_FLOOR:
            continue
        total_for_agent = agent_totals.get((key.agent_id, key.workspace_id), 1)
        weight = acc.total_count / total_for_agent  # Normalized frequency
        results.append({
            "action_name": key.action_name,
            "affinity_type": key.affinity_type,
            "workspace_id": key.workspace_id,
            "agent_id": key.agent_id,
            "intent_cluster_id": key.intent_cluster_id,
            "weight": weight,
            "confidence": wilson_lower_bound(acc.success_count, acc.total_count),
            "sample_count": acc.total_count,
        })

    return results


def _compute_gap_resolution_affinities(
    logs: List[Dict[str, Any]],
    cluster_map: Dict[int, int],
    window: timedelta = _GAP_RESOLUTION_WINDOW,
) -> List[Dict[str, Any]]:
    """PRD-232 US-011(c): the nightly gap→resolution join.

    A tool_gap row (the model hunted for a capability, or a tool-requiring turn
    ran zero platform tools) followed IN THE SAME conversation, within ``window``
    (default 24h), by a SUCCESSFUL real action is ground truth: that action
    eventually served the intent the gap recorded. Emit a
    ``succeeds_for_intent(resolving_action, gap's_cluster)`` affinity, so the
    intent cluster the user actually expressed learns the action that answered it
    — even though the gap itself executed nothing.

    Pure: the gap's cluster is read from ``cluster_map`` (its own query is
    clustered alongside every other query), so no embeddings here. No
    ``_SAMPLE_FLOOR`` — a gap→resolution is a rare, high-signal event, and its
    Wilson confidence stays conservative (a single resolution ≈ 0.2), so it never
    outranks a well-established organic affinity.
    """
    sessions: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, log in enumerate(logs):
        sessions[_derive_session_key(log)].append((idx, log))

    accum: Dict[_AffinityKey, _AffinityAccumulator] = defaultdict(_AffinityAccumulator)
    for _key, items in sessions.items():
        for i, (gap_idx, gap_log) in enumerate(items):
            if gap_log.get("action_name") != _TOOL_GAP_ACTION:
                continue
            gap_cluster = cluster_map.get(gap_idx)
            if gap_cluster is None:
                continue
            gap_time = gap_log.get("executed_at")
            # The FIRST successful real action after the gap, within the window,
            # is the one that served the intent (ordered → break past the window).
            for _res_idx, res_log in items[i + 1:]:
                action = res_log.get("action_name")
                if action in _SYNTHETIC_ACTIONS or res_log.get("status") != "success":
                    continue
                res_time = res_log.get("executed_at")
                if gap_time is not None and res_time is not None and (res_time - gap_time) > window:
                    break
                key = _AffinityKey(
                    action_name=action,
                    affinity_type="succeeds_for_intent",
                    workspace_id=gap_log.get("workspace_id"),
                    agent_id=None,
                    intent_cluster_id=gap_cluster,
                )
                accum[key].success_count += 1
                accum[key].total_count += 1
                break  # only the first resolving action is the ground truth

    results: List[Dict[str, Any]] = []
    for key, acc in accum.items():
        results.append({
            "action_name": key.action_name,
            "affinity_type": key.affinity_type,
            "workspace_id": key.workspace_id,
            "agent_id": key.agent_id,
            "intent_cluster_id": key.intent_cluster_id,
            "weight": float(acc.total_count),
            "confidence": wilson_lower_bound(acc.success_count, acc.total_count),
            "sample_count": acc.total_count,
        })
    return results


def _merge_affinities(
    base: List[Dict[str, Any]],
    extra: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sum ``sample_count`` for affinities sharing the FULL unique key
    (action, type, workspace, agent, cluster), recomputing weight + confidence.

    A gap-resolution ``succeeds_for_intent`` REINFORCES an organic one on the same
    (action, cluster) rather than colliding on ``uq_tra_full_key`` at upsert time
    (a plain upsert would clobber, not add). Only intent affinities are ever
    merged — for them weight == sample_count and confidence == wilson(n, n).
    ``agent_prefers`` (a normalized frequency the nightly owns) is never
    gap-sourced, so it passes through untouched. Pure — returns new dicts.
    """
    merged: Dict[tuple, Dict[str, Any]] = {}
    order: List[tuple] = []
    for aff in list(base) + list(extra):
        k = (
            aff["action_name"], aff["affinity_type"], aff["workspace_id"],
            aff["agent_id"], aff["intent_cluster_id"],
        )
        if k not in merged:
            merged[k] = dict(aff)
            order.append(k)
        elif aff["affinity_type"] in ("succeeds_for_intent", "fails_for_intent"):
            existing = merged[k]
            n = existing["sample_count"] + aff["sample_count"]
            existing["sample_count"] = n
            existing["weight"] = float(n)
            existing["confidence"] = wilson_lower_bound(n, n)
    return [merged[k] for k in order]


def _apply_shown_not_used_decay(
    affinities: List[Dict[str, Any]],
    logs: List[Dict[str, Any]],
    cluster_map: Dict[int, int],
    decay_factor: float = _SHOWN_DECAY_FACTOR,
    floor: float = _AFFINITY_WEIGHT_FLOOR,
) -> List[Dict[str, Any]]:
    """PRD-232 US-011(b): decay ``succeeds_for_intent`` weight for actions SHOWN in
    an intent cluster far more often than they were USED.

    ``shown`` per (action, cluster) is counted from every row's surfaced set
    (``shown_actions`` = router_decision.candidates); ``used`` is the successful
    real executions in that cluster. When an action is shown MORE than it is used,
    its intent boost is eroded geometrically by the shown-not-used EXCESS — but
    never below ``floor`` (a seeded / previously-earned affinity is dialed down,
    never deleted). Actions used at least as often as shown are untouched.

    Pure — returns new dicts; never mutates the inputs. Returns ``(affinities,
    n_decayed)``.
    """
    if not affinities:
        return affinities, 0
    shown: Dict[tuple, int] = defaultdict(int)
    used: Dict[tuple, int] = defaultdict(int)
    for idx, log in enumerate(logs):
        cluster = cluster_map.get(idx)
        if cluster is None:
            continue
        for name in (log.get("shown_actions") or []):
            shown[(name, cluster)] += 1
        action = log.get("action_name")
        if action not in _SYNTHETIC_ACTIONS and log.get("status") == "success":
            used[(action, cluster)] += 1

    out: List[Dict[str, Any]] = []
    decayed_count = 0
    for aff in affinities:
        if aff["affinity_type"] != "succeeds_for_intent":
            out.append(aff)
            continue
        key = (aff["action_name"], aff["intent_cluster_id"])
        excess = shown.get(key, 0) - used.get(key, 0)
        if excess <= 0:
            out.append(aff)
            continue
        new = dict(aff)
        new["weight"] = max(floor, aff["weight"] * (decay_factor ** excess))
        out.append(new)
        decayed_count += 1
    return out, decayed_count


def _upsert_affinities(db: Session, affinities: List[Dict[str, Any]]) -> int:
    """Upsert affinities into tool_routing_affinities using ON CONFLICT UPDATE."""
    count = 0
    now = datetime.utcnow()

    for aff in affinities:
        stmt = text("""
            INSERT INTO tool_routing_affinities
                (action_name, affinity_type, workspace_id, agent_id,
                 intent_cluster_id, weight, confidence, sample_count, last_updated)
            VALUES
                (:action_name, :affinity_type, :workspace_id, :agent_id,
                 :intent_cluster_id, :weight, :confidence, :sample_count, :last_updated)
            ON CONFLICT ON CONSTRAINT uq_tra_full_key
            DO UPDATE SET
                weight = :weight,
                confidence = :confidence,
                sample_count = :sample_count,
                last_updated = :last_updated
        """)
        db.execute(stmt, {
            "action_name": aff["action_name"],
            "affinity_type": aff["affinity_type"],
            "workspace_id": aff["workspace_id"],
            "agent_id": aff["agent_id"],
            "intent_cluster_id": aff["intent_cluster_id"],
            "weight": aff["weight"],
            "confidence": aff["confidence"],
            "sample_count": aff["sample_count"],
            "last_updated": now,
        })
        count += 1

    db.flush()
    return count
