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


@dataclass
class EdgeBuildSummary:
    """Summary returned after an edge-build run."""

    edges_built: int = 0
    affinities_built: int = 0
    intent_clusters: int = 0
    logs_processed: int = 0
    duration_ms: int = 0


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Wilson lower bound at 95% confidence."""
    if total == 0:
        return 0.0
    p = successes / total
    denominator = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    return (centre - spread) / denominator


async def build_edges(window: timedelta = timedelta(days=30)) -> EdgeBuildSummary:
    """Main entry point: read telemetry, compute edges + affinities, upsert.

    Args:
        window: How far back to look in tool_execution_logs.

    Returns:
        EdgeBuildSummary with counts of what was built.
    """
    import time

    start = time.monotonic()
    summary = EdgeBuildSummary()

    with get_db_session() as db:
        # 1. Load execution logs within window
        cutoff = datetime.utcnow() - window
        logs = _load_logs(db, cutoff)
        summary.logs_processed = len(logs)

        if not logs:
            logger.info("EdgeBuilder: no logs in window, nothing to build")
            return summary

        # 2. Compute used_after edges from sequences
        edge_data = _compute_used_after_edges(logs)
        summary.edges_built = _upsert_edges(db, edge_data)

        # 3. Compute intent clusters from query embeddings
        cluster_map = await _compute_and_upsert_clusters(db, logs)
        summary.intent_clusters = len(cluster_map)

        # 4. Compute affinities (succeeds/fails for intent, agent_prefers)
        affinities = _compute_affinities(logs, cluster_map)
        summary.affinities_built = _upsert_affinities(db, affinities)

    elapsed_ms = int((time.monotonic() - start) * 1000)
    summary.duration_ms = elapsed_ms

    logger.info(
        f"EdgeBuilder: built {summary.edges_built} edges, "
        f"{summary.affinities_built} affinities across "
        f"{summary.intent_clusters} intent clusters"
    )
    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_logs(db: Session, cutoff: datetime) -> List[Dict[str, Any]]:
    """Load tool_execution_logs rows from cutoff onwards, return as dicts."""
    rows = (
        db.query(ToolExecutionLog)
        .filter(ToolExecutionLog.executed_at >= cutoff)
        .order_by(ToolExecutionLog.executed_at.asc())
        .all()
    )
    results = []
    for row in rows:
        # Extract turn_id from router_decision JSONB
        router = row.router_decision or {}
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


def _upsert_edges(
    db: Session,
    edge_data: Dict[Tuple[str, str, Optional[str], Optional[int]], int],
) -> int:
    """Upsert edges into tool_routing_edges using ON CONFLICT UPDATE."""
    count = 0
    now = datetime.utcnow()

    for (from_action, to_action, workspace_id, agent_id), sample_count in edge_data.items():
        if sample_count < _SAMPLE_FLOOR:
            continue

        weight = float(sample_count)
        confidence = wilson_lower_bound(sample_count, sample_count)

        # Use raw SQL for ON CONFLICT upsert
        stmt = text("""
            INSERT INTO tool_routing_edges
                (from_action, to_action, edge_type, workspace_id, agent_id,
                 weight, confidence, sample_count, last_updated)
            VALUES
                (:from_action, :to_action, 'used_after', :workspace_id, :agent_id,
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
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "weight": weight,
            "confidence": confidence,
            "sample_count": sample_count,
            "last_updated": now,
        })
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

    # Get canonical embedding model key
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
    embedding_model_key = f"{provider}:{model}:{dimension}"

    # Upsert clusters - delete old ones and insert fresh (idempotent rebuild)
    # Delete existing clusters for this model key, then insert new
    db.query(ToolRoutingIntentCluster).filter(
        ToolRoutingIntentCluster.embedding_model_key == embedding_model_key
    ).delete(synchronize_session="fetch")
    db.flush()

    now = datetime.utcnow()
    cluster_db_ids: List[int] = []

    for idx in range(len(cluster_result.centroids)):
        cluster = ToolRoutingIntentCluster(
            centroid_embedding=cluster_result.centroids[idx],
            embedding_model_key=embedding_model_key,
            sample_query=cluster_result.sample_queries[idx] or "(empty)",
            action_names_hot=cluster_result.action_names_hot[idx],
            sample_count=cluster_result.sample_counts[idx],
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
