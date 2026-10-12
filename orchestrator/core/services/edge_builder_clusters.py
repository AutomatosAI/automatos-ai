"""
PRD-139 US-003 / P256-FIX-N13-2 (F409): the nightly intent-cluster rebuild.

Split out of edge_builder so the k-means never runs on the event loop. The
embedding call is the only awaited step (edge_builder._embed_queries, on the loop,
with no database session open); what it returns is an EmbeddedQueries, and
upsert_clusters — the numpy conversion, compute_intent_clusters and the cluster
rows — is plain synchronous code that build_edges runs in a worker thread.

Night 13 (11–12 Oct, F409): _kmeans ran inside an ``async`` build_edges, so the
backend sat at 105% CPU with the loop frozen ~25 minutes; every chat and /health
timed out until a restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sqlalchemy.orm import Session

from core.models.tool_routing import ToolRoutingAffinity, ToolRoutingIntentCluster
from core.services.intent_clustering import ClusterResult, compute_intent_clusters


@dataclass(frozen=True)
class EmbeddedQueries:
    """The query-bearing logs of one run and their embeddings (raw vectors: the
    numpy conversion is CPU work, so it happens in the worker thread)."""

    log_indices: Tuple[int, ...]  # index into build_edges' logs, per query
    queries: Tuple[str, ...]
    action_names: Tuple[str, ...]
    statuses: Tuple[str, ...]
    vectors: Sequence[Sequence[float]]
    model_key: str


def query_logs(logs: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    """The (index, log) pairs that carry a user query — the ones that cluster."""
    return [(i, log) for i, log in enumerate(logs) if log.get("user_query")]


def embedded_queries(
    picked: List[Tuple[int, Dict[str, Any]]],
    vectors: Sequence[Sequence[float]],
    model_key: str,
) -> EmbeddedQueries:
    """Bundle the picked logs with their embedding vectors and model key."""
    return EmbeddedQueries(
        log_indices=tuple(i for i, _ in picked),
        queries=tuple(log["user_query"] for _, log in picked),
        action_names=tuple(log["action_name"] for _, log in picked),
        statuses=tuple(log["status"] for _, log in picked),
        vectors=vectors,
        model_key=model_key,
    )


def redacted_cluster_label(action_names_hot: List[str]) -> str:
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


def upsert_clusters(db: Session, embedded: EmbeddedQueries) -> Dict[int, int]:
    """Cluster the embedded queries and rebuild the organic cluster rows.

    Synchronous and CPU-bound (build_edges runs it in a worker thread, F409).

    Returns:
        Mapping of log index -> cluster_id (DB primary key) for logs with queries.
    """
    result = compute_intent_clusters(
        embeddings=np.array(embedded.vectors, dtype=np.float32),
        queries=list(embedded.queries),
        action_names=list(embedded.action_names),
        statuses=list(embedded.statuses),
    )
    if not result.centroids:
        return {}
    _delete_organic_clusters(db, embedded.model_key)
    cluster_db_ids = _insert_clusters(db, result, embedded.model_key)
    return {
        log_index: cluster_db_ids[result.labels[label_idx]]
        for label_idx, log_index in enumerate(embedded.log_indices)
    }


def _delete_organic_clusters(db: Session, model_key: str) -> None:
    """Delete the old ORGANIC clusters for this model key (idempotent rebuild).

    PRD-232 US-007: the delete is provenance-scoped — 'seeded' cold-start clusters
    (and their affinities) are NEVER touched by the nightly, so the
    synthetic-utterance seed survives 03:00 UTC and the graph routes day-one.
    Affinities referencing the doomed organic clusters go FIRST: the FK
    (ToolRoutingAffinity.intent_cluster_id → tool_routing_intent_clusters.id, the
    ONLY live FK to this table) has no cascade, so without this a re-run would
    either FK-error on the cluster delete or strand intent affinities under dead
    cluster ids — re-runs must converge (PRD-143 S12).
    """
    existing_clusters = (
        db.query(ToolRoutingIntentCluster)
        .filter(ToolRoutingIntentCluster.embedding_model_key == model_key)
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


def _insert_clusters(db: Session, result: ClusterResult, model_key: str) -> List[int]:
    """Insert one organic cluster row per centroid; return their ids in order."""
    now = datetime.utcnow()
    cluster_db_ids: List[int] = []
    for idx, centroid in enumerate(result.centroids):
        cluster = ToolRoutingIntentCluster(
            centroid_embedding=centroid,
            embedding_model_key=model_key,
            # §6.5 privacy: the global cluster layer is text-free — redact the raw
            # user query to an action-name label (routing uses centroid + hot only).
            sample_query=redacted_cluster_label(result.action_names_hot[idx]),
            action_names_hot=result.action_names_hot[idx],
            sample_count=result.sample_counts[idx],
            provenance="organic",  # US-007: nightly-built rows are organic
            last_updated=now,
        )
        db.add(cluster)
        db.flush()
        cluster_db_ids.append(cluster.id)
    return cluster_db_ids
