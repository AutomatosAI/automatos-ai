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

Night 14 (12–13 Oct, F416): the run then went out of memory (OOMKilled at 03:50).
Every query of the 30-day window was embedded at once, held as Python float lists,
and the k-means broadcast an (N, k, D) difference tensor. Now at most
EDGE_BUILDER_CLUSTER_MAX_QUERIES of the most recent queries cluster (the embedding
matrix's row cap; a run over it logs that it sampled and its summary says so), they
are embedded EDGE_BUILDER_EMBED_BATCH_SIZE at a time straight into one float32
matrix, and the k-means works in blocks of at most EDGE_BUILDER_KMEANS_BLOCK_MB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from core.models.tool_routing import ToolRoutingAffinity, ToolRoutingIntentCluster
from core.services.intent_clustering import ClusterResult, compute_intent_clusters

logger = logging.getLogger(__name__)

# Defaults; config.EDGE_BUILDER_* overrides (F416).
_DEFAULT_CLUSTER_MAX_QUERIES = 5000
_DEFAULT_EMBED_BATCH_SIZE = 256
_DEFAULT_KMEANS_BLOCK_MB = 64
_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class EmbeddedQueries:
    """The query-bearing logs of one run (at most the cap) and their embeddings,
    one float32 row per query."""

    log_indices: Tuple[int, ...]  # index into build_edges' logs, per query
    queries: Tuple[str, ...]
    action_names: Tuple[str, ...]
    statuses: Tuple[str, ...]
    vectors: np.ndarray
    model_key: str
    total_queries: int  # query-bearing logs in the window, before the cap


def _config_int(name: str, default: int) -> int:
    from config import config

    return int(getattr(config, name, default))


def cluster_max_queries() -> int:
    """Most queries one run clusters (the embedding matrix's row cap), from config."""
    return max(1, _config_int("EDGE_BUILDER_CLUSTER_MAX_QUERIES", _DEFAULT_CLUSTER_MAX_QUERIES))


def kmeans_block_bytes() -> int:
    """Largest k-means working block, in bytes, from config."""
    return max(1, _config_int("EDGE_BUILDER_KMEANS_BLOCK_MB", _DEFAULT_KMEANS_BLOCK_MB)) * _BYTES_PER_MB


def recent_query_logs(
    logs: List[Dict[str, Any]], cap: int,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], int]:
    """The (index, log) pairs that carry a user query — the ones that cluster —
    capped to the ``cap`` most recent (logs are in executed_at order); and how
    many there were before the cap. A run over the cap logs that it sampled."""
    picked = [(i, log) for i, log in enumerate(logs) if log.get("user_query")]
    total = len(picked)
    if total > cap:
        logger.warning(
            "EdgeBuilder: %d queries in the window, over the cap of %d: "
            "clustering a sample, the %d most recent", total, cap, cap,
        )
        picked = picked[-cap:]
    return picked, total


async def embed_in_batches(embedding_manager: Any, texts: List[str]) -> np.ndarray:
    """Embed ``texts`` a configured batch at a time into one float32 matrix, so
    no full-run list of Python float vectors is ever held (F416)."""
    batch_size = max(1, _config_int("EDGE_BUILDER_EMBED_BATCH_SIZE", _DEFAULT_EMBED_BATCH_SIZE))
    blocks = []
    for start in range(0, len(texts), batch_size):
        vectors = await embedding_manager.generate_embeddings_batch(texts[start:start + batch_size])
        blocks.append(np.asarray(vectors, dtype=np.float32))
    return np.vstack(blocks)


def embedded_queries(
    picked: List[Tuple[int, Dict[str, Any]]],
    vectors: np.ndarray,
    model_key: str,
    total_queries: int,
) -> EmbeddedQueries:
    """Bundle the picked logs with their embedding vectors and model key."""
    return EmbeddedQueries(
        log_indices=tuple(i for i, _ in picked),
        queries=tuple(log["user_query"] for _, log in picked),
        action_names=tuple(log["action_name"] for _, log in picked),
        statuses=tuple(log["status"] for _, log in picked),
        vectors=vectors,
        model_key=model_key,
        total_queries=total_queries,
    )


def sampling_counts(embedded: Optional[EmbeddedQueries]) -> Dict[str, Any]:
    """The run summary's clustering counts: how many queries clustered, of how
    many, and whether that was a sample (empty when nothing was embedded)."""
    if not isinstance(embedded, EmbeddedQueries):
        return {}
    clustered = len(embedded.queries)
    return {
        "cluster_queries": clustered,
        "cluster_queries_total": embedded.total_queries,
        "cluster_sampled": embedded.total_queries > clustered,
    }


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
        embeddings=np.asarray(embedded.vectors, dtype=np.float32),
        queries=list(embedded.queries),
        action_names=list(embedded.action_names),
        statuses=list(embedded.statuses),
        max_block_bytes=kmeans_block_bytes(),
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
