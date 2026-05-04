"""
PRD-139 US-003: Intent clustering via K-means over query embeddings.

Groups user queries into semantic clusters for computing intent-level
tool affinities. Uses existing EmbeddingManager for vector generation.

Deterministic: random_state=42 guarantees reproducible centroids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum samples required to form a meaningful cluster
_MIN_SAMPLES_PER_CLUSTER = 3
# Default number of clusters — adjusted dynamically based on sample size
_DEFAULT_K = 8


@dataclass
class ClusterResult:
    """Result of clustering a set of query embeddings."""

    centroids: List[List[float]]
    labels: List[int]
    sample_queries: List[str]  # one representative query per cluster
    action_names_hot: List[List[str]]  # top actions per cluster
    sample_counts: List[int]  # observations per cluster


def compute_intent_clusters(
    embeddings: np.ndarray,
    queries: List[str],
    action_names: List[str],
    statuses: List[str],
    k: Optional[int] = None,
    max_k: int = 20,
) -> ClusterResult:
    """Run K-means clustering over query embeddings.

    Args:
        embeddings: (N, D) numpy array of query embedding vectors
        queries: parallel list of user query strings
        action_names: parallel list of action names for each observation
        statuses: parallel list of statuses ('success'/'error') per observation
        k: explicit number of clusters. If None, auto-selects based on N.
        max_k: upper bound on cluster count

    Returns:
        ClusterResult with centroids, labels, and per-cluster metadata.
    """
    n_samples = len(embeddings)
    if n_samples == 0:
        return ClusterResult(
            centroids=[], labels=[], sample_queries=[],
            action_names_hot=[], sample_counts=[],
        )

    # Auto-select k: sqrt(N/2) clamped to [2, max_k]
    if k is None:
        k = max(2, min(max_k, int(np.sqrt(n_samples / 2))))

    # Cannot have more clusters than samples
    k = min(k, n_samples)

    labels, centroids = _kmeans(embeddings, k, random_state=42)

    # Build per-cluster metadata
    sample_queries: List[str] = []
    action_names_hot: List[List[str]] = []
    sample_counts: List[int] = []

    for cluster_idx in range(k):
        mask = labels == cluster_idx
        count = int(mask.sum())
        sample_counts.append(count)

        # Representative query: closest to centroid
        if count > 0:
            cluster_indices = np.where(mask)[0]
            cluster_embeddings = embeddings[cluster_indices]
            distances = np.linalg.norm(cluster_embeddings - centroids[cluster_idx], axis=1)
            closest_idx = cluster_indices[int(np.argmin(distances))]
            sample_queries.append(queries[closest_idx])

            # Top actions (by frequency of successful execution in this cluster)
            cluster_actions: Dict[str, int] = {}
            for idx in cluster_indices:
                if statuses[idx] == "success":
                    action = action_names[idx]
                    cluster_actions[action] = cluster_actions.get(action, 0) + 1

            sorted_actions = sorted(cluster_actions.items(), key=lambda x: x[1], reverse=True)
            action_names_hot.append([a for a, _ in sorted_actions[:10]])
        else:
            sample_queries.append("")
            action_names_hot.append([])

    return ClusterResult(
        centroids=[c.tolist() for c in centroids],
        labels=labels.tolist(),
        sample_queries=sample_queries,
        action_names_hot=action_names_hot,
        sample_counts=sample_counts,
    )


def _kmeans(
    data: np.ndarray,
    k: int,
    random_state: int = 42,
    max_iterations: int = 300,
    tolerance: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal K-means implementation.

    Uses K-means++ initialization for better convergence.
    Pinned random_state ensures deterministic results across runs.

    Returns:
        (labels, centroids) where labels is (N,) int array and centroids is (k, D) array.
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = data.shape

    # K-means++ initialization
    centroids = np.empty((k, n_features), dtype=data.dtype)
    first_idx = rng.randint(0, n_samples)
    centroids[0] = data[first_idx]

    for i in range(1, k):
        # Distance from each point to nearest existing centroid
        distances = np.min(
            np.linalg.norm(data[:, np.newaxis] - centroids[:i], axis=2), axis=1
        )
        distances_sq = distances ** 2
        total = distances_sq.sum()
        if total == 0:
            # All remaining points are identical to existing centroids
            centroids[i] = data[rng.randint(0, n_samples)]
        else:
            probabilities = distances_sq / total
            chosen_idx = rng.choice(n_samples, p=probabilities)
            centroids[i] = data[chosen_idx]

    # Iterative assignment + update
    labels = np.zeros(n_samples, dtype=np.int32)
    for _ in range(max_iterations):
        # Assignment step
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        new_labels = np.argmin(distances, axis=1).astype(np.int32)

        # Update step
        new_centroids = np.empty_like(centroids)
        for j in range(k):
            mask = new_labels == j
            if mask.any():
                new_centroids[j] = data[mask].mean(axis=0)
            else:
                # Empty cluster: reinitialize to random point
                new_centroids[j] = data[rng.randint(0, n_samples)]

        # Check convergence
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        labels = new_labels

        if centroid_shift < tolerance:
            break

    return labels, centroids
