"""
PRD-143 S12 (US-005 / WS-C): one-shot tool-routing-graph seed backfill.

Populates tool_routing_edges / tool_routing_affinities from historical
tool_execution_logs by driving core/services/edge_builder.py's recompute —
the edge math (session grouping, Wilson confidence, intent clustering,
upserts) lives THERE; this module is only the entry point, the human
confirmation gate, workspace scoping, and the dry-run report.

Like a migration, this is HUMAN-APPLIED: it refuses to run without an
explicit --yes, and must never be pointed at a live/prod database by an
automated agent. Idempotent — re-running converges to the same rows.

Usage:
    cd orchestrator
    python -m scripts.seed_tool_routing_graph --yes [--workspace-id <uuid>]
        [--window-days N] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from core.database.database import get_db_session
from core.llm.embedding_manager import get_embedding_manager
from core.services import edge_builder

logger = logging.getLogger(__name__)

# Backfill default: look much further back than the nightly job's 30 days.
DEFAULT_WINDOW_DAYS = 365

# PRD-232 US-007: a seeded cluster's action_names_hot carries the action FIRST then
# its category family (same-category operator siblings), capped like the organic
# clusterer's top-10 hot list so a broad category never floods the surface.
_SEED_HOT_FAMILY_CAP = 10

REFUSAL = (
    "Refusing to run: this seed backfill is human-applied like a migration "
    "(PRD-143 US-005). Re-run with an explicit --yes once you have confirmed "
    "the target database. Use --dry-run to preview without writing."
)


@dataclass
class SeedSummary:
    """What the seed run did (or, for dry runs, would do)."""

    dry_run: bool
    logs_processed: int = 0
    edges: int = 0
    failed_edges: int = 0
    affinities: int = 0
    intent_clusters: int = 0
    meta_edges: int = 0
    seeded_clusters: int = 0  # PRD-232 US-007: synthetic-utterance cold-start clusters


async def seed_graph(
    window_days: int = DEFAULT_WINDOW_DAYS,
    workspace_id: Optional[str] = None,
    dry_run: bool = False,
) -> SeedSummary:
    """Drive edge_builder's recompute over historical logs.

    Real runs delegate wholesale to ``edge_builder.build_edges`` (idempotent
    ON CONFLICT upserts). Dry runs reuse the same load + pure compute helpers
    and write nothing — intent clustering (which both embeds and writes) only
    happens on a real run.
    """
    if dry_run:
        return _dry_run_summary(window_days, workspace_id)

    result = await edge_builder.build_edges(
        window=timedelta(days=window_days),
        workspace_id=workspace_id,
    )
    return SeedSummary(
        dry_run=False,
        logs_processed=result.logs_processed,
        edges=result.edges_built,
        failed_edges=result.failed_edges_built,
        affinities=result.affinities_built,
        intent_clusters=result.intent_clusters,
    )


def seed_metadata_edges(dry_run: bool = False) -> int:
    """Seed GLOBAL metadata cold-start edges so zero-telemetry tools are still
    graph-reachable (PRD-143 metadata_graph_seed).

    A separate concern from the telemetry backfill above: it's app-global (not
    per-workspace) and reads the registry, not the logs. Idempotent ON CONFLICT
    upsert. get_db_session commits on context exit (database.py) — matches
    edge_builder, which never calls db.commit() itself."""
    from modules.tools.discovery import get_action_registry
    from modules.tools.discovery.metadata_graph_seed import seed_meta_sibling_edges

    registry = get_action_registry()
    if dry_run:
        return seed_meta_sibling_edges(None, registry, dry_run=True)
    with get_db_session() as db:
        return seed_meta_sibling_edges(db, registry)


# ---------------------------------------------------------------------------
# PRD-232 US-007: synthetic-utterance intent-cluster seeding (cold-start)
# ---------------------------------------------------------------------------


def _seed_cluster_confidence() -> float:
    """Conservative confidence stamped on seeded affinities, from config (US-007)."""
    try:
        from config import config
        return float(getattr(config, "TOOL_ROUTING_SEED_CLUSTER_CONFIDENCE", 0.6))
    except Exception:
        return 0.6


def _load_action_categories() -> dict:
    """``{action_name: category}`` for every non-super_admin_only registered action.

    The su-only (obs) tier is excluded — the same invariant metadata_graph_seed and
    the rest of PRD-143/232 enforce: Auto's graph never surfaces an obs tool.
    """
    from modules.tools.discovery import get_action_registry

    registry = get_action_registry()
    return {
        a.name: (getattr(a, "category", "") or "")
        for a in registry.get_all()
        if not getattr(a, "super_admin_only", False)
    }


def _hot_actions_for(action_name: str, action_categories: dict) -> List[str]:
    """A seeded cluster's ``action_names_hot``: the action FIRST, then its category
    family (same-category operator siblings, sorted), capped like the organic
    clusterer's top-10 hot list."""
    category = action_categories.get(action_name, "")
    family = (
        sorted(
            n for n, c in action_categories.items()
            if c and c == category and n != action_name
        )
        if category
        else []
    )
    return [action_name] + family[: _SEED_HOT_FAMILY_CAP - 1]


async def seed_intent_clusters_from_corpus(
    dry_run: bool = False,
    *,
    corpus: Optional[dict] = None,
    action_categories: Optional[dict] = None,
) -> int:
    """Seed intent clusters from the synthetic utterance corpus (PRD-232 US-007).

    For every non-super_admin_only action that carries seed utterances (US-005),
    embed its utterances, mean-pool a centroid, and write one
    ``provenance='seeded'`` ToolRoutingIntentCluster (``action_names_hot`` = the
    action + its category family) plus one conservative global
    ``succeeds_for_intent`` affinity at the metadata floor. So a live query phrased
    like any seeded utterance matches the seeded centroid
    (GraphRouter._match_intent_cluster) and the right action surfaces day-one —
    before any telemetry accrues.

    Survives the nightly: ``edge_builder`` rebuilds only ``provenance='organic'``
    clusters, so these seeds persist across 03:00 UTC. Idempotent: existing seeded
    clusters (and their affinities) for the active model key are deleted-and-
    reinserted, so a re-run converges. NO LLM cost beyond the embeddings; the
    embeddings use the REAL EmbeddingManager (never the banned hash-random synthetic
    provider — PRD-185 S3 — which would poison every centroid). Human-applied only,
    like the rest of this module.

    ``corpus`` / ``action_categories`` are injectable for hermetic unit tests; the
    real run loads them from the corpus files and the registry.
    """
    if corpus is None:
        from modules.tools.discovery.utterance_corpus import load_utterance_corpus
        corpus = load_utterance_corpus()
    if action_categories is None:
        action_categories = _load_action_categories()

    # Seed only actions that BOTH carry utterances AND are non-su registered
    # (in action_categories). The corpus already excludes su-only actions (US-005);
    # the registry filter is defence-in-depth so a stale corpus name can never
    # seed an unregistered or su action.
    seedable = {
        name: list(utts)
        for name, utts in corpus.items()
        if utts and name in action_categories
    }
    if not seedable or dry_run:
        return len(seedable)

    embedding_manager = get_embedding_manager()
    with get_db_session() as db:
        return await _seed_clusters(db, seedable, action_categories, embedding_manager)


async def _seed_clusters(db, seedable: dict, action_categories: dict, embedding_manager) -> int:
    """Embed the corpus, write seeded clusters + affinities. Pure of any registry/
    corpus loading (its inputs are already resolved) so it is trivially testable."""
    import numpy as np

    from core.models.tool_routing import ToolRoutingAffinity, ToolRoutingIntentCluster

    model_key = edge_builder.derive_embedding_model_key(embedding_manager)

    # Idempotent: drop existing SEEDED clusters (+ their affinities) for this model
    # key, then reinsert. Provenance-scoped so it never disturbs organic rows — the
    # mirror image of the nightly's organic-only rebuild. Affinities go first (the
    # FK has no cascade), exactly like _compute_and_upsert_clusters.
    existing = (
        db.query(ToolRoutingIntentCluster)
        .filter(ToolRoutingIntentCluster.embedding_model_key == model_key)
        .filter(ToolRoutingIntentCluster.provenance == "seeded")
        .all()
    )
    if existing:
        doomed_ids = [c.id for c in existing]
        db.query(ToolRoutingAffinity).filter(
            ToolRoutingAffinity.intent_cluster_id.in_(doomed_ids)
        ).delete(synchronize_session="fetch")
        db.query(ToolRoutingIntentCluster).filter(
            ToolRoutingIntentCluster.id.in_(doomed_ids)
        ).delete(synchronize_session="fetch")
    db.flush()

    # Embed every utterance ONCE (batched), grouped back per action for mean-pool.
    names = sorted(seedable)
    flat_texts: List[str] = []
    spans: List[tuple] = []
    for name in names:
        start = len(flat_texts)
        flat_texts.extend(seedable[name])
        spans.append((start, len(flat_texts)))

    embeddings = await embedding_manager.generate_embeddings_batch(flat_texts)
    emb = np.array(embeddings, dtype=np.float32)

    now = datetime.utcnow()
    seed_conf = _seed_cluster_confidence()
    seed_affinities: List[dict] = []
    count = 0

    for name, (start, end) in zip(names, spans):
        if end <= start:
            continue
        centroid = emb[start:end].mean(axis=0).tolist()
        cluster = ToolRoutingIntentCluster(
            centroid_embedding=centroid,
            embedding_model_key=model_key,
            sample_query=seedable[name][0],
            action_names_hot=_hot_actions_for(name, action_categories),
            sample_count=len(seedable[name]),
            provenance="seeded",
            last_updated=now,
        )
        db.add(cluster)
        db.flush()  # assign cluster.id for the affinity FK
        # A conservative GLOBAL succeeds_for_intent (workspace/agent None) so the
        # seeded intent also carries a per-cluster boost on the unscoped/eval read
        # path. weight=1, sample_count=0 — metadata, not observed usage — at the
        # floor confidence, so ANY organic row of higher Wilson confidence outranks
        # it as real telemetry accrues (the seed is the floor; usage always wins).
        seed_affinities.append({
            "action_name": name,
            "affinity_type": "succeeds_for_intent",
            "workspace_id": None,
            "agent_id": None,
            "intent_cluster_id": cluster.id,
            "weight": 1.0,
            "confidence": seed_conf,
            "sample_count": 0,
        })
        count += 1

    if seed_affinities:
        edge_builder._upsert_affinities(db, seed_affinities)
    return count


def _dry_run_summary(window_days: int, workspace_id: Optional[str]) -> SeedSummary:
    """Read-only preview: load logs, run the pure edge math, count candidates."""
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    with get_db_session() as db:
        logs = edge_builder._load_logs(db, cutoff, workspace_id=workspace_id)

    edge_data = edge_builder._compute_used_after_edges(logs)
    failed_data = edge_builder._compute_failed_after_edges(logs)
    # Without intent clustering (real-run only) this counts agent_prefers
    # affinities; intent affinities are reported as 0 in a dry run.
    agent_affinities = edge_builder._compute_affinities(logs, {})

    meta_edges = seed_metadata_edges(dry_run=True)

    floor = edge_builder._SAMPLE_FLOOR
    return SeedSummary(
        dry_run=True,
        logs_processed=len(logs),
        edges=sum(1 for count in edge_data.values() if count >= floor),
        failed_edges=sum(1 for _failed, total in failed_data.values() if total >= floor),
        affinities=len(agent_affinities),
        intent_clusters=0,
        meta_edges=meta_edges,
    )


def _print_summary(summary: SeedSummary, workspace_id: Optional[str]) -> None:
    mode = "DRY RUN (nothing written)" if summary.dry_run else "SEEDED"
    scope = workspace_id or "all workspaces"
    print(f"\nTool routing graph seed — {mode}")
    print(f"  Scope:             {scope}")
    print(f"  Logs processed:    {summary.logs_processed}")
    print(f"  used_after edges:  {summary.edges}")
    print(f"  failed_after edges:{summary.failed_edges}")
    print(f"  meta_sibling edges:{summary.meta_edges}")
    print(f"  Affinities:        {summary.affinities}")
    print(f"  Intent clusters:   {summary.intent_clusters}")
    print(f"  Seeded clusters:   {summary.seeded_clusters}")
    if summary.dry_run:
        print("  Note: intent clustering/affinities are computed only on a real run.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot backfill of the tool routing graph from historical "
            "tool_execution_logs (PRD-143 US-005). Human-applied; requires --yes."
        )
    )
    parser.add_argument(
        "--workspace-id",
        default=None,
        help="Restrict the backfill to one workspace UUID",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"How far back to read logs (default {DEFAULT_WINDOW_DAYS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report candidate counts without writing anything",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required confirmation that a human is applying this to a known DB",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.yes:
        print(REFUSAL)
        return 2

    summary = asyncio.run(
        seed_graph(
            window_days=args.window_days,
            workspace_id=args.workspace_id,
            dry_run=args.dry_run,
        )
    )
    # Telemetry backfill (seed_graph), metadata cold-start, and synthetic-utterance
    # intent-cluster seeding (PRD-232 US-007) are independent seeds; a dry run only
    # counts, a real run writes all three so one human command seeds the whole graph.
    if not args.dry_run:
        summary.meta_edges = seed_metadata_edges()
        summary.seeded_clusters = asyncio.run(seed_intent_clusters_from_corpus())
    else:
        summary.seeded_clusters = asyncio.run(seed_intent_clusters_from_corpus(dry_run=True))
    _print_summary(summary, args.workspace_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
