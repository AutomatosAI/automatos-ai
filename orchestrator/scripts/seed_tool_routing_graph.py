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
from core.services import edge_builder

logger = logging.getLogger(__name__)

# Backfill default: look much further back than the nightly job's 30 days.
DEFAULT_WINDOW_DAYS = 365

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

    floor = edge_builder._SAMPLE_FLOOR
    return SeedSummary(
        dry_run=True,
        logs_processed=len(logs),
        edges=sum(1 for count in edge_data.values() if count >= floor),
        failed_edges=sum(1 for _failed, total in failed_data.values() if total >= floor),
        affinities=len(agent_affinities),
        intent_clusters=0,
    )


def _print_summary(summary: SeedSummary, workspace_id: Optional[str]) -> None:
    mode = "DRY RUN (nothing written)" if summary.dry_run else "SEEDED"
    scope = workspace_id or "all workspaces"
    print(f"\nTool routing graph seed — {mode}")
    print(f"  Scope:             {scope}")
    print(f"  Logs processed:    {summary.logs_processed}")
    print(f"  used_after edges:  {summary.edges}")
    print(f"  failed_after edges:{summary.failed_edges}")
    print(f"  Affinities:        {summary.affinities}")
    print(f"  Intent clusters:   {summary.intent_clusters}")
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
    _print_summary(summary, args.workspace_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
