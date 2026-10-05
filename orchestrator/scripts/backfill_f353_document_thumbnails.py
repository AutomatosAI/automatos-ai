"""
F353 (issue #947): draw the first page of every document made before F353
=========================================================================

New PDFs, Word documents, sheets and reports get their card picture when they are
made. This one-off run draws the ones made before, oldest first, at most
``--per-minute`` renders a minute (default 20), so it never takes the CPU or the
workspace worker from live requests. See ``modules/documents/thumbnails/backfill.py``.

Idempotent: a Deliverable that already has a picture is skipped; one whose render
failed before is skipped too, unless ``--retry-failed``.

Run it where the API runs (it reads the same disk, object storage and workspace
worker)::

    cd orchestrator
    python scripts/backfill_f353_document_thumbnails.py --dry-run
    python scripts/backfill_f353_document_thumbnails.py --workspace-id <uuid> --per-minute 30
    python scripts/backfill_f353_document_thumbnails.py --limit 200
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Make orchestrator package imports work when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database.database import SessionLocal  # noqa: E402
from modules.documents.thumbnails.backfill import DEFAULT_PER_MINUTE, backfill, find_candidates  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw first-page pictures for existing documents (F353)")
    parser.add_argument("--dry-run", action="store_true", help="Count what would be drawn; draw nothing")
    parser.add_argument("--workspace-id", default=None, help="Only this workspace (UUID)")
    parser.add_argument("--per-minute", type=int, default=DEFAULT_PER_MINUTE, help="Renders started a minute")
    parser.add_argument("--limit", type=int, default=None, help="Look at no more than this many Deliverables")
    parser.add_argument("--retry-failed", action="store_true", help="Also redraw ones whose render failed")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    return parser.parse_args()


def main() -> int:
    args = _args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    with SessionLocal() as db:
        candidates = find_candidates(db, workspace_id=args.workspace_id, limit=args.limit)
    print(f"F353 thumbnail backfill: {len(candidates)} documents, sheets and reports to look at")
    outcomes = backfill(
        SessionLocal, candidates,
        per_minute=args.per_minute, retry_failed=args.retry_failed, dry_run=args.dry_run,
    )
    print(f"Done: {outcomes}")
    return 1 if outcomes.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
