"""
Move the Auto agents still on the platform default to it (Gerard, 19 Oct 2026)
==============================================================================

The platform default for Auto is system_settings ``orchestrator_llm`` provider and
model. An admin's change of it through the System Settings API moves the Auto rows
that never chose a model (``core/llm/auto_model_sync``). This script does the same
by hand: for a default set before that existed, or after a failed move.

Default is a dry run on a READ-ONLY connection: it prints, per
``from provider/model -> to provider/model``, how many Auto rows would move, and how
many stay because a workspace chose their model. ``--apply`` moves them.

Run it where the API runs (same DATABASE_URL and keys; the target is fitted to the
keys present, as a new agent's is)::

    cd orchestrator
    python scripts/restamp_auto_models.py
    python scripts/restamp_auto_models.py --apply
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Make orchestrator package imports work when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

READ_ONLY_OPTIONS = "-c default_transaction_read_only=on"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move inheriting Auto agents to the platform default model")
    parser.add_argument("--apply", action="store_true", help="Move the rows (default: dry run, read-only)")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    return parser.parse_args()


def _read_only_session():
    """A session on a connection Postgres refuses every write on."""
    from core.database.database import DATABASE_URL

    engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args={"options": READ_ONLY_OPTIONS})
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)()


def _print_plan(counts: dict, applied: bool) -> None:
    verb = "moved" if applied else "would move"
    print(f"Platform default (orchestrator_llm): {counts['target']}")
    for route, rows in sorted(counts["by_route"].items()):
        print(f"  {route}: {rows} Auto rows {verb}")
    print(f"  {counts['moved']} {verb} in all; {counts['current']} already on it; "
          f"{counts['chosen']} left as chosen in Settings")


def _dry_run() -> dict:
    from core.llm.auto_model_sync import auto_agent_rows, plan_restamp
    from core.llm.new_agent_model import default_route

    with _read_only_session() as db:
        return plan_restamp(auto_agent_rows(db), default_route()).counts()


def _apply() -> dict:
    from core.database.database import SessionLocal
    from core.llm.auto_model_sync import restamp_inheriting_auto_agents

    with SessionLocal() as db:
        counts = restamp_inheriting_auto_agents(db)
        db.commit()
    return counts


def main() -> int:
    args = _args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from core.monitoring.request_log_guard import guard_request_logs
    guard_request_logs()  # no request URL or SDK request body in this script's logs (#1178)
    counts = _apply() if args.apply else _dry_run()
    _print_plan(counts, applied=args.apply)
    if not args.apply:
        print("Dry run: nothing changed. Re-run with --apply to move them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
