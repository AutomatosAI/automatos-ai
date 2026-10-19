"""Clerk production cutover: relink returning users to their production Clerk IDs, once.
====================================================================================

Sign-in never moves a ``users`` row to another Clerk ID. After the development
users are imported into the production Clerk instance (clerk/migration-tool, which
stores each development ``id`` as the production user's ``external_id``), this run
moves each returning row from its development ID to its production ID. The rules
are in ``core/auth/clerk_cutover.py``: the import's ID link, a verified email, at
most once per row, every move recorded in ``clerk_user_relinks``.

It needs the PRODUCTION secret key (``CLERK_SECRET_KEY``, ``sk_live_…``) and the
production database, so run it where the API runs. Without ``--apply`` it only
prints the plan. With ``--apply`` it writes the whole plan in one transaction, or
nothing. ``--rollback`` moves every recorded relink back to its development ID
(no Clerk call), for a rollback to the development instance::

    cd orchestrator
    python -m scripts.clerk_cutover_relink                       # plan only
    python -m scripts.clerk_cutover_relink --apply               # relink
    python -m scripts.clerk_cutover_relink --rollback            # rollback plan only
    python -m scripts.clerk_cutover_relink --rollback --apply    # roll back

The report prints users.id and Clerk IDs, never an email. So does a failed run: it
logs the error's type and the failing decision's IDs, never the error's message (an
``IntegrityError`` carries the statement's bound parameters, and the relink record's
include the verified email), and its session's engine hides bound parameters too.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional

# Make orchestrator package imports work when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from core.auth import clerk_cutover as cutover  # noqa: E402
from core.auth.clerk import get_clerk_auth  # noqa: E402
from core.database.database import DATABASE_URL  # noqa: E402

logger = logging.getLogger("clerk_cutover_relink")

# Clerk's GET /v1/users allows up to 500 a page; 100 keeps each response small.
PAGE_SIZE = 100
# The outcomes that need a person: everything that is neither a write nor a no-op.
QUIET_OUTCOMES = frozenset({cutover.ALREADY_LINKED, cutover.NOT_IMPORTED, cutover.ALREADY_REVERTED})


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relink returning users to their production Clerk IDs, once")
    parser.add_argument("--apply", action="store_true", help="Write the plan (one transaction); default prints it")
    parser.add_argument("--rollback", action="store_true", help="Move recorded relinks back to the development IDs")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    return parser.parse_args()


def fetch_production_users() -> List[cutover.ProductionUser]:
    """Every user of the instance the secret key belongs to, page by page, oldest first.

    A user seen on two pages (someone created mid-run shifts the offsets) counts once.
    """
    clerk = get_clerk_auth()
    users: Dict[str, cutover.ProductionUser] = {}
    offset = 0
    while True:
        page = clerk.list_users_page(PAGE_SIZE, offset)
        for payload in page:
            user = cutover.production_user(payload)
            users[user.clerk_user_id] = user
        if len(page) < PAGE_SIZE:
            return list(users.values())
        offset += PAGE_SIZE


def _report(decisions: List[cutover.Decision]) -> None:
    for outcome, count in sorted(cutover.summarize(decisions).items()):
        print(f"  {outcome}: {count}")
    for d in decisions:
        if d.outcome not in QUIET_OUTCOMES:
            print(f"  [{d.outcome}] users.id={d.user_id} old={d.old_clerk_user_id} new={d.new_clerk_user_id}")


@contextmanager
def _session() -> Iterator[Session]:
    """A session on this run's own engine, which keeps bound parameters (the verified
    email the relink record is written with) out of every SQLAlchemy error message."""
    engine = create_engine(DATABASE_URL, hide_parameters=True, pool_pre_ping=True)
    try:
        with Session(engine, autoflush=False) as db:
            yield db
    finally:
        engine.dispose()


def _failure_line(exc: BaseException, decision: Optional[cutover.Decision]) -> str:
    """The log line for a failed write: the error's type and the decision's IDs only."""
    where = (
        f"users.id={decision.user_id} old={decision.old_clerk_user_id} new={decision.new_clerk_user_id}"
        if decision else "the commit"
    )
    return f"Clerk cutover run failed ({type(exc).__name__}) at {where}; rolled back, nothing was written"


def _write(db, decisions: List[cutover.Decision], rollback: bool) -> Optional[int]:
    """Write the plan in one transaction and commit: the rows moved, or None when it failed.

    Each decision is written in turn so a failure names the one that failed. It is
    logged by type and IDs only, never ``str(exc)`` or a traceback: an
    ``IntegrityError``'s message carries the statement's bound parameters, the
    verified email among them.
    """
    write = cutover.apply_rollback if rollback else cutover.apply_relinks
    moved, current = 0, None
    try:
        for current in decisions:
            moved += write(db, [current])
        current = None
        db.commit()
    except Exception as exc:  # noqa: BLE001 — logged by type and IDs: its message can carry an email
        db.rollback()
        logger.error(_failure_line(exc, current))
        return None
    return moved


def _plan(db, rollback: bool) -> List[cutover.Decision]:
    if rollback:
        return cutover.plan_rollback(db)
    prod_users = fetch_production_users()
    imported = sum(1 for p in prod_users if p.external_id)
    print(f"Clerk instance: {len(prod_users)} users, {imported} with an external_id from the import")
    return cutover.plan_relinks(db, prod_users)


def main() -> int:
    args = _args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from core.monitoring.request_log_guard import guard_request_logs
    guard_request_logs()  # no request URL or SDK request body in this script's logs (#1178)
    with _session() as db:
        decisions = _plan(db, args.rollback)
        print("Rollback plan:" if args.rollback else "Relink plan:")
        _report(decisions)
        writes = sum(1 for d in decisions if d.outcome in cutover.WRITING_OUTCOMES)
        if not args.apply:
            db.rollback()
            print(f"Plan only: {writes} rows would move. Nothing was written; add --apply to write.")
            return 0
        moved = _write(db, decisions, args.rollback)
        if moved is None:
            return 1
    print(f"Done: {moved} rows moved and recorded in clerk_user_relinks (committed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
