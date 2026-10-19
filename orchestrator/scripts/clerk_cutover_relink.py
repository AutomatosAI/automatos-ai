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

The report prints users.id and Clerk IDs, never an email.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List

# Make orchestrator package imports work when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.auth import clerk_cutover as cutover  # noqa: E402
from core.auth.clerk import get_clerk_auth  # noqa: E402
from core.database.database import SessionLocal  # noqa: E402

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
    with SessionLocal() as db:
        decisions = _plan(db, args.rollback)
        print("Rollback plan:" if args.rollback else "Relink plan:")
        _report(decisions)
        writes = sum(1 for d in decisions if d.outcome in cutover.WRITING_OUTCOMES)
        if not args.apply:
            db.rollback()
            print(f"Plan only: {writes} rows would move. Nothing was written; add --apply to write.")
            return 0
        try:
            moved = cutover.apply_rollback(db, decisions) if args.rollback else cutover.apply_relinks(db, decisions)
            db.commit()
        except Exception:
            db.rollback()
            logger.exception("Clerk cutover run failed; rolled back, nothing was written")
            return 1
    print(f"Done: {moved} rows moved and recorded in clerk_user_relinks (committed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
