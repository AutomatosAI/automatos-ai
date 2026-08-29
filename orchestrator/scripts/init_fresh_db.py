#!/usr/bin/env python3
"""
Fresh-database initializer — the supported fresh-clone schema path (PRD-209).

Builds a complete schema on an EMPTY database exactly the way CI does on every
run — ``scripts/init_test_db.init_db()`` (``Base.metadata.create_all`` plus the
raw-DDL tables that have no SQLAlchemy model: document_chunks, codegraph_*) —
then stamps ``alembic_version`` at the single head so ``alembic upgrade heads``
is a no-op and every future migration applies incrementally.

Why not replay the migration history from empty? The forest contains 41
revisions with ``down_revision = None`` (orphan roots from the hotfix era)
that ALTER tables which, from empty, nothing has created yet — a replay dies
immediately. Re-chaining 41 legacy revisions is recorded follow-on work
(PRD-209 addendum); until then, THIS path is schema truth for new databases,
and it is proven daily: the entire CI suite runs against exactly this schema.

Existing databases (anything with an ``alembic_version`` row — prod, upgraded
locals) never come near this script; the entrypoint routes them straight to
``alembic upgrade heads``.

Usage: python -m scripts.init_fresh_db   (from /app; exits non-zero on failure)
"""

import sys

from sqlalchemy import create_engine, text

from config import config
from scripts.init_test_db import init_db


def stamp_head() -> None:
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig("alembic.ini")
    command.stamp(cfg, "heads")


def main() -> int:
    url = config.DATABASE_URL
    engine = create_engine(url)
    with engine.connect() as conn:
        has_version = conn.execute(
            text("SELECT to_regclass('alembic_version')")
        ).scalar()
        table_count = conn.execute(
            text(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema='public'"
            )
        ).scalar()

    if has_version:
        print("init_fresh_db: alembic_version exists — not a fresh database, nothing to do.")
        return 0
    if table_count and int(table_count) > 0:
        print(
            f"init_fresh_db: REFUSING — no alembic_version but {table_count} tables exist. "
            "This database is in an unknown state; initialize an empty database instead.",
            file=sys.stderr,
        )
        return 1

    print("init_fresh_db: empty database — building the CI-proven schema (create_all + raw-DDL extras)…")
    # The schema declares pgvector columns; the extension must exist first.
    # (The compose postgres user is superuser, so this succeeds locally; CI
    # creates it in a dedicated step the same way.)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    init_db()
    print("init_fresh_db: stamping alembic_version at heads…")
    stamp_head()
    print("init_fresh_db: done — schema complete and stamped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
