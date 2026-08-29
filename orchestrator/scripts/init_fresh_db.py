#!/usr/bin/env python3
"""
Fresh-database initializer — the supported fresh-clone schema path (PRD-209).

Builds a complete schema on an EMPTY database via
``scripts/generate_schema_baseline.build_schema`` — the model layer
(``init_test_db.init_db()``: create_all + raw-DDL extras) followed by a
statement-tolerant replay of the migration forest — and leaves alembic at heads,
so ``alembic upgrade heads`` is a no-op and every future migration applies
incrementally.

Why both writers? ~8 core tables (workspaces, chats, messages, system_settings…)
exist only as models, and ~50 tables (notifications, deliverables, playbooks…)
exist only in migrations; the forest's 41 orphan-root revisions cannot replay
cleanly from empty on their own. Re-chaining that lineage is recorded follow-on
work (PRD-209 addendum). No schema snapshot is committed anywhere — the
generator IS the fresh path, so nothing can rot.

Existing databases (anything with an ``alembic_version`` row — prod, upgraded
locals) never come near this script; the entrypoint routes them straight to
``alembic upgrade heads``.

Usage: python -m scripts.init_fresh_db   (from /app; exits non-zero on failure)
"""

import sys

from sqlalchemy import create_engine, text

from config import config
from scripts.generate_schema_baseline import build_schema



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

    print("init_fresh_db: empty database — building the full schema (models + tolerant migration replay)…")
    total = build_schema(engine)
    print(f"init_fresh_db: done — {total} tables, alembic at heads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
