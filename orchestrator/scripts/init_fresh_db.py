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
``alembic upgrade heads``. The one exception is a database whose own fresh
build was interrupted: it still carries ``build_schema``'s incomplete-build
marker, never served, and is resumed here — re-running the build finishes it
with the same schema a clean build produces.

Usage: python -m scripts.init_fresh_db   (from /app; exits non-zero on failure)
"""

import sys

from sqlalchemy import create_engine, text

from config import config
from scripts.generate_schema_baseline import INCOMPLETE_MARKER, build_schema

BUILD, RESUME, NOTHING, REFUSE = "build", "resume", "nothing", "refuse"


def fresh_db_action(has_marker: bool, has_version: bool, table_count: int) -> str:
    """What to do with this database. The marker wins: alembic_version exists
    from a build's first statement, so it proves nothing on its own."""
    if has_marker:
        return RESUME
    if has_version:
        return NOTHING
    return REFUSE if table_count > 0 else BUILD


def _inspect(engine) -> tuple[bool, bool, int]:
    with engine.connect() as conn:
        has_marker = conn.execute(text(f"SELECT to_regclass('{INCOMPLETE_MARKER}')")).scalar()
        has_version = conn.execute(text("SELECT to_regclass('alembic_version')")).scalar()
        table_count = conn.execute(
            text("SELECT count(*) FROM information_schema.tables WHERE table_schema='public'")
        ).scalar()
    return bool(has_marker), bool(has_version), int(table_count or 0)


def main() -> int:
    engine = create_engine(config.DATABASE_URL)
    has_marker, has_version, table_count = _inspect(engine)
    action = fresh_db_action(has_marker, has_version, table_count)

    if action == NOTHING:
        print("init_fresh_db: alembic_version exists — not a fresh database, nothing to do.")
        return 0
    if action == REFUSE:
        print(
            f"init_fresh_db: REFUSING — no alembic_version but {table_count} tables exist. "
            "This database is in an unknown state; initialize an empty database instead.",
            file=sys.stderr,
        )
        return 1
    if action == RESUME:
        print(f"init_fresh_db: a previous fresh build was interrupted ({table_count} tables, "
              f"{INCOMPLETE_MARKER} present) — resuming it…")
    else:
        print("init_fresh_db: empty database — building the full schema (models + tolerant migration replay)…")
    total = build_schema(engine)
    print(f"init_fresh_db: done — {total} tables, alembic at heads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
