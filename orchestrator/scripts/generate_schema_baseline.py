#!/usr/bin/env python3
"""
Generate the fresh-install schema baseline (PRD-209 S2, revision 2).

Schema truth lives in TWO writers: SQLAlchemy models (create_all — the only source
for ~8 core tables no migration ever creates: workspaces, chats, messages,
system_settings, …) and the Alembic forest (the only source for ~52 tables that have
no model: notifications, deliverables, playbooks, agent_skills, …). Neither alone
yields a complete database, and the forest's 41 orphan-root revisions cannot replay
cleanly from empty. So the baseline is GENERATED:

  1. init_test_db.init_db()  — create_all + raw-DDL extras (the CI-proven model layer)
  2. tolerant replay          — every revision, topological order, one at a time; a
                                 revision that fails (ALTER on a table create_all already
                                 shaped, orphan-root ordering, …) is stamped past and
                                 logged — the model layer already carries its intent.
  3. alembic_version ends at heads (every revision applied or stamped past).

There is deliberately NO committed schema dump: this generator IS the fresh path —
scripts/init_fresh_db.py (boot) and the CI from-zero gate both run build_schema(), so
there is no snapshot artifact that can rot. First boot pays ~2-3 minutes once.

Run against an EMPTY database (DATABASE_URL). Prints the skipped-revision log.
"""
import sys
from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
import functools
import re
from sqlalchemy import create_engine, text

from config import config
from scripts.init_test_db import init_db


# --- statement-granular tolerance ------------------------------------------------
# Revision-granular skipping loses sibling tables: a revision that creates a
# model-covered table AND a model-less one dies on the first DuplicateTable and takes
# the second with it. Wrap every alembic op in a SAVEPOINT so each statement succeeds
# or is skipped on its own. Failures are expected and logged (DuplicateTable /
# DuplicateColumn = the model layer already carries that intent).
_OP_NAMES = ["create_table", "drop_table", "add_column", "drop_column", "alter_column",
             "create_index", "drop_index", "create_foreign_key", "drop_constraint",
             "create_unique_constraint", "create_check_constraint", "create_primary_key",
             "rename_table", "execute", "bulk_insert"]
SKIPPED_OPS: list[str] = []


def _install_tolerant_ops() -> None:
    for name in _OP_NAMES:
        orig = getattr(Operations, name, None)
        if orig is None:
            continue

        @functools.wraps(orig)
        def wrapper(self, *a, __orig=orig, __name=name, **k):
            # NEVER drop during fresh generation: models define the current shape;
            # drops are history. A tolerated drop_table whose re-create then failed
            # silently deleted document_chunks (found 2026-08-29). Dead tables that a
            # migration would drop simply linger, harmlessly.
            if __name.startswith("drop_") or (
                __name == "execute" and a and isinstance(a[0], str)
                and re.match(r"\s*(DROP|TRUNCATE)\b", a[0], re.I)
            ):
                SKIPPED_OPS.append(f"{__name}: skipped by policy (generator never drops)")
                return None
            bind = self.get_bind()
            try:
                with bind.begin_nested():
                    return __orig(self, *a, **k)
            except Exception as exc:  # noqa: BLE001
                SKIPPED_OPS.append(f"{__name}: {str(exc).strip().splitlines()[0][:110]}")
                return None

        setattr(Operations, name, wrapper)


def _missing_tables(engine, wanted: set[str]) -> set[str]:
    with engine.connect() as conn:
        have = {r[0] for r in conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))}
    return wanted - have


def _created_by(script: ScriptDirectory) -> dict[str, str]:
    """table -> revision that create_table()s it (first seen)."""
    out: dict[str, str] = {}
    for rev in script.walk_revisions("base", "heads"):
        src = open(rev.path, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"op\.create_table\(\s*['\"]([A-Za-z_]\w*)['\"]", src):
            out.setdefault(m.group(1), rev.revision)
        for m in re.finditer(r"CREATE TABLE\s+(?:IF NOT EXISTS\s+)?[\"']?([A-Za-z_]\w*)", src, re.I):
            out.setdefault(m.group(1), rev.revision)
    return out


def _rerun_upgrade(engine, script: ScriptDirectory, rev: str) -> None:
    """Re-execute one already-stamped revision's upgrade() (tolerant ops) — for creators
    whose create_table lost an FK race in the main pass."""
    module = script.get_revision(rev).module
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            module.upgrade()


def _final_state_dropped(script: ScriptDirectory) -> set[str]:
    """Tables whose LAST touch in topological forest order is a DROP — relics that a
    cleanup migration retired for good (prd135 drop buckets, prd142 wave5, prd187 s5,
    prd195 …). A table dropped and later re-created is NOT a relic (final state =
    created). Static text parse over upgrade() bodies only."""
    state: dict[str, str] = {}
    for rev in reversed(list(script.walk_revisions("base", "heads"))):
        src = open(rev.path, encoding="utf-8", errors="replace").read()
        up = src.split("def downgrade")[0]
        for m in re.finditer(r"op\.create_table\(\s*['\"]([A-Za-z_]\w*)['\"]|CREATE TABLE\s+(?:IF NOT EXISTS\s+)?[\"']?([A-Za-z_]\w*)", up, re.I):
            state[(m.group(1) or m.group(2))] = "created"
        for m in re.finditer(r"op\.drop_table\(\s*['\"]([A-Za-z_]\w*)['\"]|DROP TABLE\s+(?:IF EXISTS\s+)?[\"']?([A-Za-z_]\w*)", up, re.I):
            state[(m.group(1) or m.group(2))] = "dropped"
        # f-string / loop drops: `for table in [...]: op.execute(f'DROP TABLE IF EXISTS {table} CASCADE')`
        if re.search(r"DROP TABLE IF EXISTS \{", up):
            for m in re.finditer(r"^\s*['\"]([a-z_]{3,})['\"],\s*$", up, re.M):
                state[m.group(1)] = "dropped"
    return {t for t, st in state.items() if st == "dropped"}


def _drop_relics(engine, script: ScriptDirectory) -> list[str]:
    """Post-replay parity pass: drop tables whose final forest state is 'dropped' and
    that neither a model nor RAW-DDL extras (init_test_db) legitimately own. Without
    this the never-drop replay policy leaves stale-shaped zombies that live code
    (written against the pre-drop shape) then trips over."""
    from sqlalchemy import inspect as _inspect
    from core.database.database import Base
    protected = set(Base.metadata.tables) | {
        "document_chunks", "codegraph_projects", "codegraph_files", "codegraph_symbols",
        "codegraph_relationships", "codegraph_query_logs", "knowledge_items", "kb_types",
        "tool_usage_logs", "agent_tool_assignments",
    }
    relics = sorted(_final_state_dropped(script) - protected)
    dropped: list[str] = []
    with engine.begin() as conn:
        existing = set(_inspect(conn).get_table_names())
        for t in relics:
            if t in existing:
                conn.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
                dropped.append(t)
    return dropped


def build_schema(engine) -> int:
    """Build the complete fresh schema on an EMPTY database and leave alembic at
    heads. Returns the table count. Used by scripts/init_fresh_db.py (the boot
    path) and by this module's CLI (the CI gate)."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))  # prd008a_sites uses uuid_generate_v4()
        # Alembic's default version table is VARCHAR(32); this repo has revision ids
        # up to 35 chars (dedupe_skills_unique_workspace_name). Prod's table is wider;
        # pre-create it wide so stamps never truncate.
        conn.execute(text("CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(255) NOT NULL PRIMARY KEY)"))

    print("== 1/2 model layer: create_all + raw-DDL extras")
    init_db()

    print("== 2/2 tolerant replay of the migration forest (topological, one revision per step)")
    cfg = AlembicConfig("alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    _install_tolerant_ops()
    order = [r.revision for r in reversed(list(script.walk_revisions("base", "heads")))]
    applied = skipped = 0
    for rev in order:
        try:
            command.upgrade(cfg, rev)
            applied += 1
        except Exception as exc:  # noqa: BLE001 — whole-revision failure (raw bind.execute etc.)
            skipped += 1
            print(f"   skip {rev}: {str(exc).strip().splitlines()[0][:150]}")
            command.stamp(cfg, rev)
    print(f"   ({len(SKIPPED_OPS)} individual ops tolerated inside applied revisions)")

    # Residual pass: tables a migration creates but that lost an ordering/FK race.
    creators = _created_by(script)
    for rnd in range(1, 4):
        missing = _missing_tables(engine, set(creators))
        if not missing:
            break
        print(f"== residual pass {rnd}: {len(missing)} migration-created tables missing — re-running their creators")
        for rev in sorted({creators[t] for t in missing}):
            before = len(SKIPPED_OPS)
            try:
                _rerun_upgrade(engine, script, rev)
            except Exception as exc:  # noqa: BLE001
                print(f"   residual {rev}: {str(exc).strip().splitlines()[0][:120]}")
            for reason in SKIPPED_OPS[before:]:
                if "already exists" not in reason and "skipped by policy" not in reason:
                    print(f"      {rev} op: {reason}")
        still = _missing_tables(engine, set(creators))
        if still == missing:
            print(f"   no progress; still missing: {sorted(still)}")
            break
    # Re-assert the model layer + raw-DDL extras: a migration's raw
    # `conn.execute(text("DROP ..."))` bypasses alembic ops and can remove an extra;
    # create_all is checkfirst and the extras are IF NOT EXISTS, so this is idempotent.
    print("== 3/4 re-asserting the model layer + raw-DDL extras")
    init_db()
    relics = _drop_relics(engine, script)
    print(f"== 4/4 relic parity pass: dropped {len(relics)} table(s) whose final forest state is DROP: {relics}")
    with engine.begin() as conn:
        total = conn.execute(text("SELECT count(*) FROM information_schema.tables WHERE table_schema='public'")).scalar()
    print(f"== done: {applied} revisions applied, {skipped} stamped past; {total} tables")
    print(f"== still-missing migration-created tables: {sorted(_missing_tables(engine, set(creators)))}")
    return int(total or 0)


def main() -> int:
    engine = create_engine(config.DATABASE_URL)
    with engine.connect() as conn:
        n = conn.execute(text("SELECT count(*) FROM information_schema.tables WHERE table_schema='public'")).scalar()
    if n:
        print(f"generate_schema_baseline: REFUSING — database has {n} tables; use an empty one.", file=sys.stderr)
        return 1
    build_schema(engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
