#!/usr/bin/env python3
"""PRD-209 S4 — schema-drift check: the structural net for the writers of schema truth.

Automatos builds its schema three ways — the Alembic migration forest (incremental,
existing databases), the model layer via ``Base.metadata.create_all`` (the OFFICIAL
fresh-database path since the 2026-08-29 S2 revision: ``scripts/init_fresh_db.py``
builds create_all + the raw-DDL extras, then stamps heads), and the handful of raw-DDL
tables that have no SQLAlchemy model (``RAW_DDL_EXTRAS`` — built by ``init_test_db``).
The old fourth writer, ``init_complete_schema.sql``, was a stale hand-maintained
snapshot (fresh clones got 107 of prod's ~152 tables) and is deleted.

When the writers disagree, a table one writer relies on is silently absent from
another. July shipped exactly that: a table migrations ``ALTER``-ed but no writer
``CREATE``-d. This check is the lane that catches the next one. It is deliberately
**pure and static** — it parses text (migration files + ``__tablename__``
declarations), needs no database, no ``Base.metadata`` import (incomplete without the
optional ML deps), and no live Alembic replay (the forest holds 41 orphan-root
revisions and cannot replay from empty — the recorded lineage-repair follow-on).

What it enforces
----------------
**Every table a migration ``ALTER``s / ``add_column``s / indexes / foreign-keys must
be ``CREATE``-d by some migration, declared as a model (``__tablename__`` — the
create_all path builds it), or listed in ``RAW_DDL_EXTRAS``.** A table only ever
``ALTER``-ed — never created by any writer the fresh path can see — crashes a fresh
database with ``relation "..." does not exist``.

The diff engine (:func:`diff_schemas`) is table- *and* column-granular and general; the
wired check uses its table dimension. The column dimension is exercised by the guard test
``test_schema_drift_detects_divergent_column`` (a model column absent from the other side
is reported as drift) and is available for a future column-level writer comparison.

Baseline
--------
``ORPHAN_ALTER_BASELINE`` is **empty**: with models counted as a legitimate writer
(they are the fresh path now), every previously-accepted create_all-only orphan is
covered. The check goes red on the next genuine orphan. Should an entry ever be
added, :func:`stale_baseline_entries` flags it for pruning the moment a writer covers
it, so the baseline can never quietly over-accept.

Exit code: ``0`` when the live orphan set is within baseline, ``1`` on any new orphan.
"""
from __future__ import annotations

import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Set

# --------------------------------------------------------------------------- paths
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VERSIONS_DIR = _REPO_ROOT / "orchestrator" / "alembic" / "versions"
_ORCH_ROOT = _REPO_ROOT / "orchestrator"

# Tables with NO SQLAlchemy model, built by raw DDL in scripts/init_test_db.py
# (and therefore by scripts/init_fresh_db.py on the fresh path).
RAW_DDL_EXTRAS: Set[str] = {
    "document_chunks",
    "codegraph_projects",
    "codegraph_files",
    "codegraph_symbols",
    "codegraph_relationships",
    "codegraph_query_logs",
    # PRD-209 drift orphans, ported from the retired init SQL (live readers exist):
    "knowledge_items",
    "tool_usage_logs",
    "kb_types",
    "agent_tool_assignments",
}

# A schema is table name -> set of column names. Column sets may be empty when a
# parser only recovers table identity (the migration-forest parsers below), which is
# all the wired orphan check needs; the column dimension is used by the diff core's
# fixture test and stands ready for a future column-level comparison.
Schema = Dict[str, Set[str]]


# ------------------------------------------------------------- the accepted baseline
# Empty since the 2026-08-29 S2 revision: models are a legitimate writer (the fresh
# path IS create_all), so the seven previously-accepted create_all-only tables are
# covered. Any future entry must carry a reason and is auto-flagged for pruning the
# moment a writer covers it (:func:`stale_baseline_entries`).
ORPHAN_ALTER_BASELINE: Dict[str, str] = {}

# Regex-parse false positives (SQL keywords a permissive identifier match can capture).
_NOT_A_TABLE = {"IF", "EXISTS", "ONLY", "TABLE", "NOT"}


@dataclass(frozen=True)
class DriftReport:
    """The difference between an ``expected`` schema and an ``actual`` one."""

    missing_tables: Set[str] = field(default_factory=set)  # in expected, not actual
    extra_tables: Set[str] = field(default_factory=set)  # in actual, not expected
    missing_columns: Dict[str, Set[str]] = field(default_factory=dict)  # per shared table
    extra_columns: Dict[str, Set[str]] = field(default_factory=dict)

    @property
    def has_drift(self) -> bool:
        return bool(
            self.missing_tables
            or self.extra_tables
            or any(self.missing_columns.values())
            or any(self.extra_columns.values())
        )


def diff_schemas(expected: Schema, actual: Schema) -> DriftReport:
    """Pure diff core. Report tables/columns present in ``expected`` but absent from
    ``actual`` (missing) and vice-versa (extra). No I/O — operates on plain dicts, so
    the guard tests feed it fixture schemas directly (no database, no live DDL)."""
    exp_tables, act_tables = set(expected), set(actual)
    missing_tables = exp_tables - act_tables
    extra_tables = act_tables - exp_tables

    missing_columns: Dict[str, Set[str]] = {}
    extra_columns: Dict[str, Set[str]] = {}
    for table in exp_tables & act_tables:
        exp_cols, act_cols = set(expected[table]), set(actual[table])
        if exp_cols - act_cols:
            missing_columns[table] = exp_cols - act_cols
        if act_cols - exp_cols:
            extra_columns[table] = act_cols - exp_cols

    return DriftReport(
        missing_tables=missing_tables,
        extra_tables=extra_tables,
        missing_columns=missing_columns,
        extra_columns=extra_columns,
    )


# ------------------------------------------------------------------ static parsers
def _clean(names: Set[str]) -> Set[str]:
    return {n for n in names if n.upper() not in _NOT_A_TABLE}


def parse_sql_tables(sql_text: str) -> Set[str]:
    """Table names from ``CREATE TABLE [IF NOT EXISTS] name`` in a SQL script."""
    found = {
        m.group(1).split(".")[-1].strip('"')
        for m in re.finditer(
            r"CREATE TABLE\s+(?:IF NOT EXISTS\s+)?([\w.\"]+)", sql_text, re.IGNORECASE
        )
    }
    return _clean(found)


def model_declared_tables(orch_root: pathlib.Path) -> Set[str]:
    """Tables declared by any SQLAlchemy model — ``__tablename__ = "x"`` — anywhere
    under ``orchestrator/`` (excluding the migration forest and tests). Static text
    scan, mirroring what ``Base.metadata.create_all`` builds on the fresh path
    without importing the model tree (which needs the optional ML deps)."""
    found: Set[str] = set()
    pattern = re.compile(
        r"__tablename__\s*(?::\s*[\w\[\]\"'. ]+)?=\s*['\"]([A-Za-z_]\w*)['\"]"
        r"|\bTable\(\s*['\"]([A-Za-z_]\w*)['\"]"  # SQLAlchemy Core association tables
    )
    for path in orch_root.rglob("*.py"):
        rel = path.relative_to(orch_root).as_posix()
        if rel.startswith(("alembic/", "tests/", "scripts/")) or "__pycache__" in rel:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "__tablename__" in text or "Table(" in text:
            found |= {(m.group(1) or m.group(2)) for m in pattern.finditer(text)}
    return _clean(found)


def _iter_migration_text(versions_dir: pathlib.Path):
    for path in sorted(versions_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        yield path.read_text(encoding="utf-8", errors="replace")


def created_tables(versions_dir: pathlib.Path) -> Set[str]:
    """Tables any migration CREATEs — ``op.create_table('x')`` or raw ``CREATE TABLE x``.
    Scanned across the whole file (both directions): a table created in either an
    upgrade or a downgrade is a table the forest knows how to build."""
    found: Set[str] = set()
    for text in _iter_migration_text(versions_dir):
        found |= {m.group(1) for m in re.finditer(r"op\.create_table\(\s*['\"]([A-Za-z_]\w*)['\"]", text)}
        found |= {
            m.group(1)
            for m in re.finditer(r"CREATE TABLE\s+(?:IF NOT EXISTS\s+)?[\"']?([A-Za-z_]\w*)", text, re.IGNORECASE)
        }
    return _clean(found)


def altered_tables(versions_dir: pathlib.Path) -> Set[str]:
    """Tables any migration ALTERs — ``add_column`` / ``alter_column`` / ``drop_column``
    (table is the 1st arg), ``create_index`` / ``create_foreign_key`` (table is the 2nd
    arg), or raw ``ALTER TABLE``. These are the tables a migration assumes already exist."""
    found: Set[str] = set()
    for text in _iter_migration_text(versions_dir):
        for verb in ("add_column", "alter_column", "drop_column"):
            found |= {m.group(1) for m in re.finditer(rf"op\.{verb}\(\s*['\"]([A-Za-z_]\w*)['\"]", text)}
        # 2nd positional arg is the table; 1st (index / constraint name, possibly an
        # ``op.f(...)`` call) is skipped by consuming up to the first comma.
        for verb in ("create_index", "create_foreign_key"):
            found |= {m.group(1) for m in re.finditer(rf"op\.{verb}\([^,]*,\s*['\"]([A-Za-z_]\w*)['\"]", text)}
        found |= {
            m.group(1)
            for m in re.finditer(r"ALTER TABLE\s+(?:IF EXISTS\s+)?(?:ONLY\s+)?[\"']?([A-Za-z_]\w*)", text, re.IGNORECASE)
        }
    return _clean(found)


def dropped_tables(versions_dir: pathlib.Path) -> Set[str]:
    """Tables some migration DROPs in its upgrade — including the loop form
    ``for table in [...]: op.execute(f'DROP TABLE IF EXISTS {table} CASCADE')`` used
    by the cleanup migrations (quoted names listed one per line)."""
    found: Set[str] = set()
    for text in _iter_migration_text(versions_dir):
        up = text.split("def downgrade")[0]
        found |= {m.group(1) for m in re.finditer(r"op\.drop_table\(\s*['\"]([A-Za-z_]\w*)['\"]", up)}
        found |= {m.group(1) for m in re.finditer(r"DROP TABLE\s+(?:IF EXISTS\s+)?[\"']?([A-Za-z_]\w*)", up, re.I)}
        if re.search(r"DROP TABLE IF EXISTS \{", up):
            found |= {m.group(1) for m in re.finditer(r"^\s*['\"]([a-z_]{3,})['\"],\s*$", up, re.M)}
    return _clean(found)


# --------------------------------------------------------------- the wired check
def orphan_alter_tables(versions_dir: pathlib.Path, orch_root: pathlib.Path) -> Set[str]:
    """Tables ALTERed by a migration but CREATEd by no writer the fresh path can see
    (not by any migration, not by a model, not in RAW_DDL_EXTRAS). Baseline not applied."""
    created = created_tables(versions_dir) | model_declared_tables(orch_root) | RAW_DDL_EXTRAS
    # A table a cleanup migration DROPs for good (prd135 buckets, prd142 wave5,
    # prd187 s5, prd195 …) is a relic: ALTERs that predate the drop are history,
    # not orphans. (Live code still referencing a relic is a different bug class —
    # see the PRD-209 addendum.)
    altered = altered_tables(versions_dir) - dropped_tables(versions_dir)
    # Expressed through the shared diff core: the "missing" (altered-but-not-created)
    # tables are exactly the orphans. Column sets are empty here (table-level check).
    report = diff_schemas(
        expected={t: set() for t in altered},
        actual={t: set() for t in created},
    )
    return report.missing_tables


def stale_baseline_entries(versions_dir: pathlib.Path, orch_root: pathlib.Path) -> Set[str]:
    """Baseline tables that are NO LONGER orphaned (some writer now CREATEs them). They
    should be pruned so the baseline never over-accepts silently."""
    return set(ORPHAN_ALTER_BASELINE) - orphan_alter_tables(versions_dir, orch_root)


def new_orphans(versions_dir: pathlib.Path, orch_root: pathlib.Path) -> Set[str]:
    """Orphans beyond the accepted baseline — the drift that reddens the check."""
    return orphan_alter_tables(versions_dir, orch_root) - set(ORPHAN_ALTER_BASELINE)


def main() -> int:
    orphans = orphan_alter_tables(_VERSIONS_DIR, _ORCH_ROOT)
    new = sorted(orphans - set(ORPHAN_ALTER_BASELINE))
    stale = sorted(set(ORPHAN_ALTER_BASELINE) - orphans)

    print(f"schema-drift check — {len(orphans)} orphan-alter table(s); baseline accepts {len(ORPHAN_ALTER_BASELINE)}.")
    if stale:
        print(
            "\nNOTE: these baseline entries are no longer orphaned (a writer now CREATEs "
            "them) — prune them from ORPHAN_ALTER_BASELINE:"
        )
        for t in stale:
            print(f"  - {t}")

    if new:
        print(
            f"\nDRIFT: {len(new)} table(s) are ALTER-ed by a migration but CREATE-d by no "
            "migration, declared by no model (__tablename__), and absent from "
            "RAW_DDL_EXTRAS. A fresh database crashes on these with 'relation does not "
            "exist'. Add each to a migration's create_table, give it a model, or (raw-DDL "
            "only) add it to init_test_db + RAW_DDL_EXTRAS:"
        )
        for t in new:
            print(f"  - {t}")
        return 1

    print("OK: no schema drift beyond the documented baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
