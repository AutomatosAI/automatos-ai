#!/usr/bin/env python3
"""PRD-209 S4 — schema-drift check: the structural net for "four writers of schema truth".

Automatos builds its schema four ways — ``init_complete_schema.sql`` (the fresh-clone
seed), the Alembic migration forest, ``Base.metadata.create_all`` at boot, and inline
``ALTER``s inside migrations. When those writers disagree, a table or column that one
writer relies on is silently absent from another. July shipped exactly that: a table
that migrations ``ALTER``-ed but no writer ``CREATE``-d, so a from-base migration replay
died with ``relation "..." does not exist`` (see ``.github/workflows/test.yml`` — the
from-zero lane "crashes on tables that are ALTER-ed but never CREATE-d in the historical
forest"). No lane caught it.

This check is that lane. It is deliberately **pure and static** — it parses text
(migration files + the init SQL), needs no database, no ``Base.metadata`` import (which
is incomplete without the optional ML deps), and no live Alembic replay (the historical
forest does not replay from base — that is *why* PRD-209 S2 boots via init-SQL+stamp
instead). Being static, it runs identically in CI and on a laptop with no services.

What it enforces
----------------
**Every table a migration ``ALTER``s / ``add_column``s / indexes / foreign-keys must
also be ``CREATE``-d by some migration or by ``init_complete_schema.sql``.** A table that
is only ever ``ALTER``-ed — never created by any writer the fresh-clone / replay path can
see — is the July failure class. It "works" today only because ``create_all`` conjures it
at boot from a model; retire ``create_all`` (PRD-209 Q3) or replay from base and it is a
hard crash.

The diff engine (:func:`diff_schemas`) is table- *and* column-granular and general; the
wired check uses its table dimension. The column dimension is exercised by the guard test
``test_schema_drift_detects_divergent_column`` (a model column absent from the other side
is reported as drift) and is available for a future column-level writer comparison.

Baseline
--------
``ORPHAN_ALTER_BASELINE`` records the orphans that already exist on ``main`` today — all
of them ``create_all``-only model tables — with the reason each is accepted. The check is
**green** when the live orphan set equals the baseline and goes **red** on the *next* new
orphan (a fresh ``ALTER``-but-never-``CREATE``). Converging the four writers to one so the
baseline can be emptied is PRD-209 Q3's follow-on; this check makes that safe by making
any regression loud. Fix an orphan (add the table to a migration or the init SQL) and its
baseline entry becomes stale — :func:`stale_baseline_entries` flags it for pruning so the
baseline can never quietly over-accept.

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
_INIT_SQL = _REPO_ROOT / "orchestrator" / "core" / "database" / "init_complete_schema.sql"

# A schema is table name -> set of column names. Column sets may be empty when a
# parser only recovers table identity (the migration-forest parsers below), which is
# all the wired orphan check needs; the column dimension is used by the diff core's
# fixture test and stands ready for a future column-level comparison.
Schema = Dict[str, Set[str]]


# ------------------------------------------------------------- the accepted baseline
# Tables the migration forest ALTERs but that NO migration and NO init SQL ever
# CREATEs — they exist at runtime only because ``create_all`` builds them from a model
# at boot. Each is the July failure class frozen at today's known set; the check bites
# on the NEXT such orphan. Converging to one writer (PRD-209 Q3) lets this shrink to {}.
ORPHAN_ALTER_BASELINE: Dict[str, str] = {
    "audit_logs": "create_all-only model (core/workspaces/audit.py); migrations index it, no CREATE writer — PRD-209 Q3",
    "composio_actions_cache": "create_all-only model; migrations ALTER it, no CREATE writer — PRD-209 Q3",
    "composio_connections": "create_all-only model; migrations ALTER it, no CREATE writer — PRD-209 Q3",
    "tool_execution_logs": "create_all-only model; migrations ALTER it, no CREATE writer — PRD-209 Q3",
    "workflow_recipes": "create_all-only model; migrations ALTER it, no CREATE writer — PRD-209 Q3",
    "workspace_invitations": "create_all-only model; migrations ALTER it, no CREATE writer — PRD-209 Q3",
    "workspace_members": "create_all-only model; migrations ALTER it, no CREATE writer — PRD-209 Q3",
}

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


# --------------------------------------------------------------- the wired check
def orphan_alter_tables(versions_dir: pathlib.Path, init_sql: pathlib.Path) -> Set[str]:
    """Tables ALTERed by a migration but CREATEd by no writer the fresh-clone / replay
    path can see (not by any migration, not by the init SQL). Baseline not yet applied."""
    created = created_tables(versions_dir) | parse_sql_tables(init_sql.read_text(encoding="utf-8"))
    altered = altered_tables(versions_dir)
    # Expressed through the shared diff core: the "missing" (altered-but-not-created)
    # tables are exactly the orphans. Column sets are empty here (table-level check).
    report = diff_schemas(
        expected={t: set() for t in altered},
        actual={t: set() for t in created},
    )
    return report.missing_tables


def stale_baseline_entries(versions_dir: pathlib.Path, init_sql: pathlib.Path) -> Set[str]:
    """Baseline tables that are NO LONGER orphaned (some writer now CREATEs them). They
    should be pruned so the baseline never over-accepts silently."""
    return set(ORPHAN_ALTER_BASELINE) - orphan_alter_tables(versions_dir, init_sql)


def new_orphans(versions_dir: pathlib.Path, init_sql: pathlib.Path) -> Set[str]:
    """Orphans beyond the accepted baseline — the drift that reddens the check."""
    return orphan_alter_tables(versions_dir, init_sql) - set(ORPHAN_ALTER_BASELINE)


def main() -> int:
    orphans = orphan_alter_tables(_VERSIONS_DIR, _INIT_SQL)
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
            "migration and not by init_complete_schema.sql. A from-base replay (and a "
            "fresh clone once create_all is retired) crashes with 'relation does not exist'. "
            "Add each to a migration's create_table or to init_complete_schema.sql:"
        )
        for t in new:
            print(f"  - {t}")
        return 1

    print("OK: no schema drift beyond the documented baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
