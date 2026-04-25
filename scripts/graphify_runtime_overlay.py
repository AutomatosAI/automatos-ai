#!/usr/bin/env python3
"""
PRD-135 Phase 4 — Runtime Overlay from ``pg_stat_statements``
==============================================================

Reads ``pg_stat_statements`` on the live DB, maps each query-shape to the
tables it touches (sqlglot), and joins runtime counts onto the Phase 2
code→DB edges. Result: each table gets a ``runtime_calls`` /
``runtime_total_ms`` / ``runtime_mean_ms`` weight.

This is the honest "what actually ran" layer. Combined with:
  - Phase 1 (pg_catalog): what exists in the DB
  - Phase 2 (code walker): what *could* be called from code
it lets us distinguish three buckets:

  1. Alive              — code edge + runtime_calls > 0
  2. Dead-in-runtime    — code edge exists, but zero calls in observation
  3. Fully dead         — no code edge, no runtime calls (drop candidates)

Outputs:
  - graphify-out/runtime.json                — raw stats rows + per-table rollup
  - graphify-out/RUNTIME_OVERLAY_REPORT.md   — heatmap + dead-in-runtime list
  - (optional) rewrites graphify-out/REPORT_dead_tables.md with runtime column

Usage:
  DATABASE_URL=postgres://… python scripts/graphify_runtime_overlay.py
  DATABASE_URL=postgres://… python scripts/graphify_runtime_overlay.py --update-dead-tables
  # reset the capture window *before* running a manual page-sweep:
  DATABASE_URL=postgres://… python scripts/graphify_runtime_overlay.py --reset
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("psycopg2 not installed. pip install psycopg2-binary", file=sys.stderr)
    sys.exit(2)

try:
    import sqlglot
    from sqlglot import exp
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "graphify-out"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

Q_PSS_STATS_PG17 = """
SELECT pss.queryid,
       pss.query,
       pss.calls,
       pss.total_exec_time AS total_ms,
       pss.mean_exec_time  AS mean_ms,
       pss.rows            AS rows_returned,
       pss.shared_blks_hit,
       pss.shared_blks_read
  FROM pg_stat_statements pss
  JOIN pg_roles r ON r.oid = pss.userid
 WHERE pss.query NOT ILIKE '%pg_stat_statements%'
   AND pss.query NOT ILIKE '%pg_catalog.%'
   AND pss.query NOT ILIKE 'COMMIT%'
   AND pss.query NOT ILIKE 'BEGIN%'
   AND pss.query NOT ILIKE 'ROLLBACK%'
   AND pss.query NOT ILIKE 'SET %'
   AND pss.query NOT ILIKE 'SHOW %'
 ORDER BY pss.total_exec_time DESC;
"""

Q_PSS_RESET_TIME = """
SELECT stats_reset FROM pg_stat_statements_info;
"""

# ---------------------------------------------------------------------------
# Query → tables (reads / writes)
# ---------------------------------------------------------------------------

# Writer operations
_WRITER_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Merge)

# Cheap regex fallback when sqlglot can't parse
_RX_FROM   = re.compile(r"\bFROM\s+(?:ONLY\s+)?(?:public\.)?([a-z_][a-z0-9_]*)", re.I)
_RX_JOIN   = re.compile(r"\bJOIN\s+(?:public\.)?([a-z_][a-z0-9_]*)", re.I)
_RX_INTO   = re.compile(r"\bINTO\s+(?:public\.)?([a-z_][a-z0-9_]*)", re.I)
_RX_UPDATE = re.compile(r"\bUPDATE\s+(?:public\.)?([a-z_][a-z0-9_]*)", re.I)
_RX_DELETE = re.compile(r"\bDELETE\s+FROM\s+(?:public\.)?([a-z_][a-z0-9_]*)", re.I)


def extract_tables(sql: str) -> tuple[set[str], set[str]]:
    """Return (reads, writes) table names for a single pg_stat_statements query."""
    reads: set[str] = set()
    writes: set[str] = set()

    if HAS_SQLGLOT:
        try:
            trees = sqlglot.parse(sql, read="postgres", error_level=sqlglot.ErrorLevel.IGNORE)
            for tree in trees:
                if tree is None:
                    continue
                for node in tree.walk():
                    nd = node[0] if isinstance(node, tuple) else node
                    if isinstance(nd, exp.Table):
                        name = nd.name
                        if not name:
                            continue
                        # Walk up to find whether this table is the write target
                        parent = nd.parent
                        is_write = False
                        while parent is not None:
                            if isinstance(parent, _WRITER_NODES):
                                # first Table under the writer is its target
                                if parent.this is nd or (
                                    isinstance(parent.this, exp.Schema) and parent.this.this is nd
                                ):
                                    is_write = True
                                break
                            parent = parent.parent
                        if is_write:
                            writes.add(name)
                        else:
                            reads.add(name)
            return reads, writes
        except Exception:
            # fall through to regex
            pass

    # Regex fallback
    upper = sql.upper().strip()
    if upper.startswith("INSERT"):
        for m in _RX_INTO.finditer(sql):
            writes.add(m.group(1).lower())
    elif upper.startswith("UPDATE"):
        for m in _RX_UPDATE.finditer(sql):
            writes.add(m.group(1).lower())
        for m in _RX_FROM.finditer(sql):
            reads.add(m.group(1).lower())
        for m in _RX_JOIN.finditer(sql):
            reads.add(m.group(1).lower())
    elif upper.startswith("DELETE"):
        for m in _RX_DELETE.finditer(sql):
            writes.add(m.group(1).lower())
        for m in _RX_FROM.finditer(sql):
            reads.add(m.group(1).lower())
    else:
        for m in _RX_FROM.finditer(sql):
            reads.add(m.group(1).lower())
        for m in _RX_JOIN.finditer(sql):
            reads.add(m.group(1).lower())

    return reads, writes


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def rollup_per_table(rows: list[dict], valid_tables: set[str]) -> dict[str, dict]:
    """Aggregate pg_stat_statements rows → per-table counters.

    Only counts tables present in `valid_tables` (public schema real tables).
    """
    agg: dict[str, dict] = defaultdict(lambda: {
        "calls": 0, "total_ms": 0.0, "mean_ms": 0.0,
        "read_calls": 0, "write_calls": 0,
        "distinct_query_shapes": 0,
    })
    query_shapes_per_table: dict[str, set] = defaultdict(set)

    unparsed = 0
    for r in rows:
        sql = r["query"] or ""
        reads, writes = extract_tables(sql)
        reads &= valid_tables
        writes &= valid_tables
        if not reads and not writes:
            unparsed += 1
            continue
        calls = int(r["calls"] or 0)
        total = float(r["total_ms"] or 0.0)
        for t in reads:
            agg[t]["calls"] += calls
            agg[t]["total_ms"] += total
            agg[t]["read_calls"] += calls
            query_shapes_per_table[t].add(r["queryid"])
        for t in writes:
            agg[t]["calls"] += calls
            agg[t]["total_ms"] += total
            agg[t]["write_calls"] += calls
            query_shapes_per_table[t].add(r["queryid"])

    for t, counters in agg.items():
        counters["distinct_query_shapes"] = len(query_shapes_per_table[t])
        counters["mean_ms"] = counters["total_ms"] / counters["calls"] if counters["calls"] else 0.0

    return dict(agg), unparsed


def load_valid_tables() -> set[str]:
    """Load real public-schema tables from the Phase 1 snapshot."""
    db_json = OUT / "db.json"
    if not db_json.exists():
        print(f"  warn: {db_json} missing; run graphify_db_scan.py first. Using catch-all.",
              file=sys.stderr)
        return set()
    snap = json.loads(db_json.read_text())
    return {t["table_name"] for t in snap.get("tables", [])}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    per_table: dict[str, dict],
    unparsed: int,
    row_count: int,
    reset_ts: str | None,
    now_ts: str,
    path: Path,
) -> None:
    lines: list[str] = []
    add = lines.append
    add("# runtime-overlay — PRD-135 Phase 4\n")
    add(f"_Captured from `pg_stat_statements` at **{now_ts}**_")
    if reset_ts:
        add(f"_Capture window opened: **{reset_ts}** (last stats_reset)_")
    add("")
    add(f"- Query shapes observed: **{row_count}**")
    add(f"- Query shapes that parsed to ≥1 public table: **{row_count - unparsed}**")
    add(f"- Tables touched: **{len(per_table)}**")
    add("")

    # Top 30 by call count
    top = sorted(per_table.items(), key=lambda kv: kv[1]["calls"], reverse=True)[:30]
    add("## Top 30 tables by runtime call count")
    add("| Table | Calls | Reads | Writes | Total ms | Mean ms | Shapes |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for tbl, c in top:
        add(f"| `{tbl}` | {c['calls']:,} | {c['read_calls']:,} | {c['write_calls']:,} "
            f"| {c['total_ms']:,.0f} | {c['mean_ms']:.2f} | {c['distinct_query_shapes']} |")
    add("")

    # Top 20 by total time
    top_time = sorted(per_table.items(), key=lambda kv: kv[1]["total_ms"], reverse=True)[:20]
    add("## Top 20 tables by total exec time")
    add("| Table | Total ms | Calls | Mean ms |")
    add("|---|---:|---:|---:|")
    for tbl, c in top_time:
        add(f"| `{tbl}` | {c['total_ms']:,.0f} | {c['calls']:,} | {c['mean_ms']:.2f} |")
    add("")

    # Hot-spot reads vs writes
    write_only = [(t, c) for t, c in per_table.items() if c["write_calls"] > 0 and c["read_calls"] == 0]
    read_only  = [(t, c) for t, c in per_table.items() if c["read_calls"] > 0 and c["write_calls"] == 0]
    add(f"## Write-only tables in window: **{len(write_only)}**")
    add(f"## Read-only tables in window: **{len(read_only)}**")
    add("")

    add("---")
    add("## Join with code→DB edges")
    add("See `REPORT_dead_tables.md` (re-run with --update-dead-tables) for the joined view: "
        "tables with zero code edges AND zero runtime calls = highest-confidence drop candidates.")

    path.write_text("\n".join(lines))


def update_dead_tables_report(
    per_table: dict[str, dict],
    reset_ts: str | None,
    now_ts: str,
) -> None:
    """Rewrite REPORT_dead_tables.md with an extra runtime_calls column."""
    target = OUT / "REPORT_dead_tables.md"
    if not target.exists():
        print("  warn: REPORT_dead_tables.md missing; skip --update-dead-tables",
              file=sys.stderr)
        return

    db_json = json.loads((OUT / "db.json").read_text())
    code_to_db = json.loads((OUT / "code_to_db.graphify.json").read_text())

    # tables with inbound code edge
    tables_with_code = set()
    for l in code_to_db["links"]:
        tgt = l.get("_tgt", "")
        if tgt.startswith("db:table:"):
            tables_with_code.add(tgt[len("db:table:"):])

    all_tables = {t["table_name"]: t for t in db_json["tables"]}
    fk_tables = set()
    # rebuild fk set
    for t in db_json["tables"]:
        fk_tables.add(t["table_name"])  # placeholder; we overwrite below
    fk_tables = set()
    # read foreign_keys
    for fk in db_json.get("foreign_keys", []):
        fk_tables.add(fk["from_table"])
        fk_tables.add(fk["to_table"])

    dead = [
        name for name in all_tables
        if name not in tables_with_code
    ]

    def is_backup(n: str) -> bool:
        return n.startswith("b_")

    dead.sort(key=lambda n: (
        - (all_tables[n]["size_bytes"] or 0),
    ))

    lines: list[str] = []
    add = lines.append
    add("# dead-tables — PRD-135 §5.3\n")
    add("Tables with **no inbound ``reads`` / ``writes`` / ``models`` code edge**.")
    add(f"Runtime column from `pg_stat_statements`, window opened **{reset_ts or '—'}** → **{now_ts}**.\n")
    add(f"Total: **{len(dead)}** of {len(all_tables)} tables.\n")
    add("| Table | Rows | Size | writes (prod) | runtime calls | runtime writes | last_autovacuum | Has FK | Is backup |")
    add("|---|---:|---:|---:|---:|---:|---|---|---|")

    for name in dead:
        t = all_tables[name]
        sz = t["size_bytes"] or 0
        size_str = f"{sz / 1024 / 1024:.1f} MB" if sz > 1024 * 1024 else f"{sz / 1024:.1f} KB"
        rt = per_table.get(name, {})
        rt_calls = rt.get("calls", 0)
        rt_writes = rt.get("write_calls", 0)
        add(
            f"| `{name}` | {t.get('row_estimate', 0):,} | {size_str} "
            f"| {(t.get('n_tup_ins') or 0) + (t.get('n_tup_upd') or 0) + (t.get('n_tup_del') or 0)} "
            f"| {rt_calls:,} | {rt_writes:,} "
            f"| {t.get('last_autovacuum') or '—'} "
            f"| {'✓' if name in fk_tables else ''} "
            f"| {'✓' if is_backup(name) else ''} |"
        )

    add("")
    add("---")
    add("## Highest-confidence drop list")
    add("Dead in code **AND** zero runtime calls during observation window.\n")
    drop_list = [n for n in dead if per_table.get(n, {}).get("calls", 0) == 0]
    add(f"**{len(drop_list)}** tables:\n")
    for n in drop_list:
        add(f"- `{n}`")
    add("")
    add("## Dead in code but touched at runtime (investigate)")
    touched = [(n, per_table[n]["calls"]) for n in dead if per_table.get(n, {}).get("calls", 0) > 0]
    touched.sort(key=lambda kv: -kv[1])
    if not touched:
        add("_None — clean._")
    else:
        add("These are called at runtime but our AST walker missed them. Likely raw SQL in strings we don't parse, or dynamic table names.\n")
        add("| Table | Runtime calls |")
        add("|---|---:|")
        for n, c in touched:
            add(f"| `{n}` | {c:,} |")

    target.write_text("\n".join(lines))
    print(f"  rewrote {target} with runtime column ({len(drop_list)} hard-drop candidates, {len(touched)} investigate)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not HAS_SQLGLOT:
        print("warn: sqlglot not installed; regex fallback only. "
              "pip install sqlglot for better coverage.", file=sys.stderr)

    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true",
                    help="Run pg_stat_statements_reset() and exit (opens a new capture window).")
    ap.add_argument("--update-dead-tables", action="store_true",
                    help="Rewrite REPORT_dead_tables.md with runtime_calls column.")
    ap.add_argument("--min-calls", type=int, default=0,
                    help="Filter out query shapes with fewer than N calls (default 0).")
    args = ap.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not set", file=sys.stderr)
        return 2

    conn = psycopg2.connect(db_url)
    conn.set_session(autocommit=True)

    try:
        if args.reset:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_stat_statements_reset()")
                cur.execute("SELECT NOW()")
                when = cur.fetchone()[0]
            print(f"  pg_stat_statements reset at {when.isoformat()}")
            return 0

        # pg_stat_statements_info.stats_reset
        reset_ts: str | None = None
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(Q_PSS_RESET_TIME)
                row = cur.fetchone()
                if row and row.get("stats_reset"):
                    reset_ts = row["stats_reset"].isoformat()
        except Exception as exc:
            print(f"  warn: could not read pg_stat_statements_info: {exc}", file=sys.stderr)

        now_ts = datetime.now(timezone.utc).isoformat()

        t0 = time.time()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(Q_PSS_STATS_PG17)
            rows = [dict(r) for r in cur.fetchall()]
        duration = round(time.time() - t0, 2)
    finally:
        conn.close()

    if args.min_calls > 0:
        rows = [r for r in rows if (r["calls"] or 0) >= args.min_calls]

    print(f"  fetched {len(rows)} query shapes from pg_stat_statements in {duration}s")

    valid_tables = load_valid_tables()
    per_table, unparsed = rollup_per_table(rows, valid_tables)
    print(f"  mapped to {len(per_table)} public tables   ({unparsed} query shapes skipped — no public-table targets)")

    # Raw runtime snapshot
    raw_out = {
        "captured_at":   now_ts,
        "stats_reset":   reset_ts,
        "query_shapes":  len(rows),
        "unparsed":      unparsed,
        "tables_touched": len(per_table),
        "per_table":     per_table,
        "top_shapes": [
            {
                "queryid":  r["queryid"],
                "calls":    r["calls"],
                "total_ms": float(r["total_ms"] or 0),
                "mean_ms":  float(r["mean_ms"] or 0),
                "query":    (r["query"] or "")[:500],
            }
            for r in sorted(rows, key=lambda r: (r["total_ms"] or 0), reverse=True)[:50]
        ],
    }
    (OUT / "runtime.json").write_text(json.dumps(raw_out, indent=2, default=str))
    print(f"  wrote {OUT / 'runtime.json'}")

    write_report(per_table, unparsed, len(rows), reset_ts, now_ts, OUT / "RUNTIME_OVERLAY_REPORT.md")
    print(f"  wrote {OUT / 'RUNTIME_OVERLAY_REPORT.md'}")

    if args.update_dead_tables:
        update_dead_tables_report(per_table, reset_ts, now_ts)

    return 0


if __name__ == "__main__":
    sys.exit(main())
