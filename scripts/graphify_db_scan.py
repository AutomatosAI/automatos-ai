#!/usr/bin/env python3
"""
PRD-135 Phase 1 — Live DB Snapshot Scanner
===========================================

Follows PRD-135 §5.2 Pass 1 + Appendix A queries verbatim.

Connects read-only to Postgres (from ``DATABASE_URL``) and emits graphify-shape
nodes + structural edges:

Node types:
  - table    (pg_class + pg_stat_user_tables)
  - column   (information_schema.columns)
  - index    (pg_index + pg_stat_user_indexes)
  - view     (pg_views)
  - function (pg_proc)
  - trigger  (pg_trigger)

Edge types:
  - column_of   column -> table
  - fk_to       table -> table
  - index_of    index -> table
  - depends_on  view -> table

Outputs:
  - graphify-out/db.json             — raw DB snapshot (human-readable)
  - graphify-out/db.graphify.json    — graphify merge fragment (nodes+links)
  - graphify-out/DB_SCAN_REPORT.md   — human summary

PRD-135 §9 Success criteria:
  - Runs in < 60s.
  - Read-only (no writes, no DDL).
  - Zero impact on production runtime.

Usage:
  DATABASE_URL=postgres://... python scripts/graphify_db_scan.py
  python scripts/graphify_db_scan.py --merge  # also write merged graph.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("psycopg2 not installed. pip install psycopg2-binary", file=sys.stderr)
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "graphify-out"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# PRD-135 Appendix A — ready-to-port queries, read-only, safe on prod.
# ---------------------------------------------------------------------------

Q_TABLES = """
SELECT c.relname                      AS table_name,
       n.nspname                      AS schema_name,
       c.reltuples::bigint            AS row_estimate,
       pg_total_relation_size(c.oid)  AS size_bytes,
       s.last_autovacuum,
       s.last_autoanalyze,
       s.n_live_tup,
       s.n_dead_tup,
       s.seq_scan,
       s.idx_scan,
       s.n_tup_ins,
       s.n_tup_upd,
       s.n_tup_del
  FROM pg_class c
  JOIN pg_namespace n ON n.oid = c.relnamespace
  LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
 WHERE c.relkind = 'r' AND n.nspname = 'public'
 ORDER BY c.relname;
"""

Q_COLUMNS = """
SELECT table_schema, table_name, column_name, ordinal_position,
       data_type, is_nullable, column_default, character_maximum_length
  FROM information_schema.columns
 WHERE table_schema = 'public'
 ORDER BY table_name, ordinal_position;
"""

Q_FKS = """
SELECT con.conname                    AS name,
       cls.relname                    AS from_table,
       ref.relname                    AS to_table,
       pg_get_constraintdef(con.oid)  AS def
  FROM pg_constraint con
  JOIN pg_class cls ON cls.oid = con.conrelid
  JOIN pg_class ref ON ref.oid = con.confrelid
  JOIN pg_namespace n ON n.oid = cls.relnamespace
 WHERE con.contype = 'f' AND n.nspname = 'public';
"""

Q_INDEXES = """
SELECT i.relname                      AS index_name,
       t.relname                      AS table_name,
       ix.indisunique                 AS is_unique,
       ix.indisprimary                AS is_primary,
       s.idx_scan                     AS scan_count,
       pg_get_indexdef(ix.indexrelid) AS def
  FROM pg_index ix
  JOIN pg_class i ON i.oid = ix.indexrelid
  JOIN pg_class t ON t.oid = ix.indrelid
  JOIN pg_namespace n ON n.oid = t.relnamespace
  LEFT JOIN pg_stat_user_indexes s ON s.indexrelid = ix.indexrelid
 WHERE t.relkind = 'r' AND n.nspname = 'public';
"""

Q_VIEWS = """
SELECT viewname AS name, definition
  FROM pg_views
 WHERE schemaname = 'public';
"""

Q_VIEW_DEPS = """
SELECT DISTINCT
       dep_cls.relname AS view_name,
       ref_cls.relname AS depends_on_table
  FROM pg_depend d
  JOIN pg_rewrite r      ON r.oid = d.objid
  JOIN pg_class  dep_cls ON dep_cls.oid = r.ev_class
  JOIN pg_class  ref_cls ON ref_cls.oid = d.refobjid
  JOIN pg_namespace n    ON n.oid = dep_cls.relnamespace
 WHERE d.classid = 'pg_rewrite'::regclass
   AND dep_cls.relkind = 'v'
   AND ref_cls.relkind IN ('r', 'v')
   AND n.nspname = 'public'
   AND dep_cls.relname <> ref_cls.relname;
"""

Q_FUNCTIONS = """
SELECT p.proname                 AS name,
       pg_get_function_arguments(p.oid) AS args,
       pg_get_function_result(p.oid)    AS returns,
       l.lanname                 AS lang
  FROM pg_proc p
  JOIN pg_namespace n ON n.oid = p.pronamespace
  JOIN pg_language l  ON l.oid = p.prolang
 WHERE n.nspname = 'public';
"""

Q_TRIGGERS = """
SELECT t.tgname                  AS name,
       c.relname                 AS table_name,
       pg_get_triggerdef(t.oid)  AS def
  FROM pg_trigger t
  JOIN pg_class c  ON c.oid = t.tgrelid
  JOIN pg_namespace n ON n.oid = c.relnamespace
 WHERE NOT t.tgisinternal AND n.nspname = 'public';
"""

Q_PG_STAT_STATEMENTS_AVAILABLE = """
SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements';
"""


def run_query(conn, sql: str) -> list[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def scan(conn) -> dict[str, Any]:
    t0 = time.time()
    snap = {
        "tables":     run_query(conn, Q_TABLES),
        "columns":    run_query(conn, Q_COLUMNS),
        "foreign_keys": run_query(conn, Q_FKS),
        "indexes":    run_query(conn, Q_INDEXES),
        "views":      run_query(conn, Q_VIEWS),
        "view_deps":  run_query(conn, Q_VIEW_DEPS),
        "functions":  run_query(conn, Q_FUNCTIONS),
        "triggers":   run_query(conn, Q_TRIGGERS),
        "pg_stat_statements_available": bool(run_query(conn, Q_PG_STAT_STATEMENTS_AVAILABLE)),
        "scan_duration_seconds": None,
    }
    snap["scan_duration_seconds"] = round(time.time() - t0, 2)
    return snap


# ---------------------------------------------------------------------------
# Graphify node/edge synthesis — matches shape of graphify-out/graph.json
# nodes: {id, label, file_type, source_file, source_location, ...}
# links: {relation, confidence, _src, _tgt, weight, source}
# ---------------------------------------------------------------------------

def tbl_id(name: str) -> str:       return f"db:table:{name}"
def col_id(t: str, c: str) -> str:  return f"db:column:{t}.{c}"
def idx_id(name: str) -> str:       return f"db:index:{name}"
def view_id(name: str) -> str:      return f"db:view:{name}"
def fn_id(name: str) -> str:        return f"db:function:{name}"
def trg_id(name: str) -> str:       return f"db:trigger:{name}"


def to_graphify(snap: dict[str, Any]) -> dict[str, list]:
    nodes: list[dict] = []
    links: list[dict] = []

    # tables
    for t in snap["tables"]:
        nodes.append({
            "id":              tbl_id(t["table_name"]),
            "label":           t["table_name"],
            "file_type":       "db_table",
            "source_file":     "postgres://public",
            "source_location": f"public.{t['table_name']}",
            "row_estimate":    t["row_estimate"],
            "size_bytes":      t["size_bytes"],
            "last_autovacuum": str(t["last_autovacuum"]) if t["last_autovacuum"] else None,
            "last_autoanalyze": str(t["last_autoanalyze"]) if t["last_autoanalyze"] else None,
            "n_live_tup":      t["n_live_tup"],
            "n_dead_tup":      t["n_dead_tup"],
            "seq_scan":        t["seq_scan"],
            "idx_scan":        t["idx_scan"],
            "norm_label":      t["table_name"],
        })

    # columns + column_of edges
    for c in snap["columns"]:
        cid = col_id(c["table_name"], c["column_name"])
        nodes.append({
            "id":              cid,
            "label":           f"{c['table_name']}.{c['column_name']}",
            "file_type":       "db_column",
            "source_file":     "postgres://public",
            "source_location": f"public.{c['table_name']}.{c['column_name']}",
            "data_type":       c["data_type"],
            "is_nullable":     c["is_nullable"],
            "column_default":  c["column_default"],
            "ordinal":         c["ordinal_position"],
            "norm_label":      f"{c['table_name']}.{c['column_name']}",
        })
        links.append({
            "relation":   "column_of",
            "confidence": "EXTRACTED",
            "_src":       cid,
            "_tgt":       tbl_id(c["table_name"]),
            "weight":     1.0,
            "source":     "pg_catalog",
        })

    # foreign keys (table -> table)
    for fk in snap["foreign_keys"]:
        links.append({
            "relation":   "fk_to",
            "confidence": "EXTRACTED",
            "_src":       tbl_id(fk["from_table"]),
            "_tgt":       tbl_id(fk["to_table"]),
            "weight":     1.0,
            "name":       fk["name"],
            "def":        fk["def"],
            "source":     "pg_catalog",
        })

    # indexes + index_of edges
    for ix in snap["indexes"]:
        iid = idx_id(ix["index_name"])
        nodes.append({
            "id":              iid,
            "label":           ix["index_name"],
            "file_type":       "db_index",
            "source_file":     "postgres://public",
            "source_location": f"public.{ix['table_name']}",
            "is_unique":       ix["is_unique"],
            "is_primary":      ix["is_primary"],
            "scan_count":      ix["scan_count"],
            "def":             ix["def"],
            "norm_label":      ix["index_name"],
        })
        links.append({
            "relation":   "index_of",
            "confidence": "EXTRACTED",
            "_src":       iid,
            "_tgt":       tbl_id(ix["table_name"]),
            "weight":     1.0,
            "source":     "pg_catalog",
        })

    # views
    for v in snap["views"]:
        nodes.append({
            "id":              view_id(v["name"]),
            "label":           v["name"],
            "file_type":       "db_view",
            "source_file":     "postgres://public",
            "source_location": f"public.{v['name']}",
            "definition":      v["definition"][:1000],  # truncate absurd views
            "norm_label":      v["name"],
        })

    # view dependencies (view -> table|view)
    for vd in snap["view_deps"]:
        # target may be table or view; assume table first, fall back to view
        target_id = tbl_id(vd["depends_on_table"])
        links.append({
            "relation":   "depends_on",
            "confidence": "EXTRACTED",
            "_src":       view_id(vd["view_name"]),
            "_tgt":       target_id,
            "weight":     1.0,
            "source":     "pg_depend",
        })

    # functions
    for f in snap["functions"]:
        nodes.append({
            "id":              fn_id(f["name"]),
            "label":           f["name"],
            "file_type":       "db_function",
            "source_file":     "postgres://public",
            "source_location": f"public.{f['name']}",
            "args":            f["args"],
            "returns":         f["returns"],
            "lang":            f["lang"],
            "norm_label":      f["name"],
        })

    # triggers
    for t in snap["triggers"]:
        tid = trg_id(t["name"])
        nodes.append({
            "id":              tid,
            "label":           t["name"],
            "file_type":       "db_trigger",
            "source_file":     "postgres://public",
            "source_location": f"public.{t['table_name']}",
            "def":             t["def"],
            "norm_label":      t["name"],
        })
        links.append({
            "relation":   "trigger_of",
            "confidence": "EXTRACTED",
            "_src":       tid,
            "_tgt":       tbl_id(t["table_name"]),
            "weight":     1.0,
            "source":     "pg_trigger",
        })

    return {"nodes": nodes, "links": links}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(snap: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    add = lines.append
    add("# PRD-135 Phase 1 — Live DB Scan Report\n")
    add(f"_Generated by `scripts/graphify_db_scan.py` in {snap['scan_duration_seconds']}s._\n")
    add(f"_pg_stat_statements available: **{snap['pg_stat_statements_available']}**_\n")
    add("")
    add("## Summary")
    add(f"- Tables: **{len(snap['tables'])}**")
    add(f"- Columns: **{len(snap['columns'])}**")
    add(f"- Foreign keys: **{len(snap['foreign_keys'])}**")
    add(f"- Indexes: **{len(snap['indexes'])}**")
    add(f"- Views: **{len(snap['views'])}**")
    add(f"- View dependencies: **{len(snap['view_deps'])}**")
    add(f"- SQL functions: **{len(snap['functions'])}**")
    add(f"- Triggers: **{len(snap['triggers'])}**")
    add("")

    # Biggest tables
    add("## Top 25 tables by size")
    add("| Table | Rows (est) | Size | idx_scan | seq_scan |")
    add("|---|---:|---:|---:|---:|")
    biggest = sorted(snap["tables"], key=lambda r: r["size_bytes"] or 0, reverse=True)[:25]
    for t in biggest:
        sz = t["size_bytes"] or 0
        size_str = f"{sz / 1024 / 1024:.1f} MB" if sz > 1024 * 1024 else f"{sz / 1024:.1f} KB"
        add(f"| `{t['table_name']}` | {t['row_estimate']:,} | {size_str} | {t['idx_scan'] or 0:,} | {t['seq_scan'] or 0:,} |")
    add("")

    # Zero-traffic tables (candidate orphans)
    zero_traffic = [
        t for t in snap["tables"]
        if (t["seq_scan"] or 0) == 0
        and (t["idx_scan"] or 0) == 0
        and (t["n_tup_ins"] or 0) == 0
    ]
    add(f"## Zero-traffic tables ({len(zero_traffic)}) — never scanned, never written")
    add("_These are the strongest dead-table candidates. Still require code→DB edges for final confirmation (Phase 2)._\n")
    add("| Table | Rows | Size | last_autovacuum |")
    add("|---|---:|---:|---|")
    for t in sorted(zero_traffic, key=lambda r: r["size_bytes"] or 0, reverse=True):
        sz = t["size_bytes"] or 0
        size_str = f"{sz / 1024 / 1024:.1f} MB" if sz > 1024 * 1024 else f"{sz / 1024:.1f} KB"
        add(f"| `{t['table_name']}` | {t['row_estimate']:,} | {size_str} | {t['last_autovacuum'] or '—'} |")
    add("")

    # Unused indexes
    unused_ix = [i for i in snap["indexes"] if (i["scan_count"] or 0) == 0 and not i["is_primary"]]
    add(f"## Unused non-primary indexes ({len(unused_ix)}) — idx_scan = 0")
    add("| Index | Table | Unique |")
    add("|---|---|---|")
    for ix in sorted(unused_ix, key=lambda r: r["table_name"])[:50]:
        add(f"| `{ix['index_name']}` | `{ix['table_name']}` | {'✓' if ix['is_unique'] else ''} |")
    if len(unused_ix) > 50:
        add(f"| …and {len(unused_ix) - 50} more | | |")
    add("")

    # Tables with no FK in or out (isolated)
    fk_tables = {fk["from_table"] for fk in snap["foreign_keys"]} | {fk["to_table"] for fk in snap["foreign_keys"]}
    isolated = [t for t in snap["tables"] if t["table_name"] not in fk_tables]
    add(f"## Isolated tables ({len(isolated)}) — no FK in or out")
    add("_Either junction tables, append-only logs, or dead schemas. Cross-check with code walker in Phase 2._\n")
    for t in sorted(isolated, key=lambda r: r["table_name"]):
        add(f"- `{t['table_name']}` ({t['row_estimate']:,} rows)")
    add("")

    # Views
    if snap["views"]:
        add("## Views")
        for v in snap["views"]:
            deps = [d["depends_on_table"] for d in snap["view_deps"] if d["view_name"] == v["name"]]
            add(f"- `{v['name']}` → depends on: {', '.join(f'`{d}`' for d in deps) or '—'}")
        add("")

    # Triggers
    if snap["triggers"]:
        add("## Triggers")
        for t in snap["triggers"]:
            add(f"- `{t['name']}` on `{t['table_name']}`")
        add("")

    add("---")
    add("## Next")
    add("- Phase 2: code→DB edge extractor (SQLAlchemy `__tablename__`, `text(\"...\")` literals, `db.query(Model)`).")
    add("- Phase 3: dead-tables, dead-routes, consolidation-candidates reports.")
    add(f"- Phase 4 (pg_stat_statements): **{'SKIPPED — extension not installed' if not snap['pg_stat_statements_available'] else 'available'}**.")

    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), help="Output directory")
    ap.add_argument("--merge-graph", action="store_true",
                    help="Merge db nodes/links into graphify-out/graph.json")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not set", file=sys.stderr)
        return 2

    print(f"Connecting to DB…")
    conn = psycopg2.connect(db_url)
    conn.set_session(readonly=True, autocommit=True)
    try:
        print("Scanning pg_catalog…")
        snap = scan(conn)
    finally:
        conn.close()

    raw_path = out_dir / "db.json"
    raw_path.write_text(json.dumps(snap, indent=2, default=str))
    print(f"  wrote {raw_path}   ({snap['scan_duration_seconds']}s)")

    fragment = to_graphify(snap)
    frag_path = out_dir / "db.graphify.json"
    frag_path.write_text(json.dumps(fragment, indent=2))
    print(f"  wrote {frag_path}   ({len(fragment['nodes'])} nodes, {len(fragment['links'])} links)")

    report_path = out_dir / "DB_SCAN_REPORT.md"
    write_report(snap, report_path)
    print(f"  wrote {report_path}")

    if args.merge_graph:
        graph_path = out_dir / "graph.json"
        if not graph_path.exists():
            print("  --merge-graph: graph.json missing; run /graphify first", file=sys.stderr)
            return 1
        graph = json.loads(graph_path.read_text())
        existing_ids = {n["id"] for n in graph["nodes"]}
        added_n = 0
        for n in fragment["nodes"]:
            if n["id"] not in existing_ids:
                graph["nodes"].append(n)
                added_n += 1
        graph["links"].extend(fragment["links"])
        backup = out_dir / "graph.before-db-merge.json"
        if not backup.exists():
            backup.write_text(graph_path.read_text())
        graph_path.write_text(json.dumps(graph, indent=2))
        print(f"  merged {added_n} new DB nodes + {len(fragment['links'])} DB links into graph.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
