#!/usr/bin/env python3
"""
PRD-135 Phase 3 — Three Reports
================================

Joins Phase 1 (db.json) + Phase 2 (code_to_db.graphify.json) + a frontend
route-reference scan to produce the three PRD-135 §5.3 reports:

  1. dead-tables        — no inbound code edge + size/rows/last_autovacuum
  2. dead-routes        — route not referenced in any frontend file, with LOC/method
  3. consolidation-candidates — table pairs ranked by column-Jaccard + FK + shared writers

Also writes a machine-readable ``graphify-out/reports.json`` for any
downstream tooling.

Usage:
  python scripts/graphify_reports.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "graphify-out"
FRONTEND = REPO_ROOT / "frontend"
ORCH_API = REPO_ROOT / "orchestrator" / "api"
ORCH_MAIN = REPO_ROOT / "orchestrator" / "main.py"

# -------- Phase 1 + 2 loaders ----------------------------------------------

def load_db() -> dict:
    p = OUT / "db.json"
    if not p.exists():
        print(f"Missing {p}. Run scripts/graphify_db_scan.py first.", file=sys.stderr)
        sys.exit(2)
    return json.loads(p.read_text())


def load_code_edges() -> list[dict]:
    p = OUT / "code_to_db.graphify.json"
    if not p.exists():
        print(f"Missing {p}. Run scripts/graphify_code_to_db.py first.", file=sys.stderr)
        sys.exit(2)
    return json.loads(p.read_text())["links"]


# -------- Route discovery (FastAPI) ----------------------------------------

ROUTER_DECL_RE = re.compile(
    r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*APIRouter\((.*?)\)", re.M | re.S,
)
ROUTE_DEC_RE = re.compile(
    r"@(\w+)\.(get|post|put|delete|patch|websocket)\(\s*['\"]([^'\"]+)['\"]",
    re.I,
)
INCLUDE_ROUTER_RE = re.compile(
    r"app\.include_router\(\s*([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*,\s*prefix\s*=\s*['\"]([^'\"]+)['\"])?",
)
IMPORT_AS_RE = re.compile(
    r"from\s+(?:orchestrator\.)?api\.([a-zA-Z_][a-zA-Z0-9_.]*)\s+import\s+(.+)",
)


def scan_routes() -> list[dict]:
    # Map alias → (module, prefix_from_import)
    main_src = ORCH_MAIN.read_text(encoding="utf-8", errors="ignore")
    alias_to_module: dict[str, str] = {}
    for m in IMPORT_AS_RE.finditer(main_src):
        module = m.group(1).split(".")[-1]
        clause = m.group(2).split("#")[0].strip()
        for piece in clause.split(","):
            piece = piece.strip().strip("()")
            if not piece:
                continue
            if " as " in piece:
                _, alias = piece.split(" as ", 1)
                alias_to_module[alias.strip()] = module
            else:
                alias_to_module[piece.strip()] = module

    mounted: dict[str, str | None] = {}  # module -> mount prefix (None = no override)
    for m in INCLUDE_ROUTER_RE.finditer(main_src):
        alias = m.group(1)
        prefix_override = m.group(2)
        mod = alias_to_module.get(alias, alias)
        mounted[mod] = prefix_override

    routes: list[dict] = []
    for path in sorted(ORCH_API.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        src = path.read_text(encoding="utf-8", errors="ignore")
        # router prefix (from APIRouter(prefix=...))
        router_prefixes: dict[str, str] = {}
        for m in ROUTER_DECL_RE.finditer(src):
            var = m.group(1)
            args = m.group(2)
            pm = re.search(r"prefix\s*=\s*['\"]([^'\"]+)['\"]", args)
            router_prefixes[var] = pm.group(1) if pm else ""
        rel = str(path.relative_to(REPO_ROOT))
        mod = path.stem
        is_mounted = mod in mounted
        mount_prefix = mounted.get(mod) or ""
        for m in ROUTE_DEC_RE.finditer(src):
            var = m.group(1)
            method = m.group(2).upper()
            route_path = m.group(3)
            prefix = router_prefixes.get(var, "")
            full = f"{mount_prefix}{prefix}{route_path}".replace("//", "/")
            line = src[:m.start()].count("\n") + 1
            routes.append({
                "file":       rel,
                "module":     mod,
                "method":     method,
                "path":       full,
                "path_parts": [p for p in full.split("/") if p and not p.startswith("{")],
                "line":       line,
                "is_mounted": is_mounted,
            })
    return routes


# -------- Frontend reference scan ------------------------------------------

FRONTEND_EXT = {".ts", ".tsx", ".js", ".jsx"}


def scan_frontend_text() -> str:
    """Concatenate all frontend source. Cheap — < 500 files usually."""
    if not FRONTEND.exists():
        return ""
    buf: list[str] = []
    for p in FRONTEND.rglob("*"):
        if p.is_dir() or p.suffix not in FRONTEND_EXT:
            continue
        if "node_modules" in p.parts or ".next" in p.parts:
            continue
        try:
            buf.append(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
    return "\n".join(buf)


def route_referenced(route: dict, frontend_src: str) -> bool:
    """True if any reasonable derivative of the route path appears in frontend."""
    path = route["path"]
    # exact string
    if f'"{path}"' in frontend_src or f"'{path}'" in frontend_src:
        return True
    # template-style: replace {id} with ${...} lookup
    if "{" in path:
        templ = re.sub(r"\{[^}]+\}", r"\\$\\{[^}]+\\}", path)
        if re.search(templ, frontend_src):
            return True
    # fallback: unique tail — e.g. /api/agents/roster → "agents/roster"
    tail_parts = route["path_parts"][-2:]
    if len(tail_parts) == 2:
        tail = "/".join(tail_parts)
        if tail in frontend_src:
            return True
    return False


# -------- Reports -----------------------------------------------------------

def make_dead_tables_report(db: dict, code_edges: list[dict]) -> tuple[list[dict], str]:
    referenced: set[str] = set()
    for e in code_edges:
        if e["_tgt"].startswith("db:table:"):
            referenced.add(e["_tgt"].split(":", 2)[-1])

    fk_tables: set[str] = set()
    for fk in db["foreign_keys"]:
        fk_tables.add(fk["from_table"])
        fk_tables.add(fk["to_table"])

    rows: list[dict] = []
    for t in db["tables"]:
        name = t["table_name"]
        if name in referenced:
            continue
        rows.append({
            "table":            name,
            "row_estimate":     t["row_estimate"],
            "size_bytes":       t["size_bytes"],
            "seq_scan":         t["seq_scan"] or 0,
            "idx_scan":         t["idx_scan"] or 0,
            "n_tup_ins":        t["n_tup_ins"] or 0,
            "n_tup_upd":        t["n_tup_upd"] or 0,
            "n_tup_del":        t["n_tup_del"] or 0,
            "last_autovacuum":  str(t["last_autovacuum"]) if t["last_autovacuum"] else None,
            "has_fk":           name in fk_tables,
            "is_backup":        name.startswith("b_"),
        })
    rows.sort(key=lambda r: (-r["size_bytes"] or 0, r["table"]))

    md = ["# dead-tables — PRD-135 §5.3\n"]
    md.append(f"Tables with **no inbound ``reads`` / ``writes`` / ``models`` code edge**. ")
    md.append(f"Live-DB row/size stats joined. Candidates for DROP pending human review.\n")
    md.append(f"Total: **{len(rows)}** of {len(db['tables'])} tables.\n")
    md.append("| Table | Rows | Size | writes (prod) | last_autovacuum | Has FK | Is backup |")
    md.append("|---|---:|---:|---:|---|---|---|")
    for r in rows:
        sz = r["size_bytes"] or 0
        size_str = f"{sz/1024/1024:.1f} MB" if sz > 1024*1024 else f"{sz/1024:.1f} KB"
        writes_prod = r["n_tup_ins"] + r["n_tup_upd"] + r["n_tup_del"]
        md.append(
            f"| `{r['table']}` | {r['row_estimate']:,} | {size_str} | "
            f"{writes_prod:,} | {r['last_autovacuum'] or '—'} | "
            f"{'✓' if r['has_fk'] else ''} | {'✓' if r['is_backup'] else ''} |"
        )
    return rows, "\n".join(md)


def make_dead_routes_report(routes: list[dict], frontend_src: str) -> tuple[list[dict], str]:
    results: list[dict] = []
    for r in routes:
        ref = route_referenced(r, frontend_src) if frontend_src else None
        results.append({
            **r,
            "frontend_reference": ref,
        })
    dead = [r for r in results if r["is_mounted"] and r["frontend_reference"] is False]
    unmounted = [r for r in results if not r["is_mounted"]]

    md = ["# dead-routes — PRD-135 §5.3\n"]
    md.append(f"- Total routes discovered: **{len(routes)}**")
    md.append(f"- Mounted routes with NO frontend reference: **{len(dead)}**")
    md.append(f"- Routes in unmounted router files: **{len(unmounted)}**\n")

    md.append("## Mounted but never referenced by frontend")
    md.append("_Advisory only — admin-only / script-only routes will appear here._\n")
    md.append("| Method | Path | File | Line |")
    md.append("|---|---|---|---:|")
    dead.sort(key=lambda r: r["path"])
    for r in dead[:300]:
        md.append(f"| {r['method']} | `{r['path']}` | `{r['file']}` | {r['line']} |")
    if len(dead) > 300:
        md.append(f"| … | +{len(dead)-300} more | | |")

    md.append("\n## Routes in unmounted router files")
    md.append("| Method | Path | File |")
    md.append("|---|---|---|")
    unmounted.sort(key=lambda r: r["file"])
    for r in unmounted:
        md.append(f"| {r['method']} | `{r['path']}` | `{r['file']}` |")
    return results, "\n".join(md)


def make_consolidation_report(db: dict, code_edges: list[dict],
                              min_jaccard: float = 0.5,
                              min_overlap_cols: int = 3) -> tuple[list[dict], str]:
    # Column sets by table
    cols: dict[str, set[str]] = defaultdict(set)
    for c in db["columns"]:
        cols[c["table_name"]].add(c["column_name"])

    # Writers by table (function that writes to it)
    writers: dict[str, set[str]] = defaultdict(set)
    for e in code_edges:
        if e["relation"] != "writes":
            continue
        tbl = e["_tgt"].split(":", 2)[-1]
        writers[tbl].add(e["_src"])

    # FK pairs
    fks: set[tuple[str, str]] = set()
    for fk in db["foreign_keys"]:
        a, b = fk["from_table"], fk["to_table"]
        fks.add((a, b))
        fks.add((b, a))

    # Exclude backup tables and alembic_version from candidate pool
    candidates = [
        t for t in cols
        if not t.startswith("b_") and t not in {"alembic_version"}
    ]

    pairs: list[dict] = []
    for i, a in enumerate(candidates):
        for b in candidates[i+1:]:
            ca, cb = cols[a], cols[b]
            if not ca or not cb:
                continue
            inter = ca & cb
            if len(inter) < min_overlap_cols:
                continue
            union = ca | cb
            j = len(inter) / len(union)
            if j < min_jaccard:
                continue
            shared_writers = writers[a] & writers[b]
            fk_link = (a, b) in fks
            score = j * 10 + len(shared_writers) * 2 + (5 if fk_link else 0)
            pairs.append({
                "table_a":        a,
                "table_b":        b,
                "jaccard":        round(j, 3),
                "shared_cols":    sorted(inter),
                "unique_to_a":    sorted(ca - cb),
                "unique_to_b":    sorted(cb - ca),
                "shared_writers": len(shared_writers),
                "fk_linked":      fk_link,
                "score":          round(score, 2),
            })

    pairs.sort(key=lambda r: -r["score"])

    md = ["# consolidation-candidates — PRD-135 §5.3\n"]
    md.append(f"Pairs of tables ranked by **column-Jaccard ≥ {min_jaccard}** + shared writers + FK link.\n")
    md.append(f"Total candidate pairs: **{len(pairs)}**\n")
    md.append("| Rank | Table A | Table B | Jaccard | Shared cols | Shared writers | FK |")
    md.append("|---:|---|---|---:|---:|---:|:---:|")
    for i, p in enumerate(pairs[:50], 1):
        md.append(
            f"| {i} | `{p['table_a']}` | `{p['table_b']}` | {p['jaccard']} | "
            f"{len(p['shared_cols'])} | {p['shared_writers']} | "
            f"{'✓' if p['fk_linked'] else ''} |"
        )
    if pairs:
        md.append("\n## Detail — top 10")
        for p in pairs[:10]:
            md.append(f"\n### `{p['table_a']}`  ⇆  `{p['table_b']}`  (score {p['score']})")
            md.append(f"- Jaccard: **{p['jaccard']}** · Shared writers: **{p['shared_writers']}** · FK-linked: **{'yes' if p['fk_linked'] else 'no'}**")
            md.append(f"- Shared columns ({len(p['shared_cols'])}): {', '.join(f'`{c}`' for c in p['shared_cols'][:20])}" + (" …" if len(p['shared_cols']) > 20 else ""))
            md.append(f"- Only in A ({len(p['unique_to_a'])}): {', '.join(f'`{c}`' for c in p['unique_to_a'][:10])}" + (" …" if len(p['unique_to_a']) > 10 else ""))
            md.append(f"- Only in B ({len(p['unique_to_b'])}): {', '.join(f'`{c}`' for c in p['unique_to_b'][:10])}" + (" …" if len(p['unique_to_b']) > 10 else ""))
    return pairs, "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_dir = Path(args.out)

    print("Loading Phase 1 DB snapshot…")
    db = load_db()
    print(f"  {len(db['tables'])} tables, {len(db['columns'])} columns, {len(db['foreign_keys'])} FKs")

    print("Loading Phase 2 code→DB edges…")
    code_edges = load_code_edges()
    print(f"  {len(code_edges)} edges")

    print("Scanning routes…")
    routes = scan_routes()
    print(f"  {len(routes)} route handlers")

    print("Scanning frontend for route references…")
    frontend_src = scan_frontend_text()
    frontend_size = len(frontend_src)
    print(f"  {frontend_size:,} chars of frontend source loaded")

    print("Generating report 1 — dead-tables…")
    dead_tables, md1 = make_dead_tables_report(db, code_edges)
    (out_dir / "REPORT_dead_tables.md").write_text(md1)

    print("Generating report 2 — dead-routes…")
    dead_routes, md2 = make_dead_routes_report(routes, frontend_src)
    (out_dir / "REPORT_dead_routes.md").write_text(md2)

    print("Generating report 3 — consolidation-candidates…")
    pairs, md3 = make_consolidation_report(db, code_edges)
    (out_dir / "REPORT_consolidation_candidates.md").write_text(md3)

    (out_dir / "reports.json").write_text(json.dumps({
        "dead_tables":             dead_tables,
        "dead_routes":             dead_routes,
        "consolidation_candidates": pairs,
    }, indent=2, default=str))

    dead_route_count = sum(
        1 for r in dead_routes
        if r["is_mounted"] and r["frontend_reference"] is False
    )
    print("\nSummary")
    print(f"  dead-tables:               {len(dead_tables)}")
    print(f"  dead-routes (mounted):     {dead_route_count}")
    print(f"  consolidation-candidates:  {len(pairs)}")
    print(f"\nReports in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
