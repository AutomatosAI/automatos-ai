#!/usr/bin/env python3
"""
PRD-135 Phase 2 — Code-to-DB Edge Extractor
============================================

Walks the orchestrator codebase with Python's ``ast`` module and emits
``function → table`` edges labelled ``reads`` / ``writes`` / ``models``.

Per PRD-135 §5.2 Pass 2:
  - SQLAlchemy class with ``__tablename__ = "..."`` → ``models`` edge.
  - Raw SQL via ``text("...")``                     → ``reads`` / ``writes`` edges
    (sqlglot extracts FROM / JOIN / INSERT / UPDATE / DELETE targets).
  - ``db.query(Model)`` / ``session.query(Model)``  → ``reads`` edge via model map.
  - ``db.add(Model(...))`` / ``session.add(...)``    → ``writes`` edge.
  - ``Model.__table__`` references                   → ``references`` edge.

Each edge carries a ``confidence`` tier:
  - ``ast_resolved``  — class reference resolved through an imported name.
  - ``static_match``  — sqlglot or regex successfully parsed literal SQL.
  - ``ambiguous``     — matched a table name but call context uncertain.

Emits ``graphify-out/code_to_db.graphify.json`` in graphify shape.
Optional ``--merge-graph`` folds it into ``graph.json``.

Does not require a DB connection. Reads ``graphify-out/db.json`` for the
authoritative table list (Phase 1 output).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    import sqlglot
    from sqlglot import exp
    HAVE_SQLGLOT = True
except ImportError:
    HAVE_SQLGLOT = False

REPO_ROOT = Path(__file__).resolve().parent.parent
ORCH = REPO_ROOT / "orchestrator"
OUT = REPO_ROOT / "graphify-out"
DB_SNAPSHOT = OUT / "db.json"


def tbl_id(name: str) -> str: return f"db:table:{name}"
def fn_id(module: str, name: str) -> str: return f"code:fn:{module}:{name}"
def cls_id(module: str, name: str) -> str: return f"code:cls:{module}:{name}"
def mod_id(module: str) -> str: return f"code:mod:{module}"


@dataclass
class ModelMap:
    """class name (fully-qualified or not) → __tablename__ value."""
    by_class: dict[str, str] = field(default_factory=dict)   # "AgentReport" -> "agent_reports"
    by_module_class: dict[tuple[str, str], str] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    tgt: str
    relation: str
    confidence: str
    source_file: str
    source_location: str
    sql_snippet: str | None = None


@dataclass
class Node:
    id: str
    label: str
    file_type: str
    source_file: str
    source_location: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "file_type": self.file_type,
            "source_file": self.source_file,
            "source_location": self.source_location,
            "norm_label": self.label,
        }


def iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts or "/tests/" in str(p) or p.name.startswith("test_"):
            continue
        yield p


def module_name(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    return ".".join(rel.parts)


# ---------------------------------------------------------------------------
# Pass A — find all SQLAlchemy models so we can resolve class → table.
# ---------------------------------------------------------------------------

def find_models(paths: list[Path]) -> tuple[ModelMap, list[Node]]:
    mm = ModelMap()
    nodes: list[Node] = []
    for p in paths:
        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src, filename=str(p))
        except (SyntaxError, UnicodeDecodeError):
            continue
        mod = module_name(p)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            tablename: str | None = None
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for t in item.targets:
                        if isinstance(t, ast.Name) and t.id == "__tablename__":
                            if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                                tablename = item.value.value
                                break
                if tablename:
                    break
            if tablename:
                mm.by_class[node.name] = tablename
                mm.by_module_class[(mod, node.name)] = tablename
                nodes.append(Node(
                    id=cls_id(mod, node.name),
                    label=node.name,
                    file_type="code_sqlalchemy_model",
                    source_file=str(p.relative_to(REPO_ROOT)),
                    source_location=f"L{node.lineno}",
                ))
    return mm, nodes


# ---------------------------------------------------------------------------
# Pass B — walk code and emit edges.
# ---------------------------------------------------------------------------

SQL_TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN|INTO|UPDATE|DELETE\s+FROM)\s+[`\"']?([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)

WRITE_KEYWORDS = ("INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "REPLACE")


def sql_tables(sql: str) -> tuple[set[str], set[str]]:
    """Return (reads, writes) table-name sets. Uses sqlglot when available,
    falls back to regex + keyword heuristic."""
    reads: set[str] = set()
    writes: set[str] = set()
    if not sql.strip():
        return reads, writes

    if HAVE_SQLGLOT:
        try:
            trees = sqlglot.parse(sql, error_level=None)
        except Exception:
            trees = []
        for tree in trees:
            if tree is None:
                continue
            # Writes: INSERT, UPDATE, DELETE, MERGE → their target tables
            if isinstance(tree, (exp.Insert, exp.Update, exp.Delete, exp.Merge)):
                this = tree.this
                if isinstance(this, exp.Schema):
                    this = this.this
                if isinstance(this, exp.Table):
                    writes.add(this.name)
            # Reads: any remaining Table ref
            for t in tree.find_all(exp.Table):
                nm = t.name
                if not nm:
                    continue
                if nm in writes:
                    continue
                reads.add(nm)
            return reads, writes
        # sqlglot parsed nothing — fall through to regex

    # Regex fallback
    upper = sql.upper()
    is_write = any(kw in upper for kw in WRITE_KEYWORDS)
    for m in SQL_TABLE_RE.finditer(sql):
        nm = m.group(1)
        # Route INTO/UPDATE/DELETE targets to writes, everything else reads.
        preceding = sql[:m.start()].upper()
        if preceding.rstrip().endswith(("INTO", "UPDATE")) or "DELETE FROM" in preceding[-30:]:
            writes.add(nm)
        else:
            (writes if is_write else reads).add(nm)
    return reads, writes


class EdgeCollector(ast.NodeVisitor):
    """Collects function → table edges by walking one module."""

    def __init__(self, mod: str, path: Path, known_tables: set[str], models: ModelMap):
        self.mod = mod
        self.path = path
        self.known_tables = known_tables
        self.models = models
        self.edges: list[Edge] = []
        self.nodes: list[Node] = []
        self.func_stack: list[str] = []
        self.class_stack: list[str] = []
        # Imported names: "AgentReport" -> source module string (may be empty)
        self.imports: dict[str, str] = {}

    # -- scope tracking ----------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._enter_fn(node)
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._enter_fn(node)
    def _enter_fn(self, node):
        self.func_stack.append(node.name)
        self.nodes.append(Node(
            id=fn_id(self.mod, self._qualname()),
            label=node.name,
            file_type="code_function",
            source_file=str(self.path.relative_to(REPO_ROOT)),
            source_location=f"L{node.lineno}",
        ))
        self.generic_visit(node)
        self.func_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def _qualname(self) -> str:
        parts = self.class_stack + self.func_stack
        return ".".join(parts) if parts else "<module>"

    def _src_id(self) -> str:
        if self.func_stack:
            return fn_id(self.mod, self._qualname())
        return mod_id(self.mod)

    # -- imports -----------------------------------------------------------
    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = mod
        self.generic_visit(node)

    # -- the interesting bit ----------------------------------------------
    def visit_Call(self, node: ast.Call):
        self._inspect_text_call(node)
        self._inspect_query_call(node)
        self._inspect_add_call(node)
        self.generic_visit(node)

    def _add_edge(self, tables: set[str], relation: str, confidence: str,
                  lineno: int, sql: str | None = None):
        src = self._src_id()
        for t in tables:
            if t not in self.known_tables:
                continue
            self.edges.append(Edge(
                src=src, tgt=tbl_id(t),
                relation=relation, confidence=confidence,
                source_file=str(self.path.relative_to(REPO_ROOT)),
                source_location=f"L{lineno}",
                sql_snippet=(sql or "")[:200] if sql else None,
            ))

    def _inspect_text_call(self, node: ast.Call):
        """Catch text("SELECT ... FROM foo") — direct and wrapped forms."""
        callee = node.func
        is_text_call = False
        if isinstance(callee, ast.Name) and callee.id == "text":
            is_text_call = True
        elif isinstance(callee, ast.Attribute) and callee.attr == "text":
            is_text_call = True
        if not is_text_call:
            return
        if not node.args:
            return
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            reads, writes = sql_tables(arg.value)
            self._add_edge(reads, "reads", "static_match", node.lineno, arg.value)
            self._add_edge(writes, "writes", "static_match", node.lineno, arg.value)
        elif isinstance(arg, ast.JoinedStr):
            # f-string — concat the constant parts, mark ambiguous.
            parts = [v.value for v in arg.values
                     if isinstance(v, ast.Constant) and isinstance(v.value, str)]
            reads, writes = sql_tables("".join(parts))
            self._add_edge(reads, "reads", "ambiguous", node.lineno, "".join(parts))
            self._add_edge(writes, "writes", "ambiguous", node.lineno, "".join(parts))

    def _inspect_query_call(self, node: ast.Call):
        """db.query(Model) / session.query(Model) → reads on Model's table."""
        callee = node.func
        if not isinstance(callee, ast.Attribute) or callee.attr != "query":
            return
        for arg in node.args:
            name = _resolve_name(arg)
            if name and name in self.models.by_class:
                self._add_edge({self.models.by_class[name]}, "reads",
                               "ast_resolved", node.lineno)

    def _inspect_add_call(self, node: ast.Call):
        """db.add(Model(...)) → writes; db.add_all([Model(...)]) → writes."""
        callee = node.func
        if not isinstance(callee, ast.Attribute):
            return
        if callee.attr not in ("add", "add_all", "merge", "delete"):
            return
        rel = "writes"
        for arg in node.args:
            # direct call Model(...)
            if isinstance(arg, ast.Call):
                name = _resolve_name(arg.func)
                if name and name in self.models.by_class:
                    self._add_edge({self.models.by_class[name]}, rel,
                                   "ast_resolved", node.lineno)
            # list of calls
            elif isinstance(arg, (ast.List, ast.Tuple)):
                for item in arg.elts:
                    if isinstance(item, ast.Call):
                        name = _resolve_name(item.func)
                        if name and name in self.models.by_class:
                            self._add_edge({self.models.by_class[name]}, rel,
                                           "ast_resolved", node.lineno)


def _resolve_name(node: ast.expr) -> str | None:
    """Best-effort resolve an AST expression to a simple name string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-graph", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    if not DB_SNAPSHOT.exists():
        print(f"Missing {DB_SNAPSHOT}. Run scripts/graphify_db_scan.py first.", file=sys.stderr)
        return 2

    db = json.loads(DB_SNAPSHOT.read_text())
    known_tables: set[str] = {t["table_name"] for t in db["tables"]}
    print(f"Loaded {len(known_tables)} tables from {DB_SNAPSHOT.name}")

    paths = list(iter_py_files(ORCH))
    print(f"Scanning {len(paths)} Python files…")

    # Pass A — models
    models, model_nodes = find_models(paths)
    print(f"  Resolved {len(models.by_class)} SQLAlchemy models (class → __tablename__)")

    # Pre-emit model edges
    edges: list[Edge] = []
    for (mod, cls), tbl in models.by_module_class.items():
        if tbl in known_tables:
            edges.append(Edge(
                src=cls_id(mod, cls),
                tgt=tbl_id(tbl),
                relation="models",
                confidence="ast_resolved",
                source_file="",   # already on the class node
                source_location="",
            ))

    # Pass B — per-file code walk
    all_nodes: list[Node] = list(model_nodes)
    for p in paths:
        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src, filename=str(p))
        except (SyntaxError, UnicodeDecodeError):
            continue
        mod = module_name(p)
        collector = EdgeCollector(mod, p, known_tables, models)
        collector.visit(tree)
        edges.extend(collector.edges)
        all_nodes.extend(collector.nodes)

    # Deduplicate nodes by id
    seen: set[str] = set()
    unique_nodes: list[dict] = []
    for n in all_nodes:
        if n.id in seen:
            continue
        seen.add(n.id)
        unique_nodes.append(n.as_dict())

    # Edges → graphify link shape
    link_payload = []
    for e in edges:
        link_payload.append({
            "relation":   e.relation,
            "confidence": e.confidence,
            "_src":       e.src,
            "_tgt":       e.tgt,
            "weight":     1.0,
            "source":     "code_walker_v1",
            "source_file": e.source_file,
            "source_location": e.source_location,
            **({"sql_snippet": e.sql_snippet} if e.sql_snippet else {}),
        })

    # Stats
    by_rel: dict[str, int] = {}
    by_conf: dict[str, int] = {}
    per_table: dict[str, dict[str, int]] = {}
    for e in edges:
        by_rel[e.relation] = by_rel.get(e.relation, 0) + 1
        by_conf[e.confidence] = by_conf.get(e.confidence, 0) + 1
        tbl = e.tgt.split(":", 2)[-1]
        per_table.setdefault(tbl, {"reads": 0, "writes": 0, "models": 0})
        per_table[tbl][e.relation] = per_table[tbl].get(e.relation, 0) + 1

    # Tables in DB with zero inbound code edges — dead-table candidates
    inbound: set[str] = {e.tgt.split(":", 2)[-1] for e in edges}
    zero_code = sorted(t for t in known_tables if t not in inbound)

    # Write outputs
    frag = {"nodes": unique_nodes, "links": link_payload}
    frag_path = out_dir / "code_to_db.graphify.json"
    frag_path.write_text(json.dumps(frag, indent=2))
    print(f"  wrote {frag_path}  ({len(unique_nodes)} nodes, {len(link_payload)} links)")

    report_path = out_dir / "CODE_TO_DB_REPORT.md"
    _write_report(report_path, by_rel, by_conf, per_table, zero_code, known_tables, len(edges))
    print(f"  wrote {report_path}")

    if args.merge_graph:
        graph_path = out_dir / "graph.json"
        graph = json.loads(graph_path.read_text())
        existing_ids = {n["id"] for n in graph["nodes"]}
        added_n = 0
        for n in unique_nodes:
            if n["id"] not in existing_ids:
                graph["nodes"].append(n)
                added_n += 1
        graph["links"].extend(link_payload)
        graph_path.write_text(json.dumps(graph, indent=2))
        print(f"  merged {added_n} new code nodes + {len(link_payload)} code→DB links into graph.json")

    print(f"\nSummary:")
    print(f"  total edges: {len(edges)}")
    print(f"  by relation: {by_rel}")
    print(f"  by confidence: {by_conf}")
    print(f"  tables with zero code edges: {len(zero_code)} / {len(known_tables)}")
    return 0


def _write_report(path, by_rel, by_conf, per_table, zero_code, known_tables, total_edges):
    lines: list[str] = []
    a = lines.append
    a("# PRD-135 Phase 2 — Code ↔ DB Edge Report\n")
    a("## Summary")
    a(f"- Total edges: **{total_edges}**")
    a(f"- By relation: {by_rel}")
    a(f"- By confidence: {by_conf}")
    a(f"- Tables with ≥1 inbound code edge: **{len(known_tables) - len(zero_code)}** / {len(known_tables)}")
    a(f"- **Tables with ZERO inbound code edges: {len(zero_code)}** — strongest dead-table candidates")
    a("")

    a("## Top 30 most-referenced tables")
    ranked = sorted(
        per_table.items(),
        key=lambda kv: sum(kv[1].values()),
        reverse=True,
    )[:30]
    a("| Table | reads | writes | models |")
    a("|---|---:|---:|---:|")
    for tbl, counts in ranked:
        a(f"| `{tbl}` | {counts.get('reads', 0)} | {counts.get('writes', 0)} | {counts.get('models', 0)} |")
    a("")

    a(f"## Tables with zero inbound code edges ({len(zero_code)})")
    a("Live in the DB, but no ``__tablename__``, no ``db.query(Model)``, no ``text(\"…FROM foo\")`` referencing them.\n")
    a("These are the strongest DROP candidates after the usual human review.\n")
    a("| Table |")
    a("|---|")
    for t in zero_code:
        a(f"| `{t}` |")
    a("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
