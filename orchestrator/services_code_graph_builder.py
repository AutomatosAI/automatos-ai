from __future__ import annotations

import ast
import os
from typing import Optional, Tuple, Dict
from sqlalchemy.orm import Session
from .models_code_graph import CodeSymbol, CodeEdge


class CodeGraphBuilder:
    def __init__(self, db: Session, project: str):
        self.db = db
        self.project = project

    def _upsert_symbol(self, file_path: str, name: str, sym_type: str, start: Optional[int], end: Optional[int], signature: Optional[str], doc: Optional[str]) -> str:
        row = CodeSymbol(
            project=self.project,
            file_path=file_path,
            symbol_name=name,
            symbol_type=sym_type,
            signature=signature,
            docstring=doc,
            start_line=start,
            end_line=end,
        )
        self.db.add(row)
        self.db.flush()
        return row.id

    def _add_edge(self, src_id: str, dst_id: str, edge_type: str) -> None:
        edge = CodeEdge(project=self.project, src_symbol_id=src_id, dst_symbol_id=dst_id, edge_type=edge_type)
        self.db.add(edge)

    def index_repo(self, root_dir: str, file_glob: Tuple[str, ...] = (".py",)) -> Dict[str, int]:
        count_symbols = 0
        count_edges = 0
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if not filename.endswith(file_glob):
                    continue
                path = os.path.join(dirpath, filename)
                s, e = self._index_python_file(path)
                count_symbols += s
                count_edges += e
        self.db.commit()
        return {"symbols": count_symbols, "edges": count_edges}

    def _index_python_file(self, path: str) -> Tuple[int, int]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception:
            return 0, 0
        try:
            tree = ast.parse(source)
        except Exception:
            return 0, 0

        # Map function defs to ids
        func_to_id: Dict[ast.AST, str] = {}
        symbols = 0
        edges = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                sid = self._upsert_symbol(path, name, "function", start, end, signature=None, doc=ast.get_docstring(node))
                func_to_id[node] = sid
                symbols += 1
            elif isinstance(node, ast.ClassDef):
                name = node.name
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                self._upsert_symbol(path, name, "class", start, end, signature=None, doc=ast.get_docstring(node))
                symbols += 1

        # Naive call edges: function calls inside each function body
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                src_id = func_to_id.get(node)
                if not src_id:
                    continue
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        # Try to capture direct Name() calls
                        callee = getattr(inner.func, "id", None) or getattr(getattr(inner.func, "attr", None), "id", None)
                        if not callee:
                            continue
                        # Best-effort: find symbol id with same name
                        dst_row = self.db.query(CodeSymbol).filter(CodeSymbol.project == self.project, CodeSymbol.symbol_name == callee).first()
                        if dst_row:
                            self._add_edge(src_id, dst_row.id, edge_type="calls")
                            edges += 1

        return symbols, edges


