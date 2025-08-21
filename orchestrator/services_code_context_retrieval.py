from __future__ import annotations

from typing import List, Dict, Any
from sqlalchemy.orm import Session
from .models_code_graph import CodeSymbol, CodeEdge


class CodeContextRetrieval:
    def __init__(self, db: Session, project: str):
        self.db = db
        self.project = project

    def find_symbols(self, query_terms: List[str], limit: int = 12) -> List[CodeSymbol]:
        q = self.db.query(CodeSymbol).filter(CodeSymbol.project == self.project)
        for term in query_terms:
            q = q.filter(CodeSymbol.symbol_name.ilike(f"%{term}%"))
        return q.limit(limit).all()

    def expand_with_edges(self, symbols: List[CodeSymbol], max_neighbors: int = 8) -> Dict[str, Any]:
        symbol_ids = [s.id for s in symbols]
        neighbors = self.db.query(CodeEdge).filter(CodeEdge.project == self.project, CodeEdge.src_symbol_id.in_(symbol_ids)).limit(max_neighbors).all()
        return {
            "seeds": symbols,
            "edges": neighbors,
        }

    def to_prompt_block(self, bundle: Dict[str, Any], max_chars: int = 3000) -> str:
        lines: List[str] = ["## CODE\n"]
        for s in bundle.get("seeds", [])[:10]:
            sig = s.signature or s.symbol_name
            lines.append(f"- {s.symbol_type} {sig} @ {s.file_path}:{s.start_line}-{s.end_line}")
        txt = "\n".join(lines)
        return txt[:max_chars]


