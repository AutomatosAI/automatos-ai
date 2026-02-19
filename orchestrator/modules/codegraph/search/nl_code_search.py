"""
Natural language code search — translates questions to graph queries
using LLM classification, executes them, and returns structured answers.

Inspired by Code-Graph-RAG's approach: Parse question → classify query type →
execute graph query → synthesize answer.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class NLCodeSearch:
    """Natural language interface to the code graph."""

    # Keyword patterns for query classification
    CALL_PATTERNS = [
        r'what (?:functions?|methods?) call',
        r'who calls',
        r'what calls',
        r'what does .+ call',
        r'callers? of',
        r'callees? of',
        r'invocations? of',
    ]

    DEPENDENCY_PATTERNS = [
        r'depends? on',
        r'dependencies? of',
        r'dependents? of',
        r'imports?',
        r'what uses',
        r'who uses',
        r'used by',
        r'requires?',
    ]

    INHERITANCE_PATTERNS = [
        r'extends',
        r'inherits?',
        r'subclass',
        r'superclass',
        r'base class',
        r'implements?',
        r'interface',
        r'children of',
        r'parent of',
    ]

    def __init__(self, db: Session):
        self.db = db

    async def query(
        self,
        question: str,
        project_id: int,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Answer a natural language question about code structure.

        Examples:
        - "What functions call the authenticate method?"
        - "Show me all classes that implement BaseHandler"
        - "What are the dependencies of the user module?"
        """
        # Verify project exists
        project = self.db.execute(
            text("SELECT id, name FROM codegraph_projects WHERE id = :pid AND workspace_id = :wid"),
            {"pid": project_id, "wid": workspace_id}
        ).fetchone()

        if not project:
            raise ValueError(f"Project {project_id} not found")

        # Step 1: Classify query type
        query_type = self._classify_query(question)

        # Step 2: Extract target symbol from question
        target_symbol = self._extract_symbol(question)

        # Step 3: Route to appropriate query handler
        if query_type == 'call_graph':
            results = await self._query_call_graph(target_symbol, project_id, workspace_id, question)
        elif query_type == 'dependency':
            results = await self._query_dependencies(target_symbol, project_id, workspace_id, question)
        elif query_type == 'inheritance':
            results = await self._query_inheritance(target_symbol, project_id, workspace_id, question)
        else:
            results = await self._semantic_search(question, project_id, workspace_id)

        # Step 4: Generate answer
        answer = self._generate_answer(question, query_type, target_symbol, results)

        return {
            "question": question,
            "answer": answer,
            "results": results,
            "query_type": query_type,
            "target_symbol": target_symbol,
            "project": project.name,
        }

    def _classify_query(self, question: str) -> str:
        """Classify question into: call_graph, dependency, inheritance, or search."""
        q = question.lower()

        for pattern in self.CALL_PATTERNS:
            if re.search(pattern, q):
                return 'call_graph'

        for pattern in self.DEPENDENCY_PATTERNS:
            if re.search(pattern, q):
                return 'dependency'

        for pattern in self.INHERITANCE_PATTERNS:
            if re.search(pattern, q):
                return 'inheritance'

        return 'search'

    def _extract_symbol(self, question: str) -> Optional[str]:
        """Extract the target symbol name from a question."""
        q = question.strip()

        # Common patterns: "What calls X?", "dependencies of X", etc.
        patterns = [
            r'(?:calls?|invokes?)\s+(?:the\s+)?[`"\']?(\w+)[`"\']?',
            r'(?:dependenc(?:y|ies)\s+of|depends?\s+on)\s+(?:the\s+)?[`"\']?(\w+)[`"\']?',
            r'(?:dependents?\s+of|what uses)\s+(?:the\s+)?[`"\']?(\w+)[`"\']?',
            r'(?:extends?|inherits?\s+from)\s+(?:the\s+)?[`"\']?(\w+)[`"\']?',
            r'(?:implements?)\s+(?:the\s+)?[`"\']?(\w+)[`"\']?',
            r'(?:callers?\s+of)\s+(?:the\s+)?[`"\']?(\w+)[`"\']?',
            r'what does\s+(?:the\s+)?[`"\']?(\w+)[`"\']?\s+call',
            r'(?:about|for|of)\s+(?:the\s+)?[`"\']?(\w+)[`"\']?\s*(?:\?|$|method|function|class)',
        ]

        for pattern in patterns:
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                return match.group(1)

        # Fallback: find PascalCase or snake_case words that look like symbol names
        symbol_matches = re.findall(r'\b([A-Z][a-zA-Z0-9]+(?:[A-Z][a-z][a-zA-Z0-9]*)*)\b', q)
        if symbol_matches:
            return symbol_matches[0]

        snake_matches = re.findall(r'\b([a-z][a-z0-9]*_[a-z0-9_]+)\b', q)
        if snake_matches:
            return snake_matches[0]

        return None

    async def _query_call_graph(
        self, symbol: Optional[str], project_id: int,
        workspace_id: Optional[str], question: str,
    ) -> List[Dict[str, Any]]:
        """Find callers and callees of a symbol."""
        if not symbol:
            return await self._semantic_search(question, project_id, workspace_id)

        results = []

        # Find the target symbol
        target_rows = self.db.execute(
            text("""
                SELECT id, name, qualified_name, file_path, symbol_type
                FROM codegraph_symbols
                WHERE project_id = :pid AND workspace_id = :wid
                AND (name = :sym OR qualified_name LIKE :sym_like)
            """),
            {"pid": project_id, "wid": workspace_id, "sym": symbol, "sym_like": f"%::{symbol}"}
        ).fetchall()

        if not target_rows:
            return [{"message": f"Symbol '{symbol}' not found in project"}]

        for target in target_rows:
            # Callers (incoming)
            callers = self.db.execute(
                text("""
                    SELECT s.name, s.qualified_name, s.file_path, s.symbol_type
                    FROM codegraph_relationships r
                    JOIN codegraph_symbols s ON s.id = r.from_symbol_id
                    WHERE r.to_symbol_id = :tid AND r.relationship_type = 'calls'
                """),
                {"tid": target.id}
            ).fetchall()

            # Callees (outgoing)
            callees = self.db.execute(
                text("""
                    SELECT s.name, s.qualified_name, s.file_path, s.symbol_type
                    FROM codegraph_relationships r
                    JOIN codegraph_symbols s ON s.id = r.to_symbol_id
                    WHERE r.from_symbol_id = :tid AND r.relationship_type = 'calls'
                """),
                {"tid": target.id}
            ).fetchall()

            results.append({
                "symbol": target.name,
                "qualified_name": target.qualified_name,
                "file": target.file_path,
                "type": target.symbol_type,
                "callers": [
                    {"name": c.name, "file": c.file_path, "type": c.symbol_type}
                    for c in callers
                ],
                "callees": [
                    {"name": c.name, "file": c.file_path, "type": c.symbol_type}
                    for c in callees
                ],
            })

        return results

    async def _query_dependencies(
        self, symbol: Optional[str], project_id: int,
        workspace_id: Optional[str], question: str,
    ) -> List[Dict[str, Any]]:
        """Find imports and dependencies of a symbol or file."""
        if not symbol:
            return await self._semantic_search(question, project_id, workspace_id)

        # Check if the query is about what uses the symbol or what the symbol uses
        q = question.lower()
        is_reverse = any(p in q for p in ['what uses', 'who uses', 'used by', 'dependents of'])

        results = []
        target_rows = self.db.execute(
            text("""
                SELECT id, name, qualified_name, file_path, symbol_type
                FROM codegraph_symbols
                WHERE project_id = :pid AND workspace_id = :wid
                AND (name = :sym OR qualified_name LIKE :sym_like)
            """),
            {"pid": project_id, "wid": workspace_id, "sym": symbol, "sym_like": f"%::{symbol}"}
        ).fetchall()

        if not target_rows:
            return [{"message": f"Symbol '{symbol}' not found in project"}]

        for target in target_rows:
            if is_reverse:
                deps = self.db.execute(
                    text("""
                        SELECT s.name, s.qualified_name, s.file_path, r.relationship_type
                        FROM codegraph_relationships r
                        JOIN codegraph_symbols s ON s.id = r.from_symbol_id
                        WHERE r.to_symbol_id = :tid AND r.relationship_type != 'external_reference'
                    """),
                    {"tid": target.id}
                ).fetchall()
            else:
                deps = self.db.execute(
                    text("""
                        SELECT s.name, s.qualified_name, s.file_path, r.relationship_type
                        FROM codegraph_relationships r
                        JOIN codegraph_symbols s ON s.id = r.to_symbol_id
                        WHERE r.from_symbol_id = :tid AND r.relationship_type != 'external_reference'
                    """),
                    {"tid": target.id}
                ).fetchall()

            results.append({
                "symbol": target.name,
                "file": target.file_path,
                "direction": "dependents" if is_reverse else "dependencies",
                "items": [
                    {"name": d.name, "file": d.file_path, "relationship": d.relationship_type}
                    for d in deps
                ],
            })

        return results

    async def _query_inheritance(
        self, symbol: Optional[str], project_id: int,
        workspace_id: Optional[str], question: str,
    ) -> List[Dict[str, Any]]:
        """Find inheritance relationships."""
        if not symbol:
            return await self._semantic_search(question, project_id, workspace_id)

        results = []
        target_rows = self.db.execute(
            text("""
                SELECT id, name, qualified_name, file_path, symbol_type
                FROM codegraph_symbols
                WHERE project_id = :pid AND workspace_id = :wid
                AND (name = :sym OR qualified_name LIKE :sym_like)
            """),
            {"pid": project_id, "wid": workspace_id, "sym": symbol, "sym_like": f"%::{symbol}"}
        ).fetchall()

        if not target_rows:
            return [{"message": f"Symbol '{symbol}' not found in project"}]

        for target in target_rows:
            # Subclasses (who extends this)
            subclasses = self.db.execute(
                text("""
                    SELECT s.name, s.qualified_name, s.file_path
                    FROM codegraph_relationships r
                    JOIN codegraph_symbols s ON s.id = r.from_symbol_id
                    WHERE r.to_symbol_id = :tid
                    AND r.relationship_type IN ('extends', 'implements')
                """),
                {"tid": target.id}
            ).fetchall()

            # Superclasses (what this extends)
            superclasses = self.db.execute(
                text("""
                    SELECT s.name, s.qualified_name, s.file_path
                    FROM codegraph_relationships r
                    JOIN codegraph_symbols s ON s.id = r.to_symbol_id
                    WHERE r.from_symbol_id = :tid
                    AND r.relationship_type IN ('extends', 'implements')
                """),
                {"tid": target.id}
            ).fetchall()

            results.append({
                "symbol": target.name,
                "file": target.file_path,
                "subclasses": [{"name": s.name, "file": s.file_path} for s in subclasses],
                "superclasses": [{"name": s.name, "file": s.file_path} for s in superclasses],
            })

        return results

    async def _semantic_search(
        self, question: str, project_id: int, workspace_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Fall back to semantic/fuzzy symbol search."""
        # Simple name-based search using ILIKE
        rows = self.db.execute(
            text("""
                SELECT name, qualified_name, file_path, symbol_type, signature, docstring
                FROM codegraph_symbols
                WHERE project_id = :pid AND workspace_id = :wid
                AND (name ILIKE :q OR qualified_name ILIKE :q OR docstring ILIKE :q)
                LIMIT 15
            """),
            {"pid": project_id, "wid": workspace_id, "q": f"%{question[:100]}%"}
        ).fetchall()

        return [
            {
                "name": r.name, "qualified_name": r.qualified_name,
                "file": r.file_path, "type": r.symbol_type,
                "signature": r.signature, "docstring": r.docstring,
            }
            for r in rows
        ]

    def _generate_answer(
        self, question: str, query_type: str,
        target_symbol: Optional[str], results: List[Dict[str, Any]],
    ) -> str:
        """Generate a natural language answer from structured results."""
        if not results:
            return f"No results found for '{question}'."

        if isinstance(results[0], dict) and 'message' in results[0]:
            return results[0]['message']

        if query_type == 'call_graph':
            parts = []
            for r in results:
                callers = r.get('callers', [])
                callees = r.get('callees', [])
                if callers:
                    caller_names = ', '.join(c['name'] for c in callers[:10])
                    parts.append(f"{r['symbol']} is called by: {caller_names}")
                if callees:
                    callee_names = ', '.join(c['name'] for c in callees[:10])
                    parts.append(f"{r['symbol']} calls: {callee_names}")
                if not callers and not callees:
                    parts.append(f"{r['symbol']} has no direct call relationships recorded.")
            return '. '.join(parts) if parts else f"Found {target_symbol} but no call relationships."

        elif query_type == 'dependency':
            parts = []
            for r in results:
                items = r.get('items', [])
                direction = r.get('direction', 'dependencies')
                if items:
                    names = ', '.join(i['name'] for i in items[:10])
                    parts.append(f"{r['symbol']} {direction}: {names}")
                else:
                    parts.append(f"{r['symbol']} has no {direction}.")
            return '. '.join(parts)

        elif query_type == 'inheritance':
            parts = []
            for r in results:
                subs = r.get('subclasses', [])
                supers = r.get('superclasses', [])
                if subs:
                    parts.append(f"{r['symbol']} is extended by: {', '.join(s['name'] for s in subs[:10])}")
                if supers:
                    parts.append(f"{r['symbol']} extends: {', '.join(s['name'] for s in supers[:10])}")
            return '. '.join(parts) if parts else f"No inheritance relationships found for {target_symbol}."

        else:
            count = len(results)
            names = ', '.join(r.get('name', '?') for r in results[:5])
            return f"Found {count} matching symbols: {names}" + ("..." if count > 5 else ".")
