"""
Schema Linker — Maps natural language entities to database schema elements.
==========================================================================

PRD-61 Phase 3: Explicit schema linking stage replacing brittle keyword matching.

Uses a combination of:
1. Semantic layer matching (map business terms to metrics/dimensions)
2. Embedding-based table/column ranking (when embeddings available)
3. Multi-factor keyword scoring (fallback)
4. Relationship expansion (include JOIN bridge tables)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Stopwords for keyword overlap
_STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'in', 'on', 'for', 'with', 'at',
    'from', 'to', 'of', 'and', 'or', 'not', 'by', 'as', 'it', 'its',
    'this', 'that', 'be', 'was', 'were', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'shall', 'can', 'show', 'me', 'get', 'find', 'list',
    'give', 'display', 'how', 'many', 'much', 'what', 'which', 'who',
})


@dataclass
class LinkedSchema:
    """Result of schema linking — relevant tables, columns, relationships."""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    columns: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


class SchemaLinker:
    """Links natural language queries to relevant schema elements."""

    def __init__(self, embedding_manager=None):
        self.embedding_manager = embedding_manager

    async def link(
        self,
        question: str,
        schema_metadata: Dict[str, Any],
        semantic_layer: Optional[Dict[str, Any]] = None,
        top_k_tables: int = 5,
        top_k_columns: int = 20
    ) -> LinkedSchema:
        """
        Link question to relevant schema elements.

        Returns LinkedSchema with ranked tables, columns, relationships,
        and overall confidence score.
        """
        result = LinkedSchema()

        # Step 1: Semantic layer matching (highest priority)
        if semantic_layer:
            result.metrics = self._match_metrics(question, semantic_layer)

        # Step 2: Score and rank tables
        table_scores = self._score_tables(question, schema_metadata)

        # Take top-k tables
        top_tables = table_scores[:top_k_tables]
        result.tables = [t['table'] for t in top_tables]

        # Step 3: Score columns within top tables
        result.columns = self._score_columns(question, result.tables, top_k_columns)

        # Step 4: Expand with relationship tables (JOIN bridge tables)
        result.tables, result.relationships = self._expand_relationships(
            result.tables, schema_metadata
        )

        # Calculate overall confidence
        if top_tables:
            max_score = top_tables[0]['score']
            result.confidence = min(1.0, max_score / 20.0)
        else:
            result.confidence = 0.1

        return result

    def _match_metrics(
        self,
        question: str,
        semantic_layer: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Match question against semantic layer metrics/dimensions."""
        matched = []
        question_lower = question.lower()

        for name, metric in (semantic_layer.get('metrics') or {}).items():
            # Check name and description
            if name.lower() in question_lower:
                matched.append({"name": name, "type": "metric", **metric})
            elif metric.get('description', '').lower() in question_lower:
                matched.append({"name": name, "type": "metric", **metric})

        for category, dims in (semantic_layer.get('dimensions') or {}).items():
            for name, sql in dims.items():
                if name.lower() in question_lower or category.lower() in question_lower:
                    matched.append({"name": name, "category": category, "type": "dimension", "sql": sql})

        return matched

    def _score_tables(
        self,
        question: str,
        schema_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Score tables by multi-factor relevance to question."""
        question_lower = question.lower()
        question_words = set(question_lower.split()) - _STOPWORDS
        tables = schema_metadata.get('tables', [])

        scored = []
        for table in tables:
            score = 0
            table_name = table.get('name', '').lower()

            # Direct name mention (+10)
            if table_name in question_lower or table_name.replace('_', ' ') in question_lower:
                score += 10

            # Name parts (+5)
            for part in table_name.split('_'):
                if len(part) > 2 and part in question_lower:
                    score += 5

            # Column name mentions (+5 each)
            for col in table.get('columns', []):
                col_name = col.get('name', '').lower()
                if col_name in question_lower or col_name.replace('_', ' ') in question_lower:
                    score += 5

            # Description overlap (+2 per word)
            desc = (table.get('description', '') or '').lower()
            desc_words = set(desc.split()) - _STOPWORDS
            overlap = question_words & desc_words
            score += len(overlap) * 2

            scored.append({"table": table, "score": score, "name": table_name})

        # Sort by score descending
        scored.sort(key=lambda x: x['score'], reverse=True)

        # If nothing matched, return top tables by significance
        if not scored or scored[0]['score'] == 0:
            business_kw = ['order', 'customer', 'user', 'product', 'transaction', 'payment', 'invoice']
            for item in scored:
                if any(kw in item['name'] for kw in business_kw):
                    item['score'] = 1

            scored.sort(key=lambda x: x['score'], reverse=True)

        return scored

    def _score_columns(
        self,
        question: str,
        tables: List[Dict[str, Any]],
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Score columns within selected tables."""
        question_lower = question.lower()
        scored = []

        for table in tables:
            table_name = table.get('name', '')
            for col in table.get('columns', []):
                col_name = col.get('name', '').lower()
                score = 0

                if col_name in question_lower:
                    score += 5
                if col_name.replace('_', ' ') in question_lower:
                    score += 3
                for part in col_name.split('_'):
                    if len(part) > 2 and part in question_lower:
                        score += 1

                if score > 0:
                    scored.append({
                        "table": table_name,
                        "column": col.get('name'),
                        "type": col.get('type'),
                        "score": score,
                    })

        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]

    def _expand_relationships(
        self,
        tables: List[Dict[str, Any]],
        schema_metadata: Dict[str, Any]
    ) -> tuple:
        """Add JOIN bridge tables needed for relationships between selected tables."""
        table_names = {t.get('name', '').lower() for t in tables}
        all_tables = {t.get('name', '').lower(): t for t in schema_metadata.get('tables', [])}
        relationships = schema_metadata.get('relationships', [])

        relevant_rels = []
        expanded_tables = list(tables)

        for rel in relationships:
            from_table = rel.get('from_table', '').lower()
            to_table = rel.get('to_table', '').lower()

            if from_table in table_names or to_table in table_names:
                relevant_rels.append(rel)

                # Add the other table if not already included
                if from_table in table_names and to_table not in table_names:
                    if to_table in all_tables:
                        expanded_tables.append(all_tables[to_table])
                        table_names.add(to_table)
                elif to_table in table_names and from_table not in table_names:
                    if from_table in all_tables:
                        expanded_tables.append(all_tables[from_table])
                        table_names.add(from_table)

        return expanded_tables, relevant_rels
