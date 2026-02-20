"""
SQL Comparator — Normalization and result comparison for benchmarks.
====================================================================

PRD-61 Phase 8: Provides SQL normalization and execution result
comparison for accurate benchmark scoring.
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SQLComparator:
    """Compare SQL queries for equivalence."""

    @staticmethod
    def normalize_sql(sql: str) -> str:
        """
        Normalize SQL for comparison.

        - Lowercase
        - Collapse whitespace
        - Remove trailing semicolons
        - Normalize alias patterns
        - Remove comments
        """
        if not sql:
            return ""

        normalized = sql.strip().lower()

        # Remove comments
        normalized = re.sub(r'--.*$', '', normalized, flags=re.MULTILINE)
        normalized = re.sub(r'/\*.*?\*/', '', normalized, flags=re.DOTALL)

        # Remove trailing semicolons
        normalized = normalized.rstrip(';').strip()

        # Collapse whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Normalize common patterns
        normalized = re.sub(r'\s*,\s*', ', ', normalized)
        normalized = re.sub(r'\s*=\s*', ' = ', normalized)
        normalized = re.sub(r'\s*<>\s*', ' <> ', normalized)
        normalized = re.sub(r'\s*!=\s*', ' <> ', normalized)

        return normalized

    @staticmethod
    def exact_match(sql_a: str, sql_b: str) -> bool:
        """Check if two SQL queries are equivalent after normalization."""
        return SQLComparator.normalize_sql(sql_a) == SQLComparator.normalize_sql(sql_b)

    @staticmethod
    def results_match(
        results_a: List[Dict[str, Any]],
        results_b: List[Dict[str, Any]],
        ignore_order: bool = True
    ) -> bool:
        """
        Check if two query results are equivalent.

        Compares row content, optionally ignoring order.
        """
        if len(results_a) != len(results_b):
            return False

        if not results_a:
            return True

        # Normalize to sorted tuples for comparison
        def row_to_tuple(row: Dict) -> tuple:
            return tuple(sorted((k, str(v)) for k, v in row.items()))

        if ignore_order:
            set_a = set(row_to_tuple(r) for r in results_a)
            set_b = set(row_to_tuple(r) for r in results_b)
            return set_a == set_b
        else:
            tuples_a = [row_to_tuple(r) for r in results_a]
            tuples_b = [row_to_tuple(r) for r in results_b]
            return tuples_a == tuples_b
