"""
NL2SQL Benchmark Runner
========================

PRD-61 Phase 8: Automated benchmarking against verified Golden SQL examples.

Tests current system by:
1. Generating SQL from each test question
2. Comparing generated SQL to verified SQL (exact match)
3. Optionally executing both and comparing results (execution match)
4. Tracking accuracy over time
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .comparator import SQLComparator

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    total: int = 0
    exact_match_rate: float = 0.0
    execution_match_rate: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class NL2SQLBenchmarkRunner:
    """Run benchmarks against verified Golden SQL examples."""

    def __init__(self, nl2sql_service=None, example_store=None):
        self.nl2sql_service = nl2sql_service
        self.example_store = example_store
        self.comparator = SQLComparator()

    async def run_benchmark(
        self,
        database_source_id: str,
        workspace_id: str,
        schema_metadata: Dict[str, Any],
        test_examples: Optional[List[Dict]] = None,
        dialect: str = "postgresql",
        test_fraction: float = 0.2,
        seed: int = 42,
        execute_sql: Optional[Callable[[str], Awaitable[List[Dict[str, Any]]]]] = None,
    ) -> BenchmarkResult:
        """
        Test current system against verified examples.

        If test_examples is None, uses a random subset of verified examples.
        """
        if test_examples is None:
            test_examples = await self._get_test_set(
                database_source_id, workspace_id, test_fraction, seed
            )

        if not test_examples:
            logger.warning("No test examples available for benchmark")
            return BenchmarkResult()

        results = []
        exact_matches = 0
        exec_matches = 0

        for example in test_examples:
            question = example.get('question', '')
            expected_sql = example.get('sql', '')

            try:
                generated_sql, explanation, metadata = self.nl2sql_service.generate_sql(
                    question=question,
                    schema_metadata=schema_metadata,
                    dialect=dialect
                )

                # Exact match stays a free secondary signal only.
                exact = self.comparator.exact_match(generated_sql, expected_sql)

                # PRD-199 S6: execution accuracy is THE metric (the Genie-style
                # benchmark-in-product): run both SQLs against the connected
                # source — through the caller-supplied executor, which
                # validates (read-only, LIMIT-capped) and runs under the
                # pipeline's EXPLAIN/timeout guards — and compare result sets
                # order-insensitively. Never again mirrored from exact match
                # (string match is the metric the field abandoned: equivalent
                # SQL that reads differently scored 0).
                error_note = None
                if execute_sql is None:
                    exec_match = None  # honest: unmeasured without an executor
                else:
                    try:
                        expected_rows = await execute_sql(expected_sql)
                    except Exception as e:  # a broken golden is a data bug — surface it
                        expected_rows = None
                        error_note = f"golden failed: {e}"
                    if expected_rows is None:
                        exec_match = False
                    else:
                        try:
                            generated_rows = await execute_sql(generated_sql)
                            exec_match = self.comparator.results_match(
                                generated_rows, expected_rows
                            )
                        except Exception as e:  # generated SQL failed on the source
                            exec_match = False
                            error_note = str(e)

                if exact:
                    exact_matches += 1
                if exec_match:
                    exec_matches += 1

                detail = {
                    'question': question,
                    'expected_sql': expected_sql,
                    'generated_sql': generated_sql,
                    'exact_match': exact,
                    'execution_match': exec_match,
                }
                if error_note:
                    detail['error'] = error_note
                results.append(detail)

            except Exception as e:
                logger.error(f"Benchmark error for '{question[:50]}...': {e}")
                results.append({
                    'question': question,
                    'expected_sql': expected_sql,
                    'generated_sql': f"ERROR: {e}",
                    'exact_match': False,
                    'execution_match': False,
                })

        total = len(results)
        return BenchmarkResult(
            total=total,
            exact_match_rate=exact_matches / total if total else 0.0,
            execution_match_rate=exec_matches / total if total else 0.0,
            details=results,
            timestamp=datetime.utcnow()
        )

    async def _get_test_set(
        self,
        database_source_id: str,
        workspace_id: str,
        fraction: float = 0.2,
        seed: int = 42
    ) -> List[Dict]:
        """Get a subset of verified examples for testing."""
        if not self.example_store:
            return []

        from core.models.database_knowledge import NL2SQLTrainingExample
        from core.database.database import SessionLocal

        db = SessionLocal()
        try:
            examples = db.query(NL2SQLTrainingExample).filter(
                NL2SQLTrainingExample.workspace_id == workspace_id,
                NL2SQLTrainingExample.database_source_id == int(database_source_id),
                NL2SQLTrainingExample.is_verified == True
            ).all()

            if not examples:
                return []

            # Deterministic random subset
            rng = random.Random(seed)
            test_size = max(1, int(len(examples) * fraction))
            test_set = rng.sample(examples, min(test_size, len(examples)))

            return [
                {"question": ex.question, "sql": ex.sql}
                for ex in test_set
            ]
        finally:
            db.close()
