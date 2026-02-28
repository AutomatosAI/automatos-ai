"""
Query Confidence Scorer
========================

PRD-61 Phase 6: Multi-factor confidence scoring for generated SQL queries.
Assigns a confidence score so users know when to trust results vs review manually.

Inspired by Dataherald's confidence scoring approach.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScoringContext:
    """Input context for confidence scoring."""
    similar_examples: List[Dict[str, Any]] = field(default_factory=list)
    linked_schema: Optional[Any] = None  # LinkedSchema or None
    sql: str = ""
    metrics_matched: List[Dict[str, Any]] = field(default_factory=list)
    validation_clean: bool = True


@dataclass
class ConfidenceScore:
    """Result of confidence scoring."""
    score: int  # 0-100
    level: str  # 'high', 'medium', 'low'
    factors: Dict[str, int] = field(default_factory=dict)
    recommendation: str = "review_sql"  # 'auto_execute', 'review_sql', 'manual_review'


class QueryConfidenceScorer:
    """Score confidence of generated SQL queries using multiple factors."""

    def score(self, context: ScoringContext) -> ConfidenceScore:
        """
        Calculate confidence score based on multiple factors.

        Factors:
            1. example_match (0-30): How well the query matches known examples
            2. schema_link (0-25): How confident the schema linking was
            3. complexity (0-20): Simpler queries = higher confidence
            4. semantic (0-15): Whether business metrics were matched
            5. validation (0-10): Whether SQL passed validation cleanly
        """
        factors = {}

        # Factor 1: Similar examples found (0-30 points)
        similar_count = len(context.similar_examples)
        best_similarity = 0.0
        if context.similar_examples:
            best_similarity = max(
                ex.get('similarity', 0) for ex in context.similar_examples
            )
        factors['example_match'] = min(30, int(similar_count * 5 + best_similarity * 20))

        # Factor 2: Schema linking confidence (0-25 points)
        schema_confidence = 0.0
        if context.linked_schema and hasattr(context.linked_schema, 'confidence'):
            schema_confidence = context.linked_schema.confidence
        elif context.linked_schema:
            schema_confidence = 0.5  # Fallback if no confidence attr
        factors['schema_link'] = min(25, int(schema_confidence * 25))

        # Factor 3: Query complexity (0-20 points, simpler = higher)
        sql_lower = context.sql.lower()
        join_count = sql_lower.count('join')
        subquery_count = max(0, sql_lower.count('select') - 1)
        complexity_penalty = (join_count * 3 + subquery_count * 5)
        factors['complexity'] = max(0, 20 - complexity_penalty)

        # Factor 4: Semantic layer coverage (0-15 points)
        metrics_count = len(context.metrics_matched)
        factors['semantic'] = min(15, metrics_count * 5)

        # Factor 5: Validation passed cleanly (0-10 points)
        factors['validation'] = 10 if context.validation_clean else 5

        total = sum(factors.values())

        # Determine level and recommendation
        if total >= 70:
            level = 'high'
            recommendation = 'auto_execute'
        elif total >= 40:
            level = 'medium'
            recommendation = 'review_sql'
        else:
            level = 'low'
            recommendation = 'manual_review'

        return ConfidenceScore(
            score=total,
            level=level,
            factors=factors,
            recommendation=recommendation
        )
