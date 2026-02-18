"""
Natural Language to SQL Service
===============================
REAL implementation with RAG-based few-shot examples,
error self-correction context, and improved schema linking.

PRD-61: NL2SQL v2 Competitive Upgrade
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import create_engine, text
import logging

logger = logging.getLogger(__name__)

# Stopwords excluded from keyword overlap scoring
_STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'because', 'but', 'and', 'or', 'if', 'while', 'that', 'what',
    'which', 'who', 'whom', 'this', 'these', 'those', 'i', 'me', 'my',
    'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it',
    'its', 'they', 'them', 'their', 'show', 'get', 'give', 'find', 'list',
})


class NaturalLanguageToSQLService:
    """
    Converts natural language queries to SQL using LLM.
    Supports few-shot examples, error correction context, and schema linking.
    """

    def __init__(self, llm_provider, example_store=None, schema_linker=None):
        self.llm_provider = llm_provider
        self.example_store = example_store
        self.schema_linker = schema_linker

    def generate_sql(
        self,
        question: str,
        schema_metadata: Dict[str, Any],
        semantic_layer: Optional[Dict[str, Any]] = None,
        dialect: str = "postgresql",
        examples: Optional[List[Dict[str, str]]] = None,
        error_context: Optional[str] = None,
        previous_attempts: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Generate SQL from natural language using LLM.

        Args:
            question: Natural language question
            schema_metadata: Database schema information
            semantic_layer: Business metrics/dimensions
            dialect: SQL dialect
            examples: Few-shot examples (question/SQL pairs)
            error_context: Previous error message for self-correction
            previous_attempts: List of {sql, error} dicts from failed attempts

        Returns:
            Tuple of (sql, explanation, metadata)
        """
        prompt = self._build_prompt(
            question=question,
            schema_metadata=schema_metadata,
            semantic_layer=semantic_layer,
            dialect=dialect,
            examples=examples,
            error_context=error_context,
            previous_attempts=previous_attempts
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            llm_response = self.llm_provider.generate_response_sync(messages)

            response_text = llm_response.content
            sql, explanation, metadata = self._parse_llm_response(response_text)

            sql = self._clean_sql(sql, dialect)

            metadata.update({
                "tables_referenced": self._extract_table_references(sql, schema_metadata),
                "question": question,
                "dialect": dialect,
                "had_error_context": error_context is not None,
                "attempt_number": len(previous_attempts) + 1 if previous_attempts else 1,
            })

            logger.info(f"Generated SQL for question: {question[:100]}...")
            return sql, explanation, metadata

        except Exception as e:
            logger.error(f"Failed to generate SQL: {str(e)}")
            return (
                f"SELECT 'Error generating SQL: {str(e)}' as error",
                f"Failed to generate SQL: {str(e)}",
                {"error": str(e), "success": False}
            )

    def _build_prompt(
        self,
        question: str,
        schema_metadata: Dict[str, Any],
        semantic_layer: Optional[Dict[str, Any]],
        dialect: str,
        examples: Optional[List[Dict[str, str]]],
        error_context: Optional[str] = None,
        previous_attempts: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build a comprehensive prompt for the LLM."""

        # Use schema linker if available, otherwise fall back to improved scoring
        if self.schema_linker:
            try:
                import asyncio
                linked = asyncio.get_event_loop().run_until_complete(
                    self.schema_linker.link(question, schema_metadata, semantic_layer)
                )
                relevant_tables = linked.tables
            except Exception:
                relevant_tables = self._get_relevant_tables(question, schema_metadata)
        else:
            relevant_tables = self._get_relevant_tables(question, schema_metadata)

        prompt_parts = []

        # System instructions
        prompt_parts.append(f"""You are an expert SQL developer. Generate {dialect} SQL queries based on natural language questions.

CRITICAL RULES:
1. ONLY use SELECT statements - no INSERT, UPDATE, DELETE, DROP, CREATE, ALTER
2. ALWAYS include a LIMIT clause (default 1000 unless specified)
3. Use proper JOIN syntax with explicit conditions
4. Handle NULL values appropriately
5. Use appropriate aggregation functions when needed
6. Consider performance - use indexes when available
7. Return results in a user-friendly format with clear column aliases

SMART QUERY INTERPRETATION:
- If user asks about data "over", "in", or "during" a time period, they likely want a TIME-SERIES breakdown (GROUP BY date/day)
- If user asks "how many" or "count", decide based on context:
  - "how many total" → single count
  - "how many per day/week/month" → grouped by time
  - "how many X happened in last N days" → likely want daily breakdown to see trends
- If user asks for "details", "breakdown", "trend", or "history" → GROUP BY the relevant dimension
- If user asks for a simple total or count without time context → return aggregate
- When in doubt about grouping, prefer showing MORE granular data (users can always aggregate, but can't disaggregate)

Database Dialect: {dialect}
""")

        # Add schema information
        prompt_parts.append("\nDATABASE SCHEMA:")
        for table in relevant_tables:
            prompt_parts.append(f"\nTable: {table.get('name')}")
            if table.get('description'):
                prompt_parts.append(f"Description: {table.get('description')}")

            prompt_parts.append("Columns:")
            for col in table.get('columns', []):
                col_info = f"  - {col['name']} ({col['type']})"
                if col.get('primary_key'):
                    col_info += " PRIMARY KEY"
                if col.get('nullable') == False:
                    col_info += " NOT NULL"
                if col.get('description'):
                    col_info += f" -- {col['description']}"
                prompt_parts.append(col_info)

            # Add sample values if available
            if table.get('columns'):
                for col in table['columns']:
                    if col.get('samples'):
                        prompt_parts.append(f"    Sample values for {col['name']}: {', '.join(str(s) for s in col['samples'][:3])}")

        # Add relationships
        relationships = schema_metadata.get('relationships', [])
        if relationships:
            prompt_parts.append("\nRELATIONSHIPS:")
            for rel in relationships:
                prompt_parts.append(f"  - {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']} ({rel.get('type', 'foreign_key')})")

        # Add semantic layer if available
        if semantic_layer:
            if semantic_layer.get('metrics'):
                prompt_parts.append("\nBUSINESS METRICS (use these when applicable):")
                for name, metric in semantic_layer['metrics'].items():
                    prompt_parts.append(f"  - {name}: {metric.get('sql')} -- {metric.get('description', '')}")

            if semantic_layer.get('dimensions'):
                prompt_parts.append("\nCOMMON DIMENSIONS:")
                for category, dims in semantic_layer.get('dimensions', {}).items():
                    for name, sql in dims.items():
                        prompt_parts.append(f"  - {category}.{name}: {sql}")

        # PRD-61 US-003: Add few-shot examples from RAG store
        if examples:
            prompt_parts.append("\n--- SIMILAR VERIFIED QUERIES (use as reference) ---")
            for ex in examples[:5]:
                prompt_parts.append(f"Question: {ex['question']}")
                prompt_parts.append(f"SQL: {ex['sql']}")
                prompt_parts.append("")

        # PRD-61 US-005: Add error correction context
        if error_context and previous_attempts:
            prompt_parts.append("\n--- PREVIOUS ATTEMPTS (fix these errors) ---")
            for attempt in previous_attempts:
                prompt_parts.append(f"Attempted SQL: {attempt['sql']}")
                prompt_parts.append(f"Error: {attempt['error']}")
                prompt_parts.append("")
            prompt_parts.append("Generate a CORRECTED SQL query that avoids these errors.\n")

        # Add the actual question
        prompt_parts.append(f"\nQUESTION: {question}")
        prompt_parts.append("\nGenerate the SQL query and explanation in this format:")
        prompt_parts.append("SQL:")
        prompt_parts.append("[your SQL query here]")
        prompt_parts.append("EXPLANATION:")
        prompt_parts.append("[brief explanation of what the query does]")

        return "\n".join(prompt_parts)

    def _get_relevant_tables(
        self,
        question: str,
        schema_metadata: Dict[str, Any],
        max_tables: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Extract relevant tables using scored approach.

        PRD-61 Bug Fix (US-007): Replaced naive keyword matching with
        multi-factor scoring. Falls back to top 5 by row_count instead
        of ALL tables when no matches found.
        """
        question_lower = question.lower()
        question_words = set(question_lower.split()) - _STOPWORDS
        tables = schema_metadata.get('tables', [])

        scored_tables = []
        for table in tables:
            score = 0
            table_name = table.get('name', '').lower()

            # Factor 1: Direct table name mention (+10)
            if table_name in question_lower or table_name.replace('_', ' ') in question_lower:
                score += 10

            # Factor 2: Table name parts match (+5 each)
            for part in table_name.split('_'):
                if len(part) > 2 and part in question_lower:
                    score += 5

            # Factor 3: Column name mentions (+5 each)
            for col in table.get('columns', []):
                col_name = col.get('name', '').lower()
                if col_name in question_lower or col_name.replace('_', ' ') in question_lower:
                    score += 5
                # Column name parts
                for part in col_name.split('_'):
                    if len(part) > 2 and part in question_lower:
                        score += 1

            # Factor 4: Description keyword overlap (+2 per word)
            desc = (table.get('description', '') or '').lower()
            comment = (table.get('comment', '') or '').lower()
            desc_words = set(f"{desc} {comment}".split()) - _STOPWORDS
            overlap = question_words & desc_words
            score += len(overlap) * 2

            if score > 0:
                scored_tables.append((score, table))

        # Sort by score descending, take top N
        scored_tables.sort(key=lambda x: x[0], reverse=True)

        if scored_tables:
            return [table for _, table in scored_tables[:max_tables]]

        # Fallback: top 5 tables by row_count (most important), NOT all tables
        tables_sorted = sorted(
            tables,
            key=lambda t: t.get('row_count', 0) or 0,
            reverse=True
        )
        # If no row_count, use business-significant table names
        if not any(t.get('row_count', 0) for t in tables_sorted):
            business_keywords = ['order', 'customer', 'user', 'product', 'transaction', 'payment', 'invoice', 'sale']
            business_tables = [
                t for t in tables
                if any(kw in t.get('name', '').lower() for kw in business_keywords)
            ]
            if business_tables:
                return business_tables[:5]

        return tables_sorted[:5]

    def _parse_llm_response(self, response: str) -> Tuple[str, str, Dict[str, Any]]:
        """Parse the LLM response to extract SQL and explanation."""
        sql = ""
        explanation = ""
        metadata = {}

        lines = response.strip().split('\n')
        in_sql = False
        in_explanation = False
        sql_lines = []
        explanation_lines = []

        for line in lines:
            line = line.strip()

            if line.upper().startswith('SQL:'):
                in_sql = True
                in_explanation = False
                continue
            elif line.upper().startswith('EXPLANATION:'):
                in_sql = False
                in_explanation = True
                continue

            if in_sql and line:
                sql_lines.append(line)
            elif in_explanation and line:
                explanation_lines.append(line)

        sql = ' '.join(sql_lines).strip()
        explanation = ' '.join(explanation_lines).strip()

        # If parsing failed, try to extract SQL from code blocks
        if not sql:
            sql_match = re.search(r'```sql?\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
            if sql_match:
                sql = sql_match.group(1).strip()
            else:
                if response.strip().upper().startswith('SELECT'):
                    sql = response.strip()

        if not explanation:
            explanation = "Query generated from natural language"

        if sql and 'SELECT' in sql.upper():
            metadata['confidence'] = 0.9 if explanation else 0.7
        else:
            metadata['confidence'] = 0.3

        return sql, explanation, metadata

    def _clean_sql(self, sql: str, dialect: str) -> str:
        """Clean and standardize the SQL query."""
        sql = sql.replace('```sql', '').replace('```', '').strip()
        sql = sql.rstrip(';')
        sql = ' '.join(sql.split())
        if 'LIMIT' not in sql.upper():
            sql = f"{sql} LIMIT 1000"
        return sql

    def _extract_table_references(
        self,
        sql: str,
        schema_metadata: Dict[str, Any]
    ) -> List[str]:
        """Extract table names referenced in the SQL."""
        tables = []
        available_tables = {t['name'].lower(): t['name']
                           for t in schema_metadata.get('tables', [])}

        sql_lower = sql.lower()
        for table_lower, table_name in available_tables.items():
            if table_lower in sql_lower:
                tables.append(table_name)

        return list(set(tables))
