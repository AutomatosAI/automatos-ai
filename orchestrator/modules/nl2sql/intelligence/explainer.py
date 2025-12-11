"""
Result Explainer
================

Generates human-readable explanations of SQL queries and their results.
Inspired by PandasAI's agent.explain() feature.
"""

import logging
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)


class ResultExplainer:
    """
    Generates explanations for SQL queries and their results.
    """
    
    def __init__(self, llm_provider=None):
        """
        Initialize explainer.
        
        Args:
            llm_provider: Optional LLM for advanced explanations
        """
        self.llm_provider = llm_provider
    
    def explain_sql(
        self,
        sql: str,
        schema_metadata: Dict[str, Any]
    ) -> str:
        """
        Explain what a SQL query does in plain English.
        
        Args:
            sql: The SQL query to explain
            schema_metadata: Database schema for context
            
        Returns:
            Human-readable explanation
        """
        # Parse SQL components
        components = self._parse_sql_components(sql)
        
        # Build explanation
        explanation_parts = []
        
        # Explain SELECT
        if components.get('select'):
            cols = components['select']
            if '*' in cols:
                explanation_parts.append("Fetching all columns")
            elif 'COUNT' in cols.upper():
                explanation_parts.append("Counting records")
            elif 'SUM' in cols.upper():
                explanation_parts.append("Summing values")
            elif 'AVG' in cols.upper():
                explanation_parts.append("Calculating average")
            else:
                explanation_parts.append(f"Selecting: {cols}")
        
        # Explain FROM
        if components.get('from'):
            explanation_parts.append(f"from the '{components['from']}' table")
        
        # Explain WHERE
        if components.get('where'):
            explanation_parts.append(f"filtered by: {components['where']}")
        
        # Explain GROUP BY
        if components.get('group_by'):
            explanation_parts.append(f"grouped by {components['group_by']}")
        
        # Explain ORDER BY
        if components.get('order_by'):
            explanation_parts.append(f"sorted by {components['order_by']}")
        
        # Explain LIMIT
        if components.get('limit'):
            explanation_parts.append(f"limited to {components['limit']} results")
        
        return ' '.join(explanation_parts)
    
    def explain_result(
        self,
        query: str,
        sql: str,
        data: List[Dict[str, Any]],
        use_llm: bool = True
    ) -> str:
        """
        Explain the results of a query in context.
        
        Args:
            query: Original natural language query
            sql: The SQL that was executed
            data: The query results
            use_llm: Whether to use LLM for explanation
            
        Returns:
            Human-readable explanation of results
        """
        if not data:
            return "The query returned no results."
        
        # Basic stats
        row_count = len(data)
        col_count = len(data[0].keys()) if data else 0
        
        # Build basic explanation
        basic_explanation = f"Found {row_count} record{'s' if row_count != 1 else ''}"
        
        if row_count > 0:
            columns = list(data[0].keys())
            
            # Analyze numeric columns
            numeric_insights = []
            for col in columns:
                values = []
                for row in data:
                    val = row.get(col)
                    if isinstance(val, (int, float)) and val is not None:
                        values.append(val)
                
                if values:
                    total = sum(values)
                    avg = total / len(values)
                    numeric_insights.append(f"{col}: total={total:,.2f}, avg={avg:,.2f}")
            
            if numeric_insights:
                basic_explanation += f". Numeric summary: {'; '.join(numeric_insights[:3])}"
        
        # Use LLM for richer explanation
        if use_llm and self.llm_provider and data:
            try:
                return self._llm_explain(query, sql, data, basic_explanation)
            except Exception as e:
                logger.warning(f"LLM explanation failed: {e}")
        
        return basic_explanation
    
    async def explain_result_async(
        self,
        query: str,
        sql: str,
        data: List[Dict[str, Any]],
        use_llm: bool = True
    ) -> str:
        """Async version of explain_result."""
        return self.explain_result(query, sql, data, use_llm)
    
    def _parse_sql_components(self, sql: str) -> Dict[str, str]:
        """
        Parse SQL into its main components.
        """
        sql_upper = sql.upper()
        components = {}
        
        # Extract SELECT
        if 'SELECT' in sql_upper:
            start = sql_upper.find('SELECT') + 6
            end = sql_upper.find('FROM') if 'FROM' in sql_upper else len(sql)
            components['select'] = sql[start:end].strip()
        
        # Extract FROM
        if 'FROM' in sql_upper:
            start = sql_upper.find('FROM') + 4
            # Find next keyword
            end = len(sql)
            for keyword in ['WHERE', 'GROUP', 'ORDER', 'LIMIT', 'JOIN']:
                pos = sql_upper.find(keyword, start)
                if pos != -1 and pos < end:
                    end = pos
            components['from'] = sql[start:end].strip()
        
        # Extract WHERE
        if 'WHERE' in sql_upper:
            start = sql_upper.find('WHERE') + 5
            end = len(sql)
            for keyword in ['GROUP', 'ORDER', 'LIMIT']:
                pos = sql_upper.find(keyword, start)
                if pos != -1 and pos < end:
                    end = pos
            components['where'] = sql[start:end].strip()
        
        # Extract GROUP BY
        if 'GROUP BY' in sql_upper:
            start = sql_upper.find('GROUP BY') + 8
            end = len(sql)
            for keyword in ['ORDER', 'LIMIT', 'HAVING']:
                pos = sql_upper.find(keyword, start)
                if pos != -1 and pos < end:
                    end = pos
            components['group_by'] = sql[start:end].strip()
        
        # Extract ORDER BY
        if 'ORDER BY' in sql_upper:
            start = sql_upper.find('ORDER BY') + 8
            end = sql_upper.find('LIMIT') if 'LIMIT' in sql_upper else len(sql)
            components['order_by'] = sql[start:end].strip()
        
        # Extract LIMIT
        if 'LIMIT' in sql_upper:
            start = sql_upper.find('LIMIT') + 5
            components['limit'] = sql[start:].strip().split()[0]
        
        return components
    
    def _llm_explain(
        self,
        query: str,
        sql: str,
        data: List[Dict[str, Any]],
        basic_explanation: str
    ) -> str:
        """
        Use LLM to generate a rich explanation.
        """
        if not self.llm_provider:
            return basic_explanation
        
        # Prepare data sample
        sample_data = data[:5]  # First 5 rows
        data_str = json.dumps(sample_data, default=str, indent=2)
        
        prompt = f"""Explain these database query results in a clear, insightful way.

Original Question: "{query}"

SQL Query: {sql}

Results ({len(data)} total rows):
{data_str}

Basic Stats: {basic_explanation}

Generate a 2-3 sentence explanation that:
1. Answers the original question directly
2. Highlights key insights or patterns
3. Notes anything surprising or important

Be concise and helpful. Don't just describe the data - interpret it."""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_provider.generate_response_sync(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM explain error: {e}")
            return basic_explanation
    
    def generate_follow_up_questions(
        self,
        query: str,
        data: List[Dict[str, Any]],
        num_questions: int = 3
    ) -> List[str]:
        """
        Suggest follow-up questions based on the results.
        """
        questions = []
        
        if not data:
            questions.append("Would you like to try a different query?")
            return questions
        
        columns = list(data[0].keys()) if data else []
        row_count = len(data)
        
        # Suggest drilling down
        if row_count > 10:
            questions.append(f"Would you like to see the top 10 results?")
        
        # Suggest grouping if not already grouped
        date_cols = [c for c in columns if any(d in c.lower() for d in ['date', 'time', 'created', 'updated'])]
        if date_cols:
            questions.append(f"Would you like to see this broken down by {date_cols[0]}?")
        
        # Suggest filtering
        if row_count > 1:
            first_col = columns[0] if columns else None
            if first_col:
                first_val = data[0].get(first_col)
                if first_val:
                    questions.append(f"Would you like to filter for a specific {first_col}?")
        
        # Suggest comparison
        numeric_cols = [c for c in columns if any(isinstance(row.get(c), (int, float)) for row in data[:5])]
        if numeric_cols:
            questions.append(f"Would you like to compare {numeric_cols[0]} trends over time?")
        
        return questions[:num_questions]

