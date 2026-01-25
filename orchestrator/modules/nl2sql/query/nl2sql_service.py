"""
Natural Language to SQL Service
===============================
REAL implementation - not mock bullshit
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import create_engine, text
import logging

logger = logging.getLogger(__name__)


class NaturalLanguageToSQLService:
    """
    Converts natural language queries to SQL using LLM
    ACTUALLY WORKS - not mock data
    """
    
    def __init__(self, llm_provider):
        """
        Initialize with actual LLM provider
        """
        self.llm_provider = llm_provider
    
    def generate_sql(
        self,
        question: str,
        schema_metadata: Dict[str, Any],
        semantic_layer: Optional[Dict[str, Any]] = None,
        dialect: str = "postgresql",
        examples: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Generate SQL from natural language using LLM
        
        Returns:
            - sql: The generated SQL query
            - explanation: Natural language explanation
            - metadata: Additional info (confidence, tables used, etc)
        """
        
        # Build the prompt with REAL schema information
        prompt = self._build_prompt(
            question=question,
            schema_metadata=schema_metadata,
            semantic_layer=semantic_layer,
            dialect=dialect,
            examples=examples
        )
        
        try:
            # Call the ACTUAL LLM (OpenAI/Anthropic/etc)
            messages = [{"role": "user", "content": prompt}]
            llm_response = self.llm_provider.generate_response_sync(messages)
            
            # Parse the LLM response
            response_text = llm_response.content
            sql, explanation, metadata = self._parse_llm_response(response_text)
            
            # Clean up the SQL
            sql = self._clean_sql(sql, dialect)
            
            # Add metadata
            metadata.update({
                "tables_referenced": self._extract_table_references(sql, schema_metadata),
                "question": question,
                "dialect": dialect
            })
            
            logger.info(f"Generated SQL for question: {question[:100]}...")
            
            return sql, explanation, metadata
            
        except Exception as e:
            logger.error(f"Failed to generate SQL: {str(e)}")
            # Return a safe fallback query
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
        examples: Optional[List[Dict[str, str]]]
    ) -> str:
        """
        Build a comprehensive prompt for the LLM
        """
        
        # Extract relevant tables based on question keywords
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
        
        # Add examples if provided
        if examples:
            prompt_parts.append("\nEXAMPLES:")
            for ex in examples[:3]:  # Limit to 3 examples
                prompt_parts.append(f"Q: {ex['question']}")
                prompt_parts.append(f"SQL: {ex['sql']}")
                prompt_parts.append("")
        
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
        max_tables: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Extract relevant tables based on question keywords
        """
        question_lower = question.lower()
        tables = schema_metadata.get('tables', [])
        
        # Score each table based on relevance
        scored_tables = []
        for table in tables:
            score = 0
            table_name = table.get('name', '').lower()
            
            # Check if table name appears in question
            if table_name in question_lower:
                score += 10
            
            # Check if any part of table name appears
            for part in table_name.split('_'):
                if part in question_lower and len(part) > 2:
                    score += 5
            
            # Check column names
            for col in table.get('columns', []):
                col_name = col.get('name', '').lower()
                if col_name in question_lower:
                    score += 3
                for part in col_name.split('_'):
                    if part in question_lower and len(part) > 2:
                        score += 1
            
            if score > 0:
                scored_tables.append((score, table))
        
        # Sort by score and return top tables
        scored_tables.sort(key=lambda x: x[0], reverse=True)
        
        # If no tables matched, include the most important ones
        if not scored_tables:
            # Look for main business tables (orders, customers, users, products, etc)
            for table in tables:
                table_name = table.get('name', '').lower()
                if any(keyword in table_name for keyword in ['order', 'customer', 'user', 'product', 'transaction', 'payment']):
                    scored_tables.append((1, table))
        
        # If still no tables, just include the first few
        if not scored_tables:
            scored_tables = [(0, table) for table in tables[:max_tables]]
        
        return [table for _, table in scored_tables[:max_tables]]
    
    def _parse_llm_response(self, response: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Parse the LLM response to extract SQL and explanation
        """
        sql = ""
        explanation = ""
        metadata = {}
        
        # Try to parse structured response
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
            import re
            sql_match = re.search(r'```sql?\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
            if sql_match:
                sql = sql_match.group(1).strip()
            else:
                # Last resort - assume the whole response is SQL if it starts with SELECT
                if response.strip().upper().startswith('SELECT'):
                    sql = response.strip()
        
        # Generate explanation if not found
        if not explanation:
            explanation = "Query generated from natural language"
        
        # Estimate confidence based on response structure
        if sql and 'SELECT' in sql.upper():
            metadata['confidence'] = 0.9 if explanation else 0.7
        else:
            metadata['confidence'] = 0.3
        
        return sql, explanation, metadata
    
    def _clean_sql(self, sql: str, dialect: str) -> str:
        """
        Clean and standardize the SQL query
        """
        # Remove markdown code blocks if present
        sql = sql.replace('```sql', '').replace('```', '').strip()
        
        # Remove any trailing semicolons
        sql = sql.rstrip(';')
        
        # Ensure single spacing
        sql = ' '.join(sql.split())
        
        # Add LIMIT if not present (safety)
        if 'LIMIT' not in sql.upper():
            sql = f"{sql} LIMIT 1000"
        
        return sql
    
    def _extract_table_references(
        self,
        sql: str,
        schema_metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Extract table names referenced in the SQL
        """
        tables = []
        available_tables = {t['name'].lower(): t['name'] 
                           for t in schema_metadata.get('tables', [])}
        
        sql_lower = sql.lower()
        
        # Look for table names in the SQL
        for table_lower, table_name in available_tables.items():
            if table_lower in sql_lower:
                tables.append(table_name)
        
        return list(set(tables))


