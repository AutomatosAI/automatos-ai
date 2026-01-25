"""
Query Rephraser
===============

Transforms vague or poorly-formed queries into precise, SQL-friendly questions.
Inspired by PandasAI's agent.rephrase_query() feature.
"""

import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryRephraser:
    """
    Rephrases natural language queries to be more precise and SQL-optimized.
    """
    
    # Common vague terms and their expansions
    VAGUE_TERMS = {
        'stuff': 'data',
        'things': 'records',
        'info': 'information',
        'stats': 'statistics',
        'numbers': 'metrics',
        'data': None,  # Keep as-is, needs context
    }
    
    # Time-related expansions
    TIME_EXPANSIONS = {
        'recent': 'in the last 7 days',
        'lately': 'in the last 30 days',
        'this week': 'from Monday to today',
        'today': 'for today',
        'yesterday': 'for yesterday',
        'now': 'current',
    }
    
    # ✨ SPARKLE: Patterns that imply time-series breakdown
    TIME_BREAKDOWN_TRIGGERS = {
        'details': 'grouped by day',
        'breakdown': 'grouped by day',
        'over time': 'grouped by day',
        'per day': 'grouped by day',
        'daily': 'grouped by day',
        'trend': 'grouped by day showing trend',
        'history': 'grouped by day',
        'timeline': 'grouped by day',
        'progression': 'grouped by day',
        'evolution': 'grouped by day over time',
        'pattern': 'grouped by day to show pattern',
        'how many': 'grouped by day',  # "how many X in last N days" → daily breakdown
        'count': 'grouped by day',      # "count of X over last N days" → daily breakdown
        'show me': 'grouped by day',    # "show me X from last N days" → daily breakdown
        'number of': 'grouped by day',  # "number of X in last N days" → daily breakdown
    }
    
    # Aggregation clarifications
    AGGREGATION_HINTS = {
        'how many': 'COUNT of',
        'total': 'SUM of',
        'average': 'AVERAGE of',
        'biggest': 'MAX of',
        'smallest': 'MIN of',
        'top': 'highest',
        'bottom': 'lowest',
    }
    
    def __init__(self, llm_provider=None):
        """
        Initialize rephraser.
        
        Args:
            llm_provider: Optional LLM for advanced rephrasing
        """
        self.llm_provider = llm_provider
    
    def rephrase(
        self,
        query: str,
        schema_metadata: Dict[str, Any],
        use_llm: bool = True
    ) -> Tuple[str, str]:
        """
        Rephrase a query to be more precise.
        
        Args:
            query: Original natural language query
            schema_metadata: Database schema information
            use_llm: Whether to use LLM for rephrasing
            
        Returns:
            Tuple of (rephrased_query, explanation)
        """
        # First, apply rule-based improvements
        improved = self._apply_rules(query, schema_metadata)
        
        # Then use LLM for smarter rephrasing if available
        if use_llm and self.llm_provider:
            try:
                llm_result = self._llm_rephrase(improved, schema_metadata)
                if llm_result:
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM rephrasing failed: {e}")
        
        # Return rule-based improvement
        explanation = "Query improved with standard expansions"
        return improved, explanation
    
    async def rephrase_async(
        self,
        query: str,
        schema_metadata: Dict[str, Any],
        use_llm: bool = True
    ) -> Tuple[str, str]:
        """Async version of rephrase."""
        return self.rephrase(query, schema_metadata, use_llm)
    
    def _apply_rules(
        self,
        query: str,
        schema_metadata: Dict[str, Any]
    ) -> str:
        """
        Apply rule-based query improvements.
        """
        improved = query
        query_lower = query.lower()
        
        # Expand vague terms
        for vague, expansion in self.VAGUE_TERMS.items():
            if vague in query_lower and expansion:
                improved = improved.replace(vague, expansion)
        
        # Expand time references
        for time_term, expansion in self.TIME_EXPANSIONS.items():
            if time_term in query_lower:
                improved = improved.replace(time_term, f"{time_term} ({expansion})")
                break  # Only expand one time reference
        
        # NOTE: We do NOT auto-add grouping anymore - let the LLM figure out intent
        # The user might want a total, daily breakdown, weekly, monthly, or by status
        # Hardcoding "grouped by day" was wrong
        
        # Clarify aggregations
        for agg_term, clarification in self.AGGREGATION_HINTS.items():
            if agg_term in query_lower:
                # Don't modify, just note for context
                pass
        
        # Add table context if missing
        tables = schema_metadata.get('tables', [])
        if tables:
            table_names = [t.get('name', '').lower() for t in tables]
            has_table_ref = any(tbl in query_lower for tbl in table_names)
            
            if not has_table_ref:
                # Try to infer table from query context
                inferred_table = self._infer_table(improved, tables)
                if inferred_table:
                    improved = f"{improved} from {inferred_table}"
        
        return improved
    
    def _infer_table(
        self,
        query: str,
        tables: list
    ) -> Optional[str]:
        """
        Try to infer which table the query is about.
        """
        query_lower = query.lower()
        
        # Keyword to table mapping
        keyword_table_map = {
            'customer': ['customers', 'customer', 'users', 'user'],
            'order': ['orders', 'order', 'transactions', 'transaction'],
            'product': ['products', 'product', 'items', 'item'],
            'sale': ['sales', 'orders', 'transactions'],
            'revenue': ['sales', 'orders', 'transactions', 'payments'],
            'user': ['users', 'user', 'customers', 'accounts'],
            'chat': ['chats', 'chat_sessions', 'conversations'],
            'session': ['sessions', 'chats', 'chat_sessions'],
            'message': ['messages', 'chats', 'chat_messages'],
            'agent': ['agents', 'agent_executions'],
            'workflow': ['workflows', 'workflow_executions'],
            'document': ['documents', 'files'],
            'payment': ['payments', 'transactions'],
        }
        
        # Find matching keyword in query
        for keyword, possible_tables in keyword_table_map.items():
            if keyword in query_lower:
                # Find matching table in schema
                for table in tables:
                    table_name = table.get('name', '').lower()
                    if table_name in possible_tables:
                        return table.get('name')
        
        return None
    
    def _llm_rephrase(
        self,
        query: str,
        schema_metadata: Dict[str, Any]
    ) -> Optional[Tuple[str, str]]:
        """
        Use LLM to intelligently rephrase the query.
        """
        if not self.llm_provider:
            return None
        
        # Build context - select RELEVANT tables, not just first 5
        all_tables = schema_metadata.get('tables', [])
        query_lower = query.lower()
        
        # Score tables by relevance to query
        scored_tables = []
        for table in all_tables:
            table_name = table.get('name', '').lower()
            score = 0
            
            # Direct name match
            if table_name in query_lower:
                score += 20
            
            # Partial name match (e.g., "workflow" in "workflow_executions")
            for part in table_name.split('_'):
                if len(part) > 2 and part in query_lower:
                    score += 10
            
            # Column match
            for col in table.get('columns', []):
                col_name = col.get('name', '').lower()
                if col_name in query_lower:
                    score += 5
            
            scored_tables.append((score, table))
        
        # Sort by score (highest first) and take top 10 relevant tables
        scored_tables.sort(key=lambda x: x[0], reverse=True)
        relevant_tables = [t for _, t in scored_tables[:10] if _ > 0] or [t for _, t in scored_tables[:5]]
        
        tables_summary = []
        for table in relevant_tables:
            cols = [c.get('name') for c in table.get('columns', [])[:5]]
            tables_summary.append(f"- {table.get('name')}: {', '.join(cols)}")
        
        prompt = f"""Rephrase this database query to be more precise and SQL-friendly.

Original query: "{query}"

Relevant database tables:
{chr(10).join(tables_summary)}

Rules:
1. Make the query specific and unambiguous
2. Add time period if relevant and not specified (default: last 30 days)
3. Specify aggregation type if counting/summing
4. ONLY reference table names that appear in the list above - NEVER invent table names
5. Keep it natural language (not SQL)
6. Do NOT change the core intent of the query

CRITICAL: If the user asks about "workflows", use the workflow_executions table. Do NOT reference other tables like codegraph.

Respond with:
REPHRASED: <your rephrased query>
EXPLANATION: <why you changed it>"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_provider.generate_response_sync(messages)
            
            # Parse response
            lines = response.content.strip().split('\n')
            rephrased = None
            explanation = None
            
            for line in lines:
                if line.startswith('REPHRASED:'):
                    rephrased = line.replace('REPHRASED:', '').strip()
                elif line.startswith('EXPLANATION:'):
                    explanation = line.replace('EXPLANATION:', '').strip()
            
            if rephrased:
                return rephrased, explanation or "Query improved by AI"
            
        except Exception as e:
            logger.error(f"LLM rephrase error: {e}")
        
        return None
    
    def suggest_alternatives(
        self,
        query: str,
        schema_metadata: Dict[str, Any],
        num_alternatives: int = 3
    ) -> list:
        """
        Suggest alternative phrasings for a query.
        
        Returns list of (query, description) tuples.
        """
        alternatives = []
        
        # Add aggregation alternatives
        if any(word in query.lower() for word in ['show', 'list', 'get']):
            alternatives.append((
                query.replace('show', 'show the count of').replace('list', 'count'),
                "As a count"
            ))
            alternatives.append((
                f"{query} grouped by date",
                "Grouped by date"
            ))
        
        # Add time-scoped alternatives
        if not any(time in query.lower() for time in ['day', 'week', 'month', 'year', 'date']):
            alternatives.append((
                f"{query} for the last 7 days",
                "Last 7 days"
            ))
            alternatives.append((
                f"{query} for this month",
                "This month"
            ))
        
        # Add limit alternative
        if 'top' not in query.lower() and 'limit' not in query.lower():
            alternatives.append((
                f"Top 10 {query}",
                "Top 10 results"
            ))
        
        return alternatives[:num_alternatives]

