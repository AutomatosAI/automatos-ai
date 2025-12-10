"""
Query Clarifier
===============

Detects ambiguous queries and generates clarification questions.
Inspired by PandasAI's agent.clarification_questions() feature.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class QueryClarifier:
    """
    Analyzes natural language queries and generates clarification questions
    when the query is ambiguous or missing key information.
    """
    
    # Common ambiguity patterns
    AMBIGUOUS_PATTERNS = {
        'time_range': {
            'keywords': ['sales', 'orders', 'revenue', 'count', 'total', 'sum', 'average', 'trend'],
            'missing': ['last', 'this', 'between', 'from', 'to', 'year', 'month', 'week', 'day', 'date'],
            'questions': [
                "What time period? (e.g., last 7 days, this month, 2024)",
                "Do you want all-time data or a specific date range?",
            ]
        },
        'grouping': {
            'keywords': ['by', 'per', 'breakdown', 'split'],
            'missing': ['product', 'region', 'category', 'customer', 'user', 'country'],
            'questions': [
                "How would you like to group the data? (e.g., by product, region, date)",
                "Do you want the total or broken down by categories?",
            ]
        },
        'entity': {
            'keywords': ['show', 'get', 'list', 'find', 'display'],
            'requires_entity': True,
            'questions': [
                "Which table or data type are you interested in?",
                "Can you be more specific about what data you need?",
            ]
        },
        'aggregation': {
            'keywords': ['how many', 'count', 'total', 'average', 'sum'],
            'questions': [
                "What would you like to count or aggregate?",
            ]
        },
        'filter': {
            'keywords': ['where', 'with', 'only', 'filter', 'having'],
            'questions': [
                "Any specific filters or conditions?",
            ]
        }
    }
    
    def __init__(self, llm_provider=None):
        """
        Initialize clarifier.
        
        Args:
            llm_provider: Optional LLM for advanced clarification
        """
        self.llm_provider = llm_provider
    
    def needs_clarification(
        self,
        query: str,
        schema_metadata: Dict[str, Any]
    ) -> bool:
        """
        Determine if a query needs clarification.
        
        Returns:
            True if query is ambiguous and needs more info
        """
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Very short queries are usually ambiguous
        if word_count <= 3:
            return True
        
        # Check for ambiguity patterns
        for pattern_name, pattern in self.AMBIGUOUS_PATTERNS.items():
            keywords = pattern.get('keywords', [])
            missing = pattern.get('missing', [])
            
            # Check if query matches pattern keywords
            has_keyword = any(kw in query_lower for kw in keywords)
            
            if has_keyword:
                # For time-based queries, check if time is specified
                if pattern_name == 'time_range':
                    has_time = any(t in query_lower for t in missing)
                    if not has_time:
                        return True
                
                # For entity queries, check if a table is referenced
                if pattern.get('requires_entity'):
                    table_names = [t.get('name', '').lower() for t in schema_metadata.get('tables', [])]
                    has_entity = any(tbl in query_lower for tbl in table_names)
                    if not has_entity:
                        return True
        
        return False
    
    def get_clarifications(
        self,
        query: str,
        schema_metadata: Dict[str, Any],
        max_questions: int = 3
    ) -> List[str]:
        """
        Generate clarification questions for an ambiguous query.
        
        Args:
            query: The natural language query
            schema_metadata: Database schema information
            max_questions: Maximum questions to return
            
        Returns:
            List of clarification questions
        """
        query_lower = query.lower()
        questions = []
        
        # Analyze what's missing
        for pattern_name, pattern in self.AMBIGUOUS_PATTERNS.items():
            keywords = pattern.get('keywords', [])
            has_keyword = any(kw in query_lower for kw in keywords)
            
            if has_keyword:
                # Add relevant questions
                for q in pattern.get('questions', []):
                    if q not in questions:
                        questions.append(q)
        
        # Add schema-aware questions
        tables = schema_metadata.get('tables', [])
        if tables:
            # Check if any table is referenced
            table_names = [t.get('name', '').lower() for t in tables]
            table_referenced = any(tbl in query_lower for tbl in table_names)
            
            if not table_referenced and len(tables) > 1:
                table_list = ", ".join([t.get('name') for t in tables[:5]])
                questions.append(f"Which data source? Available: {table_list}")
        
        # Use LLM for better clarifications if available
        if self.llm_provider and len(questions) < max_questions:
            try:
                llm_questions = self._get_llm_clarifications(query, schema_metadata)
                for q in llm_questions:
                    if q not in questions:
                        questions.append(q)
            except Exception as e:
                logger.warning(f"LLM clarification failed: {e}")
        
        return questions[:max_questions]
    
    async def get_clarifications_async(
        self,
        query: str,
        schema_metadata: Dict[str, Any],
        max_questions: int = 3
    ) -> List[str]:
        """Async version of get_clarifications."""
        # For now, delegate to sync version
        # In future, could use async LLM calls
        return self.get_clarifications(query, schema_metadata, max_questions)
    
    def _get_llm_clarifications(
        self,
        query: str,
        schema_metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Use LLM to generate smart clarification questions.
        """
        if not self.llm_provider:
            return []
        
        # Build schema summary
        tables_summary = []
        for table in schema_metadata.get('tables', [])[:5]:
            cols = [c.get('name') for c in table.get('columns', [])[:5]]
            tables_summary.append(f"{table.get('name')}: {', '.join(cols)}")
        
        prompt = f"""Given this database query and schema, generate 1-2 clarification questions to help understand what the user really wants.

Query: "{query}"

Available tables:
{chr(10).join(tables_summary)}

Generate concise, helpful clarification questions. Focus on:
- Time period if not specified
- Grouping/aggregation preferences
- Specific filters needed

Return only the questions, one per line. No explanations."""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_provider.generate_response_sync(messages)
            
            # Parse response into questions
            questions = []
            for line in response.content.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove numbering if present
                    if line[0].isdigit() and '.' in line[:3]:
                        line = line.split('.', 1)[1].strip()
                    if line.endswith('?'):
                        questions.append(line)
            
            return questions[:2]
            
        except Exception as e:
            logger.error(f"LLM clarification error: {e}")
            return []
    
    def apply_clarification(
        self,
        original_query: str,
        clarification_answers: Dict[str, str]
    ) -> str:
        """
        Apply user's clarification answers to improve the query.
        
        Args:
            original_query: The original ambiguous query
            clarification_answers: Dict of question -> answer pairs
            
        Returns:
            Improved query with clarifications applied
        """
        enhanced_query = original_query
        
        for question, answer in clarification_answers.items():
            # Add context based on the clarification
            if 'time' in question.lower() or 'period' in question.lower() or 'date' in question.lower():
                enhanced_query = f"{enhanced_query} for {answer}"
            elif 'group' in question.lower() or 'breakdown' in question.lower():
                enhanced_query = f"{enhanced_query} grouped by {answer}"
            elif 'filter' in question.lower() or 'condition' in question.lower():
                enhanced_query = f"{enhanced_query} where {answer}"
            elif 'table' in question.lower() or 'source' in question.lower() or 'data' in question.lower():
                enhanced_query = f"{enhanced_query} from {answer}"
            else:
                # Generic append
                enhanced_query = f"{enhanced_query} ({answer})"
        
        return enhanced_query

