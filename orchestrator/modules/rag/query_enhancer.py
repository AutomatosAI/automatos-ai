"""
Query Enhancement Module
========================

Implements advanced query enhancement techniques from Context-Engineering research:
- HyDE (Hypothetical Document Embeddings)
- Query Decomposition
- Cognitive-style query expansion

Based on IBM Zurich research showing 16.6% improvement in retrieval accuracy.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnhancedQuery:
    """Result of query enhancement"""
    original_query: str
    enhanced_queries: List[str]
    hypothetical_doc: Optional[str] = None
    sub_queries: List[str] = None
    key_concepts: List[str] = None
    
    def get_all_queries(self) -> List[str]:
        """Get all query variations for multi-query retrieval"""
        queries = [self.original_query]
        queries.extend(self.enhanced_queries)
        if self.sub_queries:
            queries.extend(self.sub_queries)
        return list(set(queries))  # Deduplicate


class QueryEnhancer:
    """
    Advanced query enhancement using cognitive tool patterns.
    
    Implements:
    1. HyDE - Generate hypothetical document, embed that instead
    2. Query Decomposition - Break complex queries into sub-queries
    3. Concept Extraction - Identify key concepts for targeted retrieval
    """
    
    def __init__(self, llm_provider=None):
        self._llm = llm_provider
        self._embedding_manager = None
        
    def _get_llm(self):
        """Lazy initialization of LLM"""
        if self._llm is None:
            try:
                from core.llm import create_llm_manager
                self._llm = create_llm_manager(service_name="rag")
            except Exception as e:
                logger.warning(f"LLM not available for query enhancement: {e}")
        return self._llm
    
    def _get_embedding_manager(self):
        """Lazy initialization of embedding manager"""
        if self._embedding_manager is None:
            try:
                from core.llm import create_embedding_manager
                self._embedding_manager = create_embedding_manager()
            except Exception as e:
                logger.warning(f"Embedding manager not available: {e}")
        return self._embedding_manager
    
    async def enhance_query(
        self,
        query: str,
        use_hyde: bool = True,
        use_decomposition: bool = True,
        use_expansion: bool = True
    ) -> EnhancedQuery:
        """
        Enhance a query using multiple techniques.
        
        Args:
            query: Original user query
            use_hyde: Generate hypothetical document embedding
            use_decomposition: Break into sub-queries
            use_expansion: Add synonyms and related terms
            
        Returns:
            EnhancedQuery with multiple query variations
        """
        enhanced_queries = []
        hypothetical_doc = None
        sub_queries = []
        key_concepts = []
        
        llm = self._get_llm()
        
        if llm:
            # Extract key concepts (fast, always do this)
            key_concepts = await self._extract_concepts(query, llm)
            
            if use_hyde:
                hypothetical_doc = await self._generate_hypothetical_doc(query, llm)
                if hypothetical_doc:
                    enhanced_queries.append(hypothetical_doc[:500])  # Truncate for embedding
            
            if use_decomposition:
                sub_queries = await self._decompose_query(query, llm)
                
            if use_expansion:
                expanded = await self._expand_query(query, key_concepts, llm)
                enhanced_queries.extend(expanded)
        else:
            # Fallback: simple keyword extraction
            key_concepts = self._extract_keywords(query)
        
        return EnhancedQuery(
            original_query=query,
            enhanced_queries=enhanced_queries,
            hypothetical_doc=hypothetical_doc,
            sub_queries=sub_queries,
            key_concepts=key_concepts
        )
    
    async def _generate_hypothetical_doc(self, query: str, llm) -> Optional[str]:
        """
        HyDE: Generate a hypothetical document that would answer the query.
        
        The embedding of this document often retrieves better results than
        the query embedding alone.
        """
        try:
            prompt = f"""Given this question, write a short paragraph (2-3 sentences) that would 
directly answer it. Write as if this is an excerpt from a document containing the answer.

Question: {query}

Document excerpt:"""
            
            messages = [{"role": "user", "content": prompt}]
            response_obj = await llm.generate_response(messages)
            return response_obj.content.strip()
        except Exception as e:
            logger.debug(f"HyDE generation failed: {e}")
            return None
    
    async def _decompose_query(self, query: str, llm) -> List[str]:
        """
        Break a complex query into simpler sub-queries.
        
        Based on cognitive tool patterns from IBM Zurich research.
        """
        try:
            prompt = f"""Break this complex question into 2-3 simpler, specific sub-questions.
Each sub-question should target a specific aspect of the main question.
Return only the sub-questions, one per line.

Main question: {query}

Sub-questions:"""
            
            messages = [{"role": "user", "content": prompt}]
            response_obj = await llm.generate_response(messages)
            response = response_obj.content
            
            # Parse sub-questions
            sub_queries = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove numbering like "1." or "- "
                if line and len(line) > 5:
                    cleaned = line.lstrip('0123456789.-) ').strip()
                    if cleaned and cleaned != query:
                        sub_queries.append(cleaned)
            
            return sub_queries[:3]  # Max 3 sub-queries
        except Exception as e:
            logger.debug(f"Query decomposition failed: {e}")
            return []
    
    async def _extract_concepts(self, query: str, llm) -> List[str]:
        """Extract key concepts from the query for targeted retrieval."""
        try:
            prompt = f"""Extract 3-5 key concepts or technical terms from this query.
Return only the terms, comma-separated.

Query: {query}

Key concepts:"""
            
            messages = [{"role": "user", "content": prompt}]
            response_obj = await llm.generate_response(messages)
            response = response_obj.content
            
            concepts = [c.strip() for c in response.split(',')]
            return [c for c in concepts if c and len(c) > 2][:5]
        except Exception as e:
            logger.debug(f"Concept extraction failed: {e}")
            return self._extract_keywords(query)
    
    async def _expand_query(self, query: str, concepts: List[str], llm) -> List[str]:
        """Expand query with synonyms and related terms."""
        try:
            prompt = f"""Given this query and its key concepts, generate 2 alternative phrasings.
Each alternative should capture the same intent but use different words.

Query: {query}
Key concepts: {', '.join(concepts)}

Alternative phrasings (one per line):"""
            
            messages = [{"role": "user", "content": prompt}]
            response_obj = await llm.generate_response(messages)
            response = response_obj.content
            
            alternatives = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and len(line) > 10 and line != query:
                    cleaned = line.lstrip('0123456789.-) ').strip()
                    alternatives.append(cleaned)
            
            return alternatives[:2]
        except Exception as e:
            logger.debug(f"Query expansion failed: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Simple keyword extraction fallback."""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'under', 'again', 'further', 'then', 'once',
                      'here', 'there', 'when', 'where', 'why', 'how', 'all',
                      'each', 'few', 'more', 'most', 'other', 'some', 'such',
                      'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                      'too', 'very', 's', 't', 'just', 'don', 'now', 'what'}
        
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords[:5]


# Factory function
def create_query_enhancer(llm_provider=None) -> QueryEnhancer:
    """Create a query enhancer instance."""
    return QueryEnhancer(llm_provider=llm_provider)
