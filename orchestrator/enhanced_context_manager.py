
"""
Enhanced Context Manager with Document Integration
================================================

This module extends the existing context manager to integrate with the document
management system, providing better chunking, source attribution, and relevance scoring.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
import psycopg2
from psycopg2.extras import RealDictCursor

from document_manager import DocumentManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContextResult:
    """Enhanced context result with source attribution"""
    content: str
    source_document: str
    document_id: int
    chunk_id: int
    relevance_score: float
    metadata: Dict[str, Any]
    file_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'source_document': self.source_document,
            'document_id': self.document_id,
            'chunk_id': self.chunk_id,
            'relevance_score': self.relevance_score,
            'metadata': self.metadata,
            'file_type': self.file_type
        }

@dataclass
class ContextQuery:
    """Context query with enhanced parameters"""
    query: str
    max_results: int = 10
    min_relevance: float = 0.7
    file_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    document_ids: Optional[List[int]] = None
    include_metadata: bool = True

class EnhancedContextManager:
    """Enhanced context manager with document integration"""
    
    def __init__(self, db_config: Dict[str, str], openai_api_key: str):
        self.db_config = db_config
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # self.document_manager = DocumentManager(db_config, openai_api_key)
        self.document_manager = None  # Temporarily disabled due to pgvector dependency
        
        # Context configuration
        self.default_chunk_size = 1000
        self.default_overlap = 200
        self.similarity_threshold = 0.7
        self.max_context_length = 8000
        
        # Analytics tracking
        self.usage_stats = {
            'total_queries': 0,
            'successful_retrievals': 0,
            'average_relevance': 0.0
        }
    
    async def retrieve_context(self, query: ContextQuery) -> List[ContextResult]:
        """Retrieve relevant context with enhanced filtering and scoring"""
        try:
            self.usage_stats['total_queries'] += 1
            
            # Generate query embedding
            query_embedding = await self.embeddings.aembed_query(query.query)
            
            # Build SQL query with filters
            sql_query, params = self._build_context_query(query, query_embedding)
            
            # Execute query
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(sql_query, params)
            results = cursor.fetchall()
            
            # Process results
            context_results = []
            total_relevance = 0.0
            
            for result in results:
                relevance_score = float(result['similarity'])
                
                if relevance_score >= query.min_relevance:
                    context_result = ContextResult(
                        content=result['content'],
                        source_document=result['filename'],
                        document_id=result['document_id'],
                        chunk_id=result['chunk_id'],
                        relevance_score=relevance_score,
                        metadata=result['chunk_metadata'] if query.include_metadata else {},
                        file_type=result['file_type']
                    )
                    
                    context_results.append(context_result)
                    total_relevance += relevance_score
                    
                    # Track usage for analytics
                    await self._track_context_usage(
                        result['document_id'],
                        result['chunk_id'],
                        query.query,
                        relevance_score,
                        True
                    )
            
            # Update analytics
            if context_results:
                self.usage_stats['successful_retrievals'] += 1
                self.usage_stats['average_relevance'] = (
                    self.usage_stats['average_relevance'] + (total_relevance / len(context_results))
                ) / 2
            
            # Apply intelligent ranking
            context_results = self._rank_context_results(context_results, query)
            
            cursor.close()
            conn.close()
            
            logger.info(f"Retrieved {len(context_results)} relevant contexts for query")
            return context_results[:query.max_results]
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise
    
    def _build_context_query(self, query: ContextQuery, query_embedding: List[float]) -> Tuple[str, List]:
        """Build SQL query with filters"""
        base_query = """
            SELECT 
                dc.id as chunk_id,
                dc.document_id,
                dc.content,
                dc.metadata as chunk_metadata,
                d.filename,
                d.file_type,
                d.tags,
                1 - (dc.embedding <=> %s::vector) as similarity
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.status = 'completed'
        """
        
        params = [query_embedding]
        conditions = []
        
        # Filter by file types
        if query.file_types:
            conditions.append(f"d.file_type = ANY(%s)")
            params.append(query.file_types)
        
        # Filter by tags
        if query.tags:
            conditions.append(f"d.tags && %s")
            params.append(query.tags)
        
        # Filter by specific documents
        if query.document_ids:
            conditions.append(f"d.id = ANY(%s)")
            params.append(query.document_ids)
        
        # Add conditions to query
        if conditions:
            base_query += " AND " + " AND ".join(conditions)
        
        # Order by similarity and limit
        base_query += f"""
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s
        """
        params.extend([query_embedding, query.max_results * 2])  # Get more for filtering
        
        return base_query, params
    
    def _rank_context_results(self, results: List[ContextResult], query: ContextQuery) -> List[ContextResult]:
        """Apply intelligent ranking to context results"""
        if not results:
            return results
        
        # Enhanced ranking factors
        for result in results:
            base_score = result.relevance_score
            
            # Boost score based on file type relevance
            file_type_boost = self._get_file_type_boost(result.file_type, query.query)
            
            # Boost score based on content length (prefer substantial content)
            length_boost = min(len(result.content) / 1000, 1.0) * 0.1
            
            # Boost score based on metadata richness
            metadata_boost = len(result.metadata) * 0.02
            
            # Calculate final score
            final_score = base_score + file_type_boost + length_boost + metadata_boost
            result.relevance_score = min(final_score, 1.0)
        
        # Sort by enhanced relevance score
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    def _get_file_type_boost(self, file_type: str, query: str) -> float:
        """Get relevance boost based on file type and query content"""
        query_lower = query.lower()
        
        # Boost for specific file types based on query content
        if 'code' in query_lower or 'function' in query_lower or 'implementation' in query_lower:
            if file_type == 'py':
                return 0.2
        
        if 'process' in query_lower or 'procedure' in query_lower or 'workflow' in query_lower:
            if file_type in ['md', 'docx']:
                return 0.15
        
        if 'documentation' in query_lower or 'guide' in query_lower:
            if file_type in ['md', 'txt']:
                return 0.1
        
        return 0.0
    
    async def _track_context_usage(self, document_id: int, chunk_id: int, 
                                 query: str, relevance_score: float, used: bool):
        """Track context usage for analytics"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO context_usage 
                (document_id, chunk_id, query_text, relevance_score, used_in_response, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                document_id, chunk_id, query, relevance_score, used, datetime.now()
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Failed to track context usage: {e}")
    
    async def get_context_for_task(self, task_description: str, 
                                 task_type: Optional[str] = None,
                                 max_context_length: Optional[int] = None) -> Dict[str, Any]:
        """Get optimized context for a specific task"""
        try:
            max_length = max_context_length or self.max_context_length
            
            # Create enhanced query based on task
            query = ContextQuery(
                query=task_description,
                max_results=15,  # Get more results for better selection
                min_relevance=0.6,  # Lower threshold for task context
                file_types=self._get_relevant_file_types(task_type),
                include_metadata=True
            )
            
            # Retrieve context
            context_results = await self.retrieve_context(query)
            
            # Optimize context selection for length constraints
            selected_contexts = self._optimize_context_selection(context_results, max_length)
            
            # Format context for consumption
            formatted_context = self._format_context_for_task(selected_contexts, task_description)
            
            return {
                'context': formatted_context,
                'sources': [result.to_dict() for result in selected_contexts],
                'total_sources': len(context_results),
                'selected_sources': len(selected_contexts),
                'average_relevance': np.mean([r.relevance_score for r in selected_contexts]) if selected_contexts else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting context for task: {e}")
            raise
    
    def _get_relevant_file_types(self, task_type: Optional[str]) -> Optional[List[str]]:
        """Get relevant file types based on task type"""
        if not task_type:
            return None
        
        task_type_lower = task_type.lower()
        
        if 'code' in task_type_lower or 'development' in task_type_lower:
            return ['py', 'md']
        elif 'documentation' in task_type_lower or 'analysis' in task_type_lower:
            return ['md', 'txt', 'docx']
        elif 'process' in task_type_lower or 'workflow' in task_type_lower:
            return ['md', 'docx', 'pdf']
        
        return None
    
    def _optimize_context_selection(self, results: List[ContextResult], 
                                  max_length: int) -> List[ContextResult]:
        """Optimize context selection based on length constraints and relevance"""
        if not results:
            return []
        
        selected = []
        total_length = 0
        
        # Sort by relevance score
        sorted_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        
        for result in sorted_results:
            content_length = len(result.content)
            
            # Check if adding this context would exceed the limit
            if total_length + content_length <= max_length:
                selected.append(result)
                total_length += content_length
            else:
                # Try to fit a truncated version if it's highly relevant
                if result.relevance_score > 0.9 and len(selected) < 3:
                    remaining_space = max_length - total_length
                    if remaining_space > 200:  # Minimum useful chunk size
                        truncated_result = ContextResult(
                            content=result.content[:remaining_space-3] + "...",
                            source_document=result.source_document,
                            document_id=result.document_id,
                            chunk_id=result.chunk_id,
                            relevance_score=result.relevance_score,
                            metadata=result.metadata,
                            file_type=result.file_type
                        )
                        selected.append(truncated_result)
                        break
        
        return selected
    
    def _format_context_for_task(self, contexts: List[ContextResult], 
                               task_description: str) -> str:
        """Format context for task consumption with source attribution"""
        if not contexts:
            return "No relevant context found."
        
        formatted_parts = []
        formatted_parts.append(f"# Relevant Context for: {task_description}\n")
        
        for i, context in enumerate(contexts, 1):
            formatted_parts.append(f"## Source {i}: {context.source_document}")
            formatted_parts.append(f"**File Type:** {context.file_type}")
            formatted_parts.append(f"**Relevance:** {context.relevance_score:.2f}")
            
            if context.metadata:
                formatted_parts.append(f"**Metadata:** {json.dumps(context.metadata, indent=2)}")
            
            formatted_parts.append(f"**Content:**\n{context.content}")
            formatted_parts.append("---\n")
        
        return "\n".join(formatted_parts)
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get context usage analytics"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Most used documents
            cursor.execute("""
                SELECT d.filename, d.file_type, COUNT(*) as usage_count,
                       AVG(cu.relevance_score) as avg_relevance
                FROM context_usage cu
                JOIN documents d ON cu.document_id = d.id
                WHERE cu.timestamp >= NOW() - INTERVAL '30 days'
                GROUP BY d.id, d.filename, d.file_type
                ORDER BY usage_count DESC
                LIMIT 10
            """)
            most_used_docs = cursor.fetchall()
            
            # Query patterns
            cursor.execute("""
                SELECT query_text, COUNT(*) as frequency,
                       AVG(relevance_score) as avg_relevance
                FROM context_usage
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY query_text
                ORDER BY frequency DESC
                LIMIT 10
            """)
            query_patterns = cursor.fetchall()
            
            # Performance metrics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(relevance_score) as avg_relevance,
                    COUNT(CASE WHEN used_in_response THEN 1 END) as successful_retrievals
                FROM context_usage
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
            """)
            performance = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return {
                'usage_stats': dict(self.usage_stats),
                'most_used_documents': [dict(doc) for doc in most_used_docs],
                'query_patterns': [dict(pattern) for pattern in query_patterns],
                'performance_24h': dict(performance) if performance else {},
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            raise
    
    async def update_configuration(self, config: Dict[str, Any]):
        """Update context manager configuration"""
        try:
            if 'chunk_size' in config:
                self.default_chunk_size = config['chunk_size']
            
            if 'overlap' in config:
                self.default_overlap = config['overlap']
            
            if 'similarity_threshold' in config:
                self.similarity_threshold = config['similarity_threshold']
            
            if 'max_context_length' in config:
                self.max_context_length = config['max_context_length']
            
            logger.info("Context manager configuration updated")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            raise

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_enhanced_context():
        db_config = {
            'host': 'localhost',
            'database': 'orchestrator_db',
            'user': 'postgres',
            'password': 'your_password'
        }
        
        context_manager = EnhancedContextManager(db_config, "your_openai_api_key")
        
        # Test context retrieval
        query = ContextQuery(
            query="How to implement user authentication in a web application?",
            max_results=5,
            min_relevance=0.7,
            file_types=['py', 'md']
        )
        
        results = await context_manager.retrieve_context(query)
        
        for result in results:
            print(f"Source: {result.source_document}")
            print(f"Relevance: {result.relevance_score:.2f}")
            print(f"Content: {result.content[:200]}...")
            print("---")
    
    # asyncio.run(test_enhanced_context())
