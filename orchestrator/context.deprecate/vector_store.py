
"""
Vector Store with pgvector Integration
=====================================

Advanced vector storage and retrieval using PostgreSQL with pgvector extension.
"""

import asyncio
import asyncpg
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import os

logger = logging.getLogger(__name__)

@dataclass
class VectorSearchResult:
    """Result from vector similarity search"""
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    source_file: str
    chunk_index: int
    content_type: str

class PgVectorStore:
    """PostgreSQL vector store with pgvector extension"""
    
    def __init__(self, connection_string: str = None, table_name: str = "document_embeddings"):
        self.connection_string = connection_string or os.getenv('DATABASE_URL')
        self.table_name = table_name
        self.pool = None
        self.dimension = None
    
    async def initialize(self, dimension: int = 384):
        """Initialize the vector store and create necessary tables"""
        self.dimension = dimension
        
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            # Create tables and indexes
            await self._create_tables()
            await self._create_indexes()
            
            logger.info(f"Initialized PgVectorStore with dimension {dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    async def _create_tables(self):
        """Create necessary tables for vector storage"""
        
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create main embeddings table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({self.dimension}) NOT NULL,
                    metadata JSONB DEFAULT '{{}}',
                    source_file TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content_type TEXT DEFAULT 'text',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create documents table for tracking source documents
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    file_path TEXT UNIQUE NOT NULL,
                    file_name TEXT NOT NULL,
                    file_size BIGINT,
                    content_type TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_count INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}'
                );
            """)
            
            # Create context patterns table for learning
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS context_patterns (
                    id SERIAL PRIMARY KEY,
                    pattern_name TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    embedding vector({}) NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    usage_count INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """.format(self.dimension))
            
            # Create historical tasks table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_tasks (
                    id SERIAL PRIMARY KEY,
                    task_description TEXT NOT NULL,
                    task_type TEXT,
                    context_used JSONB DEFAULT '{}',
                    outcome TEXT,
                    success BOOLEAN DEFAULT FALSE,
                    execution_time FLOAT,
                    agent_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}'
                );
            """)
    
    async def _create_indexes(self):
        """Create indexes for efficient querying"""
        
        async with self.pool.acquire() as conn:
            # Vector similarity index (HNSW for fast approximate search)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                ON {self.table_name} USING hnsw (embedding vector_cosine_ops);
            """)
            
            # Metadata indexes
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_source_file_idx 
                ON {self.table_name} (source_file);
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_content_type_idx 
                ON {self.table_name} (content_type);
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_idx 
                ON {self.table_name} USING GIN (metadata);
            """)
            
            # Documents table indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS documents_file_path_idx 
                ON documents (file_path);
            """)
            
            # Context patterns indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS context_patterns_embedding_idx 
                ON context_patterns USING hnsw (embedding vector_cosine_ops);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS context_patterns_type_idx 
                ON context_patterns (pattern_type);
            """)
    
    async def add_embeddings(self, embeddings: List[Dict[str, Any]]) -> bool:
        """
        Add embeddings to the vector store
        
        Args:
            embeddings: List of embedding dictionaries
            
        Returns:
            Success status
        """
        if not embeddings:
            return True
        
        try:
            async with self.pool.acquire() as conn:
                # Prepare data for batch insert
                values = []
                for emb in embeddings:
                    values.append((
                        emb.get('id'),
                        emb.get('text', emb.get('content', '')),
                        emb.get('embedding'),
                        json.dumps(emb.get('metadata', {})),
                        emb.get('source_file', ''),
                        emb.get('chunk_index', 0),
                        emb.get('content_type', 'text')
                    ))
                
                # Batch insert
                await conn.executemany(f"""
                    INSERT INTO {self.table_name} 
                    (id, content, embedding, metadata, source_file, chunk_index, content_type)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP;
                """, values)
                
                logger.info(f"Added {len(embeddings)} embeddings to vector store")
                return True
                
        except Exception as e:
            logger.error(f"Error adding embeddings: {str(e)}")
            return False
    
    async def similarity_search(self, 
                              query_embedding: List[float],
                              limit: int = 10,
                              similarity_threshold: float = 0.0,
                              filters: Dict[str, Any] = None) -> List[VectorSearchResult]:
        """
        Perform similarity search
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            filters: Additional filters (content_type, source_file, etc.)
            
        Returns:
            List of search results
        """
        try:
            async with self.pool.acquire() as conn:
                # Build query with optional filters
                where_conditions = ["1 - (embedding <=> $1) >= $3"]
                params = [query_embedding, limit, similarity_threshold]
                param_count = 3
                
                if filters:
                    if 'content_type' in filters:
                        param_count += 1
                        where_conditions.append(f"content_type = ${param_count}")
                        params.append(filters['content_type'])
                    
                    if 'source_file' in filters:
                        param_count += 1
                        where_conditions.append(f"source_file = ${param_count}")
                        params.append(filters['source_file'])
                    
                    if 'metadata' in filters:
                        for key, value in filters['metadata'].items():
                            param_count += 1
                            where_conditions.append(f"metadata->>${param_count-1} = ${param_count}")
                            params.extend([key, str(value)])
                            param_count += 1
                
                where_clause = " AND ".join(where_conditions)
                
                query = f"""
                    SELECT 
                        id, content, metadata, source_file, chunk_index, content_type,
                        1 - (embedding <=> $1) as similarity_score
                    FROM {self.table_name}
                    WHERE {where_clause}
                    ORDER BY embedding <=> $1
                    LIMIT $2;
                """
                
                rows = await conn.fetch(query, *params)
                
                results = []
                for row in rows:
                    result = VectorSearchResult(
                        id=row['id'],
                        content=row['content'],
                        metadata=row['metadata'] or {},
                        similarity_score=float(row['similarity_score']),
                        source_file=row['source_file'],
                        chunk_index=row['chunk_index'],
                        content_type=row['content_type']
                    )
                    results.append(result)
                
                logger.info(f"Found {len(results)} similar documents")
                return results
                
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    async def hybrid_search(self,
                          query_embedding: List[float],
                          query_text: str,
                          limit: int = 10,
                          vector_weight: float = 0.7,
                          text_weight: float = 0.3) -> List[VectorSearchResult]:
        """
        Hybrid search combining vector similarity and text search
        
        Args:
            query_embedding: Query vector
            query_text: Query text for full-text search
            limit: Maximum results
            vector_weight: Weight for vector similarity
            text_weight: Weight for text similarity
            
        Returns:
            Ranked search results
        """
        try:
            async with self.pool.acquire() as conn:
                query = f"""
                    SELECT 
                        id, content, metadata, source_file, chunk_index, content_type,
                        (1 - (embedding <=> $1)) * $3 + 
                        ts_rank(to_tsvector('english', content), plainto_tsquery('english', $2)) * $4 as combined_score,
                        1 - (embedding <=> $1) as vector_similarity,
                        ts_rank(to_tsvector('english', content), plainto_tsquery('english', $2)) as text_similarity
                    FROM {self.table_name}
                    WHERE 
                        to_tsvector('english', content) @@ plainto_tsquery('english', $2)
                        OR (1 - (embedding <=> $1)) > 0.1
                    ORDER BY combined_score DESC
                    LIMIT $5;
                """
                
                rows = await conn.fetch(query, query_embedding, query_text, 
                                      vector_weight, text_weight, limit)
                
                results = []
                for row in rows:
                    result = VectorSearchResult(
                        id=row['id'],
                        content=row['content'],
                        metadata=row['metadata'] or {},
                        similarity_score=float(row['combined_score']),
                        source_file=row['source_file'],
                        chunk_index=row['chunk_index'],
                        content_type=row['content_type']
                    )
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            # Fallback to vector search only
            return await self.similarity_search(query_embedding, limit)
    
    async def add_document_record(self, file_path: str, metadata: Dict[str, Any] = None) -> int:
        """Add a document record and return its ID"""
        try:
            async with self.pool.acquire() as conn:
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                
                result = await conn.fetchrow("""
                    INSERT INTO documents (file_path, file_name, file_size, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (file_path) DO UPDATE SET
                        processed_at = CURRENT_TIMESTAMP,
                        metadata = EXCLUDED.metadata
                    RETURNING id;
                """, file_path, file_name, file_size, json.dumps(metadata or {}))
                
                return result['id']
                
        except Exception as e:
            logger.error(f"Error adding document record: {str(e)}")
            return None
    
    async def update_document_chunk_count(self, file_path: str, chunk_count: int):
        """Update the chunk count for a document"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE documents 
                    SET chunk_count = $2 
                    WHERE file_path = $1;
                """, file_path, chunk_count)
                
        except Exception as e:
            logger.error(f"Error updating chunk count: {str(e)}")
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents"""
        try:
            async with self.pool.acquire() as conn:
                stats = await conn.fetchrow(f"""
                    SELECT 
                        COUNT(*) as total_chunks,
                        COUNT(DISTINCT source_file) as total_documents,
                        AVG(LENGTH(content)) as avg_chunk_length,
                        COUNT(DISTINCT content_type) as content_types
                    FROM {self.table_name};
                """)
                
                content_type_stats = await conn.fetch(f"""
                    SELECT content_type, COUNT(*) as count
                    FROM {self.table_name}
                    GROUP BY content_type
                    ORDER BY count DESC;
                """)
                
                return {
                    'total_chunks': stats['total_chunks'],
                    'total_documents': stats['total_documents'],
                    'avg_chunk_length': float(stats['avg_chunk_length'] or 0),
                    'content_types': stats['content_types'],
                    'content_type_distribution': {
                        row['content_type']: row['count'] 
                        for row in content_type_stats
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            return {}
    
    async def delete_document(self, source_file: str) -> bool:
        """Delete all chunks for a specific document"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(f"""
                    DELETE FROM {self.table_name} WHERE source_file = $1;
                """, source_file)
                
                await conn.execute("""
                    DELETE FROM documents WHERE file_path = $1;
                """, source_file)
                
                logger.info(f"Deleted document: {source_file}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    async def close(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Closed vector store connection pool")

# Context-aware vector operations
class ContextAwareVectorStore(PgVectorStore):
    """Extended vector store with context awareness"""
    
    async def add_context_pattern(self, pattern_name: str, pattern_type: str,
                                embedding: List[float], metadata: Dict[str, Any] = None):
        """Add a context pattern for learning"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO context_patterns 
                    (pattern_name, pattern_type, embedding, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (pattern_name) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP;
                """, pattern_name, pattern_type, embedding, json.dumps(metadata or {}))
                
        except Exception as e:
            logger.error(f"Error adding context pattern: {str(e)}")
    
    async def find_similar_patterns(self, query_embedding: List[float], 
                                  pattern_type: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar context patterns"""
        try:
            async with self.pool.acquire() as conn:
                where_clause = ""
                params = [query_embedding, limit]
                
                if pattern_type:
                    where_clause = "WHERE pattern_type = $3"
                    params.append(pattern_type)
                
                query = f"""
                    SELECT 
                        pattern_name, pattern_type, metadata,
                        success_count, usage_count,
                        1 - (embedding <=> $1) as similarity
                    FROM context_patterns
                    {where_clause}
                    ORDER BY embedding <=> $1
                    LIMIT $2;
                """
                
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error finding similar patterns: {str(e)}")
            return []
    
    async def record_task_outcome(self, task_description: str, task_type: str,
                                context_used: Dict[str, Any], success: bool,
                                execution_time: float = None, agent_used: str = None):
        """Record task execution outcome for learning"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO historical_tasks 
                    (task_description, task_type, context_used, success, execution_time, agent_used)
                    VALUES ($1, $2, $3, $4, $5, $6);
                """, task_description, task_type, json.dumps(context_used), 
                success, execution_time, agent_used)
                
        except Exception as e:
            logger.error(f"Error recording task outcome: {str(e)}")
