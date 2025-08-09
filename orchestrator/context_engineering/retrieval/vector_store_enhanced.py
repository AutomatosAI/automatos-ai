
"""
Enhanced Vector Store with pgvector Integration
===============================================

Advanced vector storage and retrieval system with PostgreSQL + pgvector backend.
Supports multiple embedding models, hybrid search, and intelligent ranking.
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncpg
import numpy as np
from datetime import datetime

# Import mathematical foundations
from ..mathematical_foundations.distance_metrics import DistanceMetrics
from ..mathematical_foundations.vector_operations import VectorOperations
from ..mathematical_foundations.statistical_analysis import StatisticalAnalysis

logger = logging.getLogger(__name__)

class SearchMode(Enum):
    """Available search modes"""
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"  # Vector + keyword
    SEMANTIC = "semantic"  # Advanced semantic search
    CONTEXTUAL = "contextual"  # Context-aware search

class RankingStrategy(Enum):
    """Ranking strategies for search results"""
    SIMILARITY = "similarity"
    RELEVANCE = "relevance"  # Combines multiple signals
    IMPORTANCE = "importance"  # Based on content importance
    RECENCY = "recency"  # Time-based ranking
    HYBRID_SCORE = "hybrid_score"  # Combination of all factors

@dataclass
class VectorDocument:
    """A document with vector representation"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime
    source: str
    document_type: str
    importance_score: float = 0.0
    
class SearchFilter:
    """Filters for vector search"""
    def __init__(
        self,
        document_types: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        min_importance: Optional[float] = None
    ):
        self.document_types = document_types or []
        self.sources = sources or []
        self.date_range = date_range
        self.metadata_filters = metadata_filters or {}
        self.min_importance = min_importance

@dataclass
class SearchResult:
    """A search result with scoring information"""
    document: VectorDocument
    similarity_score: float
    relevance_score: float
    importance_score: float
    final_score: float
    rank: int
    explanation: Dict[str, Any]  # Why this result was selected

class EnhancedVectorStore:
    """Enhanced vector store with advanced retrieval capabilities"""
    
    def __init__(
        self,
        database_url: str,
        embedding_dimension: int = 768,  # Default for many models
        similarity_function: str = "cosine",  # cosine, l2, inner_product
        table_name: str = "vector_documents"
    ):
        self.database_url = database_url
        self.embedding_dimension = embedding_dimension
        self.similarity_function = similarity_function
        self.table_name = table_name
        
        # Initialize mathematical components
        self.distance_metrics = DistanceMetrics()
        self.vector_ops = VectorOperations()
        self.stats = StatisticalAnalysis()
        
        # Connection pool
        self.pool: Optional[asyncpg.Pool] = None
        
        # Index strategies
        self.index_strategies = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops", 
            "inner_product": "vector_ip_ops"
        }
    
    async def initialize(self) -> None:
        """Initialize the vector store with database setup"""
        
        # Create connection pool
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=1,
            max_size=10,
            command_timeout=60
        )
        
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create the main table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({self.embedding_dimension}) NOT NULL,
                    metadata JSONB DEFAULT '{{}}',
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    source TEXT,
                    document_type TEXT,
                    importance_score FLOAT DEFAULT 0.0,
                    content_hash TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create indexes for performance
            await self._create_indexes(conn)
            
            # Create additional tables for advanced features
            await self._create_auxiliary_tables(conn)
        
        logger.info(f"Enhanced vector store initialized with dimension {self.embedding_dimension}")
    
    async def _create_indexes(self, conn: asyncpg.Connection) -> None:
        """Create optimized indexes for vector operations"""
        
        index_type = self.index_strategies.get(self.similarity_function, "vector_cosine_ops")
        
        # Vector index for similarity search
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
            ON {self.table_name} 
            USING ivfflat (embedding {index_type})
            WITH (lists = 100);
        """)
        
        # Metadata indexes
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_idx 
            ON {self.table_name} USING GIN (metadata);
        """)
        
        # Standard indexes
        indexes = [
            f"CREATE INDEX IF NOT EXISTS {self.table_name}_timestamp_idx ON {self.table_name} (timestamp DESC);",
            f"CREATE INDEX IF NOT EXISTS {self.table_name}_source_idx ON {self.table_name} (source);",
            f"CREATE INDEX IF NOT EXISTS {self.table_name}_document_type_idx ON {self.table_name} (document_type);",
            f"CREATE INDEX IF NOT EXISTS {self.table_name}_importance_idx ON {self.table_name} (importance_score DESC);",
        ]
        
        for index_query in indexes:
            await conn.execute(index_query)
    
    async def _create_auxiliary_tables(self, conn: asyncpg.Connection) -> None:
        """Create auxiliary tables for advanced features"""
        
        # Search analytics table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS search_analytics (
                id SERIAL PRIMARY KEY,
                query_text TEXT,
                query_embedding vector(768),
                results_count INTEGER,
                search_mode TEXT,
                ranking_strategy TEXT,
                execution_time_ms INTEGER,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                user_feedback FLOAT  -- For learning from user interactions
            );
        """)
        
        # Document relationships table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS document_relationships (
                id SERIAL PRIMARY KEY,
                source_doc_id TEXT,
                target_doc_id TEXT,
                relationship_type TEXT,  -- 'similar', 'references', 'follows', etc.
                strength FLOAT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                FOREIGN KEY (source_doc_id) REFERENCES vector_documents(id) ON DELETE CASCADE,
                FOREIGN KEY (target_doc_id) REFERENCES vector_documents(id) ON DELETE CASCADE
            );
        """)
        
        # Create indexes for auxiliary tables
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS search_analytics_timestamp_idx 
            ON search_analytics (timestamp DESC);
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS doc_relationships_source_idx 
            ON document_relationships (source_doc_id);
        """)
    
    async def add_document(self, document: VectorDocument) -> bool:
        """Add a document to the vector store"""
        
        if not self.pool:
            raise RuntimeError("Vector store not initialized")
        
        try:
            async with self.pool.acquire() as conn:
                # Convert embedding to pgvector format
                embedding_str = f"[{','.join(map(str, document.embedding))}]"
                
                # Calculate content hash for deduplication
                content_hash = str(hash(document.content))
                
                await conn.execute(f"""
                    INSERT INTO {self.table_name} 
                    (id, content, embedding, metadata, timestamp, source, document_type, 
                     importance_score, content_hash, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                    ON CONFLICT (id) 
                    DO UPDATE SET 
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        importance_score = EXCLUDED.importance_score,
                        updated_at = NOW()
                """, 
                    document.id, document.content, embedding_str, 
                    json.dumps(document.metadata), document.timestamp,
                    document.source, document.document_type, 
                    document.importance_score, content_hash
                )
            
            logger.debug(f"Added document {document.id} to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {document.id}: {e}")
            return False
    
    async def search(
        self,
        query_embedding: List[float],
        mode: SearchMode = SearchMode.VECTOR_ONLY,
        ranking_strategy: RankingStrategy = RankingStrategy.SIMILARITY,
        limit: int = 10,
        search_filter: Optional[SearchFilter] = None,
        query_text: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform advanced vector search with multiple modes and ranking"""
        
        if not self.pool:
            raise RuntimeError("Vector store not initialized")
        
        start_time = datetime.now()
        
        try:
            async with self.pool.acquire() as conn:
                # Build the search query based on mode
                if mode == SearchMode.VECTOR_ONLY:
                    results = await self._vector_search(
                        conn, query_embedding, limit * 2, search_filter
                    )
                elif mode == SearchMode.HYBRID:
                    results = await self._hybrid_search(
                        conn, query_embedding, query_text, limit * 2, search_filter
                    )
                elif mode == SearchMode.SEMANTIC:
                    results = await self._semantic_search(
                        conn, query_embedding, query_text, limit * 2, search_filter
                    )
                elif mode == SearchMode.CONTEXTUAL:
                    results = await self._contextual_search(
                        conn, query_embedding, query_text, limit * 2, search_filter
                    )
                else:
                    results = await self._vector_search(
                        conn, query_embedding, limit * 2, search_filter
                    )
                
                # Apply ranking strategy
                ranked_results = await self._rank_results(
                    results, ranking_strategy, query_embedding, query_text
                )
                
                # Apply final limit
                final_results = ranked_results[:limit]
                
                # Record analytics
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                await self._record_search_analytics(
                    conn, query_text, query_embedding, len(final_results),
                    mode.value, ranking_strategy.value, int(execution_time)
                )
                
                return final_results
                
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return []
    
    async def _vector_search(
        self,
        conn: asyncpg.Connection,
        query_embedding: List[float],
        limit: int,
        search_filter: Optional[SearchFilter]
    ) -> List[VectorDocument]:
        """Pure vector similarity search"""
        
        embedding_str = f"[{','.join(map(str, query_embedding))}]"
        
        # Build filter conditions
        where_conditions, params = self._build_filter_conditions(search_filter)
        where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
        
        # Select similarity function
        similarity_op = self._get_similarity_operator()
        
        query = f"""
            SELECT id, content, embedding, metadata, timestamp, source, 
                   document_type, importance_score,
                   (embedding {similarity_op} $1) as similarity
            FROM {self.table_name}
            {where_clause}
            ORDER BY embedding {similarity_op} $1
            LIMIT {limit}
        """
        
        # Add embedding parameter
        params.insert(0, embedding_str)
        
        rows = await conn.fetch(query, *params)
        return [self._row_to_document(row) for row in rows]
    
    async def _hybrid_search(
        self,
        conn: asyncpg.Connection,
        query_embedding: List[float],
        query_text: str,
        limit: int,
        search_filter: Optional[SearchFilter]
    ) -> List[VectorDocument]:
        """Hybrid search combining vector similarity and text search"""
        
        embedding_str = f"[{','.join(map(str, query_embedding))}]"
        
        # Build filter conditions
        where_conditions, params = self._build_filter_conditions(search_filter)
        where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
        
        similarity_op = self._get_similarity_operator()
        
        # Combine vector similarity with text search
        query = f"""
            SELECT id, content, embedding, metadata, timestamp, source, 
                   document_type, importance_score,
                   (embedding {similarity_op} $1) as vector_similarity,
                   ts_rank(to_tsvector('english', content), plainto_tsquery('english', $2)) as text_rank
            FROM {self.table_name}
            {where_clause}
            ORDER BY 
                (embedding {similarity_op} $1) * 0.7 + 
                ts_rank(to_tsvector('english', content), plainto_tsquery('english', $2)) * 0.3 DESC
            LIMIT {limit}
        """
        
        # Add parameters
        params.insert(0, embedding_str)
        params.insert(1, query_text or "")
        
        rows = await conn.fetch(query, *params)
        return [self._row_to_document(row) for row in rows]
    
    async def _semantic_search(
        self,
        conn: asyncpg.Connection,
        query_embedding: List[float],
        query_text: str,
        limit: int,
        search_filter: Optional[SearchFilter]
    ) -> List[VectorDocument]:
        """Advanced semantic search with context awareness"""
        
        # For now, implement as enhanced vector search
        # In production, this would include query expansion, semantic parsing, etc.
        return await self._vector_search(conn, query_embedding, limit, search_filter)
    
    async def _contextual_search(
        self,
        conn: asyncpg.Connection,
        query_embedding: List[float],
        query_text: str,
        limit: int,
        search_filter: Optional[SearchFilter]
    ) -> List[VectorDocument]:
        """Context-aware search considering document relationships"""
        
        # Get initial results
        initial_results = await self._vector_search(conn, query_embedding, limit // 2, search_filter)
        
        # Get related documents
        if initial_results:
            related_ids = [doc.id for doc in initial_results[:3]]  # Top 3 for context
            related_query = f"""
                SELECT DISTINCT d.id, d.content, d.embedding, d.metadata, d.timestamp, 
                       d.source, d.document_type, d.importance_score
                FROM {self.table_name} d
                JOIN document_relationships r ON (d.id = r.target_doc_id)
                WHERE r.source_doc_id = ANY($1) AND r.strength > 0.5
                ORDER BY r.strength DESC
                LIMIT {limit // 2}
            """
            
            related_rows = await conn.fetch(related_query, related_ids)
            related_docs = [self._row_to_document(row) for row in related_rows]
            
            # Combine results
            all_docs = initial_results + related_docs
            return list({doc.id: doc for doc in all_docs}.values())  # Deduplicate
        
        return initial_results
    
    async def _rank_results(
        self,
        results: List[VectorDocument],
        strategy: RankingStrategy,
        query_embedding: List[float],
        query_text: Optional[str]
    ) -> List[SearchResult]:
        """Apply ranking strategy to search results"""
        
        search_results = []
        
        for i, doc in enumerate(results):
            # Calculate similarity score
            similarity_score = self.distance_metrics.cosine_similarity(
                query_embedding, doc.embedding
            )
            
            # Calculate different scoring components
            relevance_score = similarity_score  # Base relevance
            importance_score = doc.importance_score
            
            # Apply ranking strategy
            if strategy == RankingStrategy.SIMILARITY:
                final_score = similarity_score
            elif strategy == RankingStrategy.RELEVANCE:
                final_score = relevance_score * 0.7 + importance_score * 0.3
            elif strategy == RankingStrategy.IMPORTANCE:
                final_score = importance_score * 0.6 + similarity_score * 0.4
            elif strategy == RankingStrategy.RECENCY:
                # Calculate recency score (0-1 based on age)
                now = datetime.now()
                age_days = (now - doc.timestamp).days
                recency_score = max(0.1, 1.0 / (1 + age_days / 30))  # Decay over 30 days
                final_score = recency_score * 0.5 + similarity_score * 0.5
            else:  # HYBRID_SCORE
                recency_days = (datetime.now() - doc.timestamp).days
                recency_score = max(0.1, 1.0 / (1 + recency_days / 30))
                final_score = (
                    similarity_score * 0.4 + 
                    importance_score * 0.3 + 
                    recency_score * 0.3
                )
            
            # Create explanation
            explanation = {
                "similarity_contribution": similarity_score,
                "importance_contribution": importance_score,
                "ranking_strategy": strategy.value,
                "document_age_days": (datetime.now() - doc.timestamp).days
            }
            
            search_result = SearchResult(
                document=doc,
                similarity_score=similarity_score,
                relevance_score=relevance_score,
                importance_score=importance_score,
                final_score=final_score,
                rank=i + 1,  # Will be reordered
                explanation=explanation
            )
            
            search_results.append(search_result)
        
        # Sort by final score
        search_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(search_results):
            result.rank = i + 1
        
        return search_results
    
    def _build_filter_conditions(self, search_filter: Optional[SearchFilter]) -> Tuple[List[str], List[Any]]:
        """Build SQL WHERE conditions from search filter"""
        
        conditions = []
        params = []
        param_count = 1
        
        if not search_filter:
            return conditions, params
        
        if search_filter.document_types:
            conditions.append(f"document_type = ANY(${param_count})")
            params.append(search_filter.document_types)
            param_count += 1
        
        if search_filter.sources:
            conditions.append(f"source = ANY(${param_count})")
            params.append(search_filter.sources)
            param_count += 1
        
        if search_filter.date_range:
            start_date, end_date = search_filter.date_range
            conditions.append(f"timestamp >= ${param_count} AND timestamp <= ${param_count + 1}")
            params.extend([start_date, end_date])
            param_count += 2
        
        if search_filter.min_importance is not None:
            conditions.append(f"importance_score >= ${param_count}")
            params.append(search_filter.min_importance)
            param_count += 1
        
        if search_filter.metadata_filters:
            for key, value in search_filter.metadata_filters.items():
                conditions.append(f"metadata->>'{key}' = ${param_count}")
                params.append(str(value))
                param_count += 1
        
        return conditions, params
    
    def _get_similarity_operator(self) -> str:
        """Get the appropriate similarity operator for pgvector"""
        if self.similarity_function == "cosine":
            return "<->"
        elif self.similarity_function == "l2":
            return "<->"
        elif self.similarity_function == "inner_product":
            return "<#>"
        else:
            return "<->"  # Default to cosine
    
    def _row_to_document(self, row) -> VectorDocument:
        """Convert database row to VectorDocument"""
        
        # Convert embedding from pgvector format
        embedding = list(row['embedding'])
        
        return VectorDocument(
            id=row['id'],
            content=row['content'],
            embedding=embedding,
            metadata=row['metadata'] or {},
            timestamp=row['timestamp'],
            source=row['source'] or "",
            document_type=row['document_type'] or "",
            importance_score=row['importance_score'] or 0.0
        )
    
    async def _record_search_analytics(
        self,
        conn: asyncpg.Connection,
        query_text: Optional[str],
        query_embedding: List[float],
        results_count: int,
        search_mode: str,
        ranking_strategy: str,
        execution_time_ms: int
    ) -> None:
        """Record search analytics for analysis and optimization"""
        
        try:
            embedding_str = f"[{','.join(map(str, query_embedding))}]" if query_embedding else None
            
            await conn.execute("""
                INSERT INTO search_analytics 
                (query_text, query_embedding, results_count, search_mode, 
                 ranking_strategy, execution_time_ms)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, query_text, embedding_str, results_count, search_mode, 
                 ranking_strategy, execution_time_ms)
        except Exception as e:
            logger.warning(f"Failed to record search analytics: {e}")
    
    async def add_document_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        strength: float = 1.0
    ) -> bool:
        """Add a relationship between documents"""
        
        if not self.pool:
            return False
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO document_relationships 
                    (source_doc_id, target_doc_id, relationship_type, strength)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (source_doc_id, target_doc_id, relationship_type) 
                    DO UPDATE SET strength = EXCLUDED.strength
                """, source_id, target_id, relationship_type, strength)
            return True
        except Exception as e:
            logger.error(f"Error adding document relationship: {e}")
            return False
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        
        if not self.pool:
            return {}
        
        try:
            async with self.pool.acquire() as conn:
                stats = await conn.fetchrow(f"""
                    SELECT 
                        COUNT(*) as total_documents,
                        COUNT(DISTINCT source) as unique_sources,
                        COUNT(DISTINCT document_type) as unique_types,
                        AVG(importance_score) as avg_importance,
                        MAX(timestamp) as latest_document,
                        MIN(timestamp) as oldest_document
                    FROM {self.table_name}
                """)
                
                return {
                    "total_documents": stats['total_documents'],
                    "unique_sources": stats['unique_sources'], 
                    "unique_document_types": stats['unique_types'],
                    "average_importance": float(stats['avg_importance'] or 0),
                    "latest_document": stats['latest_document'],
                    "oldest_document": stats['oldest_document']
                }
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {}
    
    async def close(self) -> None:
        """Close the vector store connection pool"""
        if self.pool:
            await self.pool.close()
