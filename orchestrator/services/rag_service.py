"""
RAG Service - Uses EXISTING ContextOptimizer and SemanticChunker
================================================================

This service wraps the existing mathematical optimization components:
- context_engineering/context_optimizer.py (Knapsack, MMR, Entropy)
- context_engineering/chunking/semantic_chunker.py (5 strategies)

NO DUPLICATE IMPLEMENTATIONS - uses what's already built.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result from RAG retrieval"""
    chunks: List[Dict[str, Any]]
    formatted_context: str
    total_tokens: int
    sources: List[str]
    query: str
    diversity_score: float = 0.0
    information_gain: float = 0.0


class RAGService:
    """
    RAG Service that uses EXISTING ContextOptimizer.
    
    Wraps:
    - context_engineering.context_optimizer.ContextOptimizer
    - context_engineering.chunking.semantic_chunker.SemanticChunker
    """
    
    def __init__(self):
        self._context_optimizer = None
        self._semantic_chunker = None
        self._embedding_manager = None
        self._initialized = False
        
    def _ensure_initialized(self):
        """Lazy initialization of components"""
        if self._initialized:
            return
            
        # Import from modules.search (re-exports from context_engineering)
        try:
            from modules.search import ContextOptimizer, ContextItem
            self._context_optimizer = ContextOptimizer()
            self._ContextItem = ContextItem
            logger.info("✅ Using modules.search.ContextOptimizer (Knapsack, MMR, Entropy)")
        except ImportError as e:
            logger.warning(f"ContextOptimizer not available: {e}")
            self._context_optimizer = None
            
        try:
            from modules.rag import SemanticChunker, ChunkingStrategy
            self._semantic_chunker = SemanticChunker(
                strategy=ChunkingStrategy.ADAPTIVE,
                target_chunk_size=500,
                min_chunk_size=100,
                max_chunk_size=1500
            )
            logger.info("✅ Using modules.rag.SemanticChunker (5 strategies)")
        except ImportError as e:
            logger.warning(f"SemanticChunker not available: {e}")
            self._semantic_chunker = None
            
        # Use centralized embedding manager
        try:
            from services.llm_provider import create_embedding_manager
            self._embedding_manager = create_embedding_manager()
        except Exception as e:
            logger.warning(f"Embedding manager not available: {e}")
            
        self._initialized = True
        
    async def retrieve(
        self,
        query: str,
        max_chunks: int = 8,
        max_tokens: int = 2000,
        diversity: float = 0.3,
        context_type: str = "chatbot"
    ) -> RAGResult:
        """
        Retrieve optimized RAG context using existing ContextOptimizer.
        
        Uses:
        - Knapsack optimization for token budget
        - MMR for diversity
        - Information theory for quality
        """
        self._ensure_initialized()
        
        # Get candidates from database
        candidates = await self._get_candidates(query, limit=max_chunks * 3)
        
        if not candidates:
            return RAGResult(
                chunks=[],
                formatted_context="No relevant context found.",
                total_tokens=0,
                sources=[],
                query=query
            )
        
        # Use existing ContextOptimizer if available
        if self._context_optimizer:
            return await self._optimize_with_context_optimizer(
                query, candidates, max_tokens, diversity
            )
        else:
            # Fallback to basic retrieval
            return self._basic_retrieval(query, candidates, max_chunks, max_tokens)
    
    # Alias for backward compatibility (used by agent_platform_tools)
    async def retrieve_context(
        self,
        query: str,
        top_k: int = 8,
        max_chunks: int = None,
        max_tokens: int = 2000,
        min_similarity: float = 0.5
    ) -> RAGResult:
        """Backward-compatible alias for retrieve()."""
        chunks = max_chunks if max_chunks is not None else top_k
        return await self.retrieve(
            query=query,
            max_chunks=chunks,
            max_tokens=max_tokens,
            diversity=0.3
        )
    
    async def _optimize_with_context_optimizer(
        self,
        query: str,
        candidates: List[Dict],
        max_tokens: int,
        diversity: float
    ) -> RAGResult:
        """Use existing ContextOptimizer for mathematical optimization"""
        
        # Convert to ContextItems
        context_items = []
        for c in candidates:
            item = self._ContextItem(
                content=c.get("content", ""),
                source=c.get("source_file", c.get("filename", "unknown")),
                context_type="documentation",
                relevance_score=c.get("similarity", 0.5),
                token_count=len(c.get("content", "")) // 4
            )
            context_items.append(item)
        
        # Use existing optimize_context method (has Knapsack + MMR)
        objective = "maximize_diversity" if diversity > 0.5 else "maximize_information"
        
        optimized = await self._context_optimizer.optimize_context(
            available_context=context_items,
            max_tokens=max_tokens,
            objective=objective
        )
        
        # Format results
        chunks = []
        for ctx in optimized.contexts:
            chunks.append({
                "content": ctx.content,
                "source_file": ctx.source,
                "similarity": ctx.relevance_score,
                "tokens": ctx.token_count
            })
        
        formatted_context = self._format_context(chunks, query)
        
        return RAGResult(
            chunks=chunks,
            formatted_context=formatted_context,
            total_tokens=optimized.total_tokens,
            sources=list(set(c["source_file"] for c in chunks)),
            query=query,
            diversity_score=optimized.diversity_score,
            information_gain=optimized.information_gain
        )
    
    def _basic_retrieval(
        self,
        query: str,
        candidates: List[Dict],
        max_chunks: int,
        max_tokens: int
    ) -> RAGResult:
        """Fallback when ContextOptimizer not available"""
        
        # Sort by similarity and take top chunks
        sorted_candidates = sorted(
            candidates, 
            key=lambda x: x.get("similarity", 0), 
            reverse=True
        )[:max_chunks]
        
        chunks = []
        total_tokens = 0
        
        for c in sorted_candidates:
            chunk_tokens = len(c.get("content", "")) // 4
            if total_tokens + chunk_tokens > max_tokens:
                break
            chunks.append({
                "content": c.get("content", ""),
                "source_file": c.get("source_file", c.get("filename", "unknown")),
                "similarity": c.get("similarity", 0),
                "tokens": chunk_tokens
            })
            total_tokens += chunk_tokens
        
        formatted_context = self._format_context(chunks, query)
        
        return RAGResult(
            chunks=chunks,
            formatted_context=formatted_context,
            total_tokens=total_tokens,
            sources=list(set(c["source_file"] for c in chunks)),
            query=query
        )
    
    async def _get_candidates(self, query: str, limit: int = 20) -> List[Dict]:
        """Get candidate chunks from database"""
        
        if not self._embedding_manager:
            return []
            
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            from database.database import get_database_url
            
            # Generate query embedding
            query_embedding = await self._embedding_manager.generate_embedding(query)
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            conn = psycopg2.connect(get_database_url())
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(f"""
                SELECT 
                    dc.id,
                    dc.content,
                    dc.parent_content,
                    dc.headers,
                    dc.metadata,
                    d.filename as source_file,
                    d.id as document_id,
                    1 - (dc.embedding <=> '{embedding_str}'::vector) as similarity
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE dc.embedding IS NOT NULL
                    AND d.status = 'completed'
                ORDER BY dc.embedding <=> '{embedding_str}'::vector
                LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(r) for r in results]
            
        except Exception as e:
            logger.error(f"Error getting candidates: {e}")
            return []
    
    def _format_context(self, chunks: List[Dict], query: str) -> str:
        """Format chunks into context string"""
        
        if not chunks:
            return "No relevant context found."
        
        parts = [f"## Retrieved Context for: {query}\n"]
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source_file", "unknown")
            similarity = chunk.get("similarity", 0)
            content = chunk.get("content", "")
            
            parts.append(f"\n### Source {i}: {source} (relevance: {similarity:.0%})")
            parts.append(content)
        
        return "\n".join(parts)
    
    async def enhance_prompt_with_context(
        self,
        original_prompt: str,
        max_context_tokens: int = 2000
    ) -> Tuple[str, Dict]:
        """
        Enhance a prompt with RAG context.
        Used by context_engineering_integrator.py
        """
        result = await self.retrieve(
            query=original_prompt,
            max_tokens=max_context_tokens
        )
        
        enhanced_prompt = f"""## Relevant Context
{result.formatted_context}

## Original Task
{original_prompt}
"""
        
        metadata = {
            "sources": result.sources,
            "total_context_tokens": result.total_tokens,
            "enhanced": len(result.chunks) > 0,
            "documents_used": len(result.sources),
            "diversity_score": result.diversity_score,
            "information_gain": result.information_gain
        }
        
        return enhanced_prompt, metadata
    
    def chunk_document(self, content: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk a document using existing SemanticChunker.
        
        Uses context_engineering/chunking/semantic_chunker.py
        NOT a duplicate implementation!
        """
        self._ensure_initialized()
        
        if self._semantic_chunker:
            # Use existing SemanticChunker with mathematical foundations
            chunks = self._semantic_chunker.chunk_text(
                content, 
                document_id=metadata.get("document_id") if metadata else None
            )
            
            return [
                {
                    "content": chunk.content,
                    "metadata": {
                        "entropy": chunk.metadata.entropy,
                        "topic_coherence": chunk.metadata.topic_coherence,
                        "semantic_density": chunk.metadata.semantic_density,
                        "importance_score": chunk.metadata.importance_score,
                        **chunk.metadata.__dict__
                    }
                }
                for chunk in chunks
            ]
        else:
            # Basic fallback
            return self._basic_chunk(content)
    
    def _basic_chunk(self, content: str, chunk_size: int = 500) -> List[Dict]:
        """Basic chunking fallback"""
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i:i + chunk_size]
            if len(chunk_content.strip()) > 50:
                chunks.append({"content": chunk_content, "metadata": {}})
        return chunks


# Singleton
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get singleton RAG service"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


# Backward compatibility aliases
UniversalRAGService = RAGService
get_universal_rag = get_rag_service

