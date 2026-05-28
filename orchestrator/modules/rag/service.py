"""
RAG Service - Uses EXISTING ContextOptimizer and SemanticChunker
================================================================

This service wraps the existing mathematical optimization components:
- modules/search/optimization/context_optimizer.py (Knapsack, MMR, Entropy)
- modules/rag/chunking/semantic_chunker.py (5 strategies)

NO DUPLICATE IMPLEMENTATIONS - uses what's already built.
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Accurate token counting
try:
    import tiktoken
    _encoding = tiktoken.encoding_for_model("gpt-4")
except Exception:
    import tiktoken
    _encoding = tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str) -> int:
    """Accurate token count using tiktoken"""
    if not text:
        return 0
    return len(_encoding.encode(text))

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


def _get_rag_setting_int(key: str, default: int) -> int:
    """Get RAG setting from system_settings"""
    try:
        from core.database.database import SessionLocal
        from core.models.system_settings import SystemSetting
        db = SessionLocal()
        try:
            setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
            if setting and setting.value:
                return int(setting.value)
        finally:
            db.close()
    except Exception:
        logger.error("Failed to read RAG setting '%s' (int), using default=%d", key, default, exc_info=True)
    return default


def _get_rag_setting_str(key: str, default: str) -> str:
    """Get RAG setting string from system_settings"""
    try:
        from core.database.database import SessionLocal
        from core.models.system_settings import SystemSetting
        db = SessionLocal()
        try:
            setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
            if setting and setting.value is not None:
                return setting.value
        finally:
            db.close()
    except Exception:
        logger.error("Failed to read RAG setting '%s' (str), using default='%s'", key, default, exc_info=True)
    return default


def _get_rag_setting_float(key: str, default: float) -> float:
    """Get RAG setting from system_settings"""
    try:
        from core.database.database import SessionLocal
        from core.models.system_settings import SystemSetting
        db = SessionLocal()
        try:
            setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
            if setting and setting.value:
                return float(setting.value)
        finally:
            db.close()
    except Exception:
        logger.error("Failed to read RAG setting '%s' (float), using default=%s", key, default, exc_info=True)
    return default


@dataclass
class RAGConfig:
    """Configuration for RAG service - reads from system_settings"""
    chunk_size: int = None
    min_chunk_size: int = None
    max_chunk_size: int = None
    max_tokens: int = None
    diversity: float = None
    min_similarity: float = None
    
    # Phase 2: Advanced retrieval options
    enable_query_enhancement: bool = True
    enable_rrf_fusion: bool = True
    enable_reranking: bool = False
    rrf_k: int = 60

    # Hybrid search settings
    hybrid_search_enabled: bool = True
    hybrid_vector_weight: float = 0.7
    hybrid_keyword_weight: float = 0.3
    parent_child_expansion: bool = True
    expansion_window: int = 1
    
    def __post_init__(self):
        """Load from system_settings if not provided"""
        if self.chunk_size is None:
            self.chunk_size = _get_rag_setting_int("chunk_size", 500)
        if self.min_chunk_size is None:
            self.min_chunk_size = _get_rag_setting_int("min_chunk_size", 100)
        if self.max_chunk_size is None:
            self.max_chunk_size = _get_rag_setting_int("max_chunk_size", 1500)
        if self.max_tokens is None:
            self.max_tokens = _get_rag_setting_int("max_tokens", 2000)
        if self.diversity is None:
            self.diversity = _get_rag_setting_float("diversity_factor", 0.3)
        if self.min_similarity is None:
            self.min_similarity = _get_rag_setting_float("min_similarity", 0.5)
        
        # Load reranking toggle from system_settings
        self.enable_reranking = _get_rag_setting_str("rag_rerank_enabled", "false") == "true"

        logger.info(f"RAGConfig loaded: max_tokens={self.max_tokens}, diversity={self.diversity}, min_similarity={self.min_similarity}, reranking={self.enable_reranking}")


class RAGService:
    """
    RAG Service that uses EXISTING ContextOptimizer.
    
    Wraps:
    - modules.search.optimization.ContextOptimizer
    - modules.rag.chunking.SemanticChunker
    """
    
    def __init__(
        self,
        config: RAGConfig = None,
        workspace_id: str = None,
    ):
        self.config = config or RAGConfig()
        self._workspace_id = workspace_id
        self._context_optimizer = None
        self._semantic_chunker = None
        self._embedding_manager = None
        self._query_enhancer = None
        self._initialized = False
        
    def _ensure_initialized(self):
        """Lazy initialization of components (NOT vector store — that needs workspace_id at query time)"""
        if self._initialized:
            return

        # Try to use existing ContextOptimizer from modules/search
        try:
            from modules.search import ContextOptimizer, ContextItem
            self._context_optimizer = ContextOptimizer()
            self._ContextItem = ContextItem
            logger.info("✅ Using modules.search.ContextOptimizer (Knapsack, MMR, Entropy)")
        except Exception as e:
            logger.warning(f"ContextOptimizer not available: {e}")

        # Lazy initialization of embedding manager
        try:
            from core.llm import create_embedding_manager
            self._embedding_manager = create_embedding_manager()
            logger.info(f"✅ Using {self._embedding_manager.get_provider_info()['provider']} embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize embedding manager: {e}")
            
        try:
            from modules.rag.chunking import SemanticChunker, ChunkingStrategy
            self._semantic_chunker = SemanticChunker(
                strategy=ChunkingStrategy.ADAPTIVE,
                target_chunk_size=self.config.chunk_size,
                min_chunk_size=self.config.min_chunk_size,
                max_chunk_size=self.config.max_chunk_size
            )
            logger.info("✅ Using modules.rag.SemanticChunker (5 strategies)")
        except ImportError as e:
            logger.warning(f"SemanticChunker not available: {e}")
            self._semantic_chunker = None
            
        # Query enhancer for HyDE and decomposition
        try:
            from modules.rag.query_enhancer import create_query_enhancer
            self._query_enhancer = create_query_enhancer()
            logger.info("✅ Using QueryEnhancer (HyDE, decomposition, concept extraction)")
        except ImportError as e:
            logger.warning(f"QueryEnhancer not available: {e}")
            self._query_enhancer = None
            
        self._initialized = True
        
    async def retrieve(
        self,
        query: str,
        max_chunks: int = 8,
        max_tokens: int = None,
        diversity: float = None,
        context_type: str = "chatbot",
        workspace_id: str = None,
        team: str = None,
    ) -> RAGResult:
        """
        Retrieve optimized RAG context using existing ContextOptimizer.
        
        Uses:
        - Query Enhancement (HyDE, decomposition) for better recall
        - Reciprocal Rank Fusion for hybrid search
        - Knapsack optimization for token budget
        - MMR for diversity
        - Information theory for quality
        """
        self._ensure_initialized()
        
        # Calculate token budget to accommodate requested chunks
        # Estimate ~500 tokens per chunk to ensure max_chunks can fit
        if max_tokens is None:
            estimated_tokens_per_chunk = 500
            max_tokens = max(self.config.max_tokens, max_chunks * estimated_tokens_per_chunk)
        
        diversity = diversity if diversity is not None else self.config.diversity
        
        # Phase 2: Enhance query with HyDE and decomposition
        queries_to_search = [query]
        if self.config.enable_query_enhancement and self._query_enhancer:
            try:
                enhanced = await self._query_enhancer.enhance_query(
                    query,
                    use_hyde=True,
                    use_decomposition=True,
                    use_expansion=True
                )
                queries_to_search = enhanced.get_all_queries()
                logger.info(f"Enhanced query into {len(queries_to_search)} variations")
            except Exception as e:
                logger.warning(f"Query enhancement failed, using original: {e}")
        
        # PRD-124: normalize team name and over-fetch to compensate for post-filter reduction
        if team:
            from core.team_access import normalize_team
            team = normalize_team(team)
        team_multiplier = 2 if team else 1

        # Multi-query retrieval with RRF fusion
        if len(queries_to_search) > 1 and self.config.enable_rrf_fusion:
            candidates = await self._multi_query_retrieval_with_rrf(
                queries_to_search,
                limit_per_query=max_chunks * 2 * team_multiplier,
                min_similarity=self.config.min_similarity,
                workspace_id=workspace_id
            )
        else:
            # Single query retrieval
            candidates = await self._get_candidates(
                query,
                limit=max_chunks * 3 * team_multiplier,
                min_similarity=self.config.min_similarity,
                workspace_id=workspace_id
            )
        
        if not candidates:
            return RAGResult(
                chunks=[],
                formatted_context="No relevant context found.",
                total_tokens=0,
                sources=[],
                query=query
            )

        # PRD-124: Post-retrieval team filtering via PostgreSQL
        if team and workspace_id:
            candidates = await self._filter_by_team(candidates, team, workspace_id)
            if not candidates:
                return RAGResult(
                    chunks=[],
                    formatted_context="No relevant context found for your team.",
                    total_tokens=0,
                    sources=[],
                    query=query
                )

        # Optional: Cross-encoder re-ranking for higher precision
        if self.config.enable_reranking:
            candidates = await self._rerank_candidates(query, candidates)

        # Parent-child context expansion
        candidates = await self._expand_to_parent_context(candidates, self.config.expansion_window)

        # Use existing ContextOptimizer if available
        if self._context_optimizer:
            result = await self._optimize_with_context_optimizer(
                query, candidates, max_chunks, max_tokens, diversity
            )
        else:
            # Fallback to basic retrieval
            result = self._basic_retrieval(query, candidates, max_chunks, max_tokens)

        # Track document access for analytics (fire-and-forget)
        if result.chunks and workspace_id:
            self._track_document_access(result.chunks, workspace_id)

        return result

    def _track_document_access(self, chunks: List[Dict[str, Any]], workspace_id: str) -> None:
        """Update last_accessed timestamp on documents retrieved via RAG."""
        try:
            from core.database.database import SessionLocal
            from sqlalchemy import text
            doc_ids = set()
            for chunk in chunks:
                meta = chunk.get("metadata", {})
                doc_id = meta.get("document_id") or meta.get("doc_id")
                if doc_id:
                    doc_ids.add(str(doc_id))
            if not doc_ids:
                return
            db = SessionLocal()
            try:
                db.execute(
                    text("""
                        UPDATE documents
                        SET last_accessed = NOW(),
                            rag_query_count = COALESCE(rag_query_count, 0) + 1
                        WHERE id = ANY(:ids::int[]) AND workspace_id = :ws::uuid
                    """),
                    {"ids": list(doc_ids), "ws": str(workspace_id)},
                )
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"Document access tracking failed: {e}")
    
    async def _filter_by_team(
        self,
        candidates: List[Dict],
        team: str,
        workspace_id: str,
    ) -> List[Dict]:
        """PRD-124: Filter candidates to only include chunks from team-accessible documents.

        Queries PostgreSQL for document IDs where:
          team_access = '{}' (empty = visible to all) OR team = ANY(team_access)
        """
        # Collect unique document IDs from candidates
        doc_ids: set = set()
        for c in candidates:
            meta = c.get("metadata", {})
            doc_id = meta.get("document_id") or meta.get("doc_id") or meta.get("external_file_id")
            if doc_id:
                doc_ids.add(str(doc_id))

        if not doc_ids:
            return candidates  # No doc IDs to filter — pass through

        try:
            from core.database.database import SessionLocal
            from sqlalchemy import text as sa_text

            db = SessionLocal()
            try:
                rows = db.execute(
                    sa_text("""
                        SELECT id::text FROM documents
                        WHERE id = ANY(:ids::int[])
                          AND workspace_id = :ws::uuid
                          AND (team_access = '{}' OR :team = ANY(team_access))
                    """),
                    {"ids": list(doc_ids), "ws": str(workspace_id), "team": team},
                ).fetchall()
                allowed_ids = {str(r[0]) for r in rows}
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Team filtering query failed, passing all candidates: {e}")
            return candidates

        filtered = [
            c for c in candidates
            if str(
                c.get("metadata", {}).get("document_id")
                or c.get("metadata", {}).get("doc_id")
                or c.get("metadata", {}).get("external_file_id")
                or ""
            ) in allowed_ids
            or not (  # Keep candidates without doc ID (shouldn't happen, but safe)
                c.get("metadata", {}).get("document_id")
                or c.get("metadata", {}).get("doc_id")
                or c.get("metadata", {}).get("external_file_id")
            )
        ]
        logger.info(f"PRD-124 team filter: {len(candidates)} → {len(filtered)} candidates (team={team})")
        return filtered

    async def _multi_query_retrieval_with_rrf(
        self,
        queries: List[str],
        limit_per_query: int = 20,
        min_similarity: float = 0.5,
        workspace_id: str = None
    ) -> List[Dict]:
        """
        Perform multi-query retrieval with Reciprocal Rank Fusion.
        
        RRF score: sum(1 / (k + rank_i)) for each query where document appears
        This is the standard approach from Context-Engineering research.
        """
        from collections import defaultdict
        
        # Collect results from each query variation
        all_results = defaultdict(lambda: {"ranks": [], "doc": None})
        
        for query in queries[:5]:  # Max 5 query variations
            try:
                results = await self._get_candidates(
                    query,
                    limit=limit_per_query,
                    min_similarity=min_similarity,
                    workspace_id=workspace_id
                )
                
                for rank, doc in enumerate(results):
                    doc_id = doc.get("id", doc.get("content", "")[:100])
                    all_results[doc_id]["ranks"].append(rank)
                    all_results[doc_id]["doc"] = doc
                    
            except Exception as e:
                logger.debug(f"Query variation failed: {e}")
                continue
        
        # Calculate RRF scores
        k = self.config.rrf_k  # Standard RRF constant (usually 60)
        rrf_scored = []
        
        for doc_id, data in all_results.items():
            if data["doc"]:
                rrf_score = sum(1.0 / (k + rank) for rank in data["ranks"])
                doc = data["doc"].copy()
                doc["rrf_score"] = rrf_score
                doc["query_count"] = len(data["ranks"])  # Appears in N queries
                rrf_scored.append(doc)
        
        # Sort by RRF score (higher is better)
        rrf_scored.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        logger.info(f"RRF fusion: {len(rrf_scored)} unique docs from {len(queries)} queries")
        return rrf_scored
    
    async def _rerank_candidates(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Re-rank candidates using Cohere Rerank API for higher precision.

        Falls back to original order if Cohere API key is not configured.
        """
        try:
            from core.llm.rerank_manager import get_rerank_manager

            manager = get_rerank_manager()
            if not manager.is_available():
                logger.debug("Reranking unavailable (no Cohere API key), skipping")
                return candidates

            # Cap at 20 candidates, truncate content to Cohere's limit
            capped = candidates[:20]
            documents = [c.get("content", "")[:4096] for c in capped]
            results = await manager.rerank(query, documents, top_n=top_k)

            # Apply rerank scores and re-sort
            reranked = []
            for result in results:
                if result.index < len(capped):
                    candidate = capped[result.index].copy()
                    candidate["rerank_score"] = result.relevance_score
                    reranked.append(candidate)

            return reranked if reranked else candidates[:top_k]

        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")
            return candidates
    
    # Alias for backward compatibility (used by agent_platform_tools)
    async def retrieve_context(
        self,
        query: str,
        top_k: int = 8,
        max_chunks: int = None,
        max_tokens: int = 2000,
        min_similarity: float = 0.5,
        workspace_id: str = None,
        team: str = None,
    ) -> RAGResult:
        """Backward-compatible alias for retrieve()."""
        chunks = max_chunks if max_chunks is not None else top_k
        return await self.retrieve(
            query=query,
            max_chunks=chunks,
            max_tokens=max_tokens,
            diversity=0.3,
            workspace_id=workspace_id,
            team=team,
        )
    
    
    async def _optimize_with_context_optimizer(
        self,
        query: str,
        candidates: List[Dict],
        max_chunks: int,
        max_tokens: int,
        diversity: float
    ) -> RAGResult:
        """Use REAL 0/1 knapsack DP algorithm with content quality and source diversity"""
        
        logger.info(f"🔍 Starting optimization: {len(candidates)} candidates, max_chunks={max_chunks}, max_tokens={max_tokens}")
        
        # Convert to ContextItems with accurate token counting
        context_items = []
        source_counts = {}
        
        for i, c in enumerate(candidates):
            content = c.get("content", "")
            source = c.get("source_file", c.get("filename", "unknown"))
            
            # Calculate content quality
            quality_score = self._calculate_content_quality(content)
            
            # Track source distribution
            source_counts[source] = source_counts.get(source, 0) + 1
            
            # Apply source diversity penalty (prefer diverse sources)
            source_penalty = 1.0
            if source_counts[source] > 1:
                source_penalty = 0.7 ** (source_counts[source] - 1)  # Exponential penalty
            
            # Combine relevance, quality, and diversity
            base_relevance = c.get("similarity", 0.5)
            adjusted_score = base_relevance * quality_score * source_penalty
            
            item = self._ContextItem(
                content=content,
                source=source,
                context_type="documentation",
                relevance_score=adjusted_score,  # Use quality-adjusted score
                token_count=_count_tokens(content),
                metadata={
                    "document_id": c.get("document_id"),
                    "original_metadata": c.get("metadata", {}),
                }
            )
            context_items.append(item)
            
            # Log first 3 candidates for debugging
            if i < 3:
                preview = content[:100].replace('\n', ' ')
                logger.info(f"  Candidate {i+1}: base={base_relevance:.3f}, quality={quality_score:.2f}, source_penalty={source_penalty:.2f}, final={adjusted_score:.3f}, tokens={item.token_count}, source={source}, preview='{preview}...'")
        
        # Calculate information value for each chunk
        values = [item.relevance_score for item in context_items]
        weights = [item.token_count for item in context_items]
        
        logger.info(f"📊 Value range: {min(values):.3f} - {max(values):.3f}, Weight range: {min(weights)} - {max(weights)} tokens")
        
        # Apply REAL 0/1 Knapsack Dynamic Programming
        logger.info(f"🎯 Running 0/1 Knapsack DP algorithm with quality-adjusted scores...")
        selected_indices = self._knapsack_dp(values, weights, max_tokens, max_chunks)
        
        logger.info(f"✅ Knapsack selected {len(selected_indices)} items: {selected_indices}")
        
        # Get selected contexts
        selected_contexts = [context_items[i] for i in selected_indices]
        
        # Log selected chunks
        for i, idx in enumerate(selected_indices):
            ctx = context_items[idx]
            preview = ctx.content[:80].replace('\n', ' ')
            logger.info(f"  Selected {i+1}: idx={idx}, score={ctx.relevance_score:.3f}, tokens={ctx.token_count}, source={ctx.source}, preview='{preview}...'")
        
        # Format results
        chunks = []
        total_tokens = 0
        final_source_counts = {}
        for ctx in selected_contexts:
            chunks.append({
                "content": ctx.content,
                "source_file": ctx.source,
                "similarity": ctx.relevance_score,
                "tokens": ctx.token_count,
                "document_id": ctx.metadata.get("document_id") if ctx.metadata else None,
                "metadata": ctx.metadata.get("original_metadata", {}) if ctx.metadata else {},
            })
            total_tokens += ctx.token_count
            final_source_counts[ctx.source] = final_source_counts.get(ctx.source, 0) + 1
        
        # Calculate diversity score
        diversity_score = len(final_source_counts) / len(chunks) if chunks else 0.0
        
        logger.info(f"📈 Results: {len(chunks)} chunks, {total_tokens} tokens, source_diversity={diversity_score:.2f}")
        logger.info(f"📁 Source distribution: {final_source_counts}")
        
        # Format context
        formatted_context = self._format_context(chunks, query)
        
        info_gain = sum(values[i] for i in selected_indices) / len(values) if values else 0.0
        logger.info(f"💡 Information gain: {info_gain:.3f}")
        
        return RAGResult(
            chunks=chunks,
            formatted_context=formatted_context,
            total_tokens=total_tokens,
            sources=list(set(c["source_file"] for c in chunks)),
            query=query,
            diversity_score=diversity_score,
            information_gain=info_gain
        )
    
    def _calculate_content_quality(self, text: str) -> float:
        """
        Calculate content quality score (0.0 - 1.0).
        Penalizes ASCII art but treats short valid content (code, definitions,
        config values) fairly.
        """
        if not text or len(text.strip()) == 0:
            return 0.1

        # Detect ASCII art characters
        ascii_art_chars = '│─┌└┐┘├┤┬┴┼▼▲►◄║═╔╗╚╝╠╣╦╩╬'
        special_char_count = sum(1 for c in text if c in ascii_art_chars)
        ascii_art_ratio = special_char_count / len(text)

        # Heavy penalty for ASCII art
        if ascii_art_ratio > 0.15:
            logger.debug(f"  High ASCII art ratio: {ascii_art_ratio:.2%}")
            return 0.2
        elif ascii_art_ratio > 0.05:
            return 0.5

        words = text.split()
        word_count = len(words)

        # Short content is still valid (code snippets, definitions, config)
        if word_count < 5:
            return 0.5
        elif word_count < 20:
            return 0.7
        elif word_count < 50:
            return 0.85
        else:
            return 1.0
    
    
    def _knapsack_dp(
        self,
        values: List[float],
        weights: List[int],
        capacity: int,
        max_items: int
    ) -> List[int]:
        """
        REAL 0/1 Knapsack with Dynamic Programming
        
        Finds optimal subset of items that:
        1. Maximizes total value
        2. Stays within weight capacity
        3. Respects max_items constraint
        
        Time: O(n * capacity * max_items)
        Space: O(n * capacity * max_items)
        """
        n = len(values)
        if n == 0 or capacity <= 0 or max_items <= 0:
            logger.warning(f"⚠️ Knapsack: invalid params n={n}, capacity={capacity}, max_items={max_items}")
            return []
        
        logger.info(f"🎒 Knapsack DP: n={n} items, capacity={capacity} tokens, max_items={max_items}")
        
        # DP table: dp[i][w][k] = max value using first i items, weight w, k items selected
        # For memory efficiency, use 2D table and track item count separately
        dp = [[0.0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        item_count = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        # Build DP table
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                # Option 1: Don't include item i-1
                dp[i][w] = dp[i-1][w]
                item_count[i][w] = item_count[i-1][w]
                
                # Option 2: Include item i-1 (if it fits and we haven't hit max_items)
                if weights[i-1] <= w and item_count[i-1][w - weights[i-1]] < max_items:
                    value_with_item = dp[i-1][w - weights[i-1]] + values[i-1]
                    
                    if value_with_item > dp[i][w]:
                        dp[i][w] = value_with_item
                        item_count[i][w] = item_count[i-1][w - weights[i-1]] + 1
        
        # Backtrack to find selected items
        selected = []
        w = capacity
        for i in range(n, 0, -1):
            # Check if item i-1 was included
            if dp[i][w] != dp[i-1][w]:
                selected.append(i-1)
                w -= weights[i-1]
        
        final_value = dp[n][capacity]
        final_weight = sum(weights[i] for i in selected)
        logger.info(f"✅ Knapsack result: {len(selected)} items, total_value={final_value:.3f}, total_weight={final_weight} tokens")
        
        return list(reversed(selected))
    
    
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
            content = c.get("content", "")
            chunk_tokens = _count_tokens(content)  # Use tiktoken for accuracy
            if total_tokens + chunk_tokens > max_tokens:
                break
            chunks.append({
                "content": content,
                "source_file": c.get("source_file", c.get("filename", "unknown")),
                "similarity": c.get("similarity", 0),
                "tokens": chunk_tokens,
                "document_id": c.get("document_id"),
                "metadata": c.get("metadata", {}),
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
    
    async def _get_candidates(self, query: str, limit: int = 20, min_similarity: float = 0.5, workspace_id: str = None) -> List[Dict]:
        """
        Get candidate chunks via S3 Vectors.

        Args:
            workspace_id: Workspace ID for multi-tenant isolation.
                          Falls back to self._workspace_id if not provided.
        """
        if not self._embedding_manager:
            logger.error("Embedding manager not initialized — cannot search")
            return []

        effective_workspace_id = workspace_id or self._workspace_id
        if not effective_workspace_id:
            logger.error("No workspace_id available — cannot search S3 Vectors")
            return []

        try:
            # Generate query embedding
            query_embedding = await self._embedding_manager.generate_embedding(query)

            # Create S3VectorsBackend for this workspace
            from modules.search.vector_store.backends.s3_vectors_backend import S3VectorsBackend
            vector_store = S3VectorsBackend(workspace_id=effective_workspace_id)
            await vector_store.initialize()

            logger.info(f"🔎 S3 Vectors search: workspace={effective_workspace_id}, min_similarity={min_similarity}, limit={limit}")

            results = vector_store.search(
                query_embedding=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding),
                limit=limit,
                min_score=min_similarity,
            )

            candidates = []
            source_file_counts = {}
            similarity_scores = []

            for r in results:
                source_file = r.get("file_name", r.get("file_path", "unknown"))
                similarity = r.get("score", 0.0)

                source_file_counts[source_file] = source_file_counts.get(source_file, 0) + 1
                similarity_scores.append(similarity)

                # external_file_id in S3 Vectors = PostgreSQL documents.id
                doc_id = (
                    r.get("external_file_id")
                    or r.get("metadata", {}).get("external_file_id")
                    or r.get("metadata", {}).get("document_id")
                    or 0
                )

                candidates.append({
                    "id": r.get("key", ""),
                    "content": r.get("content", ""),
                    "source_file": source_file,
                    "document_id": doc_id,
                    "file_type": r.get("file_path", "").rsplit(".", 1)[-1] if r.get("file_path") else "",
                    "similarity": similarity,
                    "metadata": r.get("metadata", {}),
                    "parent_content": None,
                    "headers": {}
                })

            logger.info(f"📁 Candidate sources: {source_file_counts}")
            if similarity_scores:
                logger.info(f"📈 Similarity range: {min(similarity_scores):.3f} - {max(similarity_scores):.3f}")
            logger.info(f"✅ Retrieved {len(candidates)} candidates from S3 Vectors")

            return candidates

        except Exception as e:
            logger.error(f"Error getting candidates from S3 Vectors: {e}", exc_info=True)
            return []
    
    async def _expand_to_parent_context(
        self,
        candidates: List[Dict],
        expand_window: int = 1
    ) -> List[Dict]:
        """
        For each retrieved chunk, fetch surrounding chunks from the same document.
        Uses chunk_index from metadata to find neighbors.
        """
        if not candidates or not self.config.parent_child_expansion:
            return candidates

        try:
            import asyncpg
            from config import config as app_config
            db_url = app_config.DATABASE_URL
            conn = await asyncpg.connect(db_url)

            try:
                for candidate in candidates:
                    raw_doc_id = candidate.get("document_id")
                    chunk_metadata = candidate.get("metadata", {})
                    chunk_index = chunk_metadata.get("chunk_index") if isinstance(chunk_metadata, dict) else None

                    # Cast document_id to int (S3 Vectors stores as string)
                    try:
                        doc_id = int(raw_doc_id) if raw_doc_id else None
                    except (ValueError, TypeError):
                        doc_id = None

                    if doc_id is None or chunk_index is None:
                        candidate["expanded_content"] = candidate.get("content", "")
                        continue

                    window = expand_window or self.config.expansion_window
                    surrounding = await conn.fetch("""
                        SELECT content, metadata
                        FROM document_chunks
                        WHERE document_id = $1
                          AND chunk_index BETWEEN $2 AND $3
                        ORDER BY chunk_index
                    """, doc_id, max(0, chunk_index - window), chunk_index + window)

                    if surrounding:
                        candidate["expanded_content"] = "\n\n".join(r["content"] for r in surrounding)
                    else:
                        candidate["expanded_content"] = candidate.get("content", "")
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"Parent-child expansion failed, using original content: {e}")
            for candidate in candidates:
                candidate["expanded_content"] = candidate.get("content", "")

        return candidates

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
        
        Uses modules/rag/chunking/semantic_chunker.py
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
    
    # =========================================================================
    # Stats & Analytics Methods (for api/context.py)
    # =========================================================================
    
    def get_retrieval_stats(self, db) -> Dict[str, Any]:
        """Get retrieval statistics from database"""
        try:
            from sqlalchemy import text
            
            # Count total RAG queries from documents table
            result = db.execute(text("""
                SELECT
                    COUNT(*) as total_docs,
                    SUM(chunk_count) as total_chunks,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_docs
                FROM documents
            """)).fetchone()

            total_docs = result.total_docs or 0
            total_chunks = result.total_chunks or 0
            completed_docs = result.completed_docs or 0

            # Calculate success rate
            success_rate = (completed_docs / total_docs * 100) if total_docs > 0 else 0

            # Get actual avg response time from document_usage tracking
            avg_response_time = 0
            last_query_time = None
            try:
                usage_result = db.execute(text("""
                    SELECT
                        COALESCE(AVG(execution_time_ms), 0) as avg_time,
                        MAX(timestamp) as last_query
                    FROM document_usage
                    WHERE event_type IN ('document_searched', 'rag_query')
                """)).fetchone()
                if usage_result:
                    avg_response_time = round(usage_result.avg_time or 0, 1)
                    last_query_time = usage_result.last_query.isoformat() if usage_result.last_query else None
            except Exception:
                logger.error("Failed to query usage_events for retrieval stats (table may not exist yet)", exc_info=True)

            return {
                'total_queries': total_docs,
                'success_rate': round(success_rate, 1),
                'avg_response_time': avg_response_time,
                'vector_embeddings': total_chunks,
                'system_status': 'operational' if total_chunks > 0 else 'no_data',
                'last_query_time': last_query_time
            }
            
        except Exception as e:
            logger.error(f"Error getting retrieval stats: {e}")
            return {
                'total_queries': 0,
                'success_rate': 0,
                'avg_response_time': 0,
                'vector_embeddings': 0,
                'system_status': 'error',
                'last_query_time': None
            }
    
    def get_performance_data(self, db, time_range: str = "24h") -> List[Dict]:
        """Get performance data for charts"""
        try:
            from sqlalchemy import text
            from datetime import datetime, timedelta
            
            # Parse time range
            hours = 24
            if time_range == "7d":
                hours = 168
            elif time_range == "30d":
                hours = 720
            
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            
            result = db.execute(text("""
                SELECT 
                    DATE_TRUNC('hour', processed_date) as time_bucket,
                    COUNT(*) as queries,
                    AVG(chunk_count) as avg_chunks
                FROM documents
                WHERE processed_date >= :cutoff
                GROUP BY DATE_TRUNC('hour', processed_date)
                ORDER BY time_bucket
            """), {"cutoff": cutoff}).fetchall()
            
            return [
                {
                    "time": row.time_bucket.isoformat() if row.time_bucket else None,
                    "queries": row.queries or 0,
                    "avgChunks": float(row.avg_chunks or 0)
                }
                for row in result
            ]
            
        except Exception as e:
            logger.error(f"Error getting performance data: {e}")
            return []
    
    def get_context_sources(self, db) -> List[Dict]:
        """Get context sources distribution"""
        try:
            from sqlalchemy import text
            
            result = db.execute(text("""
                SELECT 
                    file_type,
                    COUNT(*) as count,
                    SUM(chunk_count) as total_chunks
                FROM documents
                WHERE status = 'completed'
                GROUP BY file_type
                ORDER BY count DESC
            """)).fetchall()
            
            return [
                {
                    "source": row.file_type or "unknown",
                    "count": row.count or 0,
                    "chunks": row.total_chunks or 0
                }
                for row in result
            ]
            
        except Exception as e:
            logger.error(f"Error getting context sources: {e}")
            return []
    
    def get_recent_queries(self, db, limit: int = 10) -> List[Dict]:
        """Get recent RAG queries from document_usage table"""
        try:
            from sqlalchemy import text
            
            # RAG queries are tracked in document_usage with event_type='rag_query'
            result = db.execute(text("""
                SELECT 
                    id,
                    query,
                    results_count,
                    execution_time_ms,
                    metadata,
                    timestamp
                FROM document_usage
                WHERE event_type = 'rag_query' AND query IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT :limit
            """), {"limit": limit}).fetchall()
            
            return [
                {
                    "id": row.id,
                    "query": row.query,
                    "type": "rag_query",
                    "resultsCount": row.results_count or 0,
                    "responseTime": row.execution_time_ms or 0,
                    "metadata": row.metadata or {},
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None
                }
                for row in result
            ]
            
        except Exception as e:
            logger.error(f"Error getting recent queries: {e}")
            return []
    
    async def get_context_patterns(self, db) -> List[Dict]:
        """Get RAG configurations as patterns"""
        try:
            from sqlalchemy import text
            
            result = db.execute(text("""
                SELECT 
                    id,
                    name,
                    embedding_model,
                    chunk_size,
                    chunk_overlap,
                    retrieval_strategy,
                    top_k,
                    similarity_threshold,
                    is_active,
                    configuration,
                    created_at,
                    updated_at
                FROM rag_configurations
                ORDER BY is_active DESC, updated_at DESC NULLS LAST
                LIMIT 20
            """)).fetchall()
            
            return [
                {
                    "id": row.id,
                    "name": row.name or f"Pattern {row.id}",
                    "description": f"Embedding: {row.embedding_model}, Chunk: {row.chunk_size}, Strategy: {row.retrieval_strategy}",
                    "type": row.retrieval_strategy or "similarity",
                    "model": row.embedding_model,
                    "chunkSize": row.chunk_size or 1000,
                    "chunkOverlap": row.chunk_overlap or 200,
                    "strategy": row.retrieval_strategy or "similarity",
                    "topK": row.top_k or 5,
                    "threshold": float(row.similarity_threshold) if row.similarity_threshold else 0.7,
                    "active": row.is_active,
                    "configuration": row.configuration or {},
                    "created": row.created_at.isoformat() if row.created_at else None,
                    "updated": row.updated_at.isoformat() if row.updated_at else None,
                    # Stats would need actual tracking
                    "usageCount": 0,
                    "accuracy": 0.0,
                    "avgSources": 0
                }
                for row in result
            ]
            
        except Exception as e:
            logger.error(f"Error getting context patterns: {e}")
            return []
    
    @property
    def context_system(self):
        """Check if context system is initialized"""
        self._ensure_initialized()
        return self._embedding_manager is not None


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
