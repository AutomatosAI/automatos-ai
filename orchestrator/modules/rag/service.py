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
from dataclasses import dataclass, field

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
    # PRD-157 S3: numbered-citation source map — [{citation, source_file, document_id, score}]
    sources_map: List[Dict[str, Any]] = field(default_factory=list)


# PRD-157 S4: RAGConfig used to open up to 7 SessionLocal()s per instantiation
# (one per setting). Load them once, in a single session, and memoize — so the
# per-request RAGService construction costs zero DB round-trips after warm-up.
_RAG_SETTINGS_CACHE: Optional[Dict[str, str]] = None


def _load_rag_settings(force: bool = False) -> Dict[str, str]:
    """Load all RAG-related system_settings in ONE session (memoized).

    Only a successful read is cached, so a transient DB error falls back to
    defaults without poisoning the cache.
    """
    global _RAG_SETTINGS_CACHE
    if _RAG_SETTINGS_CACHE is not None and not force:
        return _RAG_SETTINGS_CACHE
    try:
        from core.database.database import SessionLocal
        from core.models.system_settings import SystemSetting

        db = SessionLocal()
        try:
            rows = db.query(SystemSetting.key, SystemSetting.value).all()
            settings = {k: v for k, v in rows if v is not None}
        finally:
            db.close()
        _RAG_SETTINGS_CACHE = settings  # cache only on success
        return settings
    except Exception:
        logger.error("Failed to load RAG settings, using defaults", exc_info=True)
        return {}


def reset_rag_settings_cache() -> None:
    """Drop the memoized RAG settings (call after changing them at runtime)."""
    global _RAG_SETTINGS_CACHE
    _RAG_SETTINGS_CACHE = None


def _get_rag_setting_int(key: str, default: int) -> int:
    raw = _load_rag_settings().get(key)
    try:
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _get_rag_setting_str(key: str, default: str) -> str:
    raw = _load_rag_settings().get(key)
    return raw if raw is not None else default


def _get_rag_setting_float(key: str, default: float) -> float:
    raw = _load_rag_settings().get(key)
    try:
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
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
        # PRD-157 S4: one initialized S3 backend per workspace, reused across
        # queries instead of rebuilt+reinitialized on every _get_candidates call.
        self._s3_backends: Dict[str, Any] = {}
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
        
        # PRD-157 S1: derive the scope through the single fail-closed choke point.
        # A team restriction without a workspace is unscoped → fail closed.
        from modules.rag.retrieval_filters import build_retrieval_filters, RetrievalScopeError
        try:
            filters = build_retrieval_filters(
                workspace_id=workspace_id,
                team=team,
                require_workspace=bool(team),
            )
        except RetrievalScopeError:
            logger.warning("retrieve() requested team scope without workspace_id — failing closed")
            return RAGResult(
                chunks=[],
                formatted_context="No relevant context found.",
                total_tokens=0,
                sources=[],
                query=query,
            )
        team = filters.team  # canonical (lowercased) team or None
        team_multiplier = 2 if filters.has_team_restriction else 1

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

        # Parent-child context expansion (PRD-172 F005: scoped to workspace).
        candidates = await self._expand_to_parent_context(
            candidates, self.config.expansion_window, workspace_id=workspace_id
        )

        # Use existing ContextOptimizer if available
        if self._context_optimizer:
            result = await self._optimize_with_context_optimizer(
                query, candidates, max_chunks, max_tokens, diversity
            )
        else:
            # Fallback to basic retrieval
            result = self._basic_retrieval(query, candidates, max_chunks, max_tokens)

        # Track document access for analytics (PRD-157 S4: fire-and-forget, run
        # off the event loop so it never blocks the retrieval response).
        if result.chunks and workspace_id:
            self._schedule_access_tracking(result.chunks, workspace_id)

        return result

    def _schedule_access_tracking(self, chunks: List[Dict[str, Any]], workspace_id: str) -> None:
        """Run the (blocking) access-tracking update without blocking the caller.

        Inside the async retrieve path this offloads to a worker thread and does
        not await it; with no running loop (sync caller) it runs inline.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            task = loop.create_task(
                asyncio.to_thread(self._track_document_access, chunks, workspace_id)
            )
            # Keep a reference so the task isn't garbage-collected mid-flight.
            self._pending_tracking = getattr(self, "_pending_tracking", set())
            self._pending_tracking.add(task)
            task.add_done_callback(lambda t: self._pending_tracking.discard(t))
        else:
            self._track_document_access(chunks, workspace_id)

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
                        WHERE id = ANY(CAST(:ids AS int[])) AND workspace_id = CAST(:ws AS uuid)
                    """),
                    {"ids": list(doc_ids), "ws": str(workspace_id)},
                )
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"Document access tracking failed: {e}")
    
    @staticmethod
    def _candidate_doc_id(c: Dict) -> Optional[str]:
        """Best-effort document id for a candidate chunk (S3 stores it as external_file_id)."""
        meta = c.get("metadata", {}) or {}
        doc_id = (
            c.get("document_id")
            or meta.get("document_id")
            or meta.get("doc_id")
            or meta.get("external_file_id")
        )
        return str(doc_id) if doc_id else None

    @staticmethod
    def _candidate_is_public(c: Dict) -> bool:
        """True when a candidate's document carries no team restriction.

        ``team_access`` empty or absent → visible workspace-wide; a non-empty list
        → team-restricted. Consulted only on the DB-error fallback, where the live
        access-check is unavailable and we degrade to candidate metadata.
        """
        meta = c.get("metadata", {}) or {}
        return not meta.get("team_access")

    async def _filter_by_team(
        self,
        candidates: List[Dict],
        team: str,
        workspace_id: str,
    ) -> List[Dict]:
        """Filter candidates to team-accessible documents through the centralized
        fail-closed scope helper (PRD-157 S1), preserving the PRD-154 S2 contract:

        * a team-restricted document is NEVER leaked;
        * a DB error degrades to **public docs only** (empty ``team_access``) and
          logs at error level — a transient hiccup must neither blank the results
          nor expose a restricted doc;
        * a candidate with no identifiable document is not document-scoped, so it
          passes through (document ``team_access`` cannot apply to it).
        """
        from modules.rag.retrieval_filters import build_retrieval_filters, allowed_document_ids

        filters = build_retrieval_filters(workspace_id=workspace_id, team=team)
        if not filters.has_team_restriction:
            return candidates  # no team → workspace-only, nothing to post-filter

        doc_ids = {d for d in (self._candidate_doc_id(c) for c in candidates) if d}
        if not doc_ids:
            return candidates  # nothing document-scoped to verify

        from core.database.database import SessionLocal

        try:
            db = SessionLocal()
            try:
                allowed_ids = allowed_document_ids(db, doc_ids, filters)
            finally:
                db.close()
        except Exception:
            # Fail CLOSED to public docs: never leak a team-restricted document on a
            # DB error, but public docs degrade gracefully (PRD-154 S2).
            logger.error(
                "team filter access-check failed; returning public docs only (fail-closed)",
                exc_info=True,
            )
            return [c for c in candidates if self._candidate_is_public(c)]

        filtered = [c for c in candidates if self._candidate_doc_id(c) in allowed_ids]
        logger.info(
            f"team filter: {len(candidates)} → {len(filtered)} candidates (team={filters.team})"
        )
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
        """Score candidates (content quality + source diversity), then select whole
        chunks under the token budget (PRD-157 S3 budgeter) and assemble numbered
        citations. Replaces the former pure-Python knapsack DP."""
        
        logger.info(f"🔍 Starting optimization: {len(candidates)} candidates, max_chunks={max_chunks}, max_tokens={max_tokens}")
        
        # Convert to ContextItems with accurate token counting
        context_items = []
        source_counts = {}
        
        for i, c in enumerate(candidates):
            # Prefer the hydrated full chunk text; fall back to the 500-char preview.
            content = c.get("expanded_content") or c.get("content", "")
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
        
        # PRD-157 S3/S4: token-budgeted whole-chunk selection replaces the
        # pure-Python knapsack DP. context_items already carry the
        # quality-adjusted relevance score and per-chunk token count computed
        # above; the budgeter accumulates whole chunks highest-score-first under
        # the token budget, then assembles numbered [1]..[n] citations.
        from modules.rag.budget import select_within_budget, assemble_with_citations

        scored_chunks = [
            {
                "content": item.content,
                "source_file": item.source,
                "similarity": item.relevance_score,
                "tokens": item.token_count,
                "document_id": item.metadata.get("document_id") if item.metadata else None,
                "metadata": item.metadata.get("original_metadata", {}) if item.metadata else {},
            }
            for item in context_items
        ]

        selection = select_within_budget(
            scored_chunks, max_tokens, max_chunks=max_chunks, score_key="similarity"
        )
        chunks = selection.chunks
        total_tokens = selection.total_tokens

        final_source_counts: Dict[str, int] = {}
        for c in chunks:
            final_source_counts[c["source_file"]] = final_source_counts.get(c["source_file"], 0) + 1
        diversity_score = len(final_source_counts) / len(chunks) if chunks else 0.0

        logger.info(
            "Token-budgeted selection: %d/%d chunks, %d tokens (budget=%d), dropped=%d, diversity=%.2f",
            len(chunks), len(scored_chunks), total_tokens, max_tokens, selection.dropped, diversity_score,
        )

        # Numbered, citation-grade context assembly [1]..[n] + source map.
        formatted_context, sources_map = assemble_with_citations(chunks, query)

        all_values = [item.relevance_score for item in context_items]
        info_gain = (
            sum(c["similarity"] for c in chunks) / len(all_values) if all_values else 0.0
        )

        return RAGResult(
            chunks=chunks,
            formatted_context=formatted_context,
            total_tokens=total_tokens,
            sources=list(set(c["source_file"] for c in chunks)),
            query=query,
            diversity_score=diversity_score,
            information_gain=info_gain,
            sources_map=sources_map,
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
            # Prefer the hydrated full chunk text; fall back to the 500-char preview.
            content = c.get("expanded_content") or c.get("content", "")
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
    
    async def _get_s3_backend(self, workspace_id: str):
        """PRD-157 S4: reuse one initialized S3VectorsBackend per workspace.

        The backend is workspace-scoped, so caching by workspace_id is safe and
        removes the boto3 client + bucket/index checks from every query (the RRF
        path calls _get_candidates up to 5x per retrieve()).
        """
        key = str(workspace_id)
        backend = self._s3_backends.get(key)
        if backend is None:
            from modules.search.vector_store.backends.s3_vectors_backend import S3VectorsBackend

            backend = S3VectorsBackend(workspace_id=key)
            await backend.initialize()
            self._s3_backends[key] = backend
        return backend

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

            # PRD-157 S4: reuse the per-workspace S3 backend (built+initialized once).
            vector_store = await self._get_s3_backend(effective_workspace_id)

            logger.info(f"🔎 S3 Vectors search: workspace={effective_workspace_id}, min_similarity={min_similarity}, limit={limit}")

            # PRD-172 F005: pass an explicit workspace_id filter so the backend
            # drops any hit not scoped to this workspace (defence-in-depth over
            # the per-workspace bucket; a shared/mis-templated bucket no longer
            # leaks cross-workspace chunks into LLM context).
            results = vector_store.search(
                query_embedding=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding),
                limit=limit,
                min_score=min_similarity,
                filters={"workspace_id": str(effective_workspace_id)},
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
        expand_window: int = 1,
        workspace_id: str = None,
    ) -> List[Dict]:
        """
        For each retrieved chunk, fetch surrounding chunks from the same document.
        Uses chunk_index from metadata to find neighbors.

        PRD-172 F005: ``workspace_id`` scopes the document_chunks hydration to
        the caller's workspace. Previously this joined document_chunks by
        (document_id, chunk_index) alone with NO workspace predicate, so on a
        shared store it could hydrate another tenant's chunk text into context.
        """
        if not candidates or not self.config.parent_child_expansion:
            return candidates

        effective_workspace_id = workspace_id or self._workspace_id

        window = expand_window or self.config.expansion_window

        # Plan each candidate's window once, collecting the union of
        # (document_id, chunk_index) keys to hydrate. The full chunk text lives
        # in document_chunks.content; candidate['content'] is only the 500-char
        # S3 preview. One batched query replaces the old query-per-candidate N+1.
        plans = []  # (candidate, doc_id|None, [chunk_index, ...]|None)
        needed = set()
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
                plans.append((candidate, None, None))
                continue

            idx_window = list(range(max(0, chunk_index - window), chunk_index + window + 1))
            for ci in idx_window:
                needed.add((doc_id, ci))
            plans.append((candidate, doc_id, idx_window))

        if not needed:
            for candidate, _, _ in plans:
                candidate["expanded_content"] = candidate.get("content", "")
            return candidates

        try:
            import asyncpg
            from config import config as app_config
            conn = await asyncpg.connect(app_config.DATABASE_URL)

            try:
                pairs = list(needed)
                doc_ids = [p[0] for p in pairs]
                chunk_idxs = [p[1] for p in pairs]
                # ONE round-trip: join the requested (doc_id, chunk_index) keys
                # against document_chunks via unnest of two parallel arrays.
                # PRD-172 F005: additionally join documents and pin
                # documents.workspace_id so a chunk is hydrated ONLY when its
                # parent document belongs to this workspace. When workspace_id
                # is unavailable ($3 IS NULL) the predicate is a no-op (only
                # reachable for a non-multi-tenant / self._workspace_id-less
                # caller), preserving prior behaviour without widening a
                # tenant's scope.
                rows = await conn.fetch("""
                    SELECT dc.document_id, dc.chunk_index, dc.content
                    FROM document_chunks dc
                    JOIN unnest($1::int[], $2::int[]) AS k(document_id, chunk_index)
                      ON dc.document_id = k.document_id
                     AND dc.chunk_index = k.chunk_index
                    JOIN documents d ON d.id = dc.document_id
                    WHERE ($3::uuid IS NULL OR d.workspace_id = $3::uuid)
                """, doc_ids, chunk_idxs, str(effective_workspace_id) if effective_workspace_id else None)
            finally:
                await conn.close()

            hydrated = {(r["document_id"], r["chunk_index"]): r["content"] for r in rows}
            for candidate, doc_id, idx_window in plans:
                if doc_id is None or idx_window is None:
                    candidate["expanded_content"] = candidate.get("content", "")
                    continue
                pieces = [hydrated[(doc_id, ci)] for ci in idx_window if (doc_id, ci) in hydrated]
                candidate["expanded_content"] = "\n\n".join(pieces) if pieces else candidate.get("content", "")
        except Exception as e:
            logger.warning(f"Parent-child expansion failed, using original content: {e}")
            for candidate in candidates:
                candidate["expanded_content"] = candidate.get("content", "")

        return candidates

    def _format_context(self, chunks: List[Dict], query: str) -> str:
        """Format chunks into a numbered-citation context string (PRD-157 S3).

        Delegates to the shared budget assembler so every retrieval path renders
        sources as ``[1]..[n]`` consistently.
        """
        from modules.rag.budget import assemble_with_citations

        formatted_context, _ = assemble_with_citations(chunks, query)
        return formatted_context
    
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
    
    def get_retrieval_stats(self, db, workspace_id=None) -> Dict[str, Any]:
        """Get retrieval statistics from database.

        PRD-172 F045: ``workspace_id`` scopes the document counts to the caller's
        workspace. Previously the ``documents`` COUNT(*) was unscoped, so every
        caller saw a platform-wide total (cross-tenant count). When
        ``workspace_id`` is None the caller is an unfiltered admin aggregate.
        """
        try:
            from sqlalchemy import text

            # Count total RAG queries from documents table, scoped to workspace.
            ws = str(workspace_id) if workspace_id is not None else None
            where = "WHERE workspace_id = :workspace_id" if ws else ""
            result = db.execute(text(f"""
                SELECT
                    COUNT(*) as total_docs,
                    SUM(chunk_count) as total_chunks,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_docs
                FROM documents
                {where}
            """), ({"workspace_id": ws} if ws else {})).fetchone()

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
                        AND metadata->>'workspace_id' = :workspace_id
                """), {"workspace_id": self._workspace_id}).fetchone()
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
                    AND metadata->>'workspace_id' = :workspace_id
                ORDER BY timestamp DESC
                LIMIT :limit
            """), {"limit": limit, "workspace_id": self._workspace_id}).fetchall()
            
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
