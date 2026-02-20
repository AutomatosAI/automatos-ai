"""
Context Engineering Integrator
==============================

Integrates existing RAG and semantic search systems with workflow execution.
Enhances subtask prompts with relevant context from knowledge base.

PHASE 1 ENHANCED: Now includes full ContextOptimizer with mathematical optimization
- Atomic→Molecular→Cellular prompt progression
- Information theory metrics
- MMR example selection
- Knapsack token optimization

PHASE 2 ENHANCED: Now includes CodeGraph integration
- Automatic code context injection for coding tasks
- Semantic code search for relevant functions/classes
- Code snippet extraction with syntax highlighting
- Smart detection of when code context is needed
"""

import logging
import httpx
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from sqlalchemy.orm import Session

# PHASE 1: Import ContextOptimizer from search module
try:
    from modules.search.optimization.context_optimizer import ContextOptimizer, ContextItem, AtomicPrompt
    CONTEXT_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ContextOptimizer not available: {e}. Falling back to basic RAG.")
    CONTEXT_OPTIMIZER_AVAILABLE = False
    # Define dummies for type hints
    @dataclass
    class ContextItem: pass
    class ContextOptimizer: pass

# PHASE 3: Import VectorStore from search module
try:
    from modules.search.vector_store.store import EnhancedVectorStore as PgVectorStore
    from modules.search.vector_store.store import SearchResult as VectorSearchResult
    from core.llm.embedding_manager import EmbeddingManager
    VECTOR_STORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"VectorStore not available: {e}. Using HTTP API fallback.")
    VECTOR_STORE_AVAILABLE = False
    class PgVectorStore: pass
    class VectorSearchResult: pass
    class EmbeddingManager: pass

logger = logging.getLogger(__name__)


@dataclass
class ContextEnhancement:
    """Context enhancement result for a subtask"""
    subtask_id: str
    original_description: str
    enhanced_prompt: str
    rag_context: Dict[str, Any]
    semantic_results: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    num_sources: int = 0
    context_quality_score: float = 0.0
    retrieval_time_ms: int = 0
    
    # PHASE 1: Mathematical optimization metrics
    used_optimization: bool = False
    information_density: float = 0.0
    optimization_score: float = 0.0
    atomic_instruction: str = ""
    examples_used: int = 0
    
    # PHASE 2: CodeGraph integration
    code_results: List[Dict[str, Any]] = field(default_factory=list)
    num_code_symbols: int = 0
    used_code_context: bool = False
    
    @property
    def context_quality(self) -> float:
        """Alias for context_quality_score for frontend compatibility"""
        return self.context_quality_score



class ContextEngineeringIntegrator:
    """
    Integrates existing RAG systems with workflow execution.
    Enhances subtask prompts with relevant context from knowledge base.
    
    Uses:
    - /api/documents/rag/retrieve for comprehensive context (primary)
    - /api/documents/search for targeted lookups (secondary)
    - /api/code-graph/search for code context (PHASE 2)
    """
    
    def __init__(
        self,
        db_session: Session,
        api_base_url: Optional[str] = None,
        use_semantic_enrichment: bool = True,
        use_optimization: bool = True,  # PHASE 1: Enable mathematical optimization
        use_code_context: bool = True   # PHASE 2: Enable CodeGraph integration
    ):
        self.db = db_session
        # Use localhost for local development, not external API
        self.api_base_url = api_base_url or os.getenv("API_URL", "http://localhost:8000")
        
        # CRITICAL FIX: Ensure api_base_url is never empty
        if not self.api_base_url or not self.api_base_url.startswith("http"):
            self.api_base_url = "http://localhost:8000"
            logger.warning(f"⚠️ api_base_url was invalid, forcing to: {self.api_base_url}")
        
        self.use_semantic_enrichment = use_semantic_enrichment
        self.use_optimization = use_optimization and CONTEXT_OPTIMIZER_AVAILABLE
        self.use_code_context = use_code_context  # PHASE 2
        self.logger = logging.getLogger(__name__)
        
        # Log the actual URL being used
        self.logger.info(f"🌐 ContextEngineeringIntegrator initialized with api_base_url: {self.api_base_url}")
        
        # Context enhancement settings
        self.rag_settings = {
            "max_chunks": 5,
            "max_tokens": 2000,
            "diversity": 0.3  # 0=relevance, 1=diversity
        }
        
        self.semantic_settings = {
            "limit": 3,
            "min_similarity": 0.75
        }
        
        # PHASE 2: CodeGraph settings
        self.codegraph_settings = {
            "limit": 5,
            "min_similarity": 0.7,
            "project_id": None  # Will use default project if None
        }
        
        # PHASE 1: Initialize Context Optimizer
        if self.use_optimization:
            try:
                self.context_optimizer = ContextOptimizer()
                logger.info("✅ Context Optimizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ContextOptimizer: {e}")
                self.context_optimizer = None
                self.use_optimization = False
        else:
            self.context_optimizer = None
            
        # PHASE 3: Initialize Vector Store for direct semantic search
        self.vector_store = None
        self.embedding_generator = None
        self.use_vector_store = VECTOR_STORE_AVAILABLE
        
        if self.use_vector_store:
            try:
                # Use centralized embedding manager (reads config from database)
                self.embedding_generator = EmbeddingManager()
                logger.info("✅ Vector Store integration ready (will initialize on first use)")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding generator: {e}")
                self.use_vector_store = False
        
        self.logger.info("ℹ️ Mathematical optimization DISABLED - using basic RAG only")
    
    async def enhance_subtasks_with_context(
        self,
        subtasks: List[Dict[str, Any]],
        workflow_description: str = "",
        workflow_tags: List[str] = None,
        workflow_context: Dict[str, Any] = None
    ) -> Dict[str, ContextEnhancement]:
        """
        Enhance all subtasks with RAG context.
        
        Args:
            subtasks: List of subtasks from RealTaskDecomposer
            workflow_description: Overall workflow description for context
            workflow_tags: Optional workflow tags for project selection (e.g., ["project:my-repo"])
            workflow_context: Optional context dict (e.g., {"codegraph_project": "my-repo"})
            
        Returns:
            Dict mapping subtask_id to ContextEnhancement
        """
        # Store workflow tags and context for project detection
        if workflow_tags:
            self.codegraph_settings["workflow_tags"] = workflow_tags
        if workflow_context:
            self.codegraph_settings["workflow_context"] = workflow_context
        
        enhancements = {}
        
        for idx, subtask in enumerate(subtasks):
            subtask_id = f"subtask_{idx}"
            
            try:
                enhancement = await self._enhance_single_subtask(
                    subtask_id,
                    subtask,
                    workflow_description
                )
                enhancements[subtask_id] = enhancement
                
                self.logger.info(
                    f"✅ Enhanced subtask {idx}: {subtask.get('description', '')[:50]} "
                    f"({enhancement.num_sources} sources, {enhancement.total_tokens} tokens)"
                )
                
            except Exception as e:
                self.logger.error(f"❌ Failed to enhance subtask {idx}: {e}")
                # Create fallback enhancement with no context
                enhancements[subtask_id] = self._create_fallback_enhancement(
                    subtask_id,
                    subtask
                )
        
        return enhancements
    
    async def _enhance_single_subtask(
        self,
        subtask_id: str,
        subtask: Dict[str, Any],
        workflow_description: str
    ) -> ContextEnhancement:
        """Enhance a single subtask with RAG and semantic search"""
        
        start_time = datetime.now()
        
        # Extract task info
        description = subtask.get("description", subtask.get("name", ""))
        agent_type = subtask.get("agent_type", "worker")
        priority = subtask.get("priority", "medium")
        
        self.logger.info(f"🔍 STAGE 3 START: Enhancing {subtask_id}")
        self.logger.info(f"  📝 Description: {description[:100]}...")
        self.logger.info(f"  🤖 Agent Type: {agent_type}")
        self.logger.info(f"  ⚡ Priority: {priority}")
        
        # 1. Build RAG query
        rag_query = self._build_rag_query(description, agent_type, workflow_description)
        self.logger.info(f"  🔍 RAG Query built: '{rag_query[:150]}...'")
        
        # 2. Retrieve RAG context (primary) - UNCHANGED
        rag_context = await self._retrieve_rag_context(rag_query)
        
        # Debug RAG retrieval results
        chunks = rag_context.get("chunks", [])
        self.logger.info(f"  📚 RAG Retrieved: {len(chunks)} chunks, {rag_context.get('total_tokens', 0)} tokens")
        if chunks:
            for i, chunk in enumerate(chunks[:2]):  # Log first 2 chunks
                similarity = chunk.get("similarity_score", chunk.get("similarity", "NO_SCORE"))
                content_preview = chunk.get("content", chunk.get("text", ""))[:80]
                self.logger.info(f"    Chunk {i+1}: similarity={similarity}, content='{content_preview}...'")
        else:
            self.logger.warning(f"  ⚠️  RAG returned NO chunks!")
        
        # 3. Optional: Semantic search for targeted enrichment
        semantic_results = []
        if self.use_semantic_enrichment and self._should_use_semantic_search(subtask):
            semantic_query = self._build_semantic_query(description, agent_type)
            semantic_results = await self._retrieve_semantic_results(semantic_query)
            self.logger.info(f"  🔍 Semantic Search: {len(semantic_results)} results")
        
        # PHASE 2: 4. Optional: CodeGraph search for code context
        code_results = []
        if self.use_code_context and self._should_use_code_context(subtask):
            code_query = self._build_code_query(description, agent_type)
            code_results = await self._retrieve_code_context(code_query)
        
        # PHASE 1/2: Apply mathematical optimization if enabled
        if self.use_optimization and self.context_optimizer:
            enhanced_prompt, optimization_metrics = await self._apply_optimization(
                description,
                agent_type,
                priority,
                rag_context,
                semantic_results,
                subtask,
                code_results  # PHASE 2: Pass code results
            )
            used_optimization = True
        else:
            # Fallback to basic prompt building
            enhanced_prompt = self._build_enhanced_prompt(
                description,
                agent_type,
                priority,
                rag_context,
                semantic_results,
                code_results  # PHASE 2: Pass code results
            )
            optimization_metrics = {
                "information_density": 0.0,
                "optimization_score": 0.0,
                "atomic_instruction": description,
                "examples_used": 0
            }
            used_optimization = False
        
        # 5. Calculate quality metrics
        total_tokens = rag_context.get("total_tokens", 0)
        num_sources = len(rag_context.get("chunks", []))
        num_code_symbols = len(code_results)  # PHASE 2
        context_quality = self._calculate_context_quality(rag_context, semantic_results, code_results)  # PHASE 2: Include code
        
        retrieval_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        self.logger.info(f"🎯 STAGE 3 COMPLETE: {subtask_id}")
        self.logger.info(f"  📊 Quality={context_quality:.2%}, Sources={num_sources}, Tokens={total_tokens}, Time={retrieval_time}ms")
        self.logger.info(f"  ✅ Optimization={'YES' if used_optimization else 'NO'}, Prompt_length={len(enhanced_prompt)} chars")
        
        return ContextEnhancement(
            subtask_id=subtask_id,
            original_description=description,
            enhanced_prompt=enhanced_prompt,
            rag_context=rag_context,
            semantic_results=semantic_results,
            total_tokens=total_tokens,
            num_sources=num_sources,
            context_quality_score=context_quality,
            retrieval_time_ms=retrieval_time,
            # PHASE 1: Add optimization metrics
            used_optimization=used_optimization,
            information_density=optimization_metrics.get("information_density", 0.0),
            optimization_score=optimization_metrics.get("optimization_score", 0.0),
            atomic_instruction=optimization_metrics.get("atomic_instruction", description),
            examples_used=optimization_metrics.get("examples_used", 0),
            # PHASE 2: Add CodeGraph metrics
            code_results=code_results,
            num_code_symbols=num_code_symbols,
            used_code_context=len(code_results) > 0
        )
    
    def _build_rag_query(
        self,
        description: str,
        agent_type: str,
        workflow_description: str
    ) -> str:
        """Build optimized query for RAG retrieval"""
        
        # Combine description with agent type for better context
        query_parts = [description]
        
        # Add agent type context
        agent_context = {
            "researcher": "research documentation analysis",
            "developer": "code implementation examples",
            "analyst": "data analysis patterns",
            "architect": "system design patterns",
            "reviewer": "code review best practices",
            "tester": "testing strategies qa",
            "writer": "documentation writing",
            "orchestrator": "workflow coordination"
        }
        
        if agent_type.lower() in agent_context:
            query_parts.append(agent_context[agent_type.lower()])
        
        # Add workflow context if relevant
        if workflow_description and len(workflow_description) > 10:
            query_parts.append(workflow_description[:100])
        
        return " ".join(query_parts)
    
    async def _retrieve_rag_context(self, query: str) -> Dict[str, Any]:
        """Call existing RAG endpoint for context retrieval"""
        
        # Use real RAG service with vector search
        try:
            from modules.rag import get_rag_service
            rag_service = get_rag_service()
            
            # Get enhanced prompt with context
            enhanced_prompt, metadata = await rag_service.enhance_prompt_with_context(
                original_prompt=query,
                max_context_tokens=2000
            )
            
            # Extract the context portion
            context_start = enhanced_prompt.find("## Relevant Context")
            context_end = enhanced_prompt.find("## Original Task")
            
            if context_start != -1 and context_end != -1:
                context = enhanced_prompt[context_start:context_end].strip()
            else:
                context = ""
            
            return {
                "query": query,
                "chunks": metadata.get("sources", []),
                "context": context,
                "total_tokens": metadata.get("total_context_tokens", 0),
                "enhanced": metadata.get("enhanced", False),
                "documents_used": metadata.get("documents_used", 0)
            }
        except Exception as e:
            self.logger.warning(f"RAG retrieval failed: {e}, returning empty context")
            return {
                "query": query,
                "chunks": [],
                "context": "",
                "total_tokens": 0,
                "note": f"RAG service error: {str(e)}"
            }
    
    def _should_use_semantic_search(self, subtask: Dict[str, Any]) -> bool:
        """Determine if semantic search would be beneficial"""
        
        description = subtask.get("description", "").lower()
        
        # Use semantic search for specific lookup keywords
        lookup_keywords = [
            "find", "locate", "search", "identify", "lookup",
            "get", "fetch", "retrieve", "extract",
            "pattern", "example", "template", "reference"
        ]
        
        return any(keyword in description for keyword in lookup_keywords)
    
    # ======================================================================
    # PHASE 2: CODE CONTEXT METHODS
    # ======================================================================
    
    def _should_use_code_context(self, subtask: Dict[str, Any]) -> bool:
        """Determine if code context from CodeGraph would be beneficial"""
        
        description = subtask.get("description", "").lower()
        agent_type = subtask.get("agent_type", "").lower()
        
        # Code-related agent types
        code_agent_types = ["developer", "architect", "reviewer", "tester"]
        if agent_type in code_agent_types:
            return True
        
        # Code-related keywords in description
        code_keywords = [
            "code", "function", "class", "method", "api", "endpoint",
            "implement", "develop", "refactor", "debug", "fix",
            "bug", "error", "issue", "module", "component",
            "authentication", "login", "database", "query",
            "test", "testing", "unit test", "integration",
            "review", "analyze", "inspect", "examine code"
        ]
        
        return any(keyword in description for keyword in code_keywords)
    
    def _build_code_query(self, description: str, agent_type: str) -> str:
        """Build focused query for CodeGraph semantic search"""
        
        # Extract key technical terms
        words = description.split()
        
        # Filter out common words and keep technical terms
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "how", "what", "where", "when", "why", "please", "can", "you", "we",
            "implement", "create", "build", "develop", "write", "make"
        }
        
        key_terms = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        
        # Focus on technical terms
        return " ".join(key_terms[:7])  # Top 7 technical terms
    
    async def _detect_project_name(self) -> Optional[str]:
        """
        Smart project detection:
        1. Check if already cached in settings
        2. Parse workflow tags for "project:name" pattern
        3. Query CodeGraph API for available projects
        4. If only 1 project, use it automatically
        5. If multiple, use first one
        """
        # Check cache
        if self.codegraph_settings.get("project_name"):
            return self.codegraph_settings["project_name"]
        
        # 1. Check workflow.context for codegraph_project (highest priority)
        workflow_context = self.codegraph_settings.get("workflow_context", {})
        if workflow_context and isinstance(workflow_context, dict):
            project_name = workflow_context.get("codegraph_project")
            if project_name:
                self.codegraph_settings["project_name"] = project_name
                self.logger.info(f"✅ Using project from workflow.context: {project_name}")
                return project_name
        
        # 2. Check workflow tags for "project:name" pattern
        workflow_tags = self.codegraph_settings.get("workflow_tags", [])
        if workflow_tags:
            for tag in workflow_tags:
                if isinstance(tag, str) and tag.startswith("project:"):
                    project_name = tag.split(":", 1)[1].strip()
                    if project_name:
                        self.codegraph_settings["project_name"] = project_name
                        self.logger.info(f"✅ Using project from tag: {project_name}")
                        return project_name
        
        try:
            # Query available projects
            url = f"{self.api_base_url}/api/code-graph/projects"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                projects = response.json()
            
            if not projects:
                self.logger.warning("⚠️ No projects indexed in CodeGraph")
                return None
            
            if len(projects) == 1:
                # Auto-use the only available project
                project_name = projects[0].get("name")
                self.codegraph_settings["project_name"] = project_name  # Cache it
                self.logger.info(f"✅ Auto-detected CodeGraph project: {project_name}")
                return project_name
            
            # Multiple projects - use first one (could be enhanced)
            project_name = projects[0].get("name")
            self.codegraph_settings["project_name"] = project_name
            self.logger.info(f"📚 Multiple projects found, using: {project_name} (add 'project:name' tag to specify)")
            return project_name
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to detect CodeGraph project: {e}")
            return None
    
    async def _retrieve_code_context(self, query: str) -> List[Dict[str, Any]]:
        """Call CodeGraph API for code symbol search"""
        
        try:
            # Smart project detection
            project_name = await self._detect_project_name()
            
            if not project_name:
                self.logger.warning("⚠️ No CodeGraph project available, skipping code search")
                return []
            
            url = f"{self.api_base_url}/api/code-graph/search/semantic"
            
            params = {
                "project": project_name,
                "q": query,
                "limit": self.codegraph_settings["limit"]
            }
            
            self.logger.info(f"🔍 CodeGraph search: '{query}' in project '{project_name}' (limit={params['limit']})")
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                self.logger.info(f"✅ CodeGraph returned {len(results)} code symbols")
                return results
                
        except Exception as e:
            self.logger.error(f"❌ CodeGraph search failed: {e}")
            return []
    
    def _build_semantic_query(self, description: str, agent_type: str) -> str:
        """Build focused query for semantic search"""
        
        # Extract key terms (simple implementation - can be enhanced)
        words = description.split()
        
        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        key_terms = [w for w in words if w.lower() not in stop_words and len(w) > 3]
        
        return " ".join(key_terms[:5])  # Top 5 key terms
    
    async def _retrieve_semantic_results(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve semantic search results using PgVectorStore (PHASE 3) or fallback to HTTP API.
        """
        # PHASE 3: Try direct vector store first
        if self.use_vector_store:
            try:
                return await self._retrieve_from_vector_store(query)
            except Exception as e:
                self.logger.warning(f"Vector store query failed, falling back to HTTP: {e}")
        
        # Fallback to HTTP API
        try:
            url = f"{self.api_base_url}/api/documents/search"
            
            params = {
                "q": query,
                "limit": self.semantic_settings["limit"]
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                results = response.json()
            
            self.logger.debug(f"Semantic search (HTTP fallback) returned {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.warning(f"Semantic search (HTTP) failed: {e}")
            return []
    
    async def _retrieve_from_vector_store(
        self,
        query: str,
        limit: int = None,
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        PHASE 3: Retrieve context from PgVectorStore using semantic search.
        
        Args:
            query: Search query  
            limit: Maximum chunks to retrieve (uses self.semantic_settings if None)
            similarity_threshold: Minimum similarity score (uses self.semantic_settings if None)
            
        Returns:
            List of context chunks
        """
        if not self.use_vector_store:
            return []
            
        limit = limit or self.semantic_settings.get("limit", 5)
        similarity_threshold = similarity_threshold or self.semantic_settings.get("min_similarity", 0.7)
            
        try:
            # Lazy initialization of vector store
            if not self.vector_store:
                from core.database.database import get_database_url as get_db_session_string
                db_string = get_db_session_string()
                self.vector_store = PgVectorStore(connection_string=db_string)
                # Read dimension from system_settings (matches embedding model)
                from modules.rag.service import _get_rag_setting_int
                vector_dim = _get_rag_setting_int("vector_store_dimensions", 4096)
                await self.vector_store.initialize(dimension=vector_dim)
                self.logger.info("✅ PgVectorStore initialized for semantic search")
                
            # Generate query embedding (generate_embeddings returns list of dicts with 'embedding' key)
            embedding_results = await self.embedding_generator.generate_embeddings([query])
            if not embedding_results:
                raise ValueError("Failed to generate query embedding")
            query_embedding = embedding_results[0].get("embedding", embedding_results[0])
            
            # Perform semantic search
            results = await self.vector_store.similarity_search(
                query_embedding=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            # Convert VectorSearchResult to dict format (compatible with existing code)
            chunks = []
            for result in results:
                chunks.append({
                    "content": result.content,
                    "source_file": result.source_file,
                    "chunk_id": result.id,
                    "similarity_score": result.similarity_score,
                    "similarity": result.similarity_score, # Alias for compatibility
                    "metadata": result.metadata,
                    "chunk_index": result.chunk_index,
                    "content_type": result.content_type,
                    "document_id": result.id  # Alias
                })
                
            self.logger.info(f"📊 PgVectorStore returned {len(chunks)} chunks (threshold={similarity_threshold})")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error retrieving from vector store: {e}")
            raise  # Re-raise so fallback can catch it
    
    def _build_enhanced_prompt(
        self,
        description: str,
        agent_type: str,
        priority: str,
        rag_context: Dict[str, Any],
        semantic_results: List[Dict[str, Any]],
        code_results: List[Dict[str, Any]] = None  # PHASE 2
    ) -> str:
        """Build enhanced prompt with injected context"""
        
        prompt_parts = []
        
        # 1. Task header
        prompt_parts.append(f"# Task: {description}")
        prompt_parts.append(f"Agent Type: {agent_type} | Priority: {priority}")
        prompt_parts.append("")
        
        # PHASE 2: 2. Code Context (highest priority for developers)
        if code_results:
            prompt_parts.append("## 📝 Relevant Code from Codebase:")
            prompt_parts.append("")
            prompt_parts.append("**IMPORTANT**: Use the following code snippets as the PRIMARY source for your analysis.")
            prompt_parts.append("These are ACTUAL implementations from the codebase - cite them directly in your response.")
            prompt_parts.append("")
            
            for idx, code in enumerate(code_results[:3], 1):
                symbol_name = code.get("symbol_name", "Unknown")
                file_path = code.get("file_path", "Unknown")
                line_number = code.get("line_number", 0)
                symbol_type = code.get("symbol_type", "symbol")
                code_snippet = code.get("code_snippet", "") or code.get("code", "")
                signature = code.get("signature", "")
                docstring = code.get("docstring", "")
                similarity = code.get("similarity", 0)
                
                prompt_parts.append(f"### Code {idx}: `{symbol_name}` ({symbol_type})")
                prompt_parts.append(f"**File**: `{file_path}:{line_number}`")
                prompt_parts.append(f"**Relevance**: {similarity:.0%}")
                if signature:
                    prompt_parts.append(f"**Signature**: `{signature}`")
                if docstring:
                    prompt_parts.append(f"**Documentation**: {docstring}")
                prompt_parts.append("")
                prompt_parts.append("```" + code.get("language", "python"))
                prompt_parts.append(code_snippet[:1500])  # Increased from 800 to 1500
                if len(code_snippet) > 1500:
                    prompt_parts.append("... (code continues)")
                prompt_parts.append("```")
                prompt_parts.append("")
        
        # 3. RAG Context (primary)
        if rag_context.get("chunks"):
            prompt_parts.append("## Relevant Context from Knowledge Base:")
            prompt_parts.append("")
            
            for idx, chunk in enumerate(rag_context["chunks"][:3], 1):
                content = chunk.get("content", "")
                source = chunk.get("source_file", chunk.get("filename", "Unknown"))
                similarity = chunk.get("similarity_score", chunk.get("similarity", 0))
                
                prompt_parts.append(f"### Source {idx}: {source} (relevance: {similarity:.0%})")
                prompt_parts.append(content[:500])  # Limit chunk size
                prompt_parts.append("")
        
        # 4. Semantic results (supplementary)
        if semantic_results:
            prompt_parts.append("## Additional References:")
            prompt_parts.append("")
            
            for idx, result in enumerate(semantic_results[:2], 1):
                content = result.get("content", "")
                source = result.get("filename", "Unknown")
                
                prompt_parts.append(f"**Reference {idx}** ({source}):")
                prompt_parts.append(content[:300])
                prompt_parts.append("")
        
        # 5. Task instruction
        prompt_parts.append("## 🎯 Your Task:")
        prompt_parts.append(f"{description}")
        prompt_parts.append("")
        
        # Add code-specific guidance if code context is present
        if code_results:
            prompt_parts.append("⚠️ **CRITICAL REQUIREMENTS**:")
            prompt_parts.append("1. You MUST analyze the actual code snippets shown above under '📝 Relevant Code from Codebase'")
            prompt_parts.append("2. Reference specific function names, class names, and code implementations")
            prompt_parts.append("3. Quote relevant lines of code in your analysis")
            prompt_parts.append("4. DO NOT write 'empty sections' or 'no code available' - the code IS PROVIDED ABOVE")
            prompt_parts.append("5. Your report should be based on REAL implementations, not hypothetical ones")
            prompt_parts.append("")
        
        prompt_parts.append("Consider all context and references provided above when completing this task.")
        
        return "\n".join(prompt_parts)
    
    def _calculate_context_quality(
        self,
        rag_context: Dict[str, Any],
        semantic_results: List[Dict[str, Any]],
        code_results: List[Dict[str, Any]] = None  # PHASE 2
    ) -> float:
        """Calculate quality score for retrieved context"""
        
        self.logger.info(f"  📊 Calculating context quality...")
        
        score = 0.0
        total_weight = 0.0
        
        # RAG context quality (50% weight)
        chunks = rag_context.get("chunks", [])
        if chunks:
            similarities = [c.get("similarity_score", c.get("similarity", 0)) for c in chunks]
            avg_similarity = sum(similarities) / len(chunks)
            score += avg_similarity * 0.5
            total_weight += 0.5
            self.logger.info(f"    RAG: {len(chunks)} chunks, similarities={similarities}, avg={avg_similarity:.3f}, weighted_score={avg_similarity * 0.5:.3f}")
        else:
            self.logger.warning(f"    RAG: NO CHUNKS (0 contribution to quality)")
        
        # Semantic results quality (20% weight)
        if semantic_results:
            similarities = [r.get("similarity", 0) for r in semantic_results]
            avg_similarity = sum(similarities) / len(semantic_results)
            score += avg_similarity * 0.2
            total_weight += 0.2
            self.logger.info(f"    Semantic: {len(semantic_results)} results, similarities={similarities}, avg={avg_similarity:.3f}, weighted_score={avg_similarity * 0.2:.3f}")
        else:
            self.logger.info(f"    Semantic: No results (0 contribution)")
        
        # PHASE 2: Code context quality (30% weight)
        if code_results:
            similarities = [c.get("similarity", 0) for c in code_results]
            avg_similarity = sum(similarities) / len(code_results)
            score += avg_similarity * 0.3
            total_weight += 0.3
            self.logger.info(f"    Code: {len(code_results)} results, similarities={similarities}, avg={avg_similarity:.3f}, weighted_score={avg_similarity * 0.3:.3f}")
        else:
            self.logger.info(f"    Code: No results (0 contribution)")
        
        # Normalize by actual weights used
        if total_weight > 0:
            score = score / total_weight
            self.logger.info(f"  ✅ Final quality: {score:.3f} (total_weight={total_weight})")
        else:
            self.logger.error(f"  ❌ CRITICAL: total_weight=0! No context sources had data!")
        
        return min(score, 1.0)
    
    def _create_fallback_enhancement(
        self,
        subtask_id: str,
        subtask: Dict[str, Any]
    ) -> ContextEnhancement:
        """Create fallback enhancement when context retrieval fails"""
        
        description = subtask.get("description", subtask.get("name", ""))
        
        return ContextEnhancement(
            subtask_id=subtask_id,
            original_description=description,
            enhanced_prompt=f"# Task: {description}\n\n## Instructions:\nPlease {description}",
            rag_context={"chunks": [], "context": "", "total_tokens": 0},
            semantic_results=[],
            total_tokens=0,
            num_sources=0,
            context_quality_score=0.0,
            retrieval_time_ms=0,
            used_optimization=False,
            information_density=0.0,
            optimization_score=0.0,
            atomic_instruction=description,
            examples_used=0,
            # PHASE 2
            code_results=[],
            num_code_symbols=0,
            used_code_context=False
        )
    
    # ======================================================================
    # PHASE 1: MATHEMATICAL OPTIMIZATION METHODS
    # ======================================================================
    
    async def _apply_optimization(
        self,
        description: str,
        agent_type: str,
        priority: str,
        rag_context: Dict[str, Any],
        semantic_results: List[Dict[str, Any]],
        subtask: Dict[str, Any],
        code_results: List[Dict[str, Any]] = None  # PHASE 2
    ) -> tuple[str, Dict[str, Any]]:
        """
        Apply full ContextOptimizer pipeline with mathematical optimization.
        
        This wraps existing RAG results with:
        - Atomic→Molecular prompt progression
        - Information theory calculations
        - MMR example selection
        - Knapsack token optimization
        """
        try:
            # Step 1: Create AtomicPrompt (basic instruction)
            atomic_prompt = AtomicPrompt(
                instruction=description,
                constraints=subtask.get("constraints", []),
                output_format="JSON"
            )
            
            # Step 2: Convert RAG results to ContextItems
            context_items = self._convert_rag_to_context_items(rag_context)
            
            # Step 3: Select examples using MMR (if available)
            # Note: For now, we'll use empty examples. Can add example library later.
            examples = []
            
            # Step 4: Optimize context with knapsack algorithm
            if context_items:
                # Use the existing optimize_context method
                optimized_result = await self.context_optimizer.optimize_context(
                    available_context=context_items,
                    max_tokens=2000,  # Limit context window
                    objective="maximize_information"
                )
                
                optimized_items = optimized_result.contexts
                
                # Construct optimized prompt
                optimized_context_str = "\n\n".join([f"### {item.source} (Relevance: {item.relevance_score:.2f})\n{item.content}" for item in optimized_items])
                
                enhanced_prompt = f"""# Task: {description}
Agent Type: {agent_type} | Priority: {priority}

## Optimized Context (Knapsack Selected):
{optimized_context_str}

## Instructions:
{description}
"""
                metrics = {
                    "information_density": optimized_result.information_gain, # Use gain as proxy for density metric in this view
                    "optimization_score": optimized_result.diversity_score,
                    "atomic_instruction": description,
                    "examples_used": 0
                }
                return enhanced_prompt, metrics
            else:
                # No context to optimize
                return self._build_enhanced_prompt(description, agent_type, priority, rag_context, semantic_results, code_results), {}

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return self._build_enhanced_prompt(description, agent_type, priority, rag_context, semantic_results, code_results), {}

    def _convert_rag_to_context_items(self, rag_context: Dict[str, Any]) -> List[ContextItem]:
        """Convert RAG chunks to ContextItem objects"""
        items = []
        chunks = rag_context.get("chunks", [])
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            # Estimate tokens if not provided (approx 4 chars per token)
            tokens = chunk.get("tokens", len(content) // 4)
            relevance = chunk.get("similarity_score", chunk.get("similarity", 0.5))
            source = chunk.get("source_file", f"chunk_{i}")
            
            items.append(ContextItem(
                content=content,
                source=source,
                context_type="documentation",
                relevance_score=relevance,
                token_count=tokens
            ))
        return items

    
    def get_enhancement_summary(
        self,
        enhancements: Dict[str, ContextEnhancement]
    ) -> Dict[str, Any]:
        """Generate summary of context enhancement results"""
        
        total_subtasks = len(enhancements)
        subtasks_with_context = sum(
            1 for e in enhancements.values() if e.num_sources > 0
        )
        
        total_tokens = sum(e.total_tokens for e in enhancements.values())
        total_sources = sum(e.num_sources for e in enhancements.values())
        avg_quality = sum(e.context_quality_score for e in enhancements.values()) / max(total_subtasks, 1)
        avg_retrieval_time = sum(e.retrieval_time_ms for e in enhancements.values()) / max(total_subtasks, 1)
        
        # PHASE 1: Add optimization metrics
        subtasks_optimized = sum(1 for e in enhancements.values() if e.used_optimization)
        avg_info_density = sum(e.information_density for e in enhancements.values()) / max(total_subtasks, 1)
        avg_opt_score = sum(e.optimization_score for e in enhancements.values()) / max(total_subtasks, 1)
        total_examples = sum(e.examples_used for e in enhancements.values())
        
        # PHASE 2: Add CodeGraph metrics
        subtasks_with_code = sum(1 for e in enhancements.values() if e.used_code_context)
        total_code_symbols = sum(e.num_code_symbols for e in enhancements.values())
        
        return {
            "total_subtasks": total_subtasks,
            "subtasks_with_context": subtasks_with_context,
            "context_coverage": subtasks_with_context / total_subtasks if total_subtasks > 0 else 0,
            "total_tokens_used": total_tokens,
            "total_sources_retrieved": total_sources,
            "avg_context_quality": avg_quality,
            "avg_retrieval_time_ms": int(avg_retrieval_time),
            # PHASE 1: Optimization metrics
            "optimization_enabled": self.use_optimization,
            "subtasks_optimized": subtasks_optimized,
            "optimization_rate": subtasks_optimized / total_subtasks if total_subtasks > 0 else 0,
            "avg_information_density": avg_info_density,
            "avg_optimization_score": avg_opt_score,
            "total_examples_used": total_examples,
            # PHASE 2: CodeGraph metrics
            "codegraph_enabled": self.use_code_context,
            "subtasks_with_code": subtasks_with_code,
            "code_context_rate": subtasks_with_code / total_subtasks if total_subtasks > 0 else 0,
            "total_code_symbols_retrieved": total_code_symbols,
            "timestamp": datetime.now().isoformat()
        }

