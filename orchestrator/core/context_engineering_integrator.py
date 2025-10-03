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
"""

import logging
import httpx
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from sqlalchemy.orm import Session

# PHASE 1: Import ContextOptimizer components
try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from context_engineering.context_optimizer import (
        ContextOptimizer,
        AtomicPrompt,
        EnhancedPrompt,
        ContextItem,
        Example
    )
    CONTEXT_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ContextOptimizer not available: {e}. Falling back to basic RAG.")
    CONTEXT_OPTIMIZER_AVAILABLE = False

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


class ContextEngineeringIntegrator:
    """
    Integrates existing RAG systems with workflow execution.
    Enhances subtask prompts with relevant context from knowledge base.
    
    Uses:
    - /api/documents/rag/retrieve for comprehensive context (primary)
    - /api/documents/search for targeted lookups (secondary)
    """
    
    def __init__(
        self,
        db_session: Session,
        api_base_url: Optional[str] = None,
        use_semantic_enrichment: bool = True,
        use_optimization: bool = True  # PHASE 1: Enable mathematical optimization
    ):
        self.db = db_session
        self.api_base_url = api_base_url or os.getenv("API_URL", "https://api.automatos.app")
        self.use_semantic_enrichment = use_semantic_enrichment
        self.use_optimization = use_optimization and CONTEXT_OPTIMIZER_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
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
        
        # PHASE 1: Initialize ContextOptimizer
        if self.use_optimization:
            try:
                self.context_optimizer = ContextOptimizer(db_session=db_session)
                self.logger.info("✅ ContextOptimizer initialized - mathematical optimization ENABLED")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ContextOptimizer: {e}. Using basic RAG only.")
                self.use_optimization = False
                self.context_optimizer = None
        else:
            self.context_optimizer = None
            self.logger.info("ℹ️ Mathematical optimization DISABLED - using basic RAG only")
    
    async def enhance_subtasks_with_context(
        self,
        subtasks: List[Dict[str, Any]],
        workflow_description: str = ""
    ) -> Dict[str, ContextEnhancement]:
        """
        Enhance all subtasks with RAG context.
        
        Args:
            subtasks: List of subtasks from RealTaskDecomposer
            workflow_description: Overall workflow description for context
            
        Returns:
            Dict mapping subtask_id to ContextEnhancement
        """
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
        
        # 1. Build RAG query
        rag_query = self._build_rag_query(description, agent_type, workflow_description)
        
        # 2. Retrieve RAG context (primary) - UNCHANGED
        rag_context = await self._retrieve_rag_context(rag_query)
        
        # 3. Optional: Semantic search for targeted enrichment
        semantic_results = []
        if self.use_semantic_enrichment and self._should_use_semantic_search(subtask):
            semantic_query = self._build_semantic_query(description, agent_type)
            semantic_results = await self._retrieve_semantic_results(semantic_query)
        
        # PHASE 1: Apply mathematical optimization if enabled
        if self.use_optimization and self.context_optimizer:
            enhanced_prompt, optimization_metrics = await self._apply_optimization(
                description,
                agent_type,
                priority,
                rag_context,
                semantic_results,
                subtask
            )
            used_optimization = True
        else:
            # Fallback to basic prompt building
            enhanced_prompt = self._build_enhanced_prompt(
                description,
                agent_type,
                priority,
                rag_context,
                semantic_results
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
        context_quality = self._calculate_context_quality(rag_context, semantic_results)
        
        retrieval_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
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
            examples_used=optimization_metrics.get("examples_used", 0)
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
        
        try:
            url = f"{self.api_base_url}/api/documents/rag/retrieve"
            
            params = {
                "query": query,
                "max_chunks": self.rag_settings["max_chunks"],
                "max_tokens": self.rag_settings["max_tokens"],
                "diversity": self.rag_settings["diversity"]
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, params=params)
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            self.logger.error(f"RAG retrieval failed: {e}")
            return {
                "query": query,
                "chunks": [],
                "context": "",
                "total_tokens": 0,
                "error": str(e)
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
    
    def _build_semantic_query(self, description: str, agent_type: str) -> str:
        """Build focused query for semantic search"""
        
        # Extract key terms (simple implementation - can be enhanced)
        words = description.split()
        
        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        key_terms = [w for w in words if w.lower() not in stop_words and len(w) > 3]
        
        return " ".join(key_terms[:5])  # Top 5 key terms
    
    async def _retrieve_semantic_results(self, query: str) -> List[Dict[str, Any]]:
        """Call semantic search endpoint for targeted results"""
        
        try:
            url = f"{self.api_base_url}/api/documents/search"
            
            params = {
                "query": query,
                "limit": self.semantic_settings["limit"],
                "min_similarity": self.semantic_settings["min_similarity"]
            }
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(url, params=params)
                response.raise_for_status()
                data = response.json()
                return data.get("results", [])
                
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def _build_enhanced_prompt(
        self,
        description: str,
        agent_type: str,
        priority: str,
        rag_context: Dict[str, Any],
        semantic_results: List[Dict[str, Any]]
    ) -> str:
        """Build enhanced prompt with injected context"""
        
        prompt_parts = []
        
        # 1. Task header
        prompt_parts.append(f"# Task: {description}")
        prompt_parts.append(f"Agent Type: {agent_type} | Priority: {priority}")
        prompt_parts.append("")
        
        # 2. RAG Context (primary)
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
        
        # 3. Semantic results (supplementary)
        if semantic_results:
            prompt_parts.append("## Additional References:")
            prompt_parts.append("")
            
            for idx, result in enumerate(semantic_results[:2], 1):
                content = result.get("content", "")
                source = result.get("filename", "Unknown")
                
                prompt_parts.append(f"**Reference {idx}** ({source}):")
                prompt_parts.append(content[:300])
                prompt_parts.append("")
        
        # 4. Task instruction
        prompt_parts.append("## Instructions:")
        prompt_parts.append(f"Please {description}")
        prompt_parts.append("")
        prompt_parts.append("Consider the context and references provided above when completing this task.")
        
        return "\n".join(prompt_parts)
    
    def _calculate_context_quality(
        self,
        rag_context: Dict[str, Any],
        semantic_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate quality score for retrieved context"""
        
        score = 0.0
        
        # RAG context quality (70% weight)
        chunks = rag_context.get("chunks", [])
        if chunks:
            avg_similarity = sum(
                c.get("similarity_score", c.get("similarity", 0))
                for c in chunks
            ) / len(chunks)
            score += avg_similarity * 0.7
        
        # Semantic results quality (30% weight)
        if semantic_results:
            avg_similarity = sum(
                r.get("similarity", 0)
                for r in semantic_results
            ) / len(semantic_results)
            score += avg_similarity * 0.3
        
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
            examples_used=0
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
        subtask: Dict[str, Any]
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
                optimized_context = await self.context_optimizer.optimize_context(
                    available_context=context_items,
                    max_tokens=4000,
                    objective="maximize_information"
                )
            else:
                # No context to optimize
                optimized_context = type('obj', (object,), {
                    'contexts': [],
                    'expected_information_gain': 0.0
                })()
            
            # Step 5: Build MolecularContext (Atomic + Examples + Context)
            molecular_context = await self.context_optimizer.build_molecular_context(
                atomic_prompt=atomic_prompt,
                examples=examples,
                context_items=optimized_context.contexts if hasattr(optimized_context, 'contexts') else [],
                max_tokens=4000
            )
            
            # Step 6: Calculate information density
            information_density = self.context_optimizer.calculate_information_density(
                molecular_context.full_prompt
            )
            
            # Step 7: Build optimization metrics
            metrics = {
                "information_density": information_density,
                "optimization_score": optimized_context.expected_information_gain if hasattr(optimized_context, 'expected_information_gain') else 0.0,
                "atomic_instruction": atomic_prompt.instruction,
                "examples_used": len(examples),
                "total_tokens": molecular_context.total_tokens,
                "context_items_used": len(optimized_context.contexts) if hasattr(optimized_context, 'contexts') else 0
            }
            
            self.logger.info(
                f"✅ Optimization applied: {information_density:.2f} density, "
                f"{molecular_context.total_tokens} tokens, "
                f"{len(optimized_context.contexts) if hasattr(optimized_context, 'contexts') else 0} context items"
            )
            
            return molecular_context.full_prompt, metrics
            
        except Exception as e:
            self.logger.error(f"❌ Optimization failed: {e}, falling back to basic prompt")
            # Fallback to basic prompt
            basic_prompt = self._build_enhanced_prompt(
                description, agent_type, priority, rag_context, semantic_results
            )
            return basic_prompt, {
                "information_density": 0.0,
                "optimization_score": 0.0,
                "atomic_instruction": description,
                "examples_used": 0
            }
    
    def _convert_rag_to_context_items(self, rag_context: Dict[str, Any]) -> List[ContextItem]:
        """Convert RAG results to ContextItem objects for optimizer"""
        context_items = []
        
        for chunk in rag_context.get("chunks", []):
            try:
                item = ContextItem(
                    text=chunk.get("content", ""),
                    source=chunk.get("source_file", chunk.get("filename", "Unknown")),
                    relevance_score=chunk.get("similarity_score", chunk.get("similarity", 0.0)),
                    metadata={
                        "chunk_id": chunk.get("chunk_id"),
                        "document_id": chunk.get("document_id"),
                        "section": chunk.get("section", "")
                    }
                )
                context_items.append(item)
            except Exception as e:
                self.logger.warning(f"Failed to convert chunk to ContextItem: {e}")
                continue
        
        return context_items
    
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
            "timestamp": datetime.now().isoformat()
        }

