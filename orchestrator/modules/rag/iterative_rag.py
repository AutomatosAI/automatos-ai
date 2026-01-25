"""
Iterative RAG with Self-Correction
===================================

Implements advanced agentic RAG patterns from Context-Engineering research:
- Iterative retrieval with gap analysis
- Self-correcting response verification
- Cognitive tools for reasoning scaffolding

Based on Agentic RAG patterns from David Kimai's Context-Engineering repo.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RetrievalQuality(Enum):
    """Assessment of retrieval quality"""
    EXCELLENT = "excellent"  # High confidence, comprehensive
    GOOD = "good"            # Sufficient for response
    INSUFFICIENT = "insufficient"  # Need more retrieval
    FAILED = "failed"        # No relevant results


@dataclass
class GapAnalysis:
    """Analysis of knowledge gaps in retrieved context"""
    has_gaps: bool
    missing_concepts: List[str]
    confidence_score: float  # 0-1
    suggested_queries: List[str]
    reasoning: str


@dataclass
class IterationResult:
    """Result from a single retrieval iteration"""
    iteration: int
    query: str
    chunks_retrieved: int
    new_chunks: int  # Chunks not seen in previous iterations
    quality: RetrievalQuality
    gap_analysis: Optional[GapAnalysis]


@dataclass
class IterativeRAGResult:
    """Complete result from iterative RAG"""
    final_context: str
    all_chunks: List[Dict[str, Any]]
    iterations: List[IterationResult]
    total_iterations: int
    converged: bool  # True if stopped due to quality, False if max iterations
    sources: List[str]
    quality_score: float


class CognitiveTool:
    """
    Base class for cognitive tools that scaffold reasoning.
    
    Based on IBM Zurich research showing 16% improvement with cognitive tools.
    """
    
    def __init__(self, llm_provider=None):
        self._llm = llm_provider
    
    def _get_llm(self):
        if self._llm is None:
            try:
                from core.llm import create_llm_provider
                self._llm = create_llm_provider()
            except Exception as e:
                logger.warning(f"LLM not available: {e}")
        return self._llm
    
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the cognitive tool to the context"""
        raise NotImplementedError


class UnderstandQueryTool(CognitiveTool):
    """Cognitive tool for understanding query intent"""
    
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the query to understand what information is needed"""
        llm = self._get_llm()
        if not llm:
            return context
        
        query = context.get("query", "")
        
        prompt = f"""Analyze this information request:

Query: {query}

Identify:
1. Primary information need (1 sentence)
2. Key concepts to search for (3-5 terms)
3. Type of answer expected (factual/explanatory/procedural)
4. Potential ambiguities or clarifications needed

Analysis:"""
        
        try:
            analysis = await llm.generate(prompt, max_tokens=300)
            context["query_understanding"] = analysis.strip()
            context["query_analyzed"] = True
        except Exception as e:
            logger.debug(f"Query understanding failed: {e}")
        
        return context


class AssessContextTool(CognitiveTool):
    """Cognitive tool for assessing retrieved context quality"""
    
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if retrieved context is sufficient to answer the query"""
        llm = self._get_llm()
        if not llm:
            return context
        
        query = context.get("query", "")
        retrieved = context.get("retrieved_context", "")
        
        prompt = f"""Assess if this context is sufficient to answer the query:

Query: {query}

Retrieved Context (truncated):
{retrieved[:2000]}

Assessment:
1. Coverage (1-10): How well does the context cover the query?
2. Gaps: What information is missing? List specific topics.
3. Confidence (1-10): How confident can we be in answering?
4. Additional queries: If gaps exist, what should we search for?

Provide your assessment:"""
        
        try:
            assessment = await llm.generate(prompt, max_tokens=400)
            context["context_assessment"] = assessment.strip()
            
            # Parse confidence score
            import re
            confidence_match = re.search(r'Confidence[:\s]*(\d+)', assessment)
            if confidence_match:
                context["confidence_score"] = int(confidence_match.group(1)) / 10
        except Exception as e:
            logger.debug(f"Context assessment failed: {e}")
        
        return context


class VerifyResponseTool(CognitiveTool):
    """Cognitive tool for verifying response against sources"""
    
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that response claims are supported by retrieved context"""
        llm = self._get_llm()
        if not llm:
            return context
        
        response = context.get("draft_response", "")
        retrieved = context.get("retrieved_context", "")
        
        prompt = f"""Verify this response against the source context:

Draft Response:
{response[:1500]}

Source Context:
{retrieved[:2000]}

Verification:
1. Supported claims: Which claims in the response are directly supported?
2. Unsupported claims: Which claims are NOT in the source (potential hallucination)?
3. Missing information: What from the source should be added?
4. Corrections needed: List any factual corrections.

Verification result:"""
        
        try:
            verification = await llm.generate(prompt, max_tokens=400)
            context["verification_result"] = verification.strip()
            
            # Check for unsupported claims
            has_unsupported = "unsupported" in verification.lower() and "none" not in verification.lower()
            context["needs_correction"] = has_unsupported
        except Exception as e:
            logger.debug(f"Response verification failed: {e}")
        
        return context


class IterativeRAGService:
    """
    Advanced RAG with iterative retrieval and self-correction.
    
    Pipeline:
    1. Understand query (cognitive tool)
    2. Initial retrieval
    3. Assess context quality (cognitive tool)
    4. If gaps → refined retrieval (repeat)
    5. Generate response
    6. Verify against sources (cognitive tool)
    7. Self-correct if needed
    """
    
    def __init__(
        self,
        rag_service=None,
        max_iterations: int = 3,
        quality_threshold: float = 0.7,
        enable_cognitive_tools: bool = True
    ):
        self._rag_service = rag_service
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.enable_cognitive_tools = enable_cognitive_tools
        
        # Cognitive tools
        self._understand_tool = UnderstandQueryTool()
        self._assess_tool = AssessContextTool()
        self._verify_tool = VerifyResponseTool()
        self._llm = None
    
    def _get_rag_service(self):
        if self._rag_service is None:
            try:
                from modules.rag import get_rag_service
                self._rag_service = get_rag_service()
            except Exception as e:
                logger.error(f"RAG service not available: {e}")
                raise
        return self._rag_service
    
    def _get_llm(self):
        if self._llm is None:
            try:
                from core.llm import create_llm_provider
                self._llm = create_llm_provider()
            except Exception as e:
                logger.warning(f"LLM not available: {e}")
        return self._llm
    
    async def retrieve_with_iteration(
        self,
        query: str,
        max_chunks: int = 10,
        max_tokens: int = 4000
    ) -> IterativeRAGResult:
        """
        Perform iterative RAG with gap analysis and self-correction.
        
        Continues retrieval until:
        - Quality threshold met
        - No new relevant chunks found
        - Max iterations reached
        """
        rag = self._get_rag_service()
        
        context = {"query": query}
        iterations = []
        all_chunks = {}  # doc_id -> chunk (deduplication)
        current_queries = [query]
        
        # Step 1: Understand query (cognitive tool)
        if self.enable_cognitive_tools:
            context = await self._understand_tool.apply(context)
            logger.info("Applied UnderstandQuery cognitive tool")
        
        # Iterative retrieval loop
        for iteration in range(self.max_iterations):
            iteration_new_chunks = 0
            
            # Retrieve for all current queries
            for q in current_queries:
                try:
                    result = await rag.retrieve(
                        query=q,
                        max_chunks=max_chunks,
                        max_tokens=max_tokens // self.max_iterations
                    )
                    
                    for chunk in result.chunks:
                        chunk_id = chunk.get("id", chunk.get("content", "")[:100])
                        if chunk_id not in all_chunks:
                            all_chunks[chunk_id] = chunk
                            iteration_new_chunks += 1
                            
                except Exception as e:
                    logger.warning(f"Retrieval failed for query '{q[:50]}': {e}")
            
            # Build current context for assessment
            current_context = "\n\n".join(
                f"[{c.get('source_file', 'unknown')}]\n{c.get('content', '')}"
                for c in all_chunks.values()
            )
            context["retrieved_context"] = current_context
            
            # Step 2: Assess context quality (cognitive tool)
            gap_analysis = None
            quality = RetrievalQuality.GOOD
            
            if self.enable_cognitive_tools:
                context = await self._assess_tool.apply(context)
                confidence = context.get("confidence_score", 0.5)
                
                if confidence >= self.quality_threshold:
                    quality = RetrievalQuality.EXCELLENT
                elif confidence >= 0.4:
                    quality = RetrievalQuality.GOOD
                else:
                    quality = RetrievalQuality.INSUFFICIENT
                
                # Extract suggested queries for next iteration
                if quality == RetrievalQuality.INSUFFICIENT:
                    gap_analysis = await self._extract_gap_analysis(context)
                    if gap_analysis.suggested_queries:
                        current_queries = gap_analysis.suggested_queries[:2]
                    else:
                        current_queries = []
            
            iteration_result = IterationResult(
                iteration=iteration + 1,
                query=", ".join(current_queries),
                chunks_retrieved=len(all_chunks),
                new_chunks=iteration_new_chunks,
                quality=quality,
                gap_analysis=gap_analysis
            )
            iterations.append(iteration_result)
            
            logger.info(f"Iteration {iteration + 1}: {len(all_chunks)} total chunks, {iteration_new_chunks} new, quality={quality.value}")
            
            # Stopping conditions
            if quality == RetrievalQuality.EXCELLENT:
                logger.info("Quality threshold met, stopping iterations")
                break
            if iteration_new_chunks == 0:
                logger.info("No new chunks found, stopping iterations")
                break
            if not current_queries:
                logger.info("No more queries to try, stopping iterations")
                break
        
        # Calculate final quality score
        quality_score = context.get("confidence_score", 0.5)
        
        return IterativeRAGResult(
            final_context=current_context,
            all_chunks=list(all_chunks.values()),
            iterations=iterations,
            total_iterations=len(iterations),
            converged=iterations[-1].quality == RetrievalQuality.EXCELLENT if iterations else False,
            sources=list(set(c.get("source_file", "unknown") for c in all_chunks.values())),
            quality_score=quality_score
        )
    
    async def _extract_gap_analysis(self, context: Dict[str, Any]) -> GapAnalysis:
        """Extract gap analysis from context assessment"""
        assessment = context.get("context_assessment", "")
        
        # Extract suggested queries using simple parsing
        suggested_queries = []
        for line in assessment.split("\n"):
            if "search" in line.lower() or "query" in line.lower():
                # Extract quoted strings or after colons
                import re
                quotes = re.findall(r'"([^"]+)"', line)
                suggested_queries.extend(quotes)
        
        return GapAnalysis(
            has_gaps=context.get("confidence_score", 0.5) < self.quality_threshold,
            missing_concepts=[],
            confidence_score=context.get("confidence_score", 0.5),
            suggested_queries=suggested_queries[:3],
            reasoning=assessment[:500]
        )
    
    async def generate_verified_response(
        self,
        query: str,
        max_chunks: int = 10,
        max_tokens: int = 4000
    ) -> Tuple[str, IterativeRAGResult]:
        """
        Generate a response with self-verification.
        
        Returns:
            Tuple of (response, iterative_rag_result)
        """
        # Iterative retrieval
        rag_result = await self.retrieve_with_iteration(query, max_chunks, max_tokens)
        
        llm = self._get_llm()
        if not llm:
            return rag_result.final_context, rag_result
        
        # Generate initial response
        prompt = f"""Based on the following context, answer the question.
Only use information from the provided context. If the context doesn't contain
enough information, say so.

Context:
{rag_result.final_context[:6000]}

Question: {query}

Answer:"""
        
        try:
            response = await llm.generate(prompt, max_tokens=1000)
            
            # Verify response (cognitive tool)
            if self.enable_cognitive_tools:
                context = {
                    "query": query,
                    "draft_response": response,
                    "retrieved_context": rag_result.final_context
                }
                context = await self._verify_tool.apply(context)
                
                if context.get("needs_correction", False):
                    logger.info("Response needs correction, regenerating...")
                    # Regenerate with verification feedback
                    correction_prompt = f"""{prompt}

Previous attempt had issues:
{context.get('verification_result', '')}

Please provide a corrected answer that only uses information from the context:"""
                    response = await llm.generate(correction_prompt, max_tokens=1000)
            
            return response.strip(), rag_result
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return rag_result.final_context, rag_result


# Factory functions
def create_iterative_rag(
    max_iterations: int = 3,
    quality_threshold: float = 0.7,
    enable_cognitive_tools: bool = True
) -> IterativeRAGService:
    """Create an iterative RAG service instance."""
    return IterativeRAGService(
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        enable_cognitive_tools=enable_cognitive_tools
    )
