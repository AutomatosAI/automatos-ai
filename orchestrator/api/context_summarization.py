"""
Context Summarization Endpoint
===============================

Implements "self-baking" from Context Engineering 2.0 research.
Compresses raw context into hierarchical summaries for efficient long-term storage.

Based on Section 5.4 of "Context Engineering 2.0" (Hua et al., 2025)
"""

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/context",
    tags=["🧠 Context Engineering"],
)


class SummarizationRequest(BaseModel):
    """Request for context summarization"""
    session_id: str = Field(..., description="Session ID to summarize")
    compression_level: str = Field(
        default="medium",
        description="Compression level: low, medium, high"
    )
    preserve_entities: bool = Field(
        default=True,
        description="Preserve named entities in summary"
    )


class SummarizationResponse(BaseModel):
    """Response with hierarchical summary"""
    session_id: str
    executive_summary: str = Field(..., description="High-level overview")
    key_facts: List[str] = Field(..., description="Important facts extracted")
    entities: List[Dict[str, Any]] = Field(..., description="Named entities")
    action_items: List[str] = Field(..., description="Actionable items")
    original_token_count: int
    summary_token_count: int
    compression_ratio: float
    timestamp: str


@router.post(
    "/summarize",
    response_model=SummarizationResponse,
    summary="Summarize Session Context",
    description="Compress session context using hierarchical summarization (Context Engineering 2.0 'self-baking')"
)
async def summarize_context(request: SummarizationRequest):
    """
    Implement hierarchical context summarization (self-baking).
    
    This endpoint demonstrates Context Engineering 2.0 principles:
    - Converts episodic memories into semantic knowledge
    - Reduces token usage while preserving key information
    - Enables long-horizon task execution
    
    Compression levels:
    - low: 70-80% of original (preserve most details)
    - medium: 40-60% of original (balanced)
    - high: 20-30% of original (executive summary only)
    """
    try:
        # Import LLM provider
        from core.llm_provider import LLMProviderManager
        from modules.memory import MemoryService
        
        # Get session context from memory
        memory_service = MemoryService()
        memories = await memory_service.retrieve(
            session_id=request.session_id,
            max_items=1000  # Get all recent memories
        )
        
        if not memories:
            raise HTTPException(
                status_code=404,
                detail=f"No context found for session {request.session_id}"
            )
        
        # Build context string
        context_text = "\n\n".join([
            f"[{mem.timestamp}] {mem.content}"
            for mem in memories
        ])
        
        original_token_count = len(context_text.split())  # Rough estimate
        
        # Determine compression ratio based on level
        compression_ratios = {
            "low": 0.75,
            "medium": 0.50,
            "high": 0.25
        }
        target_ratio = compression_ratios.get(request.compression_level, 0.50)
        
        # Create summarization prompt
        prompt = f"""Analyze this session context and create a hierarchical summary.

Session Context:
{context_text}

Create a structured summary with:
1. Executive Summary (2-3 sentences capturing the essence)
2. Key Facts (5-10 bullet points of important information)
3. Named Entities (people, tasks, concepts mentioned)
4. Action Items (what needs to be done next)

Preserve important details while compressing to approximately {int(target_ratio * 100)}% of original length.
{"Focus on preserving entities and proper nouns." if request.preserve_entities else ""}

Format your response as JSON:
{{
  "executive_summary": "...",
  "key_facts": ["fact1", "fact2", ...],
  "entities": [{{"name": "...", "type": "...", "mentions": N}}],
  "action_items": ["item1", "item2", ...]
}}
"""
        
        # Get LLM response
        llm_manager = LLMProviderManager()
        response = await llm_manager.generate(
            prompt=prompt,
            temperature=0.3,  # Lower temperature for more focused summarization
            max_tokens=int(original_token_count * target_ratio * 1.5)  # Allow some overhead
        )
        
        # Parse JSON response
        import json
        try:
            summary_data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            summary_data = {
                "executive_summary": response[:500],
                "key_facts": ["Summary generation in progress"],
                "entities": [],
                "action_items": []
            }
        
        # Calculate actual compression
        summary_text = json.dumps(summary_data)
        summary_token_count = len(summary_text.split())
        actual_ratio = summary_token_count / max(original_token_count, 1)
        
        return SummarizationResponse(
            session_id=request.session_id,
            executive_summary=summary_data.get("executive_summary", ""),
            key_facts=summary_data.get("key_facts", []),
            entities=summary_data.get("entities", []),
            action_items=summary_data.get("action_items", []),
            original_token_count=original_token_count,
            summary_token_count=summary_token_count,
            compression_ratio=round(actual_ratio, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing context: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/memory-stats",
    summary="Get Memory Statistics",
    description="Retrieve comprehensive memory system statistics"
)
async def get_memory_statistics():
    """
    Get statistics about the hierarchical memory system.
    
    Shows distribution across memory levels:
    - Immediate (7 items)
    - Working (100 items)
    - Short-term (1000 items)
    - Long-term (100k items)
    - Archival (1M items)
    """
    try:
        # TODO: Integrate with actual MemoryService
        from modules.memory import MemoryService
        
        service = MemoryService()
        stats = await service.get_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "memory_levels": stats.get("memory", {}),
            "operations": {
                "augmentation": stats.get("augmentation", {}),
                "consolidation": stats.get("consolidation", {}),
                "optimization": stats.get("optimization", {})
            },
            "research_compliance": {
                "context_engineering_version": "2.0",
                "hierarchical_memory": True,
                "self_baking": True,
                "multi_agent_sharing": False  # TODO: Implement
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
