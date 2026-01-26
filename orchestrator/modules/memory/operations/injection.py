"""
Memory Injector - Memory Retrieval and Storage for Chat
========================================================

Migrated from services/chat/memory_injector.py

Handles:
- Retrieving relevant memories for context injection
- Using ContextRetrievalEngine (advanced) or HierarchicalMemorySystem (fallback)
- Storing conversation memories for future retrieval
- Fetching recent memories for continuity
"""

import json
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Global instances
_memory_system = None
_context_retrieval_engine = None


def get_memory_system():
    """Get or create the hierarchical memory system instance."""
    global _memory_system
    if _memory_system is None:
        try:
            # Import from new module location
            from modules.memory.storage.knowledge_system import HierarchicalMemorySystem
            _memory_system = HierarchicalMemorySystem()
            logger.info("✅ MemoryInjector connected to HierarchicalMemorySystem")
        except Exception as e:
            logger.warning(f"Could not initialize memory system: {e}")
    return _memory_system


async def get_context_retrieval_engine():
    """Get or create the context retrieval engine (advanced memory)."""
    global _context_retrieval_engine
    if _context_retrieval_engine is None:
        try:
            from modules.search.retrieval.context_retrieval_engine import (
                ContextRetrievalEngine, RetrievalStrategy
            )
            from modules.search.vector_store.store import EnhancedVectorStore
            from modules.rag.chunking.semantic_chunker import SemanticChunker
            from core.database.database import get_database_url
            from modules.memory import get_embedding_dimension
            
            vector_store = EnhancedVectorStore(
                database_url=get_database_url(),
                embedding_dimension=get_embedding_dimension()
            )
            await vector_store.initialize()
            
            chunker = SemanticChunker()
            _context_retrieval_engine = ContextRetrievalEngine(
                vector_store=vector_store,
                chunker=chunker,
                default_strategy=RetrievalStrategy.ADAPTIVE
            )
            logger.info("✅ MemoryInjector connected to ContextRetrievalEngine")
        except Exception as e:
            logger.warning(f"ContextRetrievalEngine not available: {e}")
    return _context_retrieval_engine


class MemoryInjector:
    """
    Handles memory retrieval and storage for the chat service.
    Uses multiple strategies to ensure relevant context is available.
    """
    
    def __init__(self):
        self._recent_cache = {"time": 0, "data": []}
        self._cache_ttl = 60  # Cache recent memories for 60 seconds
    
    def should_retrieve_memories(self, query: str, chat_id: str) -> bool:
        """
        Intelligent decision on whether to retrieve memories for a given query.
        Skips retrieval for short greetings, thanks, or irrelevant inputs to save tokens.
        """
        if not query:
            return False
            
        # Skip for very short queries
        if len(query) < 10:
            return False
        
        query_lower = query.lower().strip()
        
        # Skip for common greetings and short acknowledgments
        greetings = {"hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye", "cool", "ok", "okay"}
        if query_lower in greetings:
            logger.info("[Memory] Skipping retrieval for greeting/short msg")
            return False
        
        # Check if query references past context - stronger signal to retrieve
        past_indicators = {"before", "earlier", "remember", "you said", "we discussed", "last time", "previously"}
        if any(ind in query_lower for ind in past_indicators):
            return True
        
        return True

    async def retrieve_relevant_memories(
        self,
        chat_id: str,
        query: str,
        workspace_id: Optional[str] = None,
        agent_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Retrieve relevant memories for injection into LLM context.
        
        Tries:
        1. ContextRetrievalEngine (advanced, adaptive strategy)
        2. HierarchicalMemorySystem (basic fallback)
        
        Args:
            chat_id: Current chat session ID
            query: The user's query
            
        Returns:
            Formatted memory context string, or None
        """
        """
        Retrieves relevant memories if the query is significant enough.
        """
        if not self.should_retrieve_memories(query, chat_id):
           return None
           
        if not query or len(query) < 5:
            return None
        
        try:
            # Try advanced retrieval engine first
            engine = await get_context_retrieval_engine()
            if engine:
                logger.info("[Memory] Trying ContextRetrievalEngine...")
                result = await self._retrieve_with_context_engine(engine, chat_id, query)
                if result:
                    return result
                logger.info("[Memory] ContextRetrievalEngine returned empty, trying basic memory...")
            
            # Fallback to basic memory system
            logger.info("[Memory] Using HierarchicalMemorySystem...")
            return await self._retrieve_with_basic_memory(
                chat_id,
                query,
                workspace_id=workspace_id,
                agent_id=agent_id
            )
            
        except Exception as e:
            logger.debug(f"Memory retrieval skipped: {e}")
            return None
    
    async def _retrieve_with_context_engine(
        self,
        engine,
        chat_id: str,
        query: str
    ) -> Optional[str]:
        """Use the ContextRetrievalEngine for advanced retrieval."""
        try:
            from modules.search.retrieval.context_retrieval_engine import (
                ContextQuery, ContextType, RetrievalStrategy
            )
            
            context_query = ContextQuery(
                text=query,
                context_types=[ContextType.HISTORICAL],
                max_results=10,
                min_relevance=0.6,
                include_metadata=True
            )
            
            result = await engine.retrieve_context(
                context_query,
                strategy=RetrievalStrategy.ADAPTIVE
            )
            
            if not result.contexts:
                return None
            
            # Format results for LLM
            memory_lines = []
            for ctx in result.contexts[:8]:
                content = ctx.metadata.get('content', {}) if ctx.metadata else {}
                
                if isinstance(content, dict) and 'user_query' in content:
                    user_q = content.get('user_query', '')[:200]
                    assistant_r = content.get('assistant_response', '')[:150]
                    mem_chat = content.get('chat_id', '')
                    
                    if user_q:
                        prefix = "↳ " if mem_chat == chat_id else ""
                        memory_lines.append(f"{prefix}You: \"{user_q}\"")
                        if assistant_r:
                            memory_lines.append(f"{prefix}Me: \"{assistant_r}\"")
                else:
                    if ctx.content and len(ctx.content) > 20:
                        memory_lines.append(f"[{ctx.context_type.value}] {ctx.content[:200]}")
            
            logger.info(f"✅ Retrieved {len(result.contexts)} memories via ContextRetrievalEngine")
            return "\n".join(memory_lines) if memory_lines else None
            
        except Exception as e:
            logger.warning(f"ContextRetrievalEngine failed: {e}")
            return None
    
    async def _retrieve_with_basic_memory(
        self,
        chat_id: str,
        query: str,
        workspace_id: Optional[str] = None,
        agent_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Retrieve memories using both semantic search and recent memories.
        Ensures continuity and personal info like names are captured.
        """
        memory_system = get_memory_system()
        if not memory_system:
            logger.warning("[Memory] HierarchicalMemorySystem not available!")
            return None
        
        all_memories = []
        seen_ids = set()
        
        # Run searches in parallel
        semantic_task = memory_system.retrieve_relevant_memories(
            agent_id=agent_id or 1,
            context=query,
            memory_types=["experience"],
            top_k=8,
            workspace_id=workspace_id
        )
        recent_task = self._get_recent_memories(limit=10, workspace_id=workspace_id)
        
        logger.info(f"[Memory] Starting parallel retrieval: Semantic + Recent")
        results = await asyncio.gather(semantic_task, recent_task, return_exceptions=True)
        
        # Process Semantic Results
        semantic_memories = []
        if isinstance(results[0], list):
            semantic_memories = results[0]
            logger.info(f"[Memory] Semantic search found {len(semantic_memories)} memories")
        else:
            logger.error(f"[Memory] Semantic search failed: {results[0]}")

        # Process Recent Results
        recent_memories = []
        if isinstance(results[1], list):
            recent_memories = results[1]
            logger.info(f"[Memory] Recent memories: {len(recent_memories)}")
        else:
            logger.error(f"[Memory] Recent memory fetch failed: {results[1]}")
            
        # Combine results
        for mem in semantic_memories:
            mem_id = mem.get('id') or str(mem.get('content', ''))[:50]
            if mem_id not in seen_ids:
                seen_ids.add(mem_id)
                all_memories.append(('semantic', mem))
                
        for mem in recent_memories:
            mem_id = mem.get('id') or str(mem.get('content', ''))[:50]
            if mem_id not in seen_ids:
                seen_ids.add(mem_id)
                all_memories.append(('recent', mem))
        
        if not all_memories:
            logger.info("[Memory] No memories found")
            return None
        
        logger.info(f"[Memory] Total unique memories: {len(all_memories)}")
        
        # Format memories for LLM
        memory_lines = []
        for source, mem in all_memories[:12]:
            content = mem.get('content', {})
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except:
                    content = {"summary": content[:200]}
            
            user_q = content.get('user_query', '')[:200]
            assistant_r = content.get('assistant_response', '')[:150]
            mem_chat = content.get('chat_id', '')
            
            if user_q:
                prefix = "[This chat] " if mem_chat == chat_id else "[Earlier] "
                memory_lines.append(f"{prefix}You: \"{user_q}\"")
                if assistant_r:
                    memory_lines.append(f"{prefix}Me: \"{assistant_r}\"")
        
        return "\n".join(memory_lines) if memory_lines else None
    
    async def _get_recent_memories(self, limit: int = 10, workspace_id: Optional[str] = None) -> List[Dict]:
        """Fetch the most recent memories regardless of semantic similarity. Cached for 60s."""
        # Check cache
        now = time.time()
        if self._recent_cache["data"] and (now - self._recent_cache["time"] < self._cache_ttl):
            logger.info("[Memory] Using cached recent memories")
            return self._recent_cache["data"][:limit]

        try:
            from core.database.database import get_db_session
            from sqlalchemy import text
            
            with get_db_session() as db:
                workspace_filter = ""
                params = {"limit": limit}
                if workspace_id:
                    workspace_filter = "AND workspace_id = :workspace_id"
                    params["workspace_id"] = str(workspace_id)

                result = db.execute(text(f"""
                    SELECT id, content, metadata, created_at
                    FROM memory_items
                    WHERE memory_type = 'experience'
                    {workspace_filter}
                    ORDER BY created_at DESC
                    LIMIT :limit
                """), params)
                
                memories = []
                for row in result:
                    content = row.content
                    if isinstance(content, str):
                        try:
                            content = json.loads(content)
                        except:
                            content = {"summary": content}
                    memories.append({
                        'id': str(row.id),
                        'content': content,
                        'metadata': row.metadata,
                        'created_at': row.created_at
                    })
                # Update cache
                self._recent_cache = {
                    "time": time.time(),
                    "data": memories
                }
                return memories
        except Exception as e:
            logger.error(f"[Memory] Direct DB query failed: {e}")
            return []
    
    async def store_conversation_memory(
        self,
        chat_id: str,
        user_message: str,
        assistant_response: str,
        workspace_id: Optional[str] = None
    ):
        """
        Store conversation as memory for future retrieval.
        
        Args:
            chat_id: Chat session ID
            user_message: The user's message
            assistant_response: The assistant's response
        """
        try:
            logger.info(f"[Memory] Storing: {user_message[:50]}...")
            memory_system = get_memory_system()
            if not memory_system:
                logger.warning("[Memory] No memory system available!")
                return
            
            experience = {
                "type": "conversation",
                "chat_id": chat_id,
                "user_query": user_message[:500],
                "assistant_response": assistant_response[:500],
                "summary": f"{user_message[:100]}",
                "timestamp": datetime.utcnow().isoformat(),
                "success": True,
                "is_novel": True,
                "goal_relevant": True
            }
            
            # Use None for agent_id when no specific agent is selected (general chat)
            # This avoids foreign key violations when agent_id=1 doesn't exist
            result = await memory_system.store_experience(
                agent_id=None,
                experience=experience,
                workspace_id=workspace_id
            )
            logger.info(f"[Memory] ✅ Stored (id={result}): {user_message[:50]}...")
            
        except Exception as e:
            logger.error(f"[Memory] ❌ Storage FAILED: {e}")
    
    def build_memory_injection_message(self, memory_context: str) -> Dict[str, str]:
        """
        Build the system message for memory injection.
        
        Args:
            memory_context: Formatted memory context string
            
        Returns:
            System message dict for LLM
        """
        return {
            "role": "system",
            "content": f"""🧠 CRITICAL - YOUR CONVERSATION MEMORY (READ THIS FIRST):

{memory_context}

INSTRUCTIONS:
- If the user asks your name, check memory for their name
- If asked "what did we talk about", summarize the memory above
- NEVER say "you haven't told me your name" if their name is in memory
- NEVER say "I don't have access to past conversations" - you DO have memory above"""
        }


# Module-level instance
_memory_injector = None

def get_memory_injector() -> MemoryInjector:
    """Get or create the global MemoryInjector instance."""
    global _memory_injector
    if _memory_injector is None:
        _memory_injector = MemoryInjector()
    return _memory_injector

