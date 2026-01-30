"""
Mem0 System Adapter
===================

Adapts the Mem0Client to matched the interface expected by MemoryInjector,
acting as a drop-in replacement for HierarchicalMemorySystem where possible.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from modules.memory.integrations.mem0_client import Mem0Client

logger = logging.getLogger(__name__)

class Mem0MemorySystem:
    """ Adapter for Mem0 to be used by MemoryInjector. """
    
    def __init__(self):
        self.client = Mem0Client()
        logger.info("Mem0MemorySystem initialized")

    def _get_scoped_user_id(self, workspace_id: Optional[str], agent_id: Optional[int] = None) -> str:
        """
        Create a scoped user ID for Mem0 to ensure isolation.
        Format: ws_{workspace_id}_agent_{agent_id} or ws_{workspace_id}_global
        """
        # Default fallback
        ws = workspace_id or "default"
        ag = agent_id or "global"
        return f"ws_{ws}_agent_{ag}"

    async def retrieve_relevant_memories(
        self,
        agent_id: int,
        context: str,
        memory_types: Optional[List[str]] = None,
        top_k: int = 10,
        workspace_id: Optional[Union[str, Any]] = None,
        min_relevance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories from Mem0.
        Matches signature of HierarchicalMemorySystem.retrieve_relevant_memories.
        """
        if not context:
            return []
            
        user_id = self._get_scoped_user_id(str(workspace_id) if workspace_id else None, agent_id)
        
        # Mem0 search
        # Note: mem0 `search` runs synchronously in the client wrapper (requests)
        # In a real async app we might want to run_in_executor, but for now direct call is ok 
        # as long as we don't block event loop too long. 
        # Ideally Mem0Client would use aiohttp.
        
        results = self.client.search(query=context, user_id=user_id, limit=top_k)
        
        # Adapt results to expected format
        adapted_memories = []
        for mem in results:
            # Mem0 result: {'id': '...', 'memory': '...', 'score': ...}
            # Expected: {'id': ..., 'content': ..., 'relevance': ...}
            
            # Filter by score if needed (though Mem0 usually handles relevance well)
            score = mem.get('score', 0)
            if score < min_relevance:
                continue
                
            adapted_memories.append({
                "id": mem.get("id"),
                "content": mem.get("memory"),  # The text content
                "relevance": score,
                "metadata": mem.get("metadata", {}),
                "created_at": mem.get("created_at"), # might need conversion
                "type": "mem0"
            })
            
        return adapted_memories

    async def store_experience(
        self,
        agent_id: Optional[int],
        experience: Dict[str, Any],
        workspace_id: Optional[Union[str, Any]] = None
    ) -> str:
        """
        Store experience/conversation in Mem0.
        """
        user_id = self._get_scoped_user_id(str(workspace_id) if workspace_id else None, agent_id)
        
        # Extract messages from experience dict
        # experience typically has: user_query, assistant_response
        messages = []
        if "user_query" in experience:
            messages.append({"role": "user", "content": experience["user_query"]})
        if "assistant_response" in experience:
            messages.append({"role": "assistant", "content": experience["assistant_response"]})
            
        if not messages:
            logger.warning("[Mem0] No conversation content to store")
            return ""
            
        # Store
        result = self.client.add(messages=messages, user_id=user_id, metadata=experience)
        
        # Return ID if available
        if isinstance(result, dict):
            return result.get("id", str(result)) 
        return str(result)
