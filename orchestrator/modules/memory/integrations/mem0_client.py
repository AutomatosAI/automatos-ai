"""
Mem0 Client Integration
=======================

Wrapper around standard mem0ai usage to connect to internal Railway instance.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

class Mem0Client:
    """
    Client for interacting with Mem0 server.
    """
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.api_url = api_url or os.getenv("MEM0_API_URL", "http://automatos-mem0-server.railway.internal")
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        
        # Ensure URL has correct format
        if not self.api_url.startswith("http"):
            self.api_url = f"https://{self.api_url}"
            
        self.api_url = self.api_url.rstrip("/")
        self.headers = {}
        if self.api_key:
            self.headers["Authorization"] = f"Token {self.api_key}"
            
        logger.info(f"Initialized Mem0Client with URL: {self.api_url}")

    def add(self, messages: List[Dict[str, str]], user_id: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Add memories from messages.
        
        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            user_id: Unique user identifier
            metadata: Optional metadata to attach
            
        Returns:
            Response dict from server
        """
        url = f"{self.api_url}/api/v1/memories/"
        
        # Adapt messages to text format expected by server
        # Server expects: {"text": "...", "user_id": "..."}
        text_content = ""
        for msg in messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            if text_content:
                text_content += "\n"
            text_content += f"{role}: {content}"
            
        payload = {
            "text": text_content,
            "user_id": user_id,
        }
        if metadata:
            payload["metadata"] = metadata
            
        # NOTE: Server validation error indicated user_id is required in QUERY params
        params = {"user_id": user_id}
            
        try:
            logger.debug(f"[Mem0] Adding memory for user {user_id}")
            resp = requests.post(url, json=payload, params=params, headers=self.headers, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"[Mem0] Failed to add memory: {e}")
            return {"error": str(e)}

    def search(self, query: str, user_id: str, limit: int = 5) -> List[Dict]:
        """
        Search for relevant memories.
        
        Args:
            query: Search query
            user_id: User identifier to scope search
            limit: Max results
            
        Returns:
            List of memory items
        """
        # Server uses GET /api/v1/memories/ with search_query param
        url = f"{self.api_url}/api/v1/memories/"
        
        params = {
            "search_query": query,
            "user_id": user_id,
            "size": limit, # Schema says 'size', not 'limit'
            "page": 1
        }
        
        try:
            logger.debug(f"[Mem0] Searching memories for user {user_id}: {query[:50]}")
            resp = requests.get(url, params=params, headers=self.headers, timeout=15)
            # Handle 404 (User not found) gracefully for search
            if resp.status_code == 404 and "User not found" in resp.text:
                 logger.warning(f"[Mem0] User {user_id} not found during search (no memories yet)")
                 return []
            
            resp.raise_for_status()
            
            # Response is Page[MemoryResponse]: {"items": [...], "total": ..., ...}
            data = resp.json()
            items = data.get("items", [])
            
            return [{"id": i["id"], "memory": i["content"], "score": i.get("score"), "metadata": i.get("metadata_")} for i in items]
            
        except Exception as e:
            logger.error(f"[Mem0] Failed to search memories: {e}")
            return []

    def get_all(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get all memories for a user."""
        url = f"{self.api_url}/api/v1/memories/"
        params = {"user_id": user_id, "size": limit, "page": 1}
        
        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=15)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
            return data.get("items", [])
        except Exception as e:
            logger.error(f"[Mem0] Failed to get memories: {e}")
            return []

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        url = f"{self.api_url}/api/v1/memories/{memory_id}" 
        
        try:
            resp = requests.delete(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"[Mem0] Failed to delete memory {memory_id}: {e}")
            return False
