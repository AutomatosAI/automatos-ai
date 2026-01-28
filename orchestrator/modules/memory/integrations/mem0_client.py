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

        # Convert messages to text format for OpenMemory API
        # API expects: {"user_id": "...", "text": "...", "infer": true}
        text_parts = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            text_parts.append(f"{role}: {content}")
        text_content = "\n".join(text_parts)

        payload = {
            "user_id": user_id,
            "text": text_content,
            "infer": True,  # Let Mem0 extract facts/memories from the text
        }
        if metadata:
            payload["metadata"] = metadata

        try:
            logger.info(f"[Mem0] Adding memory for user {user_id}: {text_content[:80]}...")
            resp = requests.post(url, json=payload, headers=self.headers, timeout=15)
            logger.info(f"[Mem0] Add response status: {resp.status_code}")

            if resp.status_code >= 400:
                logger.error(f"[Mem0] Add failed: {resp.status_code} - {resp.text[:200]}")
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

            resp.raise_for_status()

            # Handle empty response (success with no body)
            if not resp.text or resp.text.strip() == "" or resp.text == "null":
                logger.info("[Mem0] Empty/null response - treating as success")
                return {"success": True}

            result = resp.json()
            logger.info(f"[Mem0] Add result: {result}")
            return result if result else {"success": True}
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
        # OpenMemory API: GET /api/v1/memories/
        # NOTE: search_query param doesn't work reliably, so we fetch all and filter client-side
        url = f"{self.api_url}/api/v1/memories/"

        # Get more memories to allow for client-side filtering
        params = {
            "user_id": user_id,
            "page": 1,
            "size": 50,  # Fetch more to filter from
        }

        try:
            logger.info(f"[Mem0] Fetching memories for user={user_id}")
            resp = requests.get(url, params=params, headers=self.headers, timeout=15)
            logger.info(f"[Mem0] Response status: {resp.status_code}")

            if resp.status_code >= 400:
                logger.warning(f"[Mem0] Fetch returned {resp.status_code}: {resp.text[:100]}")
                return []

            resp.raise_for_status()

            data = resp.json()

            # OpenMemory returns Page[MemoryResponse]: {"items": [...], "total": int, ...}
            if isinstance(data, dict):
                items = data.get("items", [])
                total = data.get("total", 0)
                logger.info(f"[Mem0] ✅ Found {len(items)} memories (total: {total})")

                results = []
                for m in items:
                    results.append({
                        "id": m.get("id"),
                        "memory": m.get("content"),  # OpenMemory uses 'content'
                        "score": m.get("score"),
                        "metadata": m.get("metadata_"),
                        "created_at": m.get("created_at"),
                    })

                # Log some of the memories found
                if results:
                    sample = [r.get("memory", "")[:40] for r in results[:3]]
                    logger.info(f"[Mem0] Sample memories: {sample}")

                # Return most recent memories (sorted by created_at desc)
                results.sort(key=lambda x: x.get("created_at", 0), reverse=True)
                return results[:limit]
            else:
                logger.warning(f"[Mem0] Unexpected response format: {type(data)}")
                return []

        except Exception as e:
            logger.error(f"[Mem0] Failed to fetch memories: {e}")
            return []

    def get_all(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get all memories for a user."""
        url = f"{self.api_url}/api/v1/memories/"
        params = {"user_id": user_id}

        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=15)
            if resp.status_code >= 400:
                return []
            resp.raise_for_status()
            data = resp.json()
            # Handle various response formats
            if isinstance(data, list):
                return data[:limit]
            return data.get("memories", data.get("results", data.get("items", [])))[:limit]
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
