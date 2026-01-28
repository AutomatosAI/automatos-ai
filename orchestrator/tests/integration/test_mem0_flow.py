
import os
import sys
import asyncio
import requests
import json
from typing import List, Dict, Any, Optional

# STANDALONE MEM0 CLIENT FOR TESTING 
class Mem0Client:
    """Client for interacting with Mem0 server."""
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.api_url = api_url or os.getenv("MEM0_API_URL", "http://automatos-mem0-server.railway.internal")
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        
        if not self.api_url.startswith("http"):
            self.api_url = f"https://{self.api_url}"
        self.api_url = self.api_url.rstrip("/")
        
        self.headers = {}
        if self.api_key:
            self.headers["Authorization"] = f"Token {self.api_key}"
            
        print(f"Initialized Mem0Client with URL: {self.api_url}")

    def add(self, messages: List[Dict[str, str]], user_id: str, metadata: Optional[Dict] = None) -> Dict:
        url = f"{self.api_url}/api/v1/memories/"
        
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
            resp = requests.post(url, json=payload, params=params, headers=self.headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"Error adding memory: {e}")
            if e.response is not None:
                print(f"Server Response: {e.response.text}")
            return {"error": str(e)}

    def search(self, query: str, user_id: str, limit: int = 5) -> List[Dict]:
        url = f"{self.api_url}/api/v1/memories/"
        params = {"search_query": query, "user_id": user_id, "size": limit, "page": 1}
        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=5)
            if resp.status_code == 404:
                print("User not found during search (normal for new user if add failed)")
                return []
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            return [{"id": i["id"], "memory": i["content"], "score": i.get("score")} for i in items]
        except requests.exceptions.RequestException as e:
            print(f"Error searching memory: {e}")
            if e.response is not None:
                print(f"Server Response: {e.response.text}")
            return []
            
    def get_all(self, user_id: str, limit: int = 100) -> List[Dict]:
        url = f"{self.api_url}/api/v1/memories/"
        params = {"user_id": user_id, "size": limit, "page": 1}
        try:
            resp = requests.get(url, params=params, headers=self.headers, timeout=10)
            if resp.status_code == 404:
                 return []
            resp.raise_for_status()
            return resp.json().get("items", [])
        except requests.exceptions.RequestException as e:
            print(f"Error getting memories: {e}")
            if e.response is not None:
                print(f"Server Response: {e.response.text}")
            return []


async def test_mem0_client_flow():
    """Test standard Mem0 flow: Add -> Search -> Check."""
    
    # Configuration
    api_url = os.getenv("MEM0_API_URL", "http://automatos-mem0-server.railway.internal")
    print(f"\nTesting Mem0 at: {api_url}")
    
    client = Mem0Client(api_url=api_url)
    import uuid
    user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    print(f"Testing with NEW User ID: {user_id}")
    
    # 1. Add Memory
    print("1. Adding memory...")
    messages = [
        {"role": "user", "content": "My favorite color is ultraviolet."},
        {"role": "assistant", "content": "That is a unique choice!"}
    ]
    result = client.add(messages=messages, user_id=user_id, metadata={"test_run": True})
    print(f"   Result: {result}")
    
    # 2. Search Memory
    print("2. Searching memory...")
    import time
    time.sleep(2)  # Wait for indexing
    
    memories = client.search(query="favorite color", user_id=user_id)
    print(f"   Found {len(memories)} memories")
    
    found = False
    for mem in memories:
        print(f"   - {mem.get('memory')} (Score: {mem.get('score')})")
        if "ultraviolet" in mem.get("memory", "").lower():
            found = True
            
    if found:
        print("✅ SUCCESS: Found 'ultraviolet' in memory.")
    else:
        print("❌ FAILURE: Did not find 'ultraviolet' in search results.")
        all_mems = client.get_all(user_id=user_id)
        print(f"   Total user memories: {len(all_mems)}")
        
if __name__ == "__main__":
    asyncio.run(test_mem0_client_flow())
