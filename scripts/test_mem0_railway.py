
import requests
import json
import sys
import os

def test_mem0_connection(base_url):
    print(f"Testing Mem0 at: {base_url}")
    
    # 1. Health Check (Try /api/v1/config or /docs or root)
    # The server code typically has /docs. Let's try to hit the API root.
    try:
        # Try listing users/entities (from code inspection this seemed to conform to v1)
        # Or just try to search/add which is what we care about.
        
        # We saw `app.include_router(memories_router)` with prefix `/api/v1/memories` in `main.py`
        memories_url = f"{base_url}/api/v1/memories/"
        
        print(f"\n1. Health Check (GET {memories_url})...")
        # Need a user_id. The server code checks for it.
        params = {"user_id": "test_verification_user"}
        resp = requests.get(memories_url, params=params)
        
        if resp.status_code == 200:
            print("   ✅ Success! API is reachable.")
            data = resp.json()
            print(f"   Response: {json.dumps(data, indent=2)[:200]}...") 
        else:
            print(f"   ❌ Failed. Status: {resp.status_code}")
            print(f"   Body: {resp.text}")
            return

        # 2. Add Memory
        print(f"\n2. Add Memory (POST {memories_url})...")
        payload = {
            "user_id": "test_verification_user",
            "text": "The verification script is working.",
            "app": "test_script"
        }
        resp = requests.post(memories_url, json=payload)
        
        if resp.status_code == 200:
            print("   ✅ Success! Memory added.")
            result = resp.json()
            print(f"   Result: {json.dumps(result, indent=2)}")
            memory_id = result.get("id")
        else:
            print(f"   ❌ Failed. Status: {resp.status_code}")
            print(f"   Body: {resp.text}")
            return

        # 3. Verify Memory Exists (Search/Filter)
        # Based on code, we can filter by search_query
        print(f"\n3. Verify Memory (GET {memories_url})...")
        params = {
            "user_id": "test_verification_user",
            "search_query": "verification script"
        }
        resp = requests.get(memories_url, params=params)
        
        if resp.status_code == 200:
            results = resp.json()
            # It returns a Page[MemoryResponse], so checks 'items'
            items = results.get("items", [])
            if any("verification script" in m.get("content", "") for m in items):
                 print("   ✅ Success! Found the added memory.")
            else:
                 print("   ⚠️  API returned verified but memory not found in list.")
                 print(f"   Items: {json.dumps(items, indent=2)}")
        else:
            print(f"   ❌ Failed. Status: {resp.status_code}")
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_mem0_railway.py <YOUR_MEM0_RAILWAY_URL>")
        print("Example: python3 test_mem0_railway.py https://mem0-production.up.railway.app")
        sys.exit(1)
        
    url = sys.argv[1].rstrip('/')
    test_mem0_connection(url)
