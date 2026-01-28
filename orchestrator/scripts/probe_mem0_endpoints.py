
import requests
import os

base_url = os.getenv("MEM0_API_URL", "https://automatos-mem0-server-production.up.railway.app").rstrip("/")
api_key = os.getenv("MEM_API_KEY")

headers = {}
if api_key:
    headers["Authorization"] = f"Token {api_key}"

endpoints = [
    "",
    "/memories",
    "/v1/memories",
    "/api/v1/memories",
    "/search",
    "/v1/search",
    "/api/v1/search",
    "/api/v1/memories/search",
    "/docs",
    "/openapi.json"
]

print(f"Probing {base_url}...")

for ep in endpoints:
    url = f"{base_url}{ep}"
    try:
        # Try GET first
        resp = requests.get(url, headers=headers, timeout=5)
        print(f"GET  {ep.ljust(25)} -> {resp.status_code}")
        
        # If 404 or 405, try POST just in case
        if resp.status_code in [404, 405]:
             resp_post = requests.post(url, headers=headers, json={}, timeout=5)
             print(f"POST {ep.ljust(25)} -> {resp_post.status_code}")
             
    except Exception as e:
        print(f"ERR  {ep.ljust(25)} -> {e}")
