#!/usr/bin/env python3
"""
Document Upload Script
======================

Uploads all documents from the docs folder to the RAG system
using the new semantic chunking and 1024-dim embeddings.

Usage:
    python scripts/upload_docs.py

Run from the orchestrator directory.
"""

import os
import sys
import asyncio
import httpx
from pathlib import Path

# API base URL
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# Document folders to upload
DOCS_FOLDERS = [
    "/root/automatos-ai/docs",
    "/root/automatos-ai/docs/PRDS",
]

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.md', '.txt', '.pdf', '.json'}


async def upload_file(client: httpx.AsyncClient, file_path: Path) -> dict:
    """Upload a single file to the documents API"""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'application/octet-stream')}
            response = await client.post(
                f"{API_BASE}/api/documents/upload",
                files=files,
                timeout=120.0
            )
            
        if response.status_code in [200, 201]:
            result = response.json()
            print(f"  ✅ {file_path.name} -> ID: {result.get('id', 'unknown')}")
            return {"success": True, "file": file_path.name, "result": result}
        else:
            print(f"  ❌ {file_path.name} -> Error: {response.status_code} - {response.text[:100]}")
            return {"success": False, "file": file_path.name, "error": response.text}
            
    except Exception as e:
        print(f"  ❌ {file_path.name} -> Exception: {str(e)}")
        return {"success": False, "file": file_path.name, "error": str(e)}


async def upload_all_docs():
    """Upload all documents from configured folders"""
    
    print("=" * 60)
    print("📚 Automatos Document Upload Script")
    print("=" * 60)
    print(f"API: {API_BASE}")
    print(f"Folders: {DOCS_FOLDERS}")
    print(f"Extensions: {SUPPORTED_EXTENSIONS}")
    print("=" * 60)
    
    # Collect all files
    files_to_upload = []
    for folder in DOCS_FOLDERS:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"⚠️  Folder not found: {folder}")
            continue
        
        for ext in SUPPORTED_EXTENSIONS:
            files_to_upload.extend(folder_path.glob(f"*{ext}"))
    
    # Remove duplicates (in case of nested folders)
    files_to_upload = list(set(files_to_upload))
    
    print(f"\n📁 Found {len(files_to_upload)} files to upload\n")
    
    if not files_to_upload:
        print("No files found to upload!")
        return
    
    # Upload files
    results = {"success": 0, "failed": 0, "files": []}
    
    async with httpx.AsyncClient() as client:
        for file_path in sorted(files_to_upload):
            result = await upload_file(client, file_path)
            results["files"].append(result)
            if result["success"]:
                results["success"] += 1
            else:
                results["failed"] += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Upload Summary")
    print("=" * 60)
    print(f"✅ Successful: {results['success']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"📁 Total: {len(files_to_upload)}")
    
    # Trigger re-processing if needed
    if results['success'] > 0:
        print("\n🔄 Documents uploaded. Processing will happen automatically.")
        print("   Check status: curl http://localhost:8000/api/documents/reprocess-status")
    
    return results


if __name__ == "__main__":
    asyncio.run(upload_all_docs())

