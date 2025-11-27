#!/usr/bin/env python3
"""
Simple batch upload script for Automatos AI docs directory.
Uploads all documents to the API for chunking and embedding.
"""

import os
import sys
from pathlib import Path
import requests

# API URL - defaults to remote server
API_URL = os.getenv('API_URL', 'http://206.81.0.227:8000')

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.markdown', '.docx', '.doc', '.json'}

def upload_document(file_path: Path) -> dict:
    """Upload a single document"""
    url = f"{API_URL.rstrip('/')}/api/documents/upload"
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'application/octet-stream')}
            response = requests.post(url, files=files, timeout=300)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def main():
    # Get docs directory (script location/../docs)
    script_dir = Path(__file__).parent
    docs_dir = script_dir.parent / 'docs'
    
    if not docs_dir.exists():
        print(f"❌ Docs directory not found: {docs_dir}")
        sys.exit(1)
    
    # Find all markdown and text files (including subdirectories like PRDS)
    documents = []
    for ext in SUPPORTED_EXTENSIONS:
        documents.extend(docs_dir.rglob(f'*{ext}'))  # rglob = recursive glob
    
    if not documents:
        print(f"❌ No documents found in {docs_dir}")
        sys.exit(1)
    
    print(f"📚 Found {len(documents)} document(s) in {docs_dir}")
    print(f"📡 Uploading to: {API_URL}\n")
    
    successful = 0
    failed = 0
    
    for i, doc in enumerate(documents, 1):
        print(f"[{i}/{len(documents)}] 📄 {doc.name}...", end=' ', flush=True)
        
        result = upload_document(doc)
        
        if result.get('status') == 'error':
            print(f"❌ Failed: {result.get('error', 'Unknown')}")
            failed += 1
        elif result.get('status') == 'duplicate':
            print("⏭️  Already exists")
        else:
            print(f"✅ Uploaded (ID: {result.get('document_id', 'N/A')})")
            successful += 1
    
    print(f"\n✅ Success: {successful} | ❌ Failed: {failed}")
    
    if failed > 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
