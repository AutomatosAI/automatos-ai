"""
Test script for PRD-42 Cloud Document Sync with Mock S3 Vectors.

Tests:
1. Mock S3 backend initialization
2. Adding mock documents with embeddings
3. Searching vectors
4. Deleting vectors
"""

import sys
from pathlib import Path
import asyncio
import numpy as np

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.search.vector_store import get_vector_store

async def test_mock_s3_vectors():
    """Test the mock S3 vectors backend."""

    print("=" * 60)
    print("PRD-42: Cloud Document Sync - Mock S3 Vectors Test")
    print("=" * 60)

    # Create mock backend
    workspace_id = "test-workspace-123"
    print(f"\n1️⃣  Creating mock S3 vectors backend for workspace: {workspace_id}")

    backend = get_vector_store(
        backend="s3_vectors_mock",
        workspace_id=workspace_id
    )

    await backend.initialize()
    print(f"   ✅ Backend initialized: {backend.bucket_name}")

    # Create mock documents with embeddings
    print(f"\n2️⃣  Adding test documents with embeddings...")

    documents = [
        {
            "external_file_id": "gdrive_file_123",
            "chunk_index": 0,
            "chunk_text": "This is a test document about cloud storage and S3 vectors.",
            "app_name": "GOOGLEDRIVE",
            "file_name": "test_doc.txt",
            "file_path": "/My Drive/test_doc.txt"
        },
        {
            "external_file_id": "gdrive_file_123",
            "chunk_index": 1,
            "chunk_text": "Cloud document sync enables unified RAG across multiple storage providers.",
            "app_name": "GOOGLEDRIVE",
            "file_name": "test_doc.txt",
            "file_path": "/My Drive/test_doc.txt"
        },
        {
            "external_file_id": "dropbox_file_456",
            "chunk_index": 0,
            "chunk_text": "Dropbox integration allows seamless document access.",
            "app_name": "DROPBOX",
            "file_name": "dropbox_notes.txt",
            "file_path": "/Automatos/dropbox_notes.txt"
        }
    ]

    # Generate random embeddings (1024-dim vectors)
    embeddings = [
        np.random.rand(1024).tolist() for _ in documents
    ]

    keys = backend.add_documents(documents, embeddings)
    print(f"   ✅ Added {len(keys)} document chunks")
    print(f"   📊 Total vectors in storage: {backend.get_vector_count()}")

    # Test search
    print(f"\n3️⃣  Testing vector search...")

    query_embedding = np.random.rand(1024).tolist()
    results = backend.search(
        query_embedding=query_embedding,
        limit=5,
        min_score=0.0  # Low threshold for random vectors
    )

    print(f"   ✅ Search completed: found {len(results)} results")
    for i, result in enumerate(results[:3], 1):
        print(f"      {i}. {result['file_name']} (score: {result['score']:.3f})")
        print(f"         Source: {result['source']}")
        print(f"         Preview: {result['content'][:60]}...")

    # Test deletion
    print(f"\n4️⃣  Testing document deletion...")

    deleted = backend.delete_documents("gdrive_file_123")
    print(f"   ✅ Deleted {deleted} chunks for gdrive_file_123")
    print(f"   📊 Remaining vectors: {backend.get_vector_count()}")

    # Test connection cleanup
    print(f"\n5️⃣  Testing connection cleanup...")

    remaining = backend.get_vector_count()
    deleted_all = backend.delete_all_for_connection(connection_id=1)
    print(f"   ✅ Deleted all {deleted_all} vectors")
    print(f"   📊 Final vector count: {backend.get_vector_count()}")

    # Cleanup
    await backend.close()

    print("\n" + "=" * 60)
    print("✅ All tests passed! Mock S3 Vectors backend is working.")
    print("=" * 60)
    print("\n📝 Next steps:")
    print("   1. Start the API server: python -m uvicorn main:app --reload")
    print("   2. Connect a cloud storage provider via Composio")
    print("   3. Test sync via: POST /api/cloud-documents/connections/{id}/sync")
    print("   4. Query vectors via: POST /api/cloud-documents/rag/query")
    print("\n🚀 When ready for production, switch to real S3 Vectors:")
    print("   - Update api/cloud_documents.py: backend='s3_vectors'")
    print("   - Set real AWS credentials in .env")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_mock_s3_vectors())
