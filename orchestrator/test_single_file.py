#!/usr/bin/env python3
"""Test single file upload to avoid race conditions"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.rag.ingestion.manager import DocumentManager
from urllib.parse import urlparse

async def test_single_file():
    """Test uploading a single file directly"""

    database_url = os.getenv('DATABASE_URL')
    parsed = urlparse(database_url)
    db_config = {
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username,
        'password': parsed.password
    }

    # Create test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Test Document\n\n")
        f.write("This is a test document for cloud sync.\n\n")
        f.write("It contains enough text to create embeddings and test S3 Vectors storage.\n")
        test_file = f.name

    try:
        print("=" * 80)
        print("🧪 SINGLE FILE UPLOAD TEST")
        print("=" * 80)
        print(f"Test file: {test_file}")
        print()

        # Initialize DocumentManager
        doc_manager = DocumentManager(
            db_config=db_config,
            workspace_id="ae8320bc-95e1-4de1-bbe9-396bef19cbf8",
            use_s3_vectors=True,
            s3_bucket="automatos-ai"
        )
        print("✅ DocumentManager initialized")

        # Upload document
        print("📤 Uploading document...")
        document_id = await doc_manager.upload_document(
            file_path=test_file,
            filename="test_single_file.md",
            tags=["test", "single_file"],
            description="Single file test for S3 Vectors",
            created_by="test_script"
        )

        print(f"✅ Document uploaded! ID: {document_id}")

        # Check database
        import psycopg2
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, filename, status, chunk_count
            FROM documents
            WHERE id = %s
        """, (document_id,))

        row = cursor.fetchone()
        if row:
            doc_id, filename, status, chunks = row
            print()
            print("📊 DOCUMENT STATUS:")
            print(f"  ID: {doc_id}")
            print(f"  Filename: {filename}")
            print(f"  Status: {status}")
            print(f"  Chunks: {chunks}")

            if status == "completed" and chunks > 0:
                print()
                print("✅ ✅ ✅ SUCCESS! Document processed with S3 Vectors! ✅ ✅ ✅")
                return True
            else:
                print()
                print(f"❌ FAILED: Status={status}, Chunks={chunks}")
                return False
        else:
            print("❌ Document not found in database!")
            return False

    except Exception as e:
        print()
        print("❌ ERROR:")
        print(str(e))
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

if __name__ == "__main__":
    success = asyncio.run(test_single_file())
    print("\n" + "=" * 80)
    if success:
        print("✅ TEST PASSED!")
    else:
        print("❌ TEST FAILED!")
    print("=" * 80 + "\n")
    sys.exit(0 if success else 1)
