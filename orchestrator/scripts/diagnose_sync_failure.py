#!/usr/bin/env python3
"""
Diagnostic script to identify why cloud document sync processing fails.
Runs through the upload_document() flow with detailed logging to pinpoint errors.
"""

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.rag.ingestion.manager import DocumentManager
from urllib.parse import urlparse

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_document_upload():
    """Test document upload with a simple test file"""

    # Parse DATABASE_URL
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL not set")

    parsed = urlparse(database_url)
    db_config = {
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username,
        'password': parsed.password
    }

    # Check S3 Vectors config
    use_s3_vectors = os.getenv('S3_VECTORS_ENABLED', 'false').lower() == 'true'
    s3_bucket = os.getenv('S3_DOCUMENTS_BUCKET', 'automatos-ai')
    workspace_id = "1"  # Test workspace

    logger.info(f"Configuration:")
    logger.info(f"  S3 Vectors Enabled: {use_s3_vectors}")
    logger.info(f"  S3 Bucket: {s3_bucket}")
    logger.info(f"  AWS Region: {os.getenv('AWS_REGION')}")
    logger.info(f"  Workspace ID: {workspace_id}")

    # Create simple test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document for diagnosing cloud sync failures.\n")
        f.write("It contains simple text to verify the processing pipeline.\n")
        f.write("If this fails, we'll see detailed error logs.\n")
        test_file = f.name

    try:
        logger.info(f"Created test file: {test_file}")

        # Initialize DocumentManager
        logger.info("Initializing DocumentManager...")
        doc_manager = DocumentManager(
            db_config=db_config,
            workspace_id=workspace_id,
            use_s3_vectors=use_s3_vectors,
            s3_bucket=s3_bucket
        )
        logger.info("✅ DocumentManager initialized successfully")

        # Upload document
        logger.info("Starting document upload...")
        document_id = await doc_manager.upload_document(
            file_path=test_file,
            filename="test_sync_diagnostic.txt",
            tags=["diagnostic", "cloud_sync"],
            description="Diagnostic test for cloud sync",
            created_by="diagnostic_script"
        )

        logger.info(f"✅ Document uploaded successfully! ID: {document_id}")

        # Verify in database
        import psycopg2
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, filename, status, chunk_count FROM documents WHERE id = %s",
            (document_id,)
        )
        row = cursor.fetchone()

        if row:
            logger.info(f"✅ Document in database:")
            logger.info(f"  ID: {row[0]}")
            logger.info(f"  Filename: {row[1]}")
            logger.info(f"  Status: {row[2]}")
            logger.info(f"  Chunks: {row[3]}")

            if row[2] == "completed":
                logger.info("✅ SUCCESS - Document processed completely!")
            else:
                logger.error(f"❌ FAILED - Document status is '{row[2]}', expected 'completed'")
        else:
            logger.error("❌ Document not found in database!")

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}", exc_info=True)
        return False
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)
            logger.info(f"Cleaned up test file: {test_file}")

    return True


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("CLOUD DOCUMENT SYNC DIAGNOSTIC")
    logger.info("=" * 80)

    success = asyncio.run(test_document_upload())

    logger.info("=" * 80)
    if success:
        logger.info("✅ DIAGNOSTIC PASSED - System is working!")
    else:
        logger.error("❌ DIAGNOSTIC FAILED - Check logs above for errors")
    logger.info("=" * 80)

    sys.exit(0 if success else 1)
