"""
Migrate from PostgreSQL Vectors → S3 Vectors
=============================================

This script:
1. Clears ALL pgvector embeddings from document_chunks table
2. Downloads each document from S3 standard storage
3. Reprocesses through the S3 Vectors pipeline
4. Tracks progress and reports results

Prerequisites:
    - S3_VECTORS_ENABLED=true in .env
    - AWS credentials configured
    - PostgreSQL running

Usage:
    cd orchestrator
    python -m scripts.migrate_to_s3_vectors

    # Dry run (show what would happen, no changes):
    python -m scripts.migrate_to_s3_vectors --dry-run

    # Process specific document IDs only:
    python -m scripts.migrate_to_s3_vectors --doc-ids 1,2,3
"""

import asyncio
import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import psycopg2
from psycopg2.extras import RealDictCursor
import boto3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("migrate_to_s3_vectors")

# Suppress noisy loggers
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_db_config():
    """Build DB config from environment (Railway-compatible)."""
    return {
        "host": os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", "localhost")),
        "port": int(os.getenv("POSTGRES_PORT", os.getenv("DB_PORT", "5432"))),
        "dbname": os.getenv("POSTGRES_DB", os.getenv("DB_NAME", "automatos")),
        "user": os.getenv("POSTGRES_USER", os.getenv("DB_USER", "postgres")),
        "password": os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD", "")),
    }


def get_s3_client():
    """Create S3 client from environment."""
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def phase1_clear_pgvector_embeddings(db_config: dict, dry_run: bool = False):
    """
    Phase 1: Clear all pgvector embeddings from document_chunks.
    Keeps content + metadata rows (they're still useful for reference).
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Clear pgvector embeddings from PostgreSQL")
    logger.info("=" * 60)

    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    # Count existing embeddings
    cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL")
    embedding_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM document_chunks")
    total_chunks = cursor.fetchone()[0]

    logger.info(f"  Total chunks in document_chunks: {total_chunks}")
    logger.info(f"  Chunks with pgvector embeddings: {embedding_count}")

    if embedding_count == 0:
        logger.info("  No pgvector embeddings to clear. Skipping.")
        cursor.close()
        conn.close()
        return

    if dry_run:
        logger.info(f"  [DRY RUN] Would NULL out {embedding_count} embedding values")
        cursor.close()
        conn.close()
        return

    # NULL out embeddings (faster than DELETE + re-insert, preserves chunk metadata)
    logger.info(f"  Clearing {embedding_count} embeddings...")
    cursor.execute("UPDATE document_chunks SET embedding = NULL WHERE embedding IS NOT NULL")
    conn.commit()
    logger.info(f"  Cleared {cursor.rowcount} pgvector embeddings")

    cursor.close()
    conn.close()


def phase2_cleanup_orphans(db_config: dict, dry_run: bool = False):
    """
    Phase 2: Delete orphaned documents whose files are in /tmp/ (no longer exist).
    Returns count of deleted documents.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Clean up orphaned documents (lost /tmp/ files)")
    logger.info("=" * 60)

    conn = psycopg2.connect(**db_config, cursor_factory=RealDictCursor)
    cursor = conn.cursor()

    # Find documents with local /tmp/ paths (these files are gone)
    cursor.execute("""
        SELECT id, filename, file_path, chunk_count, status
        FROM documents
        WHERE file_path IS NOT NULL
          AND file_path LIKE '/tmp/%%'
        ORDER BY id
    """)
    orphans = cursor.fetchall()

    logger.info(f"  Found {len(orphans)} orphaned documents with /tmp/ paths")
    for doc in orphans:
        logger.info(f"    #{doc['id']}: {doc['filename']} [{doc['status']}] chunks={doc['chunk_count']}")

    if not orphans:
        cursor.close()
        conn.close()
        return 0

    orphan_ids = [doc['id'] for doc in orphans]

    if dry_run:
        logger.info(f"  [DRY RUN] Would delete {len(orphans)} orphaned documents and their chunks")
        cursor.close()
        conn.close()
        return 0

    # Delete chunks first (FK constraint)
    cursor.execute("DELETE FROM document_chunks WHERE document_id = ANY(%s)", (orphan_ids,))
    chunks_deleted = cursor.rowcount
    logger.info(f"  Deleted {chunks_deleted} orphaned chunks")

    # Delete the document records
    cursor.execute("DELETE FROM documents WHERE id = ANY(%s)", (orphan_ids,))
    docs_deleted = cursor.rowcount
    conn.commit()
    logger.info(f"  Deleted {docs_deleted} orphaned document records")

    cursor.close()
    conn.close()
    return docs_deleted


def phase3_get_s3_documents(db_config: dict, doc_ids: list = None):
    """
    Phase 3: Get documents stored in S3 that need reprocessing.
    Only returns documents with s3:// file paths.
    """
    conn = psycopg2.connect(**db_config, cursor_factory=RealDictCursor)
    cursor = conn.cursor()

    query = """
        SELECT id, filename, original_filename, file_type, file_size,
               file_path, status, chunk_count, workspace_id
        FROM documents
        WHERE file_path IS NOT NULL
          AND file_path LIKE 's3://%%'
    """
    params = []

    if doc_ids:
        query += " AND id = ANY(%s)"
        params.append(doc_ids)

    query += " ORDER BY id"

    cursor.execute(query, params if params else None)
    documents = cursor.fetchall()

    cursor.close()
    conn.close()
    return documents


async def phase4_reprocess_documents(
    db_config: dict, documents: list, dry_run: bool = False
):
    """
    Phase 4: Reprocess each document through S3 Vectors pipeline.
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: Reprocess documents → S3 Vectors")
    logger.info("=" * 60)
    logger.info(f"  Documents to process: {len(documents)}")

    if not documents:
        logger.info("  No documents to process.")
        return {"processed": 0, "failed": 0, "skipped": 0}

    s3_bucket = os.getenv("S3_DOCUMENTS_BUCKET", "automatos-ai")
    s3_client = get_s3_client()

    # Group documents by workspace
    workspace_docs = {}
    for doc in documents:
        ws = str(doc["workspace_id"]) if doc["workspace_id"] else "default"
        workspace_docs.setdefault(ws, []).append(doc)

    results = {"processed": 0, "failed": 0, "skipped": 0, "errors": []}

    for workspace_id, docs in workspace_docs.items():
        logger.info(f"\n  Workspace: {workspace_id} ({len(docs)} documents)")

        if dry_run:
            for doc in docs:
                logger.info(f"    [DRY RUN] Would reprocess: #{doc['id']} {doc['filename']} ({doc['file_path']})")
                results["skipped"] += 1
            continue

        # Create DocumentManager with S3 Vectors enabled
        from modules.rag.ingestion.manager import DocumentManager
        doc_manager = DocumentManager(
            db_config=db_config,
            workspace_id=workspace_id,
            use_s3_vectors=True,
            s3_bucket=s3_bucket,
        )

        for doc in docs:
            doc_id = doc["id"]
            filename = doc["filename"]
            file_path = doc["file_path"]

            logger.info(f"\n    Processing #{doc_id}: {filename}")

            try:
                # Download from S3 to temp file
                if file_path.startswith("s3://"):
                    # Parse s3://bucket/key
                    s3_parts = file_path.replace("s3://", "").split("/", 1)
                    bucket = s3_parts[0]
                    key = s3_parts[1] if len(s3_parts) > 1 else ""

                    suffix = Path(filename).suffix or ""
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp_path = tmp.name
                    tmp.close()

                    logger.info(f"      Downloading from S3: {bucket}/{key}")
                    s3_client.download_file(bucket, key, tmp_path)
                    file_size = os.path.getsize(tmp_path)
                    logger.info(f"      Downloaded {file_size} bytes")
                elif os.path.exists(file_path):
                    tmp_path = file_path
                    logger.info(f"      Using local file: {file_path}")
                else:
                    logger.warning(f"      File not found: {file_path} — skipping")
                    results["skipped"] += 1
                    continue

                # Clear old chunks for this document
                conn = psycopg2.connect(**db_config)
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM document_chunks WHERE document_id = %s", (doc_id,)
                )
                deleted = cursor.rowcount
                conn.commit()
                cursor.close()
                conn.close()
                logger.info(f"      Cleared {deleted} old chunks")

                # Detect file type
                file_type = doc_manager.processor.detect_file_type(tmp_path)
                logger.info(f"      File type: {file_type}")

                # Reprocess through the pipeline (extract → chunk → embed → S3 vectors)
                s3_key = None
                if file_path.startswith("s3://"):
                    s3_key = file_path.replace(f"s3://{bucket}/", "")

                await doc_manager._process_document(doc_id, tmp_path, file_type, s3_key)

                results["processed"] += 1
                logger.info(f"      Done — document #{doc_id} reprocessed to S3 Vectors")

                # Cleanup temp file (only if we created it)
                if file_path.startswith("s3://") and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            except Exception as e:
                logger.error(f"      FAILED: {e}", exc_info=True)
                results["failed"] += 1
                results["errors"].append({"doc_id": doc_id, "error": str(e)})

                # Mark as failed in DB
                try:
                    conn = psycopg2.connect(**db_config)
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE documents SET status = 'failed' WHERE id = %s",
                        (doc_id,),
                    )
                    conn.commit()
                    cursor.close()
                    conn.close()
                except Exception:
                    pass

    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Migrate from PostgreSQL vectors to S3 Vectors"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would happen without making changes"
    )
    parser.add_argument(
        "--doc-ids", type=str, help="Comma-separated document IDs to process (default: all)"
    )
    parser.add_argument(
        "--skip-clear", action="store_true", help="Skip clearing pgvector embeddings"
    )
    args = parser.parse_args()

    doc_ids = [int(x) for x in args.doc_ids.split(",")] if args.doc_ids else None

    # Validate environment
    s3_vectors_enabled = os.getenv("S3_VECTORS_ENABLED", "false").lower() == "true"
    if not s3_vectors_enabled:
        logger.error("S3_VECTORS_ENABLED is not set to true in .env — aborting")
        sys.exit(1)

    db_config = get_db_config()

    logger.info("=" * 60)
    logger.info("  MIGRATION: PostgreSQL Vectors → S3 Vectors")
    logger.info("=" * 60)
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info(f"  S3 Vectors: ENABLED")
    logger.info(f"  S3 Vectors Bucket: {os.getenv('S3_VECTORS_BUCKET')}")
    logger.info(f"  S3 Vectors Index: {os.getenv('S3_VECTORS_INDEX_NAME')}")
    logger.info(f"  Document IDs: {doc_ids or 'ALL'}")
    logger.info("")

    start = time.time()

    # Phase 1: Clear pgvector embeddings
    if not args.skip_clear:
        phase1_clear_pgvector_embeddings(db_config, dry_run=args.dry_run)
    else:
        logger.info("  Skipping pgvector clearing (--skip-clear)")

    # Phase 2: Clean up orphaned /tmp/ documents
    logger.info("")
    phase2_cleanup_orphans(db_config, dry_run=args.dry_run)

    # Phase 3: Get S3 documents for reprocessing
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 3: Enumerate S3 documents for reprocessing")
    logger.info("=" * 60)
    documents = phase3_get_s3_documents(db_config, doc_ids)
    logger.info(f"  Found {len(documents)} S3-backed documents")
    for doc in documents:
        status = doc["status"] or "unknown"
        logger.info(f"    #{doc['id']}: {doc['filename']} [{status}] chunks={doc['chunk_count']}")

    # Phase 4: Reprocess through S3 Vectors
    results = await phase4_reprocess_documents(db_config, documents, dry_run=args.dry_run)

    elapsed = time.time() - start

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("  MIGRATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Processed: {results['processed']}")
    logger.info(f"  Failed:    {results['failed']}")
    logger.info(f"  Skipped:   {results['skipped']}")
    logger.info(f"  Time:      {elapsed:.1f}s")

    if results["errors"]:
        logger.info("")
        logger.info("  Errors:")
        for err in results["errors"]:
            logger.info(f"    Doc #{err['doc_id']}: {err['error']}")

    logger.info("")


if __name__ == "__main__":
    asyncio.run(main())
