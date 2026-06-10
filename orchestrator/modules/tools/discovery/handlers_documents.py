"""Document handlers for PlatformActionExecutor."""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def list_documents(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models import Document

    limit = min(params.get("limit", 50), 200)

    docs = (
        db.query(Document)
        .filter(Document.workspace_id == workspace_id)
        .order_by(Document.upload_date.desc())
        .limit(limit)
        .all()
    )

    return {
        "success": True,
        "documents": [
            {
                "id": d.id,
                "filename": d.original_filename or d.filename,
                "file_type": d.file_type,
                "file_size": d.file_size,
                "status": d.status,
                "chunk_count": d.chunk_count or 0,
                "uploaded_at": d.upload_date.isoformat() if d.upload_date else None,
            }
            for d in docs
        ],
        "count": len(docs),
    }


async def delete_document(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a document -- S3 file + vector embeddings + DB record."""
    from core.models import Document

    document_id = params.get("document_id")
    if not document_id:
        return {"success": False, "error": "Missing required parameter: document_id"}

    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
        .first()
    )
    if not doc:
        return {"success": False, "error": "Document not found"}

    doc_info = {
        "id": doc.id,
        "filename": doc.original_filename or doc.filename,
    }
    cleanup_notes = []

    # Phase 1: S3 file cleanup (non-fatal)
    file_path = doc.file_path or ""
    if file_path.startswith("s3://"):
        try:
            import boto3
            parts = file_path.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
            s3 = boto3.client("s3")
            s3.delete_object(Bucket=bucket, Key=key)
            cleanup_notes.append("S3 file deleted")
        except Exception as e:
            logger.warning("[PlatformExecutor] S3 cleanup failed for doc %d: %s", doc.id, e)
            cleanup_notes.append(f"S3 cleanup failed: {e}")

    # Phase 2: Vector embedding cleanup (non-fatal)
    try:
        from modules.search.vector_store.backends.s3_vectors_backend import S3VectorsBackend
        backend = S3VectorsBackend()
        deleted = backend.delete_documents(str(doc.id))
        cleanup_notes.append(f"Vector embeddings deleted ({deleted} removed)")
    except Exception as e:
        logger.warning("[PlatformExecutor] Vector cleanup failed for doc %d: %s", doc.id, e)
        cleanup_notes.append(f"Vector cleanup failed: {e}")

    # Phase 3: DB record (cascades to document_chunks via FK)
    db.delete(doc)
    db.flush()
    cleanup_notes.append("Database record deleted")

    logger.info("[PlatformExecutor] Deleted document %s -- %s", doc_info, ", ".join(cleanup_notes))

    return {
        "success": True,
        "deleted_document": doc_info,
        "cleanup": cleanup_notes,
        "message": f"Document '{doc_info['filename']}' (ID {doc_info['id']}) deleted.",
    }


async def reprocess_document(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Re-process a document -- regenerate chunks and vector embeddings."""
    from core.models import Document

    document_id = params.get("document_id")
    if not document_id:
        return {"success": False, "error": "Missing required parameter: document_id"}

    doc = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
        .first()
    )
    if not doc:
        return {"success": False, "error": "Document not found"}

    file_path = doc.file_path or ""

    # Validate file exists
    if file_path.startswith("s3://"):
        try:
            import boto3
            parts = file_path.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
            s3 = boto3.client("s3")
            s3.head_object(Bucket=bucket, Key=key)
        except Exception as e:
            return {"success": False, "error": f"S3 file not accessible: {e}"}
    elif file_path:
        import os
        if not os.path.exists(file_path):
            return {"success": False, "error": f"Local file not found: {file_path}"}
    else:
        return {"success": False, "error": "Document has no file_path"}

    # Set status to processing
    doc.status = "processing"
    db.flush()

    # Re-process via DocumentManager
    try:
        from api.documents import get_document_manager

        dm = get_document_manager(str(workspace_id))

        # For S3 files, download to temp first
        actual_path = file_path
        if file_path.startswith("s3://"):
            import tempfile
            import boto3
            parts = file_path.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
            suffix = "." + key.rsplit(".", 1)[-1] if "." in key else ""
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            boto3.client("s3").download_file(bucket, key, tmp.name)
            actual_path = tmp.name

        new_doc_id = await dm.upload_document(
            file_path=actual_path,
            filename=doc.original_filename or doc.filename,
        )

        logger.info("[PlatformExecutor] Reprocessed document %d -> new doc %s", doc.id, new_doc_id)

        return {
            "success": True,
            "document_id": new_doc_id,
            "original_document_id": doc.id,
            "message": f"Document '{doc.original_filename or doc.filename}' reprocessed successfully.",
        }
    except Exception as e:
        doc.status = "failed"
        db.flush()
        logger.error("[PlatformExecutor] Reprocess failed for doc %d: %s", doc.id, e, exc_info=True)
        return {"success": False, "error": f"Reprocessing failed: {e}"}


_TEXT_UPLOAD_EXTENSIONS = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".json": "json",
}


async def upload_document(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a knowledge document from text content and process it into RAG (PRD-143 S10).

    Mirrors POST /api/documents/upload for the text formats Auto can supply
    (markdown/text/json): same dedupe-by-hash, same UPLOAD_DIR/MAX_UPLOAD_BYTES,
    same DocumentManager processing. Binary formats stay dashboard-only.
    """
    from pathlib import Path

    filename = (params.get("filename") or "").strip()
    content = params.get("content")
    if not filename:
        return {"success": False, "error": "Missing required parameter: filename"}
    if not content:
        return {"success": False, "error": "Missing required parameter: content"}

    ext = Path(filename).suffix.lower()
    file_type = _TEXT_UPLOAD_EXTENSIONS.get(ext)
    if file_type is None:
        return {
            "success": False,
            "error": (
                f"filename extension must be one of {sorted(_TEXT_UPLOAD_EXTENSIONS)} — "
                "this tool uploads text content; use the dashboard for binary files"
            ),
        }

    try:
        import hashlib
        import uuid as _uuid

        from api.documents import MAX_UPLOAD_BYTES, UPLOAD_DIR, get_document_manager
        from core.models import Document

        data = content.encode("utf-8")
        if len(data) > MAX_UPLOAD_BYTES:
            return {"success": False, "error": "Content too large (max 50MB)"}

        content_hash = hashlib.sha256(data).hexdigest()
        existing = (
            db.query(Document)
            .filter(
                Document.content_hash == content_hash,
                Document.workspace_id == workspace_id,
            )
            .first()
        )
        if existing:
            return {
                "success": True,
                "status": "duplicate",
                "document_id": existing.id,
                "filename": existing.filename,
                "message": "Document already exists",
            }

        UPLOAD_DIR.mkdir(exist_ok=True)
        file_path = UPLOAD_DIR / f"{_uuid.uuid4().hex}{ext}"
        file_path.write_bytes(data)

        document = Document(
            workspace_id=workspace_id,
            filename=filename,
            original_filename=filename,
            file_type=file_type,
            file_size=len(data),
            file_path=str(file_path),
            content_hash=content_hash,
            status="uploaded",
            description=params.get("description"),
            team_access=[],
            created_by="auto",
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        try:
            from modules.rag import DocumentType

            type_enum = {
                "markdown": DocumentType.MARKDOWN,
                "text": DocumentType.TEXT,
                "json": DocumentType.JSON,
            }[file_type]

            document.status = "processing"
            db.commit()

            doc_manager = get_document_manager(str(workspace_id))
            await doc_manager._process_document(document.id, str(file_path), type_enum)
            db.refresh(document)
        except Exception as exc:
            logger.error("[PlatformExecutor] upload_document processing failed for doc %s: %s",
                         document.id, exc, exc_info=True)
            document.status = "failed"
            db.commit()

        return {
            "success": document.status != "failed",
            "document_id": document.id,
            "filename": document.filename,
            "status": document.status,
            "chunk_count": document.chunk_count or 0,
        }
    except Exception as exc:
        db.rollback()
        logger.error("[PlatformExecutor] upload_document failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}
