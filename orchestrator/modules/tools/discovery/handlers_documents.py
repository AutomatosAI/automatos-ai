"""Document handlers for PlatformActionExecutor."""

import logging
from typing import Any, Dict, List, Optional
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


# ---------------------------------------------------------------------------
# PRD-157 S2: document-reading tools (read_document, grep_documents)
# ---------------------------------------------------------------------------

_READ_PAGE_TOKEN_BUDGET = 2000   # D11: token-budgeted page per read_document call
_GREP_MAX_SCAN_CHUNKS = 5000     # bound the literal scan
_GREP_SNIPPET_TOKENS = 120       # per-match snippet token budget


def _resolve_agent_team(db: Session, agent_id: Any) -> Optional[str]:
    """Look up an agent's team for retrieval scoping. None when unknown."""
    if not agent_id:
        return None
    try:
        from core.models import Agent

        row = db.query(Agent).filter(Agent.id == int(agent_id)).first()
        return getattr(row, "team", None) if row else None
    except Exception:
        return None


async def read_document(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Paged full-content reading of a document, workspace+team scoped (S1),
    token-budgeted per page (D11). Lets an agent read past the short search snippet.
    """
    from sqlalchemy import text as sa_text
    from core.models import Document
    from modules.rag.retrieval_filters import build_retrieval_filters, allowed_document_ids
    from modules.rag.budget import count_tokens

    document_id = params.get("document_id")
    if not document_id:
        return {"success": False, "error": "Missing required parameter: document_id"}
    try:
        document_id = int(document_id)
    except (TypeError, ValueError):
        return {"success": False, "error": "document_id must be an integer"}

    try:
        page = max(0, int(params.get("page") or 0))
    except (TypeError, ValueError):
        page = 0

    # S1 scope: workspace always enforced; team enforced when the agent has one.
    team = _resolve_agent_team(db, params.get("_agent_id"))
    filters = build_retrieval_filters(workspace_id=str(workspace_id), team=team)
    if str(document_id) not in allowed_document_ids(db, [document_id], filters):
        return {"success": False, "error": "Document not found or not accessible"}

    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        return {"success": False, "error": "Document not found"}

    rows = db.execute(
        sa_text(
            "SELECT chunk_index, content FROM document_chunks "
            "WHERE document_id = :doc_id ORDER BY chunk_index"
        ),
        {"doc_id": document_id},
    ).fetchall()
    if not rows:
        return {"success": False, "error": "Document has no readable content yet"}

    # Pack chunks into deterministic, token-budgeted pages (never split a chunk).
    pages: List[Dict[str, Any]] = []
    buf: List[str] = []
    buf_tokens = 0
    start_idx = rows[0].chunk_index
    prev_idx = rows[0].chunk_index
    for row in rows:
        tok = count_tokens(row.content or "")
        if buf and buf_tokens + tok > _READ_PAGE_TOKEN_BUDGET:
            pages.append({"start": start_idx, "end": prev_idx, "content": "\n\n".join(buf)})
            buf, buf_tokens, start_idx = [], 0, row.chunk_index
        buf.append(row.content or "")
        buf_tokens += tok
        prev_idx = row.chunk_index
    if buf:
        pages.append({"start": start_idx, "end": prev_idx, "content": "\n\n".join(buf)})

    total_pages = len(pages)
    # An explicit offset wins: jump to the page containing that chunk index.
    offset = params.get("offset")
    if offset is not None:
        try:
            off = int(offset)
            for idx, pg in enumerate(pages):
                if pg["start"] <= off <= pg["end"]:
                    page = idx
                    break
        except (TypeError, ValueError):
            pass
    if page >= total_pages:
        page = total_pages - 1
    current = pages[page]

    return {
        "success": True,
        "document_id": document_id,
        "source_id": document_id,
        "filename": doc.original_filename or doc.filename,
        "file_type": doc.file_type,
        "page": page,
        "total_pages": total_pages,
        "has_more": page < total_pages - 1,
        "next_page": page + 1 if page < total_pages - 1 else None,
        "chunk_range": {"start": current["start"], "end": current["end"]},
        "content": current["content"],
        "staleness": {
            "uploaded_at": doc.upload_date.isoformat() if doc.upload_date else None,
            "last_accessed": doc.last_accessed.isoformat()
            if getattr(doc, "last_accessed", None)
            else None,
        },
    }


async def grep_documents(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Regex search over document chunk text, workspace+team scoped (S1).

    The agent's own team is the security boundary; an explicit ``team`` param can
    only narrow within it (``effective_team`` prefers the agent team when set).
    """
    import re
    from sqlalchemy import text as sa_text
    from core.team_access import effective_team
    from modules.rag.retrieval_filters import build_retrieval_filters, scope_where_clause
    from modules.rag.budget import truncate_to_token_budget

    pattern = params.get("pattern")
    if not pattern or not str(pattern).strip():
        return {"success": False, "error": "Missing required parameter: pattern"}
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return {"success": False, "error": f"Invalid regular expression: {exc}"}

    try:
        limit = max(1, min(int(params.get("limit") or 20), 200))
    except (TypeError, ValueError):
        limit = 20

    agent_team = _resolve_agent_team(db, params.get("_agent_id"))
    team = effective_team(agent_team, params.get("team"))
    filters = build_retrieval_filters(workspace_id=str(workspace_id), team=team)

    # 1. resolve accessible documents (workspace + team) — no join ambiguity.
    docs_sql = f"SELECT id, filename FROM documents WHERE {scope_where_clause(filters)}"
    doc_params: Dict[str, Any] = dict(filters.sql_params())
    only_doc = params.get("document_id")
    if only_doc:
        try:
            doc_params["only_doc"] = int(only_doc)
            docs_sql += " AND id = :only_doc"
        except (TypeError, ValueError):
            return {"success": False, "error": "document_id must be an integer"}
    doc_rows = db.execute(sa_text(docs_sql), doc_params).fetchall()
    doc_names = {r.id: r.filename for r in doc_rows}
    if not doc_names:
        return {"success": True, "pattern": pattern, "matches": [], "count": 0, "scanned_chunks": 0}

    # 2. scan their chunk text (bounded) and regex-match.
    chunk_rows = db.execute(
        sa_text(
            "SELECT document_id, chunk_index, content FROM document_chunks "
            "WHERE document_id = ANY(CAST(:ids AS int[])) ORDER BY document_id, chunk_index LIMIT :scan"
        ),
        {"ids": list(doc_names.keys()), "scan": _GREP_MAX_SCAN_CHUNKS},
    ).fetchall()

    matches: List[Dict[str, Any]] = []
    for row in chunk_rows:
        content = row.content or ""
        if not rx.search(content):
            continue
        matches.append(
            {
                "document_id": row.document_id,
                "source_id": row.document_id,
                "filename": doc_names.get(row.document_id),
                "chunk_index": row.chunk_index,
                "snippet": truncate_to_token_budget(content, _GREP_SNIPPET_TOKENS),
            }
        )
        if len(matches) >= limit:
            break

    return {
        "success": True,
        "pattern": pattern,
        "matches": matches,
        "count": len(matches),
        "scanned_chunks": len(chunk_rows),
        "truncated": len(chunk_rows) >= _GREP_MAX_SCAN_CHUNKS,
    }
