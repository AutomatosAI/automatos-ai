"""Document handlers for PlatformActionExecutor."""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# platform_search_documents bounds — a passage is read in a prompt, not a viewer.
SEARCH_DOCUMENTS_DEFAULT_LIMIT = 8
SEARCH_DOCUMENTS_MAX_LIMIT = 25
SEARCH_DOCUMENTS_MAX_PASSAGE_CHARS = 1200


async def list_templates(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """PRD-167 S6: list the workspace's document templates for an agent (a social
    composition is never block-editable, PRD-251 S1.2)."""
    from core.social_templates import is_social_format
    from modules.documents.template_service import DocumentTemplateService

    service = DocumentTemplateService(db)
    templates = service.list_templates(
        workspace_id, format=params.get("format"), category=params.get("category")
    )
    return {
        "success": True,
        "templates": [
            {
                "id": str(t.id),
                "name": t.name,
                "description": t.description,
                "format": t.format,
                "category": t.category,
                "has_blocks": bool(t.blocks) and not is_social_format(t.format),
            }
            for t in templates
        ],
        "count": len(templates),
    }


async def get_template_schema(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """PRD-167 S6: describe the data a template needs (variable chips + data.* fields).

    PRD-251 S1.2: a social template's data fields are its ``variables_schema``,
    which the answer carries whole (types, defaults, claims) with its sizes.
    """
    from core.social_templates import is_social_format
    from modules.documents.blocks import collect_variable_paths, validate_blocks
    from modules.documents.template_service import DocumentTemplateService
    from modules.documents.template_summary import social_variable_names
    from modules.documents.variables import CATALOG_BY_PATH

    template_id_raw = params.get("template_id")
    if not template_id_raw:
        return {"success": False, "error": "Missing required parameter: template_id"}
    try:
        template_id = UUID(str(template_id_raw))
    except (ValueError, TypeError):
        return {"success": False, "error": f"Invalid template_id: {template_id_raw!r}"}

    service = DocumentTemplateService(db)
    template = service.get_template(template_id, workspace_id)
    if not template:
        return {"success": False, "error": "Template not found"}

    variables: List[Dict[str, Any]] = []
    data_fields: List[str] = []
    social = is_social_format(template.format)
    if social:
        data_fields = [f"data.{name}" for name in social_variable_names(template.blocks)]
    elif template.blocks:
        for path in sorted(collect_variable_paths(validate_blocks(template.blocks))):
            if path.startswith("data."):
                data_fields.append(path)
            elif path in CATALOG_BY_PATH:
                entry = CATALOG_BY_PATH[path]
                variables.append({"path": path, "label": entry["label"], "category": entry["category"]})

    schema = {
        "success": True,
        "id": str(template.id),
        "name": template.name,
        "format": template.format,
        "description": template.description,
        "uses_blocks": bool(template.blocks) and not social,
        "variables": variables,           # auto-resolved chips (user/company/brand/date)
        "data_fields": data_fields,       # data.* fields you must supply at generation
        "data_schema": template.data_schema or {},  # legacy templates
        "sample_data": template.sample_data or {},
    }
    if social:
        blocks = template.blocks if isinstance(template.blocks, dict) else {}
        schema["variables_schema"] = blocks.get("variables_schema") or {}
        schema["sizes"] = blocks.get("sizes") or []
    return schema


# ---------------------------------------------------------------------------
# PRD-251 US-115: the brand kit. Thin wrappers over modules/documents/brand_kit.py,
# the functions GET and PUT /api/documents/brand-kit call.
# ---------------------------------------------------------------------------


def _workspace(db: Session, workspace_id: UUID):
    from core.models.workspaces import Workspace

    return db.query(Workspace).filter(Workspace.id == workspace_id).first()


async def get_brand_kit_tool(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """The kit as GET /api/documents/brand-kit returns it, with the suggestions the
    Brand Kit form prefills from (an agent is not a user, so none come from one)."""
    from modules.documents import brand_kit

    workspace = _workspace(db, workspace_id)
    if workspace is None:
        return {"success": False, "error": "Workspace not found"}
    return {
        "success": True,
        "brand_kit": brand_kit.get_brand_kit(workspace.settings),
        "suggestions": brand_kit.brand_kit_suggestions(db, workspace),
    }


async def update_brand_kit_tool(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Change some kit fields through the PUT's validation and writer.

    Fails closed: a field the kit's patch does not take (a stored file, a misspelt
    name) is refused rather than dropped, and an invalid value saves nothing.
    """
    from pydantic import ValidationError

    from modules.documents import brand_kit

    fields = {k: v for k, v in params.items() if not k.startswith("_")}  # "_" keys: server-injected
    refused = sorted(set(fields) - set(brand_kit.PATCH_FIELDS))
    if refused:
        return {
            "success": False,
            "error": (
                f"platform_update_brand_kit cannot set {', '.join(refused)}; nothing saved. "
                f"It sets {', '.join(brand_kit.PATCH_FIELDS)}. The logo, logo mark and font "
                "files are uploaded by a person in the brand kit settings."
            ),
        }
    patch = {k: v for k, v in fields.items() if v is not None}
    if not patch:
        return {"success": False, "error": "Nothing to change: send at least one brand kit field."}
    workspace = _workspace(db, workspace_id)
    if workspace is None:
        return {"success": False, "error": "Workspace not found"}
    try:
        kit = brand_kit.update_brand_kit(db, workspace, patch)
    except ValidationError as exc:
        errors = brand_kit.brand_kit_errors(exc)
        reasons = "; ".join(f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in errors)
        return {"success": False, "error": f"Invalid brand kit, nothing saved. {reasons}", "errors": errors}
    return {"success": True, "brand_kit": kit, "changed": sorted(patch)}


def _whole_number(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _like_text(text: str) -> str:
    """``text`` as literal characters inside a LIKE pattern: `_` and `%` in a
    filename are not wildcards (roast_log_…, 100%-arabica.md)."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def list_documents(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """F138 (night 4): ``search`` was ignored and only the newest page came back,
    so Auto said a document "might have been deleted" after reading the newest
    20 (B6, B71). ``search`` now filters by name or description, ``offset``
    pages, and ``total`` says how many match."""
    from sqlalchemy import or_, text

    from core.models import Document

    limit = min(max(_whole_number(params.get("limit"), 50), 1), 200)
    offset = max(_whole_number(params.get("offset"), 0), 0)
    query = db.query(Document).filter(Document.workspace_id == workspace_id)
    # F155: a widget turn lists only its key's team's documents and those
    # shared with every team (the PRD-124 rule of TEAM_FILTER_CLAUSE).
    from core.security.surface import widget_team

    lock = widget_team()
    if lock:
        query = query.filter(text("(documents.team_access = '{}' OR :team = ANY(documents.team_access))")
                             .bindparams(team=lock))
    search = str(params.get("search") or "").strip()
    if search:
        pattern = f"%{_like_text(search)}%"
        query = query.filter(or_(
            Document.filename.ilike(pattern, escape="\\"),
            Document.original_filename.ilike(pattern, escape="\\"),
            Document.description.ilike(pattern, escape="\\"),
        ))
    total = query.count()
    docs = query.order_by(Document.upload_date.desc(), Document.id.desc()).offset(offset).limit(limit).all()

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
        "total": total,
        "offset": offset,
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
            from core.storage import get_s3_client

            parts = file_path.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
            s3 = get_s3_client()
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
            from core.storage import get_s3_client

            parts = file_path.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
            s3 = get_s3_client()
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
            from core.storage import get_s3_client

            parts = file_path.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
            suffix = "." + key.rsplit(".", 1)[-1] if "." in key else ""
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            get_s3_client().download_file(bucket, key, tmp.name)
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

        # F087: the same name again replaces that document (same id, new text,
        # the old source kept in its history) — never a second copy beside it.
        from services.document_versions import replace_document, replaceable_document, replaced_message

        replaced = replaceable_document(db, workspace_id, filename, [])
        if replaced is not None:
            version = await replace_document(
                db, replaced, workspace_id=workspace_id, file_path=str(file_path), file_size=len(data),
                content_hash=content_hash, file_type=file_type, replaced_by="auto",
                description=params.get("description"),
            )
            return {
                "success": replaced.status != "failed",
                "document_id": replaced.id,
                "filename": replaced.filename,
                "status": replaced.status,
                "replaced": True,
                "version": version,
                "message": replaced_message(replaced.filename, version, replaced.status),
            }

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
            team_access=_widget_team_access(),
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


def _widget_team_access() -> list:
    """F155: a widget key's upload belongs to its team, if it has one."""
    from core.security.surface import widget_team

    lock = widget_team()
    return [lock] if lock else []


def _resolve_agent_team(db: Session, agent_id: Any) -> Optional[str]:
    """The team that scopes this call's retrieval: a widget key's team lock on
    a widget turn, else the agent's team (core.team_access.retrieval_team).
    None when neither is known."""
    from core.team_access import retrieval_team

    if not agent_id:
        return retrieval_team(None)
    try:
        from core.models import Agent

        row = db.query(Agent).filter(Agent.id == int(agent_id)).first()
        return retrieval_team(getattr(row, "team", None) if row else None)
    except Exception:
        return retrieval_team(None)


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

    # F181 (night 6): a spreadsheet is counted with code on its copy in the
    # workspace (made here from the stored upload if missing), never from pages.
    from services.spreadsheet_workspace import COUNT_WITH_CODE, workspace_copies

    copies = await workspace_copies(workspace_id, doc)
    counted: Dict[str, Any] = {}
    if copies:
        counted = {"workspace_path": copies[0]["workspace_path"], "row_count": copies[0]["row_count"],
                   "count_with_code": COUNT_WITH_CODE, **({"sheets": copies} if len(copies) > 1 else {})}

    rows = db.execute(
        sa_text(
            "SELECT chunk_index, content FROM document_chunks "
            "WHERE document_id = :doc_id ORDER BY chunk_index"
        ),
        {"doc_id": document_id},
    ).fetchall()
    if not rows:
        if counted:  # its copy is readable even before the knowledge base has it
            return {"success": True, "document_id": document_id, "source_id": document_id,
                    "filename": doc.original_filename or doc.filename, "file_type": doc.file_type,
                    **counted, "content": "", "total_pages": 0, "has_more": False}
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
        **counted,
        "content": current["content"],
        "staleness": {
            "uploaded_at": doc.upload_date.isoformat() if doc.upload_date else None,
            "last_accessed": doc.last_accessed.isoformat()
            if getattr(doc, "last_accessed", None)
            else None,
        },
    }


async def search_documents(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Semantic search over knowledge-base documents.

    Night 1 (2026-09-18): agents had ``platform_search_memory`` (memories),
    ``platform_grep_documents`` (literal regex) and ``platform_read_document``
    (needs an id) — but no way to ask a *question* of the uploaded documents.
    Auto could answer from documents because it lists and reads; every agent
    that searched came back empty, in eleven tickets.

    Runs the same retrieval funnel the chat path uses, so it picks up the
    edition's vector backend (S3 Vectors on SaaS, pgvector locally), hybrid
    search and reranking, rather than a second retrieval implementation.
    """
    from core.team_access import effective_team
    from modules.rag.service import RAGService

    query = (params.get("query") or "").strip()
    if not query:
        return {"success": False, "error": "query is required"}

    limit = params.get("limit")
    try:
        limit = max(1, min(int(limit), SEARCH_DOCUMENTS_MAX_LIMIT))
    except (TypeError, ValueError):
        limit = SEARCH_DOCUMENTS_DEFAULT_LIMIT

    # Same security boundary as grep_documents: the agent's own team wins,
    # an explicit team can only narrow within it.
    team = effective_team(_resolve_agent_team(db, params.get("_agent_id")), params.get("team"))

    try:
        rag = RAGService()
        result = await rag.retrieve(
            query=query,
            max_chunks=limit,
            context_type="agent",
            workspace_id=str(workspace_id),
            team=team,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("[documents] semantic search failed: %s", e, exc_info=True)
        return {"success": False, "error": f"document search failed: {e}"}

    # RAGService chunks carry `source_file` and `document_id`; the retrieval
    # SCORE lives in sources_map (a chunk's own `similarity` is the
    # post-optimisation value and reads 1.0 for everything).
    scores = {
        str(entry.get("document_id")): entry.get("score")
        for entry in (result.sources_map or [])
        if entry.get("document_id") is not None
    }
    passages = []
    for chunk in (result.chunks or [])[:limit]:
        metadata = chunk.get("metadata") or {}
        document_id = chunk.get("document_id") or metadata.get("document_id")
        score = scores.get(str(document_id))
        if score is None:
            score = chunk.get("similarity") or 0.0
        passages.append({
            "document_id": document_id,
            "file_name": chunk.get("source_file") or metadata.get("file_name") or "",
            "chunk_index": chunk.get("chunk_index") or metadata.get("chunk_index"),
            "score": round(float(score), 4),
            "content": (chunk.get("content") or "")[:SEARCH_DOCUMENTS_MAX_PASSAGE_CHARS],
        })

    if not passages:
        return {
            "success": True,
            "results": [],
            "count": 0,
            "message": (
                "No document passages matched. The documents may not be embedded yet — "
                "platform_list_documents shows what is uploaded and its chunk count."
            ),
        }

    return {
        "success": True,
        "results": passages,
        "count": len(passages),
        "sources": result.sources or [],
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
