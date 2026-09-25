"""
Document Generation API Routes (PRD-63)
========================================

REST endpoints for template CRUD, document generation, and file serving.
"""

import logging
import os
import shutil
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.workspace_permission import require_workspace_permission
from core.auth.dependencies import RequestContext
from core.database.database import get_db
from core.media_render_client import NOT_CONFIGURED, MediaRenderError, MediaRenderUnavailable
from core.media_render_quota import RenderQuotaExceeded
from core.social_templates import InvalidVariableValues, SocialTemplateError, is_social_format
from config import config
from modules.documents.models import UnresolvedDeliverableError
from modules.documents.template_service import UnknownTemplateFormat
from api.document_brand_kit import router as brand_kit_router

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["document-generation"])
# PRD-242 S3: brand kit + logo routes live in their own module (file-size rule);
# included here so they mount under /api/documents without a new main.py mount.
router.include_router(brand_kit_router)

GENERATED_DIR = config.DOCUMENT_STORAGE_DIR

# ------------------------------------------------------------------
# Pydantic request/response models
# ------------------------------------------------------------------


class TemplateCreateRequest(BaseModel):
    name: str
    format: str  # pdf, docx, xlsx, social_image, social_video (PRD-251)
    description: Optional[str] = None
    template_content: Optional[str] = None
    data_schema: dict = Field(default_factory=dict)
    sample_data: dict = Field(default_factory=dict)
    category: str = "general"
    tags: list = Field(default_factory=list)
    blocks: Optional[dict] = None  # PRD-167 S2: canonical block-tree body


class TemplateUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    template_content: Optional[str] = None
    data_schema: Optional[dict] = None
    sample_data: Optional[dict] = None
    category: Optional[str] = None
    tags: Optional[list] = None
    blocks: Optional[dict] = None  # PRD-167 S2


class PreviewBlocksRequest(BaseModel):
    """Live-preview a block tree without persisting (PRD-167 S5)."""
    blocks: dict
    data: dict = Field(default_factory=dict)


class GenerateDocumentRequest(BaseModel):
    title: str
    format: str  # pdf, docx, xlsx, social_image, social_video (PRD-251)
    data: dict
    template_name: Optional[str] = None
    template_id: Optional[str] = None


class GenerateDocumentResponse(BaseModel):
    status: str
    filename: str
    format: str
    download_url: str
    size_kb: int
    # PRD-242 S4: a UI/API generation is a Deliverable too, and callers get the
    # links an agent would (in-app feed + a no-sign-in share link when storage
    # holds a copy).
    deliverable_id: Optional[str] = None
    app_url: Optional[str] = None
    share_url: Optional[str] = None
    template_id: Optional[str] = None
    template_name: Optional[str] = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _validate_blocks_or_422(blocks: Optional[dict]) -> Optional[dict]:
    """Validate a block body, returning the normalized dict or raising 422 with
    field-level errors (PRD-167 S2 — no silent swallow)."""
    if blocks is None:
        return None
    from modules.documents.blocks import BlockValidationError, validate_blocks

    try:
        return validate_blocks(blocks).model_dump()
    except BlockValidationError as e:
        raise HTTPException(status_code=422, detail={"message": "Invalid blocks", "errors": e.errors})


# PRD-251 S1.2: what a social template's save or render can fail with.
SOCIAL_TEMPLATE_ERRORS = (SocialTemplateError, UnknownTemplateFormat)
SOCIAL_RENDER_ERRORS = (SocialTemplateError, RenderQuotaExceeded, MediaRenderError)
RENDER_NOT_CONFIGURED = "Rendering is not configured on this server."
RENDER_UNREACHABLE = "The renderer cannot be reached right now. Try again in a few minutes."


def _template_error_422(e: ValueError) -> HTTPException:
    """A social template that breaks its contract, or an unknown format: 422 naming each problem."""
    errors = e.errors if isinstance(e, SocialTemplateError) else [{"field": "format", "message": str(e)}]
    return HTTPException(status_code=422, detail={"message": "Invalid template", "errors": errors})


def _social_render_error(e: Exception) -> HTTPException:
    """The answer for a social render that could not run (PRD-251 S1.2)."""
    if isinstance(e, InvalidVariableValues):
        return HTTPException(status_code=422, detail={"message": "Invalid template variables", "errors": e.errors})
    if isinstance(e, SocialTemplateError):
        return _template_error_422(e)
    if isinstance(e, RenderQuotaExceeded):
        return HTTPException(status_code=429, detail=str(e))
    logger.warning("Social render failed: %s (%s)", e, getattr(e, "code", None))
    if isinstance(e, MediaRenderUnavailable):
        return HTTPException(
            status_code=503, detail=RENDER_NOT_CONFIGURED if e.code == NOT_CONFIGURED else RENDER_UNREACHABLE
        )
    if isinstance(e, MediaRenderError) and e.code == "check_failed":
        return HTTPException(
            status_code=422,
            detail={"message": "The composition failed its check; nothing was rendered", "findings": list(e.findings)},
        )
    return HTTPException(status_code=502, detail=f"The renderer could not render this template ({getattr(e, 'code', 'error')}).")


# ------------------------------------------------------------------
# Template CRUD
# ------------------------------------------------------------------


@router.post("/templates", status_code=201, dependencies=[Depends(require_workspace_permission("documents:create"))])
async def create_template(
    body: TemplateCreateRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create a new document template.

    PRD-251 S1.2: a social template's composition is checked by the service
    (variables_schema, sizes, the brand rule); it answers 422 naming each problem.
    """
    from modules.documents.template_service import DocumentTemplateService

    # PRD-167 S2: validate the block body up-front; malformed blocks return 422 with
    # field-level errors (no silent swallow). A social composition is no block tree.
    normalized_blocks = body.blocks if is_social_format(body.format) else _validate_blocks_or_422(body.blocks)

    service = DocumentTemplateService(db)
    try:
        template = service.create_template(
            workspace_id=ctx.workspace_id,
            name=body.name,
            format=body.format,
            description=body.description,
            template_content=body.template_content,
            data_schema=body.data_schema,
            sample_data=body.sample_data,
            category=body.category,
            tags=body.tags,
            created_by=str(ctx.user.id) if ctx.user and ctx.user.id else None,
            blocks=normalized_blocks,
        )
    except SOCIAL_TEMPLATE_ERRORS as e:
        raise _template_error_422(e)
    return {
        "id": str(template.id),
        "name": template.name,
        "format": template.format,
        "category": template.category,
        "version": template.version,
        "has_blocks": bool(template.blocks),
    }


@router.get("/templates")
async def list_templates(
    format: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List document templates for the workspace.

    PRD-242 S2: each entry also says whether it is block-editable
    (``has_blocks``), a platform starter (``is_starter``), and which ``data.*``
    fields an agent must supply (``data_fields``) — the gallery renders those
    instead of a bare name.
    """
    from modules.documents.template_service import DocumentTemplateService
    from modules.documents.template_summary import summarize_template

    service = DocumentTemplateService(db)
    templates = service.list_templates(
        workspace_id=ctx.workspace_id, format=format, category=category
    )
    return [summarize_template(t) for t in templates]


@router.get("/templates/presets")
async def list_template_presets(ctx: RequestContext = Depends(get_request_context_hybrid)):
    """The layout each category starts from (PRD-243) — what "New template → Letter"
    loads, what the starters are seeded from. Declared BEFORE ``/templates/{id}`` so
    the literal segment is not swallowed by the UUID path parameter."""
    from modules.documents.presets import PRESETS, preset_payload

    return [preset_payload(p) for p in PRESETS]


@router.get("/templates/{template_id}")
async def get_template(
    template_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get a single template with full details."""
    from modules.documents.template_service import DocumentTemplateService

    from modules.documents.template_summary import summarize_template

    service = DocumentTemplateService(db)
    template = service.get_template(template_id, ctx.workspace_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return {
        **summarize_template(template),
        "template_content": template.template_content,
        "template_file_path": template.template_file_path,
        "blocks": template.blocks,
    }


@router.put("/templates/{template_id}", dependencies=[Depends(require_workspace_permission("documents:update"))])
async def update_template(
    template_id: UUID,
    body: TemplateUpdateRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update a document template (a social template's new composition is checked
    by the service, PRD-251 S1.2)."""
    from modules.documents.template_service import DocumentTemplateService

    service = DocumentTemplateService(db)
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    # PRD-167 S2: validate blocks before persisting (422 with field-level errors).
    # A social template's blocks are a composition, checked by the service instead.
    if "blocks" in updates:
        current = service.get_template(template_id, ctx.workspace_id)
        if not current:
            raise HTTPException(status_code=404, detail="Template not found")
        if not is_social_format(current.format):
            updates["blocks"] = _validate_blocks_or_422(updates["blocks"])
    try:
        template = service.update_template(template_id, ctx.workspace_id, **updates)
    except SOCIAL_TEMPLATE_ERRORS as e:
        raise _template_error_422(e)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"id": str(template.id), "name": template.name, "updated": True}


@router.delete("/templates/{template_id}", dependencies=[Depends(require_workspace_permission("documents:delete"))])
async def delete_template(
    template_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Soft-delete a document template."""
    from modules.documents.template_service import DocumentTemplateService

    service = DocumentTemplateService(db)
    deleted = service.delete_template(template_id, ctx.workspace_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"deleted": True}


@router.post("/templates/{template_id}/preview", dependencies=[Depends(require_workspace_permission("documents:create"))])
async def preview_template(
    template_id: UUID,
    data: Optional[dict] = None,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Generate a preview using sample data or provided data."""
    from modules.documents.generation_service import DocumentGenerationService
    from modules.documents.template_service import DocumentTemplateService

    tmpl_service = DocumentTemplateService(db)
    template = tmpl_service.get_template(template_id, ctx.workspace_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    preview_data = data if data else template.sample_data
    if not preview_data:
        raise HTTPException(status_code=400, detail="No data provided and template has no sample_data")

    gen_service = DocumentGenerationService(db, ctx.workspace_id)
    try:
        result = await gen_service.generate(
            title=f"Preview_{template.name}",
            format=template.format,
            data=preview_data,
            workspace_id=ctx.workspace_id,
            template_id=template.id,
            user_id=resolve_user_pk(db, ctx),
        )
    except UnresolvedDeliverableError as e:
        # P2-09 S3: the finalisation gate — tell the caller WHICH variables
        # blocked the file so they can fill the data / brand kit / template.
        # (The Studio live preview, /templates/preview-blocks, stays the
        # visible-marker surface and is not gated.)
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Document blocked: template variables did not resolve",
                "unresolved": e.unresolved,
                "unknown": e.unknown,
            },
        )
    except SOCIAL_RENDER_ERRORS as e:
        raise _social_render_error(e)
    except Exception as e:
        logger.error(f"Document preview failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Document preview failed")

    return {"preview_url": result.download_url, "filename": result.filename}


@router.post("/templates/upload", dependencies=[Depends(require_workspace_permission("documents:create"))])
async def upload_docx_template(
    name: str = Form(...),
    category: str = Form("general"),
    description: str = Form(None),
    file: UploadFile = File(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Upload a .docx template file."""
    safe_filename = os.path.basename(file.filename or "")
    if not safe_filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")

    from modules.documents.template_service import DocumentTemplateService

    # Save file (use basename to prevent path traversal)
    upload_dir = os.path.join(GENERATED_DIR, str(ctx.workspace_id), "templates")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, safe_filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Extract variables from .docx (basic detection)
    variables = []
    try:
        from docxtpl import DocxTemplate
        doc = DocxTemplate(file_path)
        variables = list(doc.get_undeclared_template_variables())
    except Exception:
        pass

    # Auto-generate schema from variables
    schema = {
        "type": "object",
        "properties": {var: {"type": "string"} for var in variables},
        "required": variables,
    }

    service = DocumentTemplateService(db)
    template = service.create_template(
        workspace_id=ctx.workspace_id,
        name=name,
        format="docx",
        description=description,
        template_file_path=file_path,
        data_schema=schema,
        category=category,
        created_by=str(ctx.user.id) if ctx.user and ctx.user.id else None,
    )

    return {
        "id": str(template.id),
        "name": template.name,
        "format": "docx",
        "variables_detected": variables,
    }


# ------------------------------------------------------------------
# Document Generation
# ------------------------------------------------------------------


@router.post("/generate", response_model=GenerateDocumentResponse, dependencies=[Depends(require_workspace_permission("documents:create"))])
async def generate_document(
    body: GenerateDocumentRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Generate a document from data + template.

    PRD-242 S4: the file is registered as a Deliverable (it used to vanish
    into the download link — only agent generations reached the feed), and the
    response carries the same links an agent gets: the in-app feed and a
    no-sign-in share link when object storage holds a copy.
    """
    from modules.documents.generation_service import DocumentGenerationService, deliverables_app_url

    try:
        template_uuid = UUID(body.template_id) if body.template_id else None
    except ValueError:
        raise HTTPException(status_code=400, detail="template_id must be a UUID")

    service = DocumentGenerationService(db, ctx.workspace_id)
    try:
        result = await service.generate(
            title=body.title,
            format=body.format,
            data=body.data,
            workspace_id=ctx.workspace_id,
            template_name=body.template_name,
            template_id=template_uuid,
            user_id=resolve_user_pk(db, ctx),
        )
    except UnresolvedDeliverableError as e:
        # P2-09 S3: a Deliverable with [[unresolved]]/unknown variables is
        # blocked at finalisation — surface the offending paths, loudly.
        logger.warning(f"Document generation blocked by unresolved variables: {e}")
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Document blocked: template variables did not resolve",
                "unresolved": e.unresolved,
                "unknown": e.unknown,
            },
        )
    except SOCIAL_RENDER_ERRORS as e:
        # PRD-251 S1.2: a social format renders through media-render.
        raise _social_render_error(e)
    except (ValueError, FileNotFoundError) as e:
        logger.warning(f"Document generation validation error: {e}")
        raise HTTPException(status_code=400, detail="Invalid document generation request")
    except ImportError as e:
        logger.error(f"Document generation import error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    except Exception as e:
        logger.exception("Document generation failed")
        raise HTTPException(status_code=500, detail="Internal server error")

    registration = service.register_as_deliverable(
        result,
        title=body.title,
        source_type="document",
        template_id=UUID(result.template_id) if result.template_id else None,
    )
    return GenerateDocumentResponse(
        status="success",
        filename=result.filename,
        format=result.format,
        download_url=result.download_url,
        size_kb=result.size // 1024,
        deliverable_id=(registration or {}).get("deliverable_id"),
        app_url=deliverables_app_url(),
        share_url=service.share_link(result),
        template_id=result.template_id,
        template_name=result.template_name,
    )


# ------------------------------------------------------------------
# Variables + live preview (PRD-167 S3 / S5). Brand kit: api/document_brand_kit.py
# ------------------------------------------------------------------


@router.get("/variables")
async def list_variables(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return the variable catalog with resolved sample values for this workspace/user.

    Drives the editor's variable-chip picker (PRD-167 S3). Each entry carries the
    catalog label/sample plus the *actual* resolved value where available, so authors
    see what a chip will render to.
    """
    from modules.documents.variables import CATALOG
    from modules.documents.variables.resolver import VariableResolver

    resolver = VariableResolver(db)
    paths = [e["path"] for e in CATALOG]
    # PRD-242 S1: ``ctx.user.id`` is the Clerk string / operator email, never the
    # integer PK — handing it to ``User.id`` 500'd this endpoint in both editions.
    resolved = resolver.resolve(ctx.workspace_id, resolve_user_pk(db, ctx), paths)
    entries = [
        {
            "path": e["path"],
            "category": e["category"],
            "label": e["label"],
            "sample": e["sample"],
            "value": resolved.values.get(e["path"]),
            "resolved": e["path"] in resolved.values,
        }
        for e in CATALOG
    ]
    # Group by category for the picker UI.
    grouped: dict = {}
    for entry in entries:
        grouped.setdefault(entry["category"], []).append(entry)
    return {"variables": entries, "by_category": grouped}


@router.post("/preview-blocks", dependencies=[Depends(require_workspace_permission("documents:create"))])
async def preview_blocks(
    body: PreviewBlocksRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Render a block tree to HTML for live preview, without persisting (PRD-167 S5).

    Returns the rendered HTML plus the list of unresolved/unknown variable paths so the
    editor can surface inline errors instead of shipping blanks.
    """
    from modules.documents.blocks import (
        BlockValidationError,
        collect_variable_paths,
        render_document_html,
        validate_blocks,
    )
    from modules.documents.brand_kit import get_brand_kit
    from modules.documents.brand_logo import brand_kit_for_render
    from modules.documents.variables.resolver import VariableResolver
    from core.models.workspaces import Workspace

    try:
        block_doc = validate_blocks(body.blocks)
    except BlockValidationError as e:
        raise HTTPException(status_code=422, detail={"message": "Invalid blocks", "errors": e.errors})

    paths = collect_variable_paths(block_doc)
    resolver = VariableResolver(db)
    resolved = resolver.resolve(
        ctx.workspace_id, resolve_user_pk(db, ctx), paths, extra_data=body.data
    )
    ws = db.query(Workspace).filter(Workspace.id == ctx.workspace_id).first()
    # Render-ready kit: an uploaded logo is inlined so the live preview shows it.
    brand_kit = brand_kit_for_render(get_brand_kit(getattr(ws, "settings", None)))
    rendered = render_document_html(block_doc, resolved.values, brand_kit, title="Preview", data=body.data)
    return {
        "html": rendered.html,
        "unresolved": rendered.unresolved,
        "unknown": resolved.unknown,
    }


# ------------------------------------------------------------------
# File Serving
# ------------------------------------------------------------------

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    # PRD-251 S1.2: a rendered social template.
    ".mp4": "video/mp4",
    ".png": "image/png",
}


@router.get("/generated/{filename}")
async def serve_generated_file(
    filename: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Serve a generated document file for download.

    Tries local filesystem first, then falls back to S3 (containers are ephemeral).
    """
    from pathlib import Path

    # Validate filename to prevent path traversal
    if os.path.basename(filename) != filename or os.path.sep in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    base_dir = Path(GENERATED_DIR) / str(ctx.workspace_id) / "generated"
    file_path = str((base_dir / filename).resolve())
    if not file_path.startswith(str(base_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Try local filesystem first
    if os.path.exists(file_path):
        ext = os.path.splitext(filename)[1].lower()
        media_type = MIME_TYPES.get(ext, "application/octet-stream")
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # Fallback: redirect to S3 presigned URL
    try:
        from config import config as app_config
        from core.storage import get_public_s3_client, get_s3_client, is_storage_configured

        if not is_storage_configured():
            raise HTTPException(status_code=404, detail="File not found")

        bucket = app_config.S3_DOCUMENTS_BUCKET
        s3_key = f"workspaces/{ctx.workspace_id}/generated-documents/{filename}"

        # Check the object exists
        get_s3_client().head_object(Bucket=bucket, Key=s3_key)

        # Presign against the browser-reachable host and redirect (PRD-151 US-006)
        from fastapi.responses import RedirectResponse
        presigned_url = get_public_s3_client().generate_presigned_url(
            "get_object",
            Params={
                "Bucket": bucket,
                "Key": s3_key,
                "ResponseContentDisposition": f'attachment; filename="{filename}"',
            },
            ExpiresIn=3600,
        )
        return RedirectResponse(url=presigned_url, status_code=302)

    except Exception as e:
        logger.warning(f"S3 fallback failed for {filename}: {e}")
        raise HTTPException(status_code=404, detail="File not found")
