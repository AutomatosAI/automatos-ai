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
from core.auth.workspace_permission import require_workspace_permission
from core.auth.dependencies import RequestContext
from core.database.database import get_db
from config import config
from modules.documents.models import UnresolvedDeliverableError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["document-generation"])

GENERATED_DIR = config.DOCUMENT_STORAGE_DIR

# ------------------------------------------------------------------
# Pydantic request/response models
# ------------------------------------------------------------------


class TemplateCreateRequest(BaseModel):
    name: str
    format: str  # pdf, docx, xlsx
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


class BrandKitUpdateRequest(BaseModel):
    name: Optional[str] = None
    tagline: Optional[str] = None
    logo_url: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    accent_color: Optional[str] = None
    text_color: Optional[str] = None
    font_family: Optional[str] = None
    company: Optional[dict] = None


class PreviewBlocksRequest(BaseModel):
    """Live-preview a block tree without persisting (PRD-167 S5)."""
    blocks: dict
    data: dict = Field(default_factory=dict)


class GenerateDocumentRequest(BaseModel):
    title: str
    format: str  # pdf, docx, xlsx
    data: dict
    template_name: Optional[str] = None
    template_id: Optional[str] = None


class GenerateDocumentResponse(BaseModel):
    status: str
    filename: str
    format: str
    download_url: str
    size_kb: int


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


# ------------------------------------------------------------------
# Template CRUD
# ------------------------------------------------------------------


@router.post("/templates", status_code=201, dependencies=[Depends(require_workspace_permission("documents:create"))])
async def create_template(
    body: TemplateCreateRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create a new document template."""
    from modules.documents.template_service import DocumentTemplateService

    # PRD-167 S2: validate the block body up-front; malformed blocks return 422 with
    # field-level errors (no silent swallow).
    normalized_blocks = _validate_blocks_or_422(body.blocks)

    service = DocumentTemplateService(db)
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
    return {
        "id": str(template.id),
        "name": template.name,
        "format": template.format,
        "category": template.category,
        "version": template.version,
    }


@router.get("/templates")
async def list_templates(
    format: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List document templates for the workspace."""
    from modules.documents.template_service import DocumentTemplateService

    service = DocumentTemplateService(db)
    templates = service.list_templates(
        workspace_id=ctx.workspace_id, format=format, category=category
    )
    return [
        {
            "id": str(t.id),
            "name": t.name,
            "description": t.description,
            "format": t.format,
            "category": t.category,
            "tags": t.tags or [],
            "version": t.version,
            "data_schema": t.data_schema,
            "sample_data": t.sample_data,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in templates
    ]


@router.get("/templates/{template_id}")
async def get_template(
    template_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get a single template with full details."""
    from modules.documents.template_service import DocumentTemplateService

    service = DocumentTemplateService(db)
    template = service.get_template(template_id, ctx.workspace_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return {
        "id": str(template.id),
        "name": template.name,
        "description": template.description,
        "format": template.format,
        "category": template.category,
        "tags": template.tags or [],
        "version": template.version,
        "template_content": template.template_content,
        "template_file_path": template.template_file_path,
        "data_schema": template.data_schema,
        "sample_data": template.sample_data,
        "blocks": template.blocks,
        "created_at": template.created_at.isoformat() if template.created_at else None,
        "updated_at": template.updated_at.isoformat() if template.updated_at else None,
    }


@router.put("/templates/{template_id}", dependencies=[Depends(require_workspace_permission("documents:update"))])
async def update_template(
    template_id: UUID,
    body: TemplateUpdateRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update a document template."""
    from modules.documents.template_service import DocumentTemplateService

    service = DocumentTemplateService(db)
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    # PRD-167 S2: validate blocks before persisting (422 with field-level errors).
    if "blocks" in updates:
        updates["blocks"] = _validate_blocks_or_422(updates["blocks"])
    template = service.update_template(template_id, ctx.workspace_id, **updates)
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
            user_id=ctx.user.id if ctx.user else None,
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
    """Generate a document from data + template."""
    from modules.documents.generation_service import DocumentGenerationService

    service = DocumentGenerationService(db, ctx.workspace_id)
    try:
        result = await service.generate(
            title=body.title,
            format=body.format,
            data=body.data,
            workspace_id=ctx.workspace_id,
            template_name=body.template_name,
            template_id=UUID(body.template_id) if body.template_id else None,
            user_id=ctx.user.id if ctx.user else None,
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
    except (ValueError, FileNotFoundError) as e:
        logger.warning(f"Document generation validation error: {e}")
        raise HTTPException(status_code=400, detail="Invalid document generation request")
    except ImportError as e:
        logger.error(f"Document generation import error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    except Exception as e:
        logger.exception("Document generation failed")
        raise HTTPException(status_code=500, detail="Internal server error")

    return GenerateDocumentResponse(
        status="success",
        filename=result.filename,
        format=result.format,
        download_url=result.download_url,
        size_kb=result.size // 1024,
    )


# ------------------------------------------------------------------
# Variables + Brand Kit (PRD-167 S3 / S4)
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
    resolved = resolver.resolve(
        ctx.workspace_id, ctx.user.id if ctx.user else None, paths
    )
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


@router.get("/brand-kit")
async def get_brand_kit_endpoint(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return the workspace brand kit (defaults merged in)."""
    from core.models.workspaces import Workspace
    from modules.documents.brand_kit import get_brand_kit

    ws = db.query(Workspace).filter(Workspace.id == ctx.workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return get_brand_kit(ws.settings)


@router.put("/brand-kit", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def update_brand_kit_endpoint(
    body: BrandKitUpdateRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update the workspace brand kit (validated, persisted on workspace.settings)."""
    from pydantic import ValidationError

    from core.models.workspaces import Workspace
    from modules.documents.brand_kit import BRAND_KIT_SETTINGS_KEY, get_brand_kit, validate_brand_kit

    ws = db.query(Workspace).filter(Workspace.id == ctx.workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")

    existing = (ws.settings or {}).get(BRAND_KIT_SETTINGS_KEY)
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        new_kit = validate_brand_kit(patch, existing)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={"message": "Invalid brand kit", "errors": e.errors()})

    # Reassign settings (not in-place mutate) so SQLAlchemy tracks the JSONB change.
    ws.settings = {**(ws.settings or {}), BRAND_KIT_SETTINGS_KEY: new_kit}
    db.commit()
    return new_kit


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
    from modules.documents.variables.resolver import VariableResolver
    from core.models.workspaces import Workspace

    try:
        block_doc = validate_blocks(body.blocks)
    except BlockValidationError as e:
        raise HTTPException(status_code=422, detail={"message": "Invalid blocks", "errors": e.errors})

    paths = collect_variable_paths(block_doc)
    resolver = VariableResolver(db)
    resolved = resolver.resolve(
        ctx.workspace_id, ctx.user.id if ctx.user else None, paths, extra_data=body.data
    )
    ws = db.query(Workspace).filter(Workspace.id == ctx.workspace_id).first()
    brand_kit = get_brand_kit(getattr(ws, "settings", None))
    rendered = render_document_html(block_doc, resolved.values, brand_kit, title="Preview")
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
        import boto3
        from botocore.config import Config as BotoConfig

        if not app_config.AWS_ACCESS_KEY_ID or not app_config.AWS_SECRET_ACCESS_KEY:
            raise HTTPException(status_code=404, detail="File not found")

        bucket = app_config.S3_DOCUMENTS_BUCKET
        s3_key = f"workspaces/{ctx.workspace_id}/generated-documents/{filename}"

        boto_cfg = BotoConfig(
            region_name=app_config.AWS_REGION or "us-east-1",
            signature_version="v4",
        )
        client = boto3.client(
            "s3",
            aws_access_key_id=app_config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=app_config.AWS_SECRET_ACCESS_KEY,
            config=boto_cfg,
        )

        # Check the object exists
        client.head_object(Bucket=bucket, Key=s3_key)

        # Generate presigned URL and redirect
        from fastapi.responses import RedirectResponse
        presigned_url = client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": bucket,
                "Key": s3_key,
                "ResponseContentDisposition": f'attachment; filename="{filename}"',
            },
            ExpiresIn=3600,
        )
        return RedirectResponse(url=presigned_url, status_code=302)

    except ImportError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.warning(f"S3 fallback failed for {filename}: {e}")
        raise HTTPException(status_code=404, detail="File not found")
