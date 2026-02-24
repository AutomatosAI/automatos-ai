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
from core.auth.dependencies import RequestContext
from core.database.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["document-generation"])

GENERATED_DIR = os.environ.get("DOCUMENT_STORAGE_DIR", "documents")

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


class TemplateUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    template_content: Optional[str] = None
    data_schema: Optional[dict] = None
    sample_data: Optional[dict] = None
    category: Optional[str] = None
    tags: Optional[list] = None


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
# Template CRUD
# ------------------------------------------------------------------


@router.post("/templates", status_code=201)
async def create_template(
    body: TemplateCreateRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create a new document template."""
    from modules.documents.template_service import DocumentTemplateService

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
        created_by=str(ctx.user_id) if ctx.user_id else None,
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
    template = service.get_template(template_id)
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
        "created_at": template.created_at.isoformat() if template.created_at else None,
        "updated_at": template.updated_at.isoformat() if template.updated_at else None,
    }


@router.put("/templates/{template_id}")
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
    template = service.update_template(template_id, **updates)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"id": str(template.id), "name": template.name, "updated": True}


@router.delete("/templates/{template_id}")
async def delete_template(
    template_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Soft-delete a document template."""
    from modules.documents.template_service import DocumentTemplateService

    service = DocumentTemplateService(db)
    deleted = service.delete_template(template_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"deleted": True}


@router.post("/templates/{template_id}/preview")
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
    template = tmpl_service.get_template(template_id)
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
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"preview_url": result.download_url, "filename": result.filename}


@router.post("/templates/upload")
async def upload_docx_template(
    name: str = Form(...),
    category: str = Form("general"),
    description: str = Form(None),
    file: UploadFile = File(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Upload a .docx template file."""
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")

    from modules.documents.template_service import DocumentTemplateService

    # Save file
    upload_dir = os.path.join(GENERATED_DIR, str(ctx.workspace_id), "templates")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
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
        created_by=str(ctx.user_id) if ctx.user_id else None,
    )

    return {
        "id": str(template.id),
        "name": template.name,
        "format": "docx",
        "variables_detected": variables,
        "file_path": file_path,
    }


# ------------------------------------------------------------------
# Document Generation
# ------------------------------------------------------------------


@router.post("/generate", response_model=GenerateDocumentResponse)
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
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
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
    file_path = os.path.join(GENERATED_DIR, str(ctx.workspace_id), "generated", filename)

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

        bucket = os.getenv("S3_DOCUMENTS_BUCKET", "automatos-ai")
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
