"""
Admin Plugin API Endpoints (PRD-42: US-009)
============================================

Provides REST APIs for admin plugin management:
- Upload plugins (multipart zip + security scan)
- Approve / reject plugins
- View security scan results
- Deactivate plugins
- List pending plugins
- Import plugins from GitHub repos
- Hard-delete plugins
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.marketplace_plugins import (
    AgentAssignedPlugin,
    MarketplacePlugin,
    PluginSecurityScan,
    PluginSyncHistory,
    WorkspaceEnabledPlugin,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/plugins", tags=["Admin Plugins"])


# ===================================================================
# Helpers
# ===================================================================

def _is_admin(ctx: RequestContext) -> bool:
    """Check whether the current user has admin privileges."""
    if not ctx.user:
        return False
    return getattr(ctx.user, "system_role", "user") == "admin"


def _assert_admin(ctx: RequestContext) -> None:
    """Raise 403 if the current user is not an admin."""
    if not _is_admin(ctx):
        raise HTTPException(status_code=403, detail="Admin access required")


# ===================================================================
# Pydantic Models
# ===================================================================

class GitHubImportBody(BaseModel):
    github_url: str = Field(..., min_length=1, description="GitHub repo URL to import")


class GitHubImportResult(BaseModel):
    slug: str
    plugin_id: Optional[str] = None
    security_status: Optional[str] = None
    approval_status: Optional[str] = None
    status: str  # "uploaded", "already_exists", "error"
    error: Optional[str] = None


class PluginUploadOut(BaseModel):
    plugin_id: str
    scan_id: Optional[str] = None
    slug: str
    name: str
    version: str
    security_status: str
    approval_status: str
    status: str = "scanning"


class RejectBody(BaseModel):
    reason: str = Field(..., min_length=1, description="Reason for rejection")


class PendingPluginOut(BaseModel):
    id: str
    slug: str
    name: str
    version: str
    description: Optional[str] = None
    source_type: Optional[str] = None
    security_status: str
    created_at: Optional[str] = None
    risk_score: Optional[int] = None
    static_findings_count: int = 0
    llm_summary: Optional[str] = None

    class Config:
        from_attributes = True


class ScanResultOut(BaseModel):
    id: str
    plugin_slug: str
    plugin_version: str
    static_scan_status: Optional[str] = None
    static_findings: Optional[List[Dict[str, Any]]] = None
    blocked_patterns_found: Optional[List[str]] = None
    llm_scan_status: Optional[str] = None
    llm_risk_score: Optional[int] = None
    llm_findings: Optional[List[Dict[str, Any]]] = None
    llm_summary: Optional[str] = None
    llm_model_used: Optional[str] = None
    llm_tokens_used: Optional[int] = None
    overall_verdict: Optional[str] = None
    scanned_at: Optional[str] = None
    scanned_by: Optional[str] = None

    class Config:
        from_attributes = True


# ===================================================================
# Endpoints
# ===================================================================

@router.post("/upload", response_model=PluginUploadOut, status_code=202)
async def upload_plugin(
    file: UploadFile = File(...),
    source_type: str = Form("upload"),
    source_url: Optional[str] = Form(None),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Upload a plugin zip, run security scans, and create a pending plugin record."""
    _assert_admin(ctx)

    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        if not file.filename.endswith(".zip"):
            raise HTTPException(status_code=400, detail="Only .zip files are accepted")

        content = await file.read()

        # 10 MB limit
        max_size = 10 * 1024 * 1024
        if len(content) > max_size:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        # Build services
        from core.services.marketplace_s3 import MarketplaceS3Service
        from core.services.plugin_security_scanner import PluginScanService
        from core.services.plugin_upload_service import PluginUploadService

        s3_service = MarketplaceS3Service()
        scan_service = PluginScanService(db)
        upload_service = PluginUploadService(db, s3_service, scan_service)

        uploaded_by = str(getattr(ctx.user, "id", "admin"))

        plugin = await upload_service.upload_plugin(
            zip_bytes=content,
            source_type=source_type,
            source_url=source_url,
            uploaded_by=uploaded_by,
        )

        return PluginUploadOut(
            plugin_id=str(plugin.id),
            scan_id=str(plugin.security_scan_id) if plugin.security_scan_id else None,
            slug=plugin.slug,
            name=plugin.name,
            version=plugin.version,
            security_status=plugin.security_status or "pending",
            approval_status=plugin.approval_status or "pending",
            status="scanning",
        )

    except ValueError as e:
        logger.warning(f"Plugin upload validation error: {e}")
        raise HTTPException(status_code=400, detail="Invalid plugin upload")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error uploading plugin: %s", e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to upload plugin")


@router.post("/{plugin_id}/approve")
async def approve_plugin(
    plugin_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Approve a pending plugin for the marketplace."""
    _assert_admin(ctx)

    try:
        plugin = db.query(MarketplacePlugin).filter(
            MarketplacePlugin.id == plugin_id,
        ).first()

        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        if plugin.approval_status == "approved":
            raise HTTPException(status_code=400, detail="Plugin is already approved")

        plugin.approval_status = "approved"
        plugin.approved_by = str(getattr(ctx.user, "id", "admin"))
        plugin.approved_at = datetime.utcnow()

        # Create sync history record
        history = PluginSyncHistory(
            action="approve",
            plugin_id=plugin.id,
            plugin_slug=plugin.slug,
            status="completed",
            completed_at=datetime.utcnow(),
            details={
                "previous_status": "pending",
                "approved_by": plugin.approved_by,
            },
            performed_by=plugin.approved_by,
        )
        db.add(history)
        db.commit()

        return {
            "success": True,
            "message": f"Plugin '{plugin.name}' approved",
            "plugin_id": str(plugin.id),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error approving plugin %s: %s", plugin_id, e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to approve plugin")


@router.post("/{plugin_id}/reject")
async def reject_plugin(
    plugin_id: UUID,
    body: RejectBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Reject a pending plugin with a reason."""
    _assert_admin(ctx)

    try:
        plugin = db.query(MarketplacePlugin).filter(
            MarketplacePlugin.id == plugin_id,
        ).first()

        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        plugin.approval_status = "rejected"
        plugin.rejection_reason = body.reason

        performed_by = str(getattr(ctx.user, "id", "admin"))

        # Create sync history record
        history = PluginSyncHistory(
            action="reject",
            plugin_id=plugin.id,
            plugin_slug=plugin.slug,
            status="completed",
            completed_at=datetime.utcnow(),
            details={
                "reason": body.reason,
                "rejected_by": performed_by,
            },
            performed_by=performed_by,
        )
        db.add(history)
        db.commit()

        return {
            "success": True,
            "message": f"Plugin '{plugin.name}' rejected",
            "plugin_id": str(plugin.id),
            "reason": body.reason,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error rejecting plugin %s: %s", plugin_id, e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to reject plugin")


@router.get("/{plugin_id}/scan", response_model=ScanResultOut)
async def get_scan_results(
    plugin_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return security scan results (static + LLM) for a plugin."""
    _assert_admin(ctx)

    try:
        plugin = db.query(MarketplacePlugin).filter(
            MarketplacePlugin.id == plugin_id,
        ).first()

        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        if not plugin.security_scan_id:
            raise HTTPException(status_code=404, detail="No security scan found for this plugin")

        scan = db.query(PluginSecurityScan).filter(
            PluginSecurityScan.id == plugin.security_scan_id,
        ).first()

        if not scan:
            raise HTTPException(status_code=404, detail="Security scan record not found")

        return ScanResultOut(
            id=str(scan.id),
            plugin_slug=scan.plugin_slug,
            plugin_version=scan.plugin_version,
            static_scan_status=scan.static_scan_status,
            static_findings=scan.static_findings,
            blocked_patterns_found=scan.blocked_patterns_found,
            llm_scan_status=scan.llm_scan_status,
            llm_risk_score=scan.llm_risk_score,
            llm_findings=scan.llm_findings,
            llm_summary=scan.llm_summary,
            llm_model_used=scan.llm_model_used,
            llm_tokens_used=scan.llm_tokens_used,
            overall_verdict=scan.overall_verdict,
            scanned_at=scan.scanned_at.isoformat() if scan.scanned_at else None,
            scanned_by=scan.scanned_by,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching scan for plugin %s: %s", plugin_id, e)
        raise HTTPException(status_code=500, detail="Failed to fetch scan results")


@router.post("/{plugin_id}/deactivate")
async def deactivate_plugin(
    plugin_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Deactivate a plugin and cascade-remove all workspace enablements and agent assignments."""
    _assert_admin(ctx)

    try:
        plugin = db.query(MarketplacePlugin).filter(
            MarketplacePlugin.id == plugin_id,
        ).first()

        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        plugin.is_active = False

        performed_by = str(getattr(ctx.user, "id", "admin"))

        # Cascade: remove all WorkspaceEnabledPlugin records for this plugin
        workspace_count = db.query(WorkspaceEnabledPlugin).filter(
            WorkspaceEnabledPlugin.plugin_id == plugin_id,
        ).count()
        db.query(WorkspaceEnabledPlugin).filter(
            WorkspaceEnabledPlugin.plugin_id == plugin_id,
        ).delete(synchronize_session="fetch")

        # Cascade: remove all AgentAssignedPlugin records for this plugin
        agent_count = db.query(AgentAssignedPlugin).filter(
            AgentAssignedPlugin.plugin_id == plugin_id,
        ).count()
        db.query(AgentAssignedPlugin).filter(
            AgentAssignedPlugin.plugin_id == plugin_id,
        ).delete(synchronize_session="fetch")

        # Create sync history record with cascade details
        history = PluginSyncHistory(
            action="deactivate",
            plugin_id=plugin.id,
            plugin_slug=plugin.slug,
            status="completed",
            completed_at=datetime.utcnow(),
            details={
                "deactivated_by": performed_by,
                "workspaces_affected": workspace_count,
                "agents_affected": agent_count,
            },
            performed_by=performed_by,
        )
        db.add(history)
        db.commit()

        return {
            "success": True,
            "message": f"Plugin '{plugin.name}' deactivated",
            "plugin_id": str(plugin.id),
            "plugin_name": plugin.name,
            "workspaces_affected": workspace_count,
            "agents_affected": agent_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deactivating plugin %s: %s", plugin_id, e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to deactivate plugin")


@router.get("/pending")
async def list_pending_plugins(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List plugins with approval_status='pending'. Paginated."""
    _assert_admin(ctx)

    try:
        query = db.query(MarketplacePlugin).filter(
            MarketplacePlugin.approval_status == "pending",
        ).order_by(desc(MarketplacePlugin.created_at))

        total = query.count()
        offset = (page - 1) * limit
        plugins = query.offset(offset).limit(limit).all()

        items = []
        for p in plugins:
            # Load scan if available
            risk_score = None
            static_count = 0
            llm_summary = None
            if p.security_scan_id:
                scan = db.query(PluginSecurityScan).filter(
                    PluginSecurityScan.id == p.security_scan_id,
                ).first()
                if scan:
                    risk_score = scan.llm_risk_score
                    static_count = len(scan.static_findings) if scan.static_findings else 0
                    llm_summary = scan.llm_summary

            items.append(PendingPluginOut(
                id=str(p.id),
                slug=p.slug,
                name=p.name,
                version=p.version,
                description=p.description,
                source_type=p.source_type,
                security_status=p.security_status or "pending",
                created_at=p.created_at.isoformat() if p.created_at else None,
                risk_score=risk_score,
                static_findings_count=static_count,
                llm_summary=llm_summary,
            ))

        return {
            "items": [item.model_dump() for item in items],
            "total": total,
            "page": page,
            "limit": limit,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error listing pending plugins: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list pending plugins")


@router.post("/import-github", status_code=200)
async def import_github(
    body: GitHubImportBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Import plugins from a GitHub repo URL.

    Clones the repo, discovers plugins, bridges manifests, uploads through
    the standard pipeline, and auto-approves safe plugins.
    """
    _assert_admin(ctx)

    # Lazy-import harvest helpers to avoid coupling at module level
    try:
        from scripts.harvest_plugins import (
            clone_repo,
            discover_plugins,
            bridge_to_manifest,
            create_plugin_zip,
        )
    except ImportError as e:
        logger.error("Harvest helpers not available: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )

    from core.services.marketplace_s3 import MarketplaceS3Service
    from core.services.plugin_security_scanner import PluginScanService
    from core.services.plugin_upload_service import PluginUploadService

    url = body.github_url.strip()
    if not url.startswith("https://github.com/"):
        raise HTTPException(status_code=400, detail="URL must be a GitHub repository (https://github.com/...)")

    # Normalise: ensure .git suffix
    repo_url = url if url.endswith(".git") else url + ".git"
    # Derive repo name from URL (e.g. "user/repo")
    repo_name = url.replace("https://github.com/", "").rstrip("/")
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    tmp_dir = Path(tempfile.mkdtemp(prefix="github_import_"))
    results: List[GitHubImportResult] = []

    try:
        repo_dir = tmp_dir / "repo"
        if not clone_repo(repo_url, repo_dir):
            raise HTTPException(status_code=400, detail=f"Failed to clone repository: {url}")

        all_plugins = discover_plugins(repo_dir, repo_name)
        if not all_plugins:
            raise HTTPException(status_code=400, detail="No plugins discovered in this repository")

        s3_service = MarketplaceS3Service()
        scan_service = PluginScanService(db)
        upload_service = PluginUploadService(db, s3_service, scan_service)
        uploaded_by = str(getattr(ctx.user, "id", "admin"))

        for plugin in all_plugins:
            slug = plugin.slug

            # Check existing
            existing = db.query(MarketplacePlugin).filter(
                MarketplacePlugin.slug == slug,
            ).first()
            if existing:
                results.append(GitHubImportResult(
                    slug=slug,
                    plugin_id=str(existing.id),
                    security_status=existing.security_status,
                    approval_status=existing.approval_status,
                    status="already_exists",
                ))
                continue

            try:
                manifest = bridge_to_manifest(
                    plugin.source_manifest, plugin.content_dir, slug
                )
                zip_bytes = create_plugin_zip(
                    plugin.content_dir, manifest, plugin.source_manifest
                )

                base_url = url.rstrip("/").replace(".git", "")
                plugin_record = await upload_service.upload_plugin(
                    zip_bytes=zip_bytes,
                    source_type="github",
                    source_url=base_url,
                    uploaded_by=uploaded_by,
                )

                # Auto-approve safe plugins
                if plugin_record.security_status == "safe":
                    plugin_record.approval_status = "approved"
                    plugin_record.approved_by = uploaded_by
                    plugin_record.approved_at = datetime.utcnow()
                    db.commit()

                results.append(GitHubImportResult(
                    slug=slug,
                    plugin_id=str(plugin_record.id),
                    security_status=plugin_record.security_status,
                    approval_status=plugin_record.approval_status,
                    status="uploaded",
                ))

            except Exception as e:
                logger.error("Failed to import plugin %s: %s", slug, e)
                try:
                    db.rollback()
                except Exception:
                    pass
                results.append(GitHubImportResult(
                    slug=slug,
                    status="error",
                    error="Plugin import failed",
                ))

        return {
            "plugins": [r.model_dump() for r in results],
            "total": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("GitHub import error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.delete("/{plugin_id}")
async def delete_plugin(
    plugin_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Hard-delete a plugin: remove S3 files, all DB records, and junction tables."""
    _assert_admin(ctx)

    try:
        plugin = db.query(MarketplacePlugin).filter(
            MarketplacePlugin.id == plugin_id,
        ).first()

        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        slug = plugin.slug
        version = plugin.version

        # Delete S3 files
        from core.services.marketplace_s3 import MarketplaceS3Service
        s3_service = MarketplaceS3Service()
        try:
            await s3_service.delete_plugin(slug, version)
        except Exception as e:
            logger.warning("S3 delete for %s@%s failed (continuing): %s", slug, version, e)

        # Delete junction records
        db.query(WorkspaceEnabledPlugin).filter(
            WorkspaceEnabledPlugin.plugin_id == plugin_id,
        ).delete(synchronize_session="fetch")

        db.query(AgentAssignedPlugin).filter(
            AgentAssignedPlugin.plugin_id == plugin_id,
        ).delete(synchronize_session="fetch")

        # Delete sync history
        db.query(PluginSyncHistory).filter(
            PluginSyncHistory.plugin_id == plugin_id,
        ).delete(synchronize_session="fetch")

        # Delete security scan
        if plugin.security_scan_id:
            db.query(PluginSecurityScan).filter(
                PluginSecurityScan.id == plugin.security_scan_id,
            ).delete(synchronize_session="fetch")

        # Delete the plugin itself
        db.delete(plugin)
        db.commit()

        return {
            "deleted": True,
            "slug": slug,
            "plugin_id": str(plugin_id),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting plugin %s: %s", plugin_id, e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete plugin")


@router.post("/backfill-categories")
async def backfill_categories(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Backfill category_id on all plugins that don't have one set.

    Uses the same keyword-matching logic as the upload pipeline.
    """
    _assert_admin(ctx)

    try:
        from core.services.plugin_upload_service import _auto_categorise

        plugins = db.query(MarketplacePlugin).filter(
            MarketplacePlugin.category_id.is_(None),
        ).all()

        updated = 0
        results = []
        for p in plugins:
            cat_id = _auto_categorise(
                p.slug, p.name, p.tags or [], p.description or "", db,
            )
            if cat_id:
                p.category_id = cat_id
                updated += 1
                results.append({"slug": p.slug, "category_id": str(cat_id)})

        db.commit()

        return {
            "total_checked": len(plugins),
            "updated": updated,
            "results": results,
        }

    except Exception as e:
        logger.exception("Error backfilling categories: %s", e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to backfill categories")
