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
    # PRD-174 F043: shared admin check — super_admin ⊇ admin when the plane is on.
    from core.auth.roles import caller_is_admin
    return caller_is_admin(ctx.user)


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

        # PRD-71: Materialize SKILL.md files from the plugin into Skill records
        materialized_ids = []
        try:
            from core.services.skill_materializer import SkillMaterializer
            materializer = SkillMaterializer(db)
            materialized_ids = await materializer.materialize_plugin(plugin)
        except Exception as mat_err:
            logger.warning(
                "Skill materialization failed for plugin %s (non-blocking): %s",
                plugin.slug, mat_err,
            )

        return {
            "success": True,
            "message": f"Plugin '{plugin.name}' approved",
            "plugin_id": str(plugin.id),
            "materialized_skill_ids": materialized_ids,
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
    """Import plugins AND/OR standalone skills from a GitHub repo URL.

    Clones the repo, discovers plugins (manifest.json) and standalone
    SKILL.md files. Plugins go through the standard pipeline; standalone
    skills are imported directly as Skill DB records.
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

        # --- Phase 1: Discover plugins (manifest-based packages) ---
        all_plugins = discover_plugins(repo_dir, repo_name)

        if all_plugins:
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

        # --- Phase 2: Discover standalone SKILL.md files ---
        skill_results = _import_standalone_skills(repo_dir, repo_name, url, db)
        results.extend(skill_results)

        if not results:
            raise HTTPException(
                status_code=400,
                detail="No plugins or skills discovered in this repository",
            )

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


def _import_standalone_skills(
    repo_dir: Path,
    repo_name: str,
    source_url: str,
    db: Session,
) -> List[GitHubImportResult]:
    """Discover and import standalone SKILL.md files from a cloned repo.

    Creates Skill records directly (workspace_id=NULL for marketplace).
    Uses the skill_loader's parse + security scan logic.
    """
    from modules.agents.services.skill_loader import (
        parse_yaml_frontmatter,
        scan_for_dangerous_patterns,
    )
    from core.models.core import Skill

    skill_files = list(repo_dir.rglob("SKILL.md"))
    if not skill_files:
        return []

    results: List[GitHubImportResult] = []
    source_tag = f"github:{repo_name}"

    for skill_file in skill_files:
        try:
            content = skill_file.read_text(encoding="utf-8")
            yaml_data, markdown_body = parse_yaml_frontmatter(content)

            if not yaml_data:
                logger.warning("Skipping %s — no YAML frontmatter", skill_file)
                continue

            skill_name = yaml_data.get("name")
            description = yaml_data.get("description", "")
            if not skill_name:
                logger.warning("Skipping %s — missing 'name' in frontmatter", skill_file)
                continue

            # Security scan
            is_safe, dangerous_matches = scan_for_dangerous_patterns(content)
            if not is_safe:
                logger.warning(
                    "Skill %s blocked — dangerous patterns: %s",
                    skill_name, dangerous_matches,
                )
                results.append(GitHubImportResult(
                    slug=skill_name,
                    status="error",
                    error=f"Security scan failed: {dangerous_matches}",
                ))
                continue

            # Upsert: match on name + source
            existing = db.query(Skill).filter(
                Skill.name == skill_name,
                Skill.skill_source == source_tag,
            ).first()

            if existing:
                existing.description = description
                existing.skill_version = yaml_data.get("version", "1.0.0")
                existing.prompt_template = markdown_body
                existing.tags = yaml_data.get("tags", [])
                existing.skill_metadata = yaml_data
                existing.workspace_id = None  # marketplace/global
                skill = existing
                status = "updated"
            else:
                skill = Skill(
                    name=skill_name,
                    description=description,
                    skill_type=yaml_data.get("skill_type", "custom"),
                    category=yaml_data.get("category", "general"),
                    skill_version=yaml_data.get("version", "1.0.0"),
                    skill_source=source_tag,
                    prompt_template=markdown_body,
                    tags=yaml_data.get("tags", []),
                    skill_metadata=yaml_data,
                    is_active=True,
                    workspace_id=None,  # marketplace/global
                )
                db.add(skill)
                status = "uploaded"

            db.flush()

            results.append(GitHubImportResult(
                slug=skill_name,
                plugin_id=str(skill.id),
                security_status="safe",
                approval_status="approved",
                status=status,
            ))

        except Exception as e:
            logger.error("Failed to import skill from %s: %s", skill_file, e, exc_info=True)
            results.append(GitHubImportResult(
                slug=str(skill_file.stem),
                status="error",
                error=str(e),
            ))

    if results:
        try:
            db.commit()
        except Exception as e:
            logger.error("Failed to commit skill imports: %s", e)
            db.rollback()

    logger.info(
        "Imported %d standalone skills from %s",
        sum(1 for r in results if r.status in ("uploaded", "updated")),
        repo_name,
    )
    return results


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
