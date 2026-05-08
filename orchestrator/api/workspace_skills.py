"""
Workspace Skill Enablement API Endpoints (PRD-71)
==================================================

Provides REST APIs for workspace owners to manage skills:
- List skills available to a workspace (forked + enabled marketplace)
- Enable a marketplace skill for a workspace
- Disable a skill for a workspace (cascades to agent assignments)
- Browse all marketplace skills (for adding)
- Create a workspace-owned skill from pasted/uploaded content
- Edit a workspace skill (forks marketplace skills on first edit)
- Delete a workspace-owned skill

All edits run plugin_security_scanner.quick_scan() — critical findings block,
high findings warn-and-confirm.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workspaces", tags=["Workspace Skills"])

# Severity levels that block save unless explicitly acknowledged.
_BLOCKING_SEVERITIES = {"critical"}
_WARNING_SEVERITIES = {"high"}


def _is_admin(ctx: RequestContext) -> bool:
    """System admins can operate on any workspace (mirrors admin_plugins.py)."""
    return getattr(ctx.user, "system_role", "user") == "admin"


def _assert_workspace_access(ctx: RequestContext, workspace_id: UUID) -> None:
    """Allow own workspace or system admin; otherwise 403."""
    if ctx.workspace_id != workspace_id and not _is_admin(ctx):
        raise HTTPException(status_code=403, detail="Access denied: workspace mismatch")


# ===================================================================
# Pydantic Models
# ===================================================================

class EnableSkillBody(BaseModel):
    skill_id: int = Field(..., description="ID of the marketplace skill to enable")


class WorkspaceSkillOut(BaseModel):
    skill_id: int
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    skill_version: Optional[str] = None
    tags: Optional[List[str]] = None
    estimated_tokens: int = 0
    skill_source: Optional[str] = None
    enabled_at: Optional[str] = None
    # Origin: 'marketplace' = enabled via junction (read-only original),
    # 'workspace' = workspace-owned (forked or user-created, editable).
    origin: str = "marketplace"
    # If origin='workspace' and this skill was forked, the marketplace id it came from.
    forked_from_skill_id: Optional[int] = None
    # Number of agents in this workspace that have this skill assigned.
    assigned_agent_count: int = 0

    class Config:
        from_attributes = True


class ScannerFinding(BaseModel):
    """A single security finding surfaced to the UI."""
    type: str
    severity: str
    line: int
    description: str
    matched_text: str


class CreateSkillBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1, description="SKILL.md content (frontmatter + body)")
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    skill_type: Optional[str] = Field(None, description="cognitive | technical | communication")
    acknowledge_warnings: bool = Field(False, description="Allow save when only 'high'-severity findings exist.")


class UpdateSkillBody(BaseModel):
    content: str = Field(..., min_length=1)
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    acknowledge_warnings: bool = False


class MarketplaceSkillOut(BaseModel):
    """Skill available in the marketplace (workspace_id IS NULL)."""
    id: int
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    skill_version: Optional[str] = None
    tags: Optional[List[str]] = None
    estimated_tokens: int = 0
    skill_source: Optional[str] = None
    is_enabled: bool = False

    class Config:
        from_attributes = True


# ===================================================================
# Endpoints
# ===================================================================

@router.get("/{workspace_id}/skills", response_model=None)
async def list_workspace_skills(
    workspace_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    List all skills available to a workspace:
    - Marketplace skills enabled via junction (origin='marketplace', read-only)
    - Workspace-owned skills (origin='workspace', editable — forked or user-created)
    """
    _assert_workspace_access(ctx, workspace_id)

    try:
        from sqlalchemy import func
        from core.models.core import Agent, Skill, agent_skills as agent_skills_table
        from core.models.marketplace_plugins import WorkspaceEnabledSkill

        items: List[WorkspaceSkillOut] = []
        seen_skill_ids: set[int] = set()

        # Pre-compute assignment counts per skill_id, scoped to agents in this workspace,
        # so the "assigned / unassigned" filter and the remove-when-unused affordance
        # don't need an N+1 round trip from the UI.
        workspace_agent_ids_subq = (
            db.query(Agent.id)
            .filter(Agent.workspace_id == workspace_id)
            .subquery()
        )
        assignment_rows = (
            db.query(agent_skills_table.c.skill_id, func.count().label("agent_count"))
            .filter(agent_skills_table.c.agent_id.in_(workspace_agent_ids_subq))
            .group_by(agent_skills_table.c.skill_id)
            .all()
        )
        assigned_count_by_skill: Dict[int, int] = {row.skill_id: row.agent_count for row in assignment_rows}

        # Workspace-owned skills (forked or user-created)
        owned_skills = (
            db.query(Skill)
            .filter(
                Skill.workspace_id == workspace_id,
                Skill.is_active == True,
            )
            .order_by(Skill.name)
            .all()
        )
        for skill in owned_skills:
            tokens = (len(skill.prompt_template) // 4) if skill.prompt_template else 0
            metadata = skill.skill_metadata if isinstance(skill.skill_metadata, dict) else {}
            items.append(WorkspaceSkillOut(
                skill_id=skill.id,
                name=skill.name,
                description=skill.description,
                category=skill.category,
                skill_version=skill.skill_version,
                tags=skill.tags,
                estimated_tokens=tokens,
                skill_source=skill.skill_source,
                enabled_at=skill.created_at.isoformat() if skill.created_at else None,
                origin="workspace",
                forked_from_skill_id=metadata.get("forked_from_skill_id"),
                assigned_agent_count=assigned_count_by_skill.get(skill.id, 0),
            ))
            seen_skill_ids.add(skill.id)

        # Marketplace skills enabled via junction (skip any already shadowed by a fork
        # that points back at the same marketplace id — see PATCH/fork below)
        forked_origin_ids = {
            i.forked_from_skill_id for i in items if i.forked_from_skill_id is not None
        }
        rows = (
            db.query(WorkspaceEnabledSkill, Skill)
            .join(Skill, Skill.id == WorkspaceEnabledSkill.skill_id)
            .filter(WorkspaceEnabledSkill.workspace_id == workspace_id)
            .all()
        )
        for wes, skill in rows:
            if skill.id in seen_skill_ids or skill.id in forked_origin_ids:
                continue
            tokens = (len(skill.prompt_template) // 4) if skill.prompt_template else 0
            items.append(WorkspaceSkillOut(
                skill_id=skill.id,
                name=skill.name,
                description=skill.description,
                category=skill.category,
                skill_version=skill.skill_version,
                tags=skill.tags,
                estimated_tokens=tokens,
                skill_source=skill.skill_source,
                enabled_at=wes.enabled_at.isoformat() if wes.enabled_at else None,
                origin="marketplace",
                assigned_agent_count=assigned_count_by_skill.get(skill.id, 0),
            ))

        return {"items": [item.model_dump() for item in items]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing workspace skills for %s: %s", workspace_id, e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{workspace_id}/skills/available", response_model=None)
async def list_available_skills(
    workspace_id: UUID,
    q: Optional[str] = Query(None, description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List marketplace skills available to enable (workspace_id IS NULL, is_active=True)."""
    _assert_workspace_access(ctx, workspace_id)

    try:
        from core.models.core import Skill
        from core.models.marketplace_plugins import WorkspaceEnabledSkill

        # Get all marketplace/global skills
        query = db.query(Skill).filter(
            Skill.workspace_id.is_(None),
            Skill.is_active == True,
        )

        if q:
            search = f"%{q}%"
            query = query.filter(
                (Skill.name.ilike(search)) | (Skill.description.ilike(search))
            )

        if category:
            query = query.filter(Skill.category == category)

        all_skills = query.order_by(Skill.name).all()

        # Check which are already enabled for this workspace
        enabled_ids = set(
            row[0] for row in
            db.query(WorkspaceEnabledSkill.skill_id)
            .filter(WorkspaceEnabledSkill.workspace_id == workspace_id)
            .all()
        )

        items = []
        for skill in all_skills:
            tokens = len(skill.prompt_template or "") // 4 if skill.prompt_template else 0
            items.append(MarketplaceSkillOut(
                id=skill.id,
                name=skill.name,
                description=skill.description,
                category=skill.category,
                skill_version=skill.skill_version,
                tags=skill.tags,
                estimated_tokens=tokens,
                skill_source=skill.skill_source,
                is_enabled=skill.id in enabled_ids,
            ))

        return {
            "items": [item.model_dump() for item in items],
            "total": len(items),
            "enabled_count": sum(1 for i in items if i.is_enabled),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing available skills for %s: %s", workspace_id, e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{workspace_id}/skills", status_code=201)
async def enable_skill(
    workspace_id: UUID,
    body: EnableSkillBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Enable a marketplace skill for a workspace."""
    _assert_workspace_access(ctx, workspace_id)

    try:
        from core.models.core import Skill
        from core.models.marketplace_plugins import WorkspaceEnabledSkill

        # Validate skill exists, is marketplace/global, and is active
        skill = db.query(Skill).filter(
            Skill.id == body.skill_id,
            Skill.workspace_id.is_(None),
            Skill.is_active == True,
        ).first()

        if not skill:
            raise HTTPException(
                status_code=404,
                detail="Skill not found or not available in marketplace",
            )

        # Check if already enabled
        existing = db.query(WorkspaceEnabledSkill).filter(
            WorkspaceEnabledSkill.workspace_id == workspace_id,
            WorkspaceEnabledSkill.skill_id == body.skill_id,
        ).first()

        if existing:
            raise HTTPException(status_code=409, detail="Skill is already enabled for this workspace")

        # Resolve user id
        enabled_by = None
        user_id_str = getattr(ctx.user, "id", None)
        if user_id_str:
            try:
                enabled_by = int(user_id_str)
            except (ValueError, TypeError):
                enabled_by = None

        junction = WorkspaceEnabledSkill(
            workspace_id=workspace_id,
            skill_id=body.skill_id,
            enabled_by=enabled_by,
        )
        db.add(junction)
        db.commit()

        return {
            "success": True,
            "message": f"Skill '{skill.name}' enabled for workspace",
            "skill_id": skill.id,
            "workspace_id": str(workspace_id),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error enabling skill for workspace %s: %s", workspace_id, e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{workspace_id}/skills/{skill_id}")
async def disable_skill(
    workspace_id: UUID,
    skill_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Disable a skill for a workspace. Also removes agent assignments."""
    _assert_workspace_access(ctx, workspace_id)

    try:
        from core.models.core import Agent, agent_skills as agent_skills_table
        from core.models.marketplace_plugins import WorkspaceEnabledSkill

        junction = db.query(WorkspaceEnabledSkill).filter(
            WorkspaceEnabledSkill.workspace_id == workspace_id,
            WorkspaceEnabledSkill.skill_id == skill_id,
        ).first()

        if not junction:
            raise HTTPException(status_code=404, detail="Skill is not enabled for this workspace")

        # Remove agent assignments for this skill from agents in this workspace
        workspace_agent_ids = [
            a.id for a in db.query(Agent.id).filter(
                Agent.workspace_id == workspace_id,
            ).all()
        ]

        removed_agent_count = 0
        if workspace_agent_ids:
            from sqlalchemy import and_, delete
            stmt = delete(agent_skills_table).where(
                and_(
                    agent_skills_table.c.agent_id.in_(workspace_agent_ids),
                    agent_skills_table.c.skill_id == skill_id,
                )
            )
            result = db.execute(stmt)
            removed_agent_count = result.rowcount

        db.delete(junction)
        db.commit()

        return {
            "success": True,
            "message": "Skill disabled for workspace",
            "skill_id": skill_id,
            "workspace_id": str(workspace_id),
            "agents_unassigned": removed_agent_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error disabling skill %s for workspace %s: %s", skill_id, workspace_id, e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


# ===================================================================
# Workspace Skill Editor: View / Create / Update (fork-on-edit) / Delete
# ===================================================================

def _classify_findings(findings: list) -> Dict[str, Any]:
    """Bucket scanner findings by severity and decide whether to block."""
    blocking = [f for f in findings if f.severity in _BLOCKING_SEVERITIES]
    warnings = [f for f in findings if f.severity in _WARNING_SEVERITIES]
    return {
        "blocking": blocking,
        "warnings": warnings,
        "all": findings,
    }


def _findings_to_payload(findings: list) -> List[Dict[str, Any]]:
    """Serialize StaticFinding list to the UI-facing shape."""
    return [
        {
            "type": f.type,
            "severity": f.severity,
            "line": f.line,
            "description": f.description,
            "matched_text": f.matched_text,
        }
        for f in findings
    ]


def _resolve_user_id(ctx: RequestContext, db: Session) -> Optional[int]:
    """Look up the integer users.id for the current Clerk user (or None)."""
    from core.models.core import User as UserModel

    clerk_id = getattr(ctx.user, "id", None)
    if not clerk_id:
        return None

    # ctx.user.id may itself already be a numeric DB id depending on auth path
    try:
        return int(clerk_id)
    except (TypeError, ValueError):
        pass

    user = db.query(UserModel).filter(UserModel.clerk_user_id == clerk_id).first()
    if not user and getattr(ctx.user, "email", None):
        user = db.query(UserModel).filter(UserModel.email == ctx.user.email).first()
    return user.id if user else None


def _invalidate_skill_cache(skill_name: str, db: Session) -> None:
    """Clear SkillLoader caches so updated content is served immediately."""
    try:
        from modules.agents.services.skill_loader import get_skill_loader
        loader = get_skill_loader(db)
        loader.metadata_cache.pop(skill_name, None)
        loader.core_content_cache.pop(skill_name, None)
    except Exception:  # SkillLoader not initialized — nothing to clear
        pass


def _scan_or_raise(content: str, filename: str, acknowledge_warnings: bool) -> List[Any]:
    """
    Run the static scanner. Raises 422 with findings on critical issues, or on
    high-severity issues unless the caller has acknowledged them.
    """
    from core.services.plugin_security_scanner import quick_scan

    findings = quick_scan(content, filename=filename)
    bucketed = _classify_findings(findings)

    if bucketed["blocking"]:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "blocked",
                "message": "Skill contains critical security issues. Please remove the flagged patterns and try again.",
                "findings": _findings_to_payload(findings),
            },
        )

    if bucketed["warnings"] and not acknowledge_warnings:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "warnings",
                "message": "Skill contains high-severity findings. Review and resubmit with acknowledge_warnings=true to save anyway.",
                "findings": _findings_to_payload(findings),
            },
        )

    return findings


def _apply_frontmatter(skill, content: str, body: Any) -> str:
    """
    Parse SKILL.md frontmatter and overlay onto the skill record.
    Body fields (description / category / tags) override frontmatter when set.
    Returns the raw markdown body to store in prompt_template.
    """
    from modules.agents.services.skill_loader import parse_yaml_frontmatter

    yaml_data, markdown_body = parse_yaml_frontmatter(content)
    yaml_data = yaml_data or {}

    # Frontmatter sets defaults; explicit body fields win.
    skill.description = body.description or yaml_data.get("description") or skill.description
    skill.category = body.category or yaml_data.get("category") or skill.category
    incoming_tags = body.tags if body.tags is not None else yaml_data.get("tags")
    if incoming_tags is not None:
        skill.tags = incoming_tags

    if yaml_data.get("version"):
        skill.skill_version = str(yaml_data["version"])

    # Persist parsed metadata for downstream consumers (skill_loader.load_skill_metadata)
    existing_metadata = skill.skill_metadata if isinstance(skill.skill_metadata, dict) else {}
    skill.skill_metadata = {**existing_metadata, **{k: v for k, v in yaml_data.items() if k != "forked_from_skill_id"}}
    # Restore lineage if it was already there
    if "forked_from_skill_id" in existing_metadata:
        skill.skill_metadata["forked_from_skill_id"] = existing_metadata["forked_from_skill_id"]

    # Store the full SKILL.md (frontmatter + body) — that's what skill_loader expects
    return content


@router.get("/{workspace_id}/skills/{skill_id}/content")
async def get_workspace_skill_content(
    workspace_id: UUID,
    skill_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Return the raw SKILL.md content for a workspace skill (forked or enabled marketplace).
    Used by the editor modal to populate the textarea.
    """
    _assert_workspace_access(ctx, workspace_id)

    from core.models.core import Skill
    from core.models.marketplace_plugins import WorkspaceEnabledSkill

    skill = db.query(Skill).filter(Skill.id == skill_id, Skill.is_active == True).first()
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")

    # Authorization: workspace can read its own skill, or any marketplace skill it has enabled.
    if skill.workspace_id is not None and skill.workspace_id != workspace_id:
        raise HTTPException(status_code=403, detail="Skill does not belong to this workspace")

    if skill.workspace_id is None:
        enabled = db.query(WorkspaceEnabledSkill).filter(
            WorkspaceEnabledSkill.workspace_id == workspace_id,
            WorkspaceEnabledSkill.skill_id == skill_id,
        ).first()
        if not enabled:
            raise HTTPException(status_code=403, detail="Skill is not enabled for this workspace")

    metadata = skill.skill_metadata if isinstance(skill.skill_metadata, dict) else {}
    return {
        "skill_id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "category": skill.category,
        "tags": skill.tags or [],
        "skill_version": skill.skill_version,
        "skill_source": skill.skill_source,
        "content": skill.prompt_template or "",
        "origin": "workspace" if skill.workspace_id is not None else "marketplace",
        "forked_from_skill_id": metadata.get("forked_from_skill_id"),
        "editable": skill.workspace_id is not None,
    }


@router.post("/{workspace_id}/skills/create", status_code=201)
async def create_workspace_skill(
    workspace_id: UUID,
    body: CreateSkillBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Create a new workspace-owned skill from pasted or uploaded SKILL.md content.
    Runs the security scanner; 422 if blocked.
    """
    _assert_workspace_access(ctx, workspace_id)

    from core.models.core import Skill
    from core.models.marketplace_plugins import WorkspaceEnabledSkill

    findings = _scan_or_raise(body.content, filename=f"{body.name}.md", acknowledge_warnings=body.acknowledge_warnings)

    # Reject duplicate names within the same workspace
    duplicate = db.query(Skill).filter(
        Skill.workspace_id == workspace_id,
        Skill.name == body.name,
        Skill.is_active == True,
    ).first()
    if duplicate:
        raise HTTPException(status_code=409, detail=f"A skill named '{body.name}' already exists in this workspace")

    user_id = _resolve_user_id(ctx, db)

    skill = Skill(
        name=body.name,
        description=body.description or "",
        skill_type=body.skill_type or "cognitive",
        category=body.category,
        tags=body.tags,
        prompt_template=body.content,
        skill_source="workspace-user",
        skill_version="1.0.0",
        workspace_id=workspace_id,
        is_active=True,
        created_by=str(user_id) if user_id else None,
        skill_metadata={},
    )
    _apply_frontmatter(skill, body.content, body)

    db.add(skill)
    db.flush()

    # Auto-enable for the workspace so it shows up in the listing immediately
    db.add(WorkspaceEnabledSkill(
        workspace_id=workspace_id,
        skill_id=skill.id,
        enabled_by=user_id,
    ))
    db.commit()

    logger.info("Created workspace skill '%s' (id=%d) for workspace %s", skill.name, skill.id, workspace_id)

    return {
        "success": True,
        "skill_id": skill.id,
        "name": skill.name,
        "warnings": _findings_to_payload([f for f in findings if f.severity in _WARNING_SEVERITIES]),
    }


@router.patch("/{workspace_id}/skills/{skill_id}")
async def update_workspace_skill(
    workspace_id: UUID,
    skill_id: int,
    body: UpdateSkillBody,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Update a workspace skill's content.

    If the target is a marketplace skill (workspace_id IS NULL), this performs
    fork-on-edit: clones the skill into the workspace, records lineage in
    skill_metadata.forked_from_skill_id, migrates any existing agent_skills
    assignments to the fork, and disables the marketplace junction.

    Returns the (possibly new) skill_id of the workspace-owned record.
    """
    _assert_workspace_access(ctx, workspace_id)

    from core.models.core import Agent, Skill, agent_skills as agent_skills_table
    from core.models.marketplace_plugins import WorkspaceEnabledSkill

    target = db.query(Skill).filter(Skill.id == skill_id, Skill.is_active == True).first()
    if not target:
        raise HTTPException(status_code=404, detail="Skill not found")

    # Authorization
    if target.workspace_id is not None and target.workspace_id != workspace_id:
        raise HTTPException(status_code=403, detail="Skill does not belong to this workspace")

    if target.workspace_id is None:
        # Must be enabled for this workspace to be forkable
        enabled = db.query(WorkspaceEnabledSkill).filter(
            WorkspaceEnabledSkill.workspace_id == workspace_id,
            WorkspaceEnabledSkill.skill_id == skill_id,
        ).first()
        if not enabled:
            raise HTTPException(status_code=403, detail="Skill is not enabled for this workspace")

    findings = _scan_or_raise(body.content, filename=f"{target.name}.md", acknowledge_warnings=body.acknowledge_warnings)

    user_id = _resolve_user_id(ctx, db)

    if target.workspace_id is None:
        # Fork-on-edit
        fork = Skill(
            name=target.name,
            description=target.description,
            skill_type=target.skill_type,
            category=target.category,
            tags=list(target.tags) if target.tags else None,
            prompt_template=body.content,
            skill_source="workspace-fork",
            skill_version=target.skill_version,
            workspace_id=workspace_id,
            is_active=True,
            created_by=str(user_id) if user_id else None,
            skill_metadata={"forked_from_skill_id": target.id},
        )
        _apply_frontmatter(fork, body.content, body)
        db.add(fork)
        db.flush()

        # Migrate existing agent_skills assignments to the fork (within this workspace)
        from sqlalchemy import and_
        workspace_agent_ids = [
            a.id for a in db.query(Agent.id).filter(Agent.workspace_id == workspace_id).all()
        ]
        migrated = 0
        if workspace_agent_ids:
            existing = db.execute(
                agent_skills_table.select().where(
                    and_(
                        agent_skills_table.c.agent_id.in_(workspace_agent_ids),
                        agent_skills_table.c.skill_id == target.id,
                    )
                )
            ).fetchall()
            for row in existing:
                db.execute(agent_skills_table.delete().where(
                    and_(
                        agent_skills_table.c.agent_id == row.agent_id,
                        agent_skills_table.c.skill_id == target.id,
                    )
                ))
                db.execute(agent_skills_table.insert().values(
                    agent_id=row.agent_id, skill_id=fork.id,
                ))
                migrated += 1

        # Replace marketplace junction with fork-aware view: drop the old junction so
        # listing surfaces the fork instead. (Listing already hides marketplace skills
        # whose ids are forked_from_skill_id of a workspace skill — but dropping the
        # junction is cleaner and ensures the user can re-enable the original later.)
        db.query(WorkspaceEnabledSkill).filter(
            WorkspaceEnabledSkill.workspace_id == workspace_id,
            WorkspaceEnabledSkill.skill_id == target.id,
        ).delete()

        db.commit()
        _invalidate_skill_cache(fork.name, db)
        logger.info(
            "Forked marketplace skill %d -> workspace skill %d (name=%s, %d agent assignments migrated)",
            target.id, fork.id, fork.name, migrated,
        )
        return {
            "success": True,
            "skill_id": fork.id,
            "forked": True,
            "forked_from_skill_id": target.id,
            "agents_migrated": migrated,
            "warnings": _findings_to_payload([f for f in findings if f.severity in _WARNING_SEVERITIES]),
        }

    # In-place update (already workspace-owned)
    target.prompt_template = body.content
    _apply_frontmatter(target, body.content, body)
    db.commit()
    _invalidate_skill_cache(target.name, db)
    logger.info("Updated workspace skill %d (name=%s)", target.id, target.name)
    return {
        "success": True,
        "skill_id": target.id,
        "forked": False,
        "warnings": _findings_to_payload([f for f in findings if f.severity in _WARNING_SEVERITIES]),
    }


@router.delete("/{workspace_id}/skills/{skill_id}/owned")
async def delete_workspace_owned_skill(
    workspace_id: UUID,
    skill_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Delete a workspace-owned skill (forked or user-created). Refuses to delete
    marketplace skills — use the existing DELETE /skills/{skill_id} (disable) for those.
    """
    _assert_workspace_access(ctx, workspace_id)

    from core.models.core import Agent, Skill, agent_skills as agent_skills_table
    from core.models.marketplace_plugins import WorkspaceEnabledSkill
    from sqlalchemy import and_, delete

    skill = db.query(Skill).filter(Skill.id == skill_id).first()
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    if skill.workspace_id != workspace_id:
        raise HTTPException(status_code=403, detail="Skill is not workspace-owned")

    # Drop agent assignments first (FK)
    workspace_agent_ids = [
        a.id for a in db.query(Agent.id).filter(Agent.workspace_id == workspace_id).all()
    ]
    if workspace_agent_ids:
        db.execute(delete(agent_skills_table).where(
            and_(
                agent_skills_table.c.agent_id.in_(workspace_agent_ids),
                agent_skills_table.c.skill_id == skill_id,
            )
        ))

    # Drop junction (auto-enabled on create)
    db.query(WorkspaceEnabledSkill).filter(
        WorkspaceEnabledSkill.workspace_id == workspace_id,
        WorkspaceEnabledSkill.skill_id == skill_id,
    ).delete()

    name = skill.name
    db.delete(skill)
    db.commit()
    _invalidate_skill_cache(name, db)

    logger.info("Deleted workspace-owned skill %d (name=%s) for workspace %s", skill_id, name, workspace_id)
    return {"success": True, "skill_id": skill_id}
