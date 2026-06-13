"""
Skill Editing Handlers
=======================

Implements platform_get_skill_content, platform_create_workspace_skill,
platform_update_skill, and platform_delete_workspace_skill.

Mirrors the auth, scanner, and fork-on-edit semantics of
api/workspace_skills.py so behaviour is identical whether a user clicks
the UI button or an agent calls the tool. Workspace authorisation is
implicit: the executor passes the calling agent's workspace_id, and these
handlers refuse to read or mutate skills outside that boundary.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


_BLOCKING_SEVERITIES = {"critical"}
_WARNING_SEVERITIES = {"high"}


# ---------------------------------------------------------------------------
# Local helpers (mirrors workspace_skills.py — kept here to avoid pulling in
# FastAPI HTTPException semantics into a tool handler)
# ---------------------------------------------------------------------------

def _findings_to_payload(findings: list) -> List[Dict[str, Any]]:
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


def _scan(content: str, filename: str, acknowledge_warnings: bool) -> Dict[str, Any]:
    """Run the security scanner. Returns a dict with status: ok | blocked | warnings."""
    try:
        from core.services.plugin_security_scanner import quick_scan
    except Exception as e:
        logger.warning("[skills] Scanner unavailable, allowing save: %s", e)
        return {"status": "ok", "findings": []}

    findings = quick_scan(content, filename=filename)
    blocking = [f for f in findings if f.severity in _BLOCKING_SEVERITIES]
    warnings = [f for f in findings if f.severity in _WARNING_SEVERITIES]

    if blocking:
        return {
            "status": "blocked",
            "findings": _findings_to_payload(findings),
        }
    if warnings and not acknowledge_warnings:
        return {
            "status": "warnings",
            "findings": _findings_to_payload(warnings),
        }
    return {"status": "ok", "findings": _findings_to_payload(warnings)}


def _apply_frontmatter(skill, content: str, params: Dict[str, Any]) -> None:
    """Parse SKILL.md frontmatter and overlay onto the skill record."""
    from modules.agents.services.skill_loader import parse_yaml_frontmatter

    yaml_data, _ = parse_yaml_frontmatter(content)
    yaml_data = yaml_data or {}

    skill.description = params.get("description") or yaml_data.get("description") or skill.description
    # category must never be NULL: the prod `skills.category` column is NOT NULL
    # (the model says nullable=True, but prod schema drifted), and an agent that
    # omits category with no frontmatter would otherwise dead-end the INSERT.
    # A skill should be categorised anyway — fall back to 'general'.
    skill.category = (
        params.get("category") or yaml_data.get("category") or skill.category or "general"
    )

    incoming_tags = params.get("tags")
    if incoming_tags is None:
        incoming_tags = yaml_data.get("tags")
    if incoming_tags is not None:
        skill.tags = incoming_tags

    if yaml_data.get("version"):
        skill.skill_version = str(yaml_data["version"])

    existing_metadata = skill.skill_metadata if isinstance(skill.skill_metadata, dict) else {}
    skill.skill_metadata = {
        **existing_metadata,
        **{k: v for k, v in yaml_data.items() if k != "forked_from_skill_id"},
    }
    if "forked_from_skill_id" in existing_metadata:
        skill.skill_metadata["forked_from_skill_id"] = existing_metadata["forked_from_skill_id"]


def _invalidate_skill_cache(skill_name: str, db: Session) -> None:
    """Clear SkillLoader caches so updated content is served immediately."""
    try:
        from modules.agents.services.skill_loader import get_skill_loader
        loader = get_skill_loader(db)
        loader.metadata_cache.pop(skill_name, None)
        loader.core_content_cache.pop(skill_name, None)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def get_skill_content(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Read full SKILL.md content. Workspace can read its own skills + any marketplace skill enabled for it."""
    from core.models.core import Skill
    from core.models.marketplace_plugins import WorkspaceEnabledSkill

    skill_id = params.get("skill_id")
    skill_name = params.get("skill_name")

    if not skill_id and not skill_name:
        return {"success": False, "error": "Provide skill_id or skill_name"}

    query = db.query(Skill).filter(Skill.is_active == True)
    if skill_id:
        query = query.filter(Skill.id == skill_id)
    else:
        query = query.filter(Skill.name.ilike(f"%{skill_name}%"))

    # Restrict to skills this workspace can see: own forks + marketplace
    query = query.filter(
        (Skill.workspace_id.is_(None)) | (Skill.workspace_id == workspace_id)
    )

    skill = query.first()
    if not skill:
        return {"success": False, "error": "Skill not found or not accessible to this workspace"}

    # If marketplace, must be enabled for this workspace
    if skill.workspace_id is None:
        enabled = (
            db.query(WorkspaceEnabledSkill)
            .filter(
                WorkspaceEnabledSkill.workspace_id == workspace_id,
                WorkspaceEnabledSkill.skill_id == skill.id,
            )
            .first()
        )
        if not enabled:
            return {
                "success": False,
                "error": (
                    f"Skill '{skill.name}' is a marketplace skill but is not enabled for this workspace. "
                    "Use platform_install_skill first."
                ),
            }

    metadata = skill.skill_metadata if isinstance(skill.skill_metadata, dict) else {}
    return {
        "success": True,
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


async def create_workspace_skill(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new workspace-owned skill from SKILL.md content."""
    from core.models.core import Skill
    from core.models.marketplace_plugins import WorkspaceEnabledSkill

    name = (params.get("name") or "").strip()
    content = params.get("content") or ""
    if not name:
        return {"success": False, "error": "Missing required parameter: name"}
    if not content:
        return {"success": False, "error": "Missing required parameter: content"}
    if len(name) > 255:
        return {"success": False, "error": "Skill name exceeds 255 characters"}

    acknowledge_warnings = bool(params.get("acknowledge_warnings", False))
    scan_result = _scan(content, filename=f"{name}.md", acknowledge_warnings=acknowledge_warnings)
    if scan_result["status"] != "ok":
        return {
            "success": False,
            "status": scan_result["status"],
            "findings": scan_result["findings"],
            "error": (
                "Skill contains critical security issues — fix the flagged patterns and try again."
                if scan_result["status"] == "blocked" else
                "Skill contains high-severity findings — review and resubmit with acknowledge_warnings=true."
            ),
        }

    duplicate = (
        db.query(Skill)
        .filter(
            Skill.workspace_id == workspace_id,
            Skill.name == name,
            Skill.is_active == True,
        )
        .first()
    )
    if duplicate:
        return {
            "success": False,
            "error": f"A skill named '{name}' already exists in this workspace (id={duplicate.id})",
        }

    skill = Skill(
        name=name,
        description=params.get("description") or "",
        skill_type=params.get("skill_type") or "cognitive",
        category=params.get("category"),
        tags=params.get("tags"),
        prompt_template=content,
        skill_source="workspace-user",
        skill_version="1.0.0",
        workspace_id=workspace_id,
        is_active=True,
        skill_metadata={},
    )
    _apply_frontmatter(skill, content, params)

    db.add(skill)
    db.flush()

    db.add(WorkspaceEnabledSkill(workspace_id=workspace_id, skill_id=skill.id))
    db.commit()

    logger.info(
        "[skills] Created workspace skill '%s' (id=%d) for workspace %s",
        skill.name, skill.id, workspace_id,
    )

    return {
        "success": True,
        "skill_id": skill.id,
        "name": skill.name,
        "warnings": scan_result["findings"],
        "message": f"Skill '{skill.name}' created and enabled for this workspace.",
    }


async def update_skill(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Update a skill's content. Forks-on-edit if the target is a marketplace skill."""
    from sqlalchemy import and_
    from core.models.core import Agent, Skill, agent_skills as agent_skills_table
    from core.models.marketplace_plugins import WorkspaceEnabledSkill

    skill_id = params.get("skill_id")
    content = params.get("content") or ""
    if not skill_id:
        return {"success": False, "error": "Missing required parameter: skill_id"}
    if not content:
        return {"success": False, "error": "Missing required parameter: content"}

    target = db.query(Skill).filter(Skill.id == skill_id, Skill.is_active == True).first()
    if not target:
        return {"success": False, "error": f"Skill {skill_id} not found"}

    # Authorization: must be marketplace (and enabled) or workspace-owned by this workspace
    if target.workspace_id is not None and target.workspace_id != workspace_id:
        return {"success": False, "error": "Skill does not belong to this workspace"}

    if target.workspace_id is None:
        enabled = (
            db.query(WorkspaceEnabledSkill)
            .filter(
                WorkspaceEnabledSkill.workspace_id == workspace_id,
                WorkspaceEnabledSkill.skill_id == skill_id,
            )
            .first()
        )
        if not enabled:
            return {
                "success": False,
                "error": (
                    f"Skill '{target.name}' is a marketplace skill but is not enabled for this workspace. "
                    "Use platform_install_skill first."
                ),
            }

    acknowledge_warnings = bool(params.get("acknowledge_warnings", False))
    scan_result = _scan(content, filename=f"{target.name}.md", acknowledge_warnings=acknowledge_warnings)
    if scan_result["status"] != "ok":
        return {
            "success": False,
            "status": scan_result["status"],
            "findings": scan_result["findings"],
            "error": (
                "Skill contains critical security issues — fix the flagged patterns and try again."
                if scan_result["status"] == "blocked" else
                "Skill contains high-severity findings — review and resubmit with acknowledge_warnings=true."
            ),
        }

    if target.workspace_id is None:
        # Fork-on-edit
        fork = Skill(
            name=target.name,
            description=target.description,
            skill_type=target.skill_type,
            category=target.category,
            tags=list(target.tags) if target.tags else None,
            prompt_template=content,
            skill_source="workspace-fork",
            skill_version=target.skill_version,
            workspace_id=workspace_id,
            is_active=True,
            skill_metadata={"forked_from_skill_id": target.id},
        )
        _apply_frontmatter(fork, content, params)
        db.add(fork)
        db.flush()

        # Migrate agent_skills assignments within this workspace
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

        # Drop the marketplace junction so listing surfaces the fork
        db.query(WorkspaceEnabledSkill).filter(
            WorkspaceEnabledSkill.workspace_id == workspace_id,
            WorkspaceEnabledSkill.skill_id == target.id,
        ).delete()

        db.commit()
        _invalidate_skill_cache(fork.name, db)
        logger.info(
            "[skills] Forked marketplace skill %d -> workspace skill %d (name=%s, %d agent assignments migrated)",
            target.id, fork.id, fork.name, migrated,
        )
        return {
            "success": True,
            "skill_id": fork.id,
            "name": fork.name,
            "forked": True,
            "forked_from_skill_id": target.id,
            "agents_migrated": migrated,
            "warnings": scan_result["findings"],
            "message": f"Forked marketplace skill '{target.name}' into your workspace and applied edits.",
        }

    # In-place update (already workspace-owned)
    target.prompt_template = content
    _apply_frontmatter(target, content, params)
    db.commit()
    _invalidate_skill_cache(target.name, db)
    logger.info("[skills] Updated workspace skill %d (name=%s)", target.id, target.name)
    return {
        "success": True,
        "skill_id": target.id,
        "name": target.name,
        "forked": False,
        "warnings": scan_result["findings"],
        "message": f"Skill '{target.name}' updated.",
    }


async def delete_workspace_skill(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a workspace-owned skill. Refuses for marketplace skills."""
    from sqlalchemy import and_, delete
    from core.models.core import Agent, Skill, agent_skills as agent_skills_table
    from core.models.marketplace_plugins import WorkspaceEnabledSkill

    skill_id = params.get("skill_id")
    if not skill_id:
        return {"success": False, "error": "Missing required parameter: skill_id"}

    skill = db.query(Skill).filter(Skill.id == skill_id).first()
    if not skill:
        return {"success": False, "error": f"Skill {skill_id} not found"}
    if skill.workspace_id is None:
        return {
            "success": False,
            "error": (
                f"Skill '{skill.name}' is a marketplace skill and cannot be deleted by a workspace. "
                "Use platform_install_skill (disable) to remove it from this workspace."
            ),
        }
    if skill.workspace_id != workspace_id:
        return {"success": False, "error": "Skill does not belong to this workspace"}

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

    db.query(WorkspaceEnabledSkill).filter(
        WorkspaceEnabledSkill.workspace_id == workspace_id,
        WorkspaceEnabledSkill.skill_id == skill_id,
    ).delete()

    name = skill.name
    db.delete(skill)
    db.commit()
    _invalidate_skill_cache(name, db)

    logger.info("[skills] Deleted workspace skill %d (name=%s) for workspace %s", skill_id, name, workspace_id)
    return {"success": True, "skill_id": skill_id, "name": name, "message": f"Skill '{name}' deleted."}
