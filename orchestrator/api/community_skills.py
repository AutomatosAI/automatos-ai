"""
Community Skills API (PRD-55 US-016)
=====================================

Search, install, and manage community skills from skills.sh ecosystem.
Admin-only endpoints for browsing the skills.sh marketplace.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/skills/community", tags=["community-skills"])

# Security: regex for safe skill names — alphanumeric, hyphens, underscores only.
# Prevents path traversal via .., /, \, and other special characters.
_SAFE_SKILL_NAME_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$')


def _validate_skill_name(name: str) -> str:
    """Validate skill_name against path traversal attacks (OWASP A01:2021 Broken Access Control).

    Only allows alphanumeric characters, hyphens and underscores. Rejects empty
    names and names containing path separators (.., /, \\, etc.).
    """
    if not name or not _SAFE_SKILL_NAME_RE.match(name):
        raise HTTPException(
            400,
            "Invalid skill name: only alphanumeric characters, hyphens, and underscores are allowed",
        )
    return name

# skills.sh API base URL
SKILLS_SH_API = "https://api.skills.sh/v1"


@router.get("/search")
async def search_skills(
    q: str = Query(..., min_length=1, description="Search query"),
    category: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=50),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Search for community skills on skills.sh."""
    if ctx.user.role not in ("admin", "owner"):
        raise HTTPException(403, "Only admin users can browse community skills")

    import requests

    try:
        params = {"q": q, "limit": limit}
        if category:
            params["category"] = category

        resp = requests.get(f"{SKILLS_SH_API}/skills/search", params=params, timeout=10)

        if resp.status_code == 200:
            results = resp.json()
            skills = results.get("skills", results.get("results", []))
            return {
                "skills": [
                    {
                        "name": s.get("name", ""),
                        "description": s.get("description", ""),
                        "author": s.get("author", "unknown"),
                        "installs": s.get("installs", s.get("install_count", 0)),
                        "category": s.get("category", "general"),
                        "version": s.get("version", "1.0.0"),
                        "url": s.get("url", s.get("repository", "")),
                        "rating": s.get("rating"),
                    }
                    for s in skills[:limit]
                ],
                "total": results.get("total", len(skills)),
            }

        return {"skills": [], "total": 0}

    except Exception as e:
        logger.error("Skills search failed: %s", e)
        return {"skills": [], "total": 0, "error": "skills.sh API unavailable"}


@router.post("/install")
async def install_skill(
    payload: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Install a community skill. Downloads, scans, and saves to workspace."""
    if ctx.user.role not in ("admin", "owner"):
        raise HTTPException(403, "Only admin users can install community skills")

    skill_name = payload.get("skill_name", "").strip()
    force = payload.get("force", False)
    if not skill_name:
        raise HTTPException(400, "skill_name is required")
    # SECURITY: reject path traversal characters in skill_name
    _validate_skill_name(skill_name)

    import requests

    # 1. Fetch from skills.sh
    try:
        resp = requests.get(f"{SKILLS_SH_API}/skills/{skill_name}", timeout=15)
        if resp.status_code != 200:
            raise HTTPException(404, f"Skill '{skill_name}' not found on skills.sh")
        skill_data = resp.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to fetch from skills.sh: %s", e)
        raise HTTPException(502, "Failed to fetch from skills.sh")

    skill_content = skill_data.get("content", skill_data.get("skill_md", ""))
    if not skill_content:
        raise HTTPException(400, "Skill has no content")

    # 2. Security scan
    scan_result = None
    risk_score = 0
    try:
        from core.services.plugin_security_scanner import get_plugin_scanner
        scanner = get_plugin_scanner()
        scan_result = await scanner.scan_content(
            content=skill_content,
            filename=f"{skill_name}/SKILL.md",
            content_type="skill",
        )
        risk_score = scan_result.get("risk_score", 0) if scan_result else 0
    except Exception as scan_err:
        logger.warning("Security scan unavailable: %s", scan_err)

    if not force:
        if risk_score >= 70:
            return {
                "status": "blocked",
                "risk_score": risk_score,
                "reason": "Skill blocked by security scanner",
                "findings": scan_result.get("findings", []) if scan_result else [],
            }
        if risk_score >= 20:
            return {
                "status": "review_required",
                "risk_score": risk_score,
                "findings": scan_result.get("findings", []) if scan_result else [],
                "skill_name": skill_name,
                "message": "Admin confirmation required",
            }

    # 3. Save to workspace skills directory
    workspace_dir = Path(os.path.expanduser("~/.automatos/skills")) / str(ctx.workspace_id)
    skill_dir = workspace_dir / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    (skill_dir / "SKILL.md").write_text(skill_content)
    meta = {
        "name": skill_name,
        "source": "skills.sh",
        "installed_at": datetime.utcnow().isoformat(),
        "installed_by": ctx.user.id,
        "version": skill_data.get("version", "1.0.0"),
        "author": skill_data.get("author", "unknown"),
        "risk_score": risk_score,
    }
    (skill_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    logger.info("Installed skill '%s' for workspace %s (risk_score=%s)", skill_name, ctx.workspace_id, risk_score)
    return {"status": "installed", "skill_name": skill_name, "risk_score": risk_score}


@router.get("/installed")
async def list_installed_skills(
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """List all installed skills (workspace + built-in)."""
    results = []

    # Workspace skills (from skills.sh or manual)
    workspace_dir = Path(os.path.expanduser("~/.automatos/skills")) / str(ctx.workspace_id)
    if workspace_dir.exists():
        for skill_dir in sorted(workspace_dir.iterdir()):
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                meta = _read_meta(skill_dir)
                results.append({
                    "name": skill_dir.name,
                    "source": meta.get("source", "workspace"),
                    "version": meta.get("version", "unknown"),
                    "author": meta.get("author", "unknown"),
                    "installed_at": meta.get("installed_at"),
                    "risk_score": meta.get("risk_score"),
                    "removable": True,
                })

    # Built-in skills (shipped with Automatos)
    builtin_dir = Path(__file__).parent.parent / "skills"
    if builtin_dir.exists():
        for skill_dir in sorted(builtin_dir.iterdir()):
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                results.append({
                    "name": skill_dir.name,
                    "source": "built-in",
                    "version": "1.0.0",
                    "author": "Automatos",
                    "installed_at": None,
                    "risk_score": 0,
                    "removable": False,
                })

    return {"skills": results, "total": len(results)}


@router.delete("/{skill_name}")
async def uninstall_skill(
    skill_name: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Uninstall a workspace skill."""
    if ctx.user.role not in ("admin", "owner"):
        raise HTTPException(403, "Only admin users can uninstall skills")

    # SECURITY: reject path traversal characters in skill_name
    _validate_skill_name(skill_name)

    workspace_dir = Path(os.path.expanduser("~/.automatos/skills")) / str(ctx.workspace_id)
    skill_dir = workspace_dir / skill_name

    if not skill_dir.exists():
        raise HTTPException(404, f"Skill '{skill_name}' not found")

    import shutil
    shutil.rmtree(skill_dir)
    logger.info("Uninstalled skill '%s' from workspace %s", skill_name, ctx.workspace_id)
    return {"status": "uninstalled", "skill_name": skill_name}


def _read_meta(skill_dir: Path) -> dict:
    meta_file = skill_dir / "metadata.json"
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text())
        except Exception:
            pass
    return {}
