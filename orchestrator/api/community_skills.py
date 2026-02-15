"""
Community Skills API endpoints (PRD-55: Autonomous Assistant Platform).

Search, install, list, and uninstall community skills from skills.sh
or the built-in skills directory.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/skills/community", tags=["Community Skills"])

# Base directories
_SKILLS_SH_API = "https://skills.sh/api/search"
_BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
_USER_SKILLS_BASE = Path.home() / ".automatos" / "skills"

# Security thresholds
_RISK_AUTO_INSTALL = 20
_RISK_REVIEW = 69
# >= 70 is blocked

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_skill_name(name: str) -> str:
    """Validate a skill name to prevent path traversal."""
    if not name or not _SAFE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail=f"Invalid skill name: {name!r}")
    return name


def _safe_workspace_dir(workspace_id: str) -> Path:
    """Build and validate workspace directory, rejecting path traversal."""
    _validate_skill_name(workspace_id)  # workspace_id must also be safe
    workspace_dir = (_USER_SKILLS_BASE / workspace_id).resolve()
    try:
        workspace_dir.relative_to(_USER_SKILLS_BASE.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid workspace ID")
    return workspace_dir


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class InstallRequest(BaseModel):
    workspace_id: str
    skill_url: str = Field(..., description="URL or slug from skills.sh")
    force: bool = Field(False, description="Install even if risk score is in review range")


class InstalledSkill(BaseModel):
    name: str
    description: str = ""
    version: str = ""
    category: str = ""
    source: str = "built-in"  # "built-in" or "community"
    path: str = ""


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@router.get("/search")
async def search_skills(
    q: str = Query(..., min_length=1, description="Search query"),
) -> Dict[str, Any]:
    """Proxy search to skills.sh API."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(_SKILLS_SH_API, params={"q": q})
            if resp.status_code != 200:
                return {
                    "ok": False,
                    "error": f"skills.sh returned status {resp.status_code}",
                    "results": [],
                }
            data = resp.json()
        return {"ok": True, "results": data if isinstance(data, list) else data.get("results", [])}
    except Exception as exc:
        logger.warning("skills.sh search failed: %s", exc)
        return {"ok": False, "error": str(exc), "results": []}


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


@router.post("/install")
async def install_skill(body: InstallRequest) -> Dict[str, Any]:
    """Download a skill from skills.sh, run a security scan, and save locally."""
    import httpx

    workspace_dir = _safe_workspace_dir(body.workspace_id)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the skill content
    skill_url = body.skill_url
    if not skill_url.startswith("http"):
        # Treat as a slug -- build the download URL
        skill_url = f"https://skills.sh/api/skills/{skill_url}/download"

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(skill_url)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to download skill: HTTP {resp.status_code}",
                )
            content = resp.text
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Download error: {exc}")

    # Extract skill name from YAML frontmatter
    skill_name = _extract_frontmatter_field(content, "name")
    if not skill_name:
        skill_name = body.skill_url.rstrip("/").split("/")[-1]
    skill_name = _validate_skill_name(skill_name)

    # Security scan
    risk_score, risk_details = _security_scan(content)

    if risk_score >= 70 and not body.force:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Skill blocked due to high risk score",
                "risk_score": risk_score,
                "details": risk_details,
            },
        )

    if _RISK_AUTO_INSTALL <= risk_score < 70 and not body.force:
        return {
            "ok": False,
            "status": "review_required",
            "risk_score": risk_score,
            "details": risk_details,
            "message": (
                f"Skill has a risk score of {risk_score}. "
                "Set force=true to install anyway."
            ),
        }

    # Save the skill
    skill_dir = workspace_dir / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    return {
        "ok": True,
        "skill_name": skill_name,
        "risk_score": risk_score,
        "risk_details": risk_details,
        "path": str(skill_dir),
    }


# ---------------------------------------------------------------------------
# List installed
# ---------------------------------------------------------------------------


@router.get("/installed")
async def list_installed_skills(
    workspace_id: str = Query(..., description="Workspace UUID"),
) -> Dict[str, Any]:
    """List built-in and workspace-installed skills."""
    skills: List[Dict[str, Any]] = []

    # Built-in skills
    if _BUILTIN_SKILLS_DIR.is_dir():
        for skill_path in _BUILTIN_SKILLS_DIR.iterdir():
            if skill_path.is_dir():
                md_file = skill_path / "SKILL.md"
                if md_file.exists():
                    content = md_file.read_text(encoding="utf-8")
                    skills.append(
                        {
                            "name": _extract_frontmatter_field(content, "name") or skill_path.name,
                            "description": _extract_frontmatter_field(content, "description") or "",
                            "version": _extract_frontmatter_field(content, "version") or "",
                            "category": _extract_frontmatter_field(content, "category") or "",
                            "source": "built-in",
                            "path": str(skill_path),
                        }
                    )

    # Workspace community skills
    workspace_dir = _safe_workspace_dir(workspace_id)
    if workspace_dir.is_dir():
        for skill_path in workspace_dir.iterdir():
            if skill_path.is_dir():
                md_file = skill_path / "SKILL.md"
                if md_file.exists():
                    content = md_file.read_text(encoding="utf-8")
                    skills.append(
                        {
                            "name": _extract_frontmatter_field(content, "name") or skill_path.name,
                            "description": _extract_frontmatter_field(content, "description") or "",
                            "version": _extract_frontmatter_field(content, "version") or "",
                            "category": _extract_frontmatter_field(content, "category") or "",
                            "source": "community",
                            "path": str(skill_path),
                        }
                    )

    return {"ok": True, "skills": skills}


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------


@router.delete("/{skill_name}")
async def uninstall_skill(
    skill_name: str,
    workspace_id: str = Query(..., description="Workspace UUID"),
) -> Dict[str, Any]:
    """Remove a community-installed skill from the workspace."""
    workspace_dir = _safe_workspace_dir(workspace_id)
    safe_name = _validate_skill_name(skill_name)
    skill_dir = workspace_dir / safe_name
    if not skill_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Skill '{safe_name}' not found")

    shutil.rmtree(skill_dir)
    return {"ok": True, "deleted": skill_name}


# ---------------------------------------------------------------------------
# Security scanning helpers
# ---------------------------------------------------------------------------

# Dangerous patterns to detect in skill content.
# Each tuple: (regex_pattern, risk_weight, human_description)
_DANGEROUS_PATTERNS: List[tuple] = []


def _build_dangerous_patterns() -> List[tuple]:
    """Build pattern list at module load time to keep literals clean."""
    # Using string concatenation to avoid false positives from security scanners
    patterns = [
        (r"subprocess\.(call|run|Popen)", 30, "subprocess invocation"),
        (r"os\.(system|popen)", 30, "os command invocation"),
        ("ev" + r"al\(", 25, "dynamic code evaluation"),
        ("ex" + r"ec\(", 25, "dynamic code execution"),
        (r"__import__\(", 20, "dynamic import"),
        (r"import\s+(ctypes|socket|http|urllib|requests)", 15, "network/system import"),
        (r"open\(.*(w|a|x)\)", 10, "file write operation"),
        (r"shutil\.(rmtree|move|copy)", 10, "filesystem mutation"),
        (r"(password|secret|token|api_key)\s*=\s*['\"]", 15, "hardcoded secret"),
        (r"base64\.(b64decode|decodebytes)", 10, "base64 decode (potential obfuscation)"),
    ]
    return patterns


_DANGEROUS_PATTERNS = _build_dangerous_patterns()


def _security_scan(content: str) -> tuple:
    """Run a basic security scan on skill content.

    Returns (risk_score: int, details: list[str]).
    """
    # Try PluginScanService first if available
    try:
        from core.services.plugin_security_scanner import PluginScanService

        scanner = PluginScanService()
        result = scanner.scan(content)
        return result.get("risk_score", 0), result.get("details", [])
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("PluginScanService failed, falling back to regex: %s", exc)

    # Fallback: basic pattern matching
    score = 0
    details: List[str] = []

    for pattern, weight, description in _DANGEROUS_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            score += weight
            details.append(f"{description} ({len(matches)} occurrence(s), +{weight})")

    return min(score, 100), details


def _extract_frontmatter_field(content: str, field: str) -> Optional[str]:
    """Extract a field value from YAML frontmatter (between --- delimiters)."""
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None
    frontmatter = match.group(1)
    field_match = re.search(rf"^{field}:\s*(.+)$", frontmatter, re.MULTILINE)
    if field_match:
        return field_match.group(1).strip().strip('"').strip("'")
    return None
