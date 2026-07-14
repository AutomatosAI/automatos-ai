"""
Skill-runtime handlers (PRD-202)
================================

The runtime side of the Agent Skills loading model, distinct from the CRUD
edit handlers in ``handlers_skills.py``:

  * ``load_skill``  (S2) — trigger-based L2 activation. The prompt lists attached
    skills at L1 (name + description) only; the model calls this to pull a
    skill's full body into context for a turn.
  * ``run_skill_script`` (S3) — L3 script execution via the WORKSPACE WORKER.
    Materializes the skill's bundle into the worker filesystem and runs the
    script there (sandboxed, per-workspace, token-gated); only the OUTPUT
    returns to context — never the script source.
  * ``set_skill_script_execution`` (S4) — workspace-admin enablement gate for
    L3. Import/read a skill freely; run its scripts only after an admin enables
    it (scanner-pass required, audited). Import != executable.

All handlers share the platform-tool signature ``(db, workspace_id, params)``
and are workspace-scoped: the executor passes the calling agent's workspace_id
and these refuse to touch skills outside that boundary.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared: resolve a workspace-visible skill by name or id
# ---------------------------------------------------------------------------

def _resolve_visible_skill(db: Session, workspace_id, name: str = "", skill_id=None):
    """Return the workspace-visible skill (own fork or global/marketplace).

    Prefers a workspace-owned row over a global one of the same name.
    """
    from core.models.core import Skill

    query = db.query(Skill).filter(Skill.is_active == True)  # noqa: E712
    if skill_id:
        query = query.filter(Skill.id == skill_id)
    elif name:
        query = query.filter(Skill.name.ilike(name))
    else:
        return None

    query = query.filter(
        (Skill.workspace_id.is_(None)) | (Skill.workspace_id == workspace_id)
    )
    # Workspace-owned fork wins over the global original of the same name.
    return query.order_by(Skill.workspace_id.isnot(None).desc()).first()


# ---------------------------------------------------------------------------
# S2 — load_skill: trigger-based L2 activation
# ---------------------------------------------------------------------------

async def load_skill(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Load a skill's full L2 body into context for this turn (trigger).

    The prompt lists attached skills at L1 (name + description); the model calls
    this when its task matches a skill's description. Returns the SKILL.md body.
    """
    name = (params.get("name") or params.get("skill_name") or "").strip()
    if not name:
        return {"success": False, "error": "Provide the skill 'name' to load."}

    skill = _resolve_visible_skill(db, workspace_id, name=name)
    if not skill:
        return {
            "success": False,
            "error": f"Skill '{name}' not found or not available to this workspace.",
        }

    body = skill.prompt_template
    if not body or not str(body).strip():
        try:
            from modules.agents.services.skill_loader import get_skill_loader

            body = get_skill_loader(db).load_skill_core(skill.name, db=db)
        except Exception:
            logger.warning("[load_skill] loader fallback failed for '%s'", skill.name, exc_info=True)
            body = None

    if not body or not str(body).strip():
        return {"success": False, "error": f"Skill '{skill.name}' has no loadable instructions."}

    logger.info("[load_skill] activated '%s' (id=%s) for ws=%s", skill.name, skill.id, workspace_id)
    return {
        "success": True,
        "skill": skill.name,
        "skill_id": skill.id,
        "content": str(body).strip(),
        "message": (
            f"Loaded skill '{skill.name}'. Its full instructions are now in "
            "context for this turn — follow them for the current task."
        ),
    }


# ---------------------------------------------------------------------------
# S3 — run_skill_script: L3 execution via the workspace worker
# ---------------------------------------------------------------------------

def _infer_interpreter(script_rel: str) -> str:
    s = script_rel.lower()
    if s.endswith(".py"):
        return "python"
    if s.endswith(".sh"):
        return "bash"
    if s.endswith(".js"):
        return "node"
    return "python"


def _safe_name(name: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:100] or "skill"


def _cap(text: Optional[str], limit: int) -> str:
    if not text:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated, {len(text) - limit} more chars]"


async def run_skill_script(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run a skill's bundled script in the workspace worker; return OUTPUT only.

    Materializes the skill's ``SkillFile`` bundle into the per-workspace worker
    filesystem and invokes ``WorkspaceClient.exec_command`` — the existing
    sandboxed, per-workspace, ``WORKER_INTERNAL_TOKEN``-gated, wall-clock-capped
    exec path (NOT the in-process ActionExecutor). Only stdout/stderr return to
    context; the script source never does. Gated on per-workspace L3 enablement
    (S4): import/read is free, running is opt-in.
    """
    import shlex

    from config import config
    from core.services.skill_l3_execution import is_l3_execution_enabled
    from core.workspace_client import WorkspaceClient
    from modules.agents.services.skill_portability import collect_skill_bundle

    skill_name = (params.get("skill") or params.get("skill_name") or params.get("name") or "").strip()
    script = (params.get("script") or "").strip()
    args = params.get("args") or ""
    interpreter = (params.get("interpreter") or "").strip()

    if not skill_name:
        return {"success": False, "error": "Provide the 'skill' whose script to run."}
    if not script:
        return {"success": False, "error": "Provide the 'script' filename to run."}

    skill = _resolve_visible_skill(db, workspace_id, name=skill_name)
    if not skill:
        return {"success": False, "error": f"Skill '{skill_name}' not found or not available to this workspace."}

    # --- S4 gate: L3 execution must be enabled for this skill in this workspace ---
    if not is_l3_execution_enabled(db, workspace_id, skill.id):
        return {
            "success": False,
            "error": (
                f"L3 script execution is not enabled for skill '{skill.name}' in this workspace. "
                "A workspace admin must enable it first (platform_set_skill_script_execution) — "
                "importing/reading a skill is always allowed, but running its scripts is opt-in."
            ),
            "enablement_required": True,
        }

    bundle = collect_skill_bundle(getattr(skill, "filesystem_path", None))
    if not bundle:
        return {"success": False, "error": f"Skill '{skill.name}' has no bundled scripts to run."}

    # Resolve the script within the bundle (scripts/<name> or <name>).
    script_rel = None
    for candidate in (f"scripts/{script}", script):
        if candidate in bundle:
            script_rel = candidate
            break
    if script_rel is None:
        return {
            "success": False,
            "error": f"Script '{script}' not found in skill '{skill.name}' bundle.",
            "available_scripts": [p for p in bundle if p.startswith("scripts/") or p.endswith((".py", ".sh", ".js"))],
        }

    # --- Materialize the bundle into the worker filesystem (per-workspace jail) ---
    client = WorkspaceClient(str(workspace_id))
    base = f".skills/{_safe_name(skill.name)}"
    for rel, content in bundle.items():
        write = await client.write_file(f"{base}/{rel}", content)
        if not write.get("success", False):
            return {
                "success": False,
                "error": f"Failed to materialize skill bundle into the workspace worker: {write.get('error')}",
            }

    # --- Execute in the sandbox; only OUTPUT returns to context ---
    interp = interpreter or _infer_interpreter(script_rel)
    command = f"{interp} {shlex.quote(script_rel)}"
    if args:
        command += f" {args}"

    logger.info("[run_skill_script] ws=%s skill='%s' script='%s' -> worker exec", workspace_id, skill.name, script_rel)
    result = await client.exec_command(
        command=command,
        cwd=base,
        timeout=config.SKILL_SCRIPT_TIMEOUT_SECONDS,
    )

    if not result.get("success", True) and "error" in result and "stdout" not in result:
        # Transport/worker error (unreachable, etc.) — no script output produced.
        return {"success": False, "skill": skill.name, "script": script_rel, "error": result.get("error")}

    cap = config.SKILL_SCRIPT_OUTPUT_MAX_CHARS
    return {
        "success": True,
        "skill": skill.name,
        "script": script_rel,
        "exit_code": result.get("exit_code", result.get("returncode")),
        "stdout": _cap(result.get("stdout") or result.get("output"), cap),
        "stderr": _cap(result.get("stderr"), cap),
    }


# ---------------------------------------------------------------------------
# S4 — set_skill_script_execution: workspace-admin L3 enablement (audited)
# ---------------------------------------------------------------------------

async def set_skill_script_execution(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Enable/disable L3 script execution for a skill in this workspace (admin).

    Enabling re-runs the security scanner on the skill body and refuses on a
    critical finding (scanner-pass required). Every change is audited via
    ``SkillAuditLog``. import/read stays free; running is what this gates.
    """
    from core.services.skill_l3_execution import set_l3_execution_enabled

    if "enabled" not in params:
        return {"success": False, "error": "Provide 'enabled' (true to enable L3 execution, false to disable)."}
    enabled = bool(params.get("enabled"))
    skill_id = params.get("skill_id")
    skill_name = (params.get("skill_name") or params.get("skill") or "").strip()

    skill = _resolve_visible_skill(db, workspace_id, name=skill_name, skill_id=skill_id)
    if not skill:
        return {"success": False, "error": "Skill not found or not available to this workspace."}

    actor = str(params.get("_agent_name") or params.get("_agent_id") or "workspace-admin")

    # Scanner-pass required at enable-time (import != executable).
    if enabled:
        try:
            from core.services.plugin_security_scanner import quick_scan

            findings = quick_scan(skill.prompt_template or "", filename=f"{skill.name}/SKILL.md")
            critical = [f for f in findings if f.severity == "critical"]
            if critical:
                return {
                    "success": False,
                    "error": (
                        f"Cannot enable L3 execution for '{skill.name}' — the security scanner "
                        "found critical issues. Fix them and re-import before enabling."
                    ),
                    "findings": [{"severity": f.severity, "description": f.description, "line": f.line} for f in critical],
                }
        except Exception:
            logger.warning("[set_skill_script_execution] scanner unavailable — refusing enable (fail-closed)", exc_info=True)
            return {"success": False, "error": "Security scanner unavailable — refusing to enable L3 execution (fail-closed)."}

    try:
        enabled_ids = set_l3_execution_enabled(db, workspace_id, skill.id, enabled, actor=actor)
        db.commit()
    except ValueError as e:
        db.rollback()
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "skill": skill.name,
        "skill_id": skill.id,
        "l3_execution_enabled": enabled,
        "enabled_skill_ids": enabled_ids,
        "message": (
            f"L3 script execution {'ENABLED' if enabled else 'DISABLED'} for skill "
            f"'{skill.name}' in this workspace."
        ),
    }
