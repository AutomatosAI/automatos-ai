"""
PRD-230 US-006 — Package platform tools (handlers).
===================================================

Three tools, the 3-file pattern (schemas in ``actions_packages.py``, dispatch in
``platform_executor.py``):

  - ``platform_search_packages``            — rank packages against business signals.
  - ``platform_install_package``            — install a package's full closure (US-005).
  - ``platform_install_marketplace_agent``  — install one agent + its closure (US-005).

Every install goes through the US-005 installer, so D1/D2/D3 are inherited. Two
tool-layer policies live here:

  D6/FR-3 one-package-during-onboarding: while onboarding is non-terminal, a
    SECOND ``platform_install_package`` returns honest copy; unrestricted once
    onboarding is terminal.
  D9 over-quota: when a package would exceed the tier's agent cap, the tool
    returns the honest plan conversation + recommendation — NEVER a silent block
    and NEVER a partial install (the closure is atomic; the quota is checked
    BEFORE any registration).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List
from uuid import UUID

logger = logging.getLogger(__name__)

ONE_PACKAGE_DURING_ONBOARDING = (
    "You can install one package during onboarding — let's get this one set up "
    "together, then add more any time from the marketplace after setup."
)


# --------------------------------------------------------------------------- #
# Small DB reads (kept as named helpers so tests can stub them without a DB)
# --------------------------------------------------------------------------- #


def _load_workspace(db: Any, workspace_id: UUID) -> Any:
    from core.models.workspaces import Workspace

    return db.query(Workspace).filter(Workspace.id == workspace_id).first()


def _workspace_agent_count(db: Any, workspace_id: UUID) -> int:
    from core.models.core import Agent

    return (
        db.query(Agent)
        .filter(Agent.workspace_id == workspace_id, Agent.owner_type == "workspace")
        .count()
    )


def _package_agent_refs(package: Any) -> set:
    """Distinct agent members a package installs (the headline quota count)."""
    return {
        str(m.get("ref") or m.get("id"))
        for m in (getattr(package, "members", None) or [])
        if isinstance(m, dict) and m.get("type") == "agent"
    }


# --------------------------------------------------------------------------- #
# platform_search_packages
# --------------------------------------------------------------------------- #


def _member_counts(package: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for m in (getattr(package, "members", None) or []):
        if isinstance(m, dict):
            t = m.get("type", "?")
            counts[t] = counts.get(t, 0) + 1
    return counts


def _match_summary(match: Any) -> Dict[str, Any]:
    pkg = match.package
    manifest = getattr(pkg, "setup_manifest", None) or {}
    return {
        "slug": pkg.slug,
        "name": pkg.name,
        "description": pkg.description,
        "score": match.score,
        "reasons": match.reasons,
        "contents": _member_counts(pkg),  # {"agent": 4, "playbook": 1, ...}
        "required_connects": manifest.get("required_connects", []),
        "showcase": bool(getattr(pkg, "showcase", False)),
    }


async def search_packages(db: Any, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """platform_search_packages — rank marketplace packages against business signals."""
    from services.marketplace_packages import list_packages, match_by_signals

    signals = {
        "platforms": params.get("platforms") or [],
        "urls": params.get("urls") or [],
        "text": params.get("text") or params.get("query") or "",
        "vertical_tags": params.get("vertical_tags") or [],
    }
    matches = match_by_signals(signals, list_packages(db))
    return {
        "success": True,
        "matches": [_match_summary(m) for m in matches],
        "count": len(matches),
    }


# --------------------------------------------------------------------------- #
# platform_install_package
# --------------------------------------------------------------------------- #


def _over_quota_response(db: Any, workspace: Any, package: Any, package_agents: int,
                         current: int, max_agents: int) -> Dict[str, Any]:
    from services.plan_tiers import recommend_plan

    segment = (getattr(workspace, "onboarding", None) or {}).get("segment") if workspace else None
    plan, reason = recommend_plan(segment, team_size=current + package_agents)
    plan_name = getattr(workspace, "plan", None) or "basic"
    return {
        "success": False,
        "over_quota": True,
        "message": (
            f"'{package.name}' installs {package_agents} agents; your {plan_name} plan "
            f"includes {max_agents}. Nothing's been installed yet — let's pick a plan "
            f"that fits ({plan}, because {reason}), then I'll set the whole team up."
        ),
        "plan_recommendation": plan,
        "package_agents": package_agents,
        "current_agents": current,
        "max_agents": max_agents,
    }


def _check_quota(db: Any, workspace: Any, workspace_id: UUID, package: Any) -> Dict[str, Any]:
    """D9: does installing this package exceed the tier's agent cap? Read-only —
    NEVER installs. Returns {ok: True} or the honest over-quota response."""
    from services.plan_tiers import get_tier

    plan = (getattr(workspace, "plan", None) or "basic") if workspace else "basic"
    tier = get_tier(plan) or {}
    max_agents = int(tier.get("max_agents", 0) or 0)
    package_agents = len(_package_agent_refs(package))
    if max_agents <= 0:  # 0 = unlimited
        return {"ok": True}
    current = _workspace_agent_count(db, workspace_id)
    if current + package_agents > max_agents:
        return {"ok": False, "response": _over_quota_response(
            db, workspace, package, package_agents, current, max_agents)}
    return {"ok": True}


async def install_package_tool(db: Any, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """platform_install_package — install a package's full closure (US-005),
    honoring the one-package-during-onboarding restriction (D6) and the over-quota
    honest conversation (D9). Provide slug."""
    slug = params.get("slug")
    if not slug:
        return {"success": False, "error": "Missing required parameter: slug"}

    from services import onboarding_state
    from services.marketplace_packages import get_by_slug
    from services.package_installer import PackageInstallError, install_package

    workspace = _load_workspace(db, workspace_id)
    onboarding_active = (
        workspace is not None
        and onboarding_state.current_stage(workspace) not in onboarding_state.TERMINAL_STAGES
    )

    # D6/FR-3 — one package during onboarding.
    if onboarding_active and onboarding_state.onboarding_package_installed(workspace):
        return {"success": False, "onboarding_restricted": True,
                "message": ONE_PACKAGE_DURING_ONBOARDING}

    package = get_by_slug(db, slug)
    if package is None:
        return {"success": False, "error": f"Package not found: {slug}"}

    # D9 — over-quota is an honest conversation, checked BEFORE any registration.
    quota = _check_quota(db, workspace, workspace_id, package)
    if not quota["ok"]:
        return quota["response"]

    try:
        manifest = await install_package(db, workspace_id, slug, user_id=None)
    except PackageInstallError as exc:
        return {"success": False, "error": str(exc)}

    if onboarding_active:
        onboarding_state.record_package_event(db, workspace, "package_installed", slug, commit=False)
    db.commit()

    return {
        "success": True,
        "slug": slug,
        "message": f"Installed '{package.name}' — {len(manifest.added)} added, "
                   f"{len(manifest.required_connects)} to connect.",
        **manifest.to_dict(),
    }


# --------------------------------------------------------------------------- #
# platform_install_marketplace_agent
# --------------------------------------------------------------------------- #


async def install_marketplace_agent_tool(db: Any, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """platform_install_marketplace_agent — install one marketplace agent with its
    full closure (US-005). Provide agent_id or agent_name."""
    agent_id = params.get("agent_id")
    agent_name = params.get("agent_name")
    if agent_id is None and not agent_name:
        return {"success": False, "error": "Provide agent_id or agent_name"}

    from services.package_installer import PackageInstallError, install_marketplace_agent

    ref = str(agent_id) if agent_id is not None else str(agent_name)
    try:
        manifest = await install_marketplace_agent(db, workspace_id, ref, user_id=None)
    except PackageInstallError as exc:
        return {"success": False, "error": str(exc)}
    db.commit()

    return {
        "success": True,
        "message": f"Installed agent + {len(manifest.added) - 1} dependencies, "
                   f"{len(manifest.required_connects)} to connect.",
        **manifest.to_dict(),
    }
