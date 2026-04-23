"""Marketplace discovery & workspace inventory handlers for PlatformActionExecutor (PRD-71)."""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def browse_marketplace_plugins(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Browse/search approved marketplace plugins."""
    from core.models.marketplace_plugins import (
        MarketplacePlugin, PluginCategory, WorkspaceEnabledPlugin,
    )

    query = db.query(MarketplacePlugin).filter(
        MarketplacePlugin.approval_status == "approved",
        MarketplacePlugin.is_active == True,
    )

    search = (params.get("search") or "").strip()
    if search:
        like = f"%{search}%"
        query = query.filter(
            MarketplacePlugin.name.ilike(like)
            | MarketplacePlugin.description.ilike(like)
            | MarketplacePlugin.slug.ilike(like)
        )

    category_slug = params.get("category")
    if category_slug:
        cat = db.query(PluginCategory).filter(
            PluginCategory.slug == category_slug,
        ).first()
        if cat:
            query = query.filter(MarketplacePlugin.category_id == cat.id)

    limit = min(params.get("limit", 20), 50)
    plugins = query.order_by(MarketplacePlugin.enable_count.desc()).limit(limit).all()

    # Cross-reference enabled plugins for this workspace
    enabled_ids = set()
    try:
        rows = (
            db.query(WorkspaceEnabledPlugin.plugin_id)
            .filter(WorkspaceEnabledPlugin.workspace_id == workspace_id)
            .all()
        )
        enabled_ids = {r.plugin_id for r in rows}
    except Exception:
        pass

    # Resolve category names in one query
    cat_ids = {p.category_id for p in plugins if p.category_id}
    cat_map = {}
    if cat_ids:
        cats = db.query(PluginCategory).filter(PluginCategory.id.in_(cat_ids)).all()
        cat_map = {c.id: c.name for c in cats}

    return {
        "success": True,
        "plugins": [
            {
                "id": str(p.id),
                "slug": p.slug,
                "name": p.name,
                "description": (p.description or "")[:200],
                "category": cat_map.get(p.category_id),
                "skills_count": p.skills_count or 0,
                "enable_count": p.enable_count or 0,
                "is_featured": p.is_featured,
                "is_enabled": p.id in enabled_ids,
            }
            for p in plugins
        ],
        "count": len(plugins),
    }


async def browse_marketplace_agents(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Browse/search marketplace agent templates (owner_type='marketplace')."""
    from core.models.core import Agent, agent_skills, Skill

    query = db.query(Agent).filter(
        Agent.owner_type == "marketplace",
        Agent.status == "active",
    )

    search = (params.get("search") or "").strip()
    if search:
        like = f"%{search}%"
        query = query.filter(
            Agent.name.ilike(like)
            | Agent.description.ilike(like)
            | Agent.marketplace_category.ilike(like)
        )

    category = params.get("category")
    if category:
        query = query.filter(Agent.marketplace_category.ilike(f"%{category}%"))

    limit = min(params.get("limit", 20), 50)
    agents = query.order_by(Agent.install_count.desc()).limit(limit).all()

    # Check which marketplace agents are already cloned into this workspace
    cloned_from_ids = set()
    try:
        rows = (
            db.query(Agent.cloned_from_id)
            .filter(
                Agent.workspace_id == workspace_id,
                Agent.cloned_from_id.isnot(None),
            )
            .all()
        )
        cloned_from_ids = {r.cloned_from_id for r in rows}
    except Exception:
        pass

    results = []
    for a in agents:
        # Get skill names
        skill_names = []
        try:
            if a.skills:
                skill_names = [s.name for s in a.skills if s.name]
        except Exception:
            pass

        # Extract model info
        model_id = None
        if a.model_config and isinstance(a.model_config, dict):
            model_id = a.model_config.get("model_id")

        results.append({
            "id": a.id,
            "name": a.name,
            "description": (a.description or "")[:300],
            "category": a.marketplace_category,
            "model": model_id,
            "skills": skill_names,
            "tags": a.tags or [],
            "install_count": a.install_count or 0,
            "is_featured": a.is_featured,
            "is_installed": a.id in cloned_from_ids,
        })

    return {
        "success": True,
        "agents": results,
        "count": len(results),
    }


async def browse_marketplace_skills(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Browse/search global marketplace skills (workspace_id IS NULL)."""
    from core.models.core import Skill
    from core.models.marketplace_plugins import WorkspaceEnabledSkill

    query = db.query(Skill).filter(
        Skill.workspace_id.is_(None),
        Skill.is_active == True,
    )

    search = (params.get("search") or "").strip()
    if search:
        like = f"%{search}%"
        query = query.filter(
            Skill.name.ilike(like) | Skill.description.ilike(like)
        )

    category = params.get("category")
    if category:
        query = query.filter(Skill.category == category)

    limit = min(params.get("limit", 20), 50)
    skills = query.order_by(Skill.name).limit(limit).all()

    # Cross-reference enabled skills for this workspace
    enabled_ids = set()
    try:
        rows = (
            db.query(WorkspaceEnabledSkill.skill_id)
            .filter(WorkspaceEnabledSkill.workspace_id == workspace_id)
            .all()
        )
        enabled_ids = {r.skill_id for r in rows}
    except Exception:
        pass

    return {
        "success": True,
        "skills": [
            {
                "id": s.id,
                "name": s.name,
                "description": (s.description or "")[:200],
                "category": s.category,
                "skill_type": s.skill_type,
                "estimated_tokens": len(s.prompt_template or "") // 4,
                "is_enabled": s.id in enabled_ids,
            }
            for s in skills
        ],
        "count": len(skills),
    }


async def list_workspace_plugins(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List plugins enabled for this workspace."""
    from core.models.marketplace_plugins import (
        WorkspaceEnabledPlugin, MarketplacePlugin, PluginCategory,
    )

    rows = (
        db.query(WorkspaceEnabledPlugin, MarketplacePlugin)
        .join(MarketplacePlugin, WorkspaceEnabledPlugin.plugin_id == MarketplacePlugin.id)
        .filter(WorkspaceEnabledPlugin.workspace_id == workspace_id)
        .order_by(WorkspaceEnabledPlugin.enabled_at.desc())
        .all()
    )

    # Resolve category names
    cat_ids = {mp.category_id for _, mp in rows if mp.category_id}
    cat_map = {}
    if cat_ids:
        cats = db.query(PluginCategory).filter(PluginCategory.id.in_(cat_ids)).all()
        cat_map = {c.id: c.name for c in cats}

    return {
        "success": True,
        "plugins": [
            {
                "id": str(mp.id),
                "slug": mp.slug,
                "name": mp.name,
                "description": (mp.description or "")[:200],
                "category": cat_map.get(mp.category_id),
                "skills_count": mp.skills_count or 0,
                "enabled_at": wep.enabled_at.isoformat() if wep.enabled_at else None,
            }
            for wep, mp in rows
        ],
        "count": len(rows),
    }


async def list_workspace_skills(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List skills enabled for this workspace."""
    from core.models.core import Skill
    from core.models.marketplace_plugins import WorkspaceEnabledSkill

    rows = (
        db.query(WorkspaceEnabledSkill, Skill)
        .join(Skill, WorkspaceEnabledSkill.skill_id == Skill.id)
        .filter(WorkspaceEnabledSkill.workspace_id == workspace_id)
        .order_by(WorkspaceEnabledSkill.enabled_at.desc())
        .all()
    )

    return {
        "success": True,
        "skills": [
            {
                "id": skill.id,
                "name": skill.name,
                "description": (skill.description or "")[:200],
                "category": skill.category,
                "skill_type": skill.skill_type,
                "estimated_tokens": len(skill.prompt_template or "") // 4,
                "enabled_at": wes.enabled_at.isoformat() if wes.enabled_at else None,
            }
            for wes, skill in rows
        ],
        "count": len(rows),
    }


async def list_workspace_models(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List LLM models installed for this workspace + default models."""
    from core.models.core import LLMModel, WorkspaceModel

    # Workspace-installed models
    installed = (
        db.query(WorkspaceModel, LLMModel)
        .join(LLMModel, WorkspaceModel.model_id == LLMModel.id)
        .filter(
            WorkspaceModel.workspace_id == workspace_id,
            WorkspaceModel.is_active == True,
        )
        .all()
    )

    installed_llm_ids = {wm.model_id for wm, _ in installed}

    # Default models (available to all workspaces, not already in installed set)
    defaults = (
        db.query(LLMModel)
        .filter(
            LLMModel.is_default == True,
            LLMModel.status == "active",
            ~LLMModel.id.in_(installed_llm_ids) if installed_llm_ids else True,
        )
        .all()
    )

    models = []
    for wm, llm in installed:
        models.append({
            "model_id": llm.model_id,
            "display_name": llm.display_name,
            "provider": llm.provider,
            "input_cost_per_1k": llm.input_cost_per_1k_tokens,
            "output_cost_per_1k": llm.output_cost_per_1k_tokens,
            "context_length": llm.context_window,
            "supports_tools": llm.supports_functions,
            "supports_vision": llm.supports_vision,
            "category": llm.category,
            "source": wm.source,
            "installed_at": wm.installed_at.isoformat() if wm.installed_at else None,
        })

    for llm in defaults:
        models.append({
            "model_id": llm.model_id,
            "display_name": llm.display_name,
            "provider": llm.provider,
            "input_cost_per_1k": llm.input_cost_per_1k_tokens,
            "output_cost_per_1k": llm.output_cost_per_1k_tokens,
            "context_length": llm.context_window,
            "supports_tools": llm.supports_functions,
            "supports_vision": llm.supports_vision,
            "category": llm.category,
            "source": "default",
            "installed_at": None,
        })

    return {
        "success": True,
        "models": models,
        "count": len(models),
    }


async def install_plugin(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Enable a marketplace plugin for this workspace."""
    from core.models.marketplace_plugins import (
        MarketplacePlugin, WorkspaceEnabledPlugin,
    )

    plugin_id = params.get("plugin_id")
    plugin_slug = params.get("plugin_slug")

    if not plugin_id and not plugin_slug:
        return {"success": False, "error": "Provide plugin_id or plugin_slug"}

    # Resolve plugin
    query = db.query(MarketplacePlugin)
    if plugin_id:
        from uuid import UUID as _UUID
        query = query.filter(MarketplacePlugin.id == _UUID(str(plugin_id)))
    else:
        query = query.filter(MarketplacePlugin.slug == plugin_slug)

    plugin = query.first()
    if not plugin:
        return {"success": False, "error": "Plugin not found"}

    if plugin.approval_status != "approved" or not plugin.is_active:
        return {"success": False, "error": "Plugin is not approved or inactive"}

    # Idempotency check
    existing = (
        db.query(WorkspaceEnabledPlugin)
        .filter(
            WorkspaceEnabledPlugin.workspace_id == workspace_id,
            WorkspaceEnabledPlugin.plugin_id == plugin.id,
        )
        .first()
    )
    if existing:
        return {
            "success": True,
            "already_enabled": True,
            "plugin": {"id": str(plugin.id), "slug": plugin.slug, "name": plugin.name},
            "message": f"Plugin '{plugin.name}' is already enabled for this workspace.",
        }

    # Create junction record
    junction = WorkspaceEnabledPlugin(
        workspace_id=workspace_id,
        plugin_id=plugin.id,
    )
    db.add(junction)

    # Increment enable_count
    plugin.enable_count = (plugin.enable_count or 0) + 1
    db.flush()

    logger.info(
        "[PlatformExecutor] Installed plugin '%s' (id=%s) for workspace %s",
        plugin.name, plugin.id, workspace_id,
    )

    return {
        "success": True,
        "plugin": {"id": str(plugin.id), "slug": plugin.slug, "name": plugin.name},
        "message": f"Plugin '{plugin.name}' enabled for this workspace.",
    }


async def install_skill(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Enable a marketplace skill for this workspace."""
    from core.models.core import Skill
    from core.models.marketplace_plugins import WorkspaceEnabledSkill

    skill_id = params.get("skill_id")
    skill_name = params.get("skill_name")

    if not skill_id and not skill_name:
        return {"success": False, "error": "Provide skill_id or skill_name"}

    # Resolve skill
    query = db.query(Skill).filter(Skill.workspace_id.is_(None))
    if skill_id:
        query = query.filter(Skill.id == skill_id)
    else:
        query = query.filter(Skill.name.ilike(f"%{skill_name}%"))

    skill = query.first()
    if not skill:
        return {"success": False, "error": "Marketplace skill not found"}

    if not skill.is_active:
        return {"success": False, "error": "Skill is inactive"}

    # Idempotency check
    existing = (
        db.query(WorkspaceEnabledSkill)
        .filter(
            WorkspaceEnabledSkill.workspace_id == workspace_id,
            WorkspaceEnabledSkill.skill_id == skill.id,
        )
        .first()
    )
    if existing:
        return {
            "success": True,
            "already_enabled": True,
            "skill": {"id": skill.id, "name": skill.name},
            "message": f"Skill '{skill.name}' is already enabled for this workspace.",
        }

    # Create junction record
    junction = WorkspaceEnabledSkill(
        workspace_id=workspace_id,
        skill_id=skill.id,
    )
    db.add(junction)
    db.flush()

    logger.info(
        "[PlatformExecutor] Installed skill '%s' (id=%d) for workspace %s",
        skill.name, skill.id, workspace_id,
    )

    return {
        "success": True,
        "skill": {"id": skill.id, "name": skill.name},
        "message": f"Skill '{skill.name}' enabled for this workspace.",
    }


async def install_model(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Install an LLM model for this workspace from the OpenRouter catalog."""
    from core.models.core import LLMModel, WorkspaceModel
    from core.models.openrouter_cache import OpenRouterModelCache

    model_id = params.get("model_id")
    if not model_id:
        return {"success": False, "error": "Missing required parameter: model_id"}

    # Find or create LLMModel from OpenRouter cache
    # Try exact match first, then suffix match (seed data may omit provider prefix)
    llm = db.query(LLMModel).filter(LLMModel.model_id == model_id).first()
    if not llm:
        # Suffix match: "llama-3.3-70b-instruct" → "meta-llama/llama-3.3-70b-instruct"
        llm = db.query(LLMModel).filter(LLMModel.model_id.endswith(f"/{model_id}")).first()

    if not llm:
        cached = db.query(OpenRouterModelCache).filter(
            OpenRouterModelCache.model_id == model_id,
        ).first()
        if not cached:
            # Suffix match on OpenRouter cache
            cached = db.query(OpenRouterModelCache).filter(
                OpenRouterModelCache.model_id.endswith(f"/{model_id}"),
            ).first()
        if not cached:
            return {"success": False, "error": f"Model '{model_id}' not found in OpenRouter catalog"}

        llm = LLMModel(
            provider=cached.provider,
            model_id=cached.model_id,
            display_name=cached.display_name,
            description=cached.description,
            model_family=cached.provider,
            context_window=cached.context_length or 0,
            max_output_tokens=cached.max_completion_tokens or 0,
            input_cost_per_1k_tokens=(cached.prompt_cost or 0) * 1000,
            output_cost_per_1k_tokens=(cached.completion_cost or 0) * 1000,
            supports_functions=cached.supports_tools or False,
            supports_vision=cached.supports_vision or False,
            supports_streaming=cached.supports_streaming if cached.supports_streaming is not None else True,
            status="active",
            tier="aggregator",
            category=cached.category,
            tags=cached.tags or [],
            capabilities={},
            recommended_for=[],
            external_id=cached.model_id,
        )
        db.add(llm)
        db.flush()
        logger.info("[PlatformExecutor] Auto-created LLMModel from cache: %s", model_id)

    # Check for existing workspace install
    existing = (
        db.query(WorkspaceModel)
        .filter(
            WorkspaceModel.workspace_id == workspace_id,
            WorkspaceModel.model_id == llm.id,
        )
        .first()
    )

    if existing:
        if existing.is_active:
            return {
                "success": True,
                "already_installed": True,
                "model": {
                    "model_id": llm.model_id,
                    "display_name": llm.display_name,
                    "provider": llm.provider,
                },
                "message": f"Model '{llm.display_name}' is already installed.",
            }
        # Re-activate
        existing.is_active = True
        db.flush()
        logger.info("[PlatformExecutor] Re-activated model '%s' for workspace %s", model_id, workspace_id)
        return {
            "success": True,
            "reactivated": True,
            "model": {
                "model_id": llm.model_id,
                "display_name": llm.display_name,
                "provider": llm.provider,
            },
            "message": f"Model '{llm.display_name}' re-activated for this workspace.",
        }

    # Create new workspace install
    wm = WorkspaceModel(
        workspace_id=workspace_id,
        model_id=llm.id,
        source="marketplace",
    )
    db.add(wm)

    # Increment install_count
    llm.install_count = (llm.install_count or 0) + 1
    db.flush()

    logger.info(
        "[PlatformExecutor] Installed model '%s' for workspace %s",
        model_id, workspace_id,
    )

    return {
        "success": True,
        "model": {
            "model_id": llm.model_id,
            "display_name": llm.display_name,
            "provider": llm.provider,
        },
        "message": f"Model '{llm.display_name}' installed for this workspace.",
    }
