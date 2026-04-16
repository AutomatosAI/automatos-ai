"""
Cascading Marketplace Installs
==============================

When a marketplace agent or recipe is installed, this module auto-installs
all child dependencies: LLM models, skills, tools, and (for recipes)
referenced agents.

The only manual step left for users is connecting OAuth apps (Gmail, Slack, etc.).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class CascadeResult:
    """Collects everything that was auto-installed during a cascading install."""
    cloned_items: List[Dict[str, Any]] = field(default_factory=list)
    installed_dependencies: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def merge(self, other: "CascadeResult") -> None:
        self.cloned_items.extend(other.cloned_items)
        self.installed_dependencies.extend(other.installed_dependencies)
        self.warnings.extend(other.warnings)


# ---------------------------------------------------------------------------
# OAuth check helper
# ---------------------------------------------------------------------------

def check_oauth_requirements(db: Session, app_names: List[str]) -> Dict[str, bool]:
    """
    For each Composio app name, check whether it requires OAuth.
    Returns {app_name: True/False} where True means OAuth is required.
    """
    if not app_names:
        return {}

    from core.models.composio_cache import ComposioAppCache

    results: Dict[str, bool] = {}
    for name in app_names:
        upper_name = name.upper()
        cache_entry = (
            db.query(ComposioAppCache)
            .filter(
                (ComposioAppCache.app_name == upper_name)
                | (ComposioAppCache.app_slug == name.lower())
            )
            .first()
        )
        if cache_entry and cache_entry.auth_schemes:
            schemes = cache_entry.auth_schemes
            needs_oauth = any(
                s.upper() in ("OAUTH2", "OAUTH1", "OAUTH")
                for s in (schemes if isinstance(schemes, list) else [])
            )
            results[upper_name] = needs_oauth
        else:
            # Unknown app — assume it may need auth, warn to be safe
            results[upper_name] = True
    return results


# ---------------------------------------------------------------------------
# Clone helper (extracted from marketplace.py agent install)
# ---------------------------------------------------------------------------

def clone_agent_to_workspace(
    db: Session,
    workspace_id: UUID,
    marketplace_agent,  # Agent model instance
    user_id_int: Optional[int] = None,
):
    """
    Clone a marketplace agent into a workspace.
    Returns the cloned Agent instance (already flushed, has an id).
    """
    from core.models.core import Agent

    # Name collision check
    name_exists = db.query(Agent).filter(
        Agent.name == marketplace_agent.name,
        Agent.workspace_id == workspace_id,
        Agent.owner_type == 'workspace',
    ).first() is not None

    agent_name = f"{marketplace_agent.name}-copy" if name_exists else marketplace_agent.name

    cloned = Agent(
        name=agent_name,
        description=marketplace_agent.description,
        agent_type=marketplace_agent.agent_type,
        configuration=marketplace_agent.configuration,
        model_config=marketplace_agent.model_config,
        tags=marketplace_agent.tags,
        status=marketplace_agent.status,
        owner_type='workspace',
        owner_id=str(workspace_id),
        workspace_id=workspace_id,
        created_by_user_id=user_id_int,
        cloned_from_id=marketplace_agent.id,
        original_creator_id=marketplace_agent.original_creator_id,
        is_approved=True,
        is_featured=False,
        install_count=0,
        version=marketplace_agent.version,
    )
    db.add(cloned)
    db.flush()

    # Copy skills M2M relationship
    if marketplace_agent.skills:
        cloned.skills = list(marketplace_agent.skills)

    return cloned, agent_name


# ---------------------------------------------------------------------------
# Agent dependency cascade
# ---------------------------------------------------------------------------

async def cascade_agent_dependencies(
    db: Session,
    workspace_id: UUID,
    marketplace_agent,
    cloned_agent,
) -> CascadeResult:
    """
    After cloning a marketplace agent, auto-install its dependencies:
      1. LLM model → workspace
      2. Skills → workspace
      3. Plugins → workspace + assigned to cloned agent
      4. Tools → assigned to cloned agent (both runtime + display tables)
      5. OAuth warnings for tools that need connection
    """
    from .handlers_marketplace import install_model, install_plugin, install_skill

    result = CascadeResult()

    # --- 1. Install LLM model ---
    model_id = None
    if marketplace_agent.model_config and isinstance(marketplace_agent.model_config, dict):
        model_id = marketplace_agent.model_config.get("model_id")

    # Fallback: check configuration dict or marketplace_items metadata
    if not model_id:
        config = getattr(marketplace_agent, 'configuration', None)
        if isinstance(config, dict):
            model_id = config.get("model_id")
    if not model_id:
        # Last resort: look up in marketplace_items table by name
        mi_row = db.execute(
            text("SELECT metadata FROM marketplace_items WHERE type='agent' AND name=:name LIMIT 1"),
            {"name": marketplace_agent.name},
        ).first()
        if mi_row and mi_row[0]:
            mi_meta = mi_row[0] if isinstance(mi_row[0], dict) else {}
            model_id = mi_meta.get("model_id")

    if model_id:
        # Write model_config onto cloned agent so the model selector pre-selects it
        if not cloned_agent.model_config or not cloned_agent.model_config.get("model_id"):
            cloned_agent.model_config = {"model_id": model_id}
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(cloned_agent, "model_config")
            db.flush()

        try:
            model_result = await install_model(db, workspace_id, {"model_id": model_id})
            status = "already_installed" if model_result.get("already_installed") else "installed"
            if model_result.get("reactivated"):
                status = "reactivated"
            if not model_result.get("success"):
                status = "failed"
                result.warnings.append(f"Failed to install model {model_id}: {model_result.get('error', 'unknown')}")
            result.installed_dependencies.append({
                "type": "model",
                "name": model_id,
                "status": status,
            })
        except Exception as e:
            logger.warning("Cascade: failed to install model %s: %s", model_id, e)
            result.installed_dependencies.append({"type": "model", "name": model_id, "status": "failed"})
            result.warnings.append(f"Failed to install model {model_id}: {e}")

    # --- 2. Install skills to workspace ---
    if marketplace_agent.skills:
        for skill in marketplace_agent.skills:
            try:
                skill_result = await install_skill(db, workspace_id, {"skill_id": skill.id})
                status = "already_installed" if skill_result.get("already_enabled") else "installed"
                if not skill_result.get("success"):
                    status = "failed"
                result.installed_dependencies.append({
                    "type": "skill",
                    "name": skill.name,
                    "status": status,
                })
            except Exception as e:
                logger.warning("Cascade: failed to install skill %s: %s", skill.name, e)
                result.installed_dependencies.append({"type": "skill", "name": skill.name, "status": "failed"})

    # --- 3. Install plugins to workspace + assign to cloned agent ---
    await _cascade_plugins(db, workspace_id, marketplace_agent, cloned_agent, result, install_plugin)

    # --- 4. Copy tool assignments ---
    tool_names = _copy_tool_assignments(db, marketplace_agent.id, cloned_agent.id, workspace_id)

    for tool_name in tool_names:
        result.installed_dependencies.append({
            "type": "tool",
            "name": tool_name,
            "status": "assigned",
        })

    # --- 5. OAuth warnings ---
    if tool_names:
        oauth_map = check_oauth_requirements(db, tool_names)
        for tool_name, needs_oauth in oauth_map.items():
            if needs_oauth:
                # Update the dependency entry with oauth info
                for dep in result.installed_dependencies:
                    if dep.get("type") == "tool" and dep.get("name") == tool_name:
                        dep["oauth_required"] = True
                result.warnings.append(
                    f"{tool_name} requires an OAuth connection. "
                    f"Connect it at Settings \u2192 Integrations."
                )

    logger.info(
        "[CascadeInstaller] Agent '%s' (id=%d) → %d deps installed, %d warnings",
        cloned_agent.name, cloned_agent.id,
        len(result.installed_dependencies), len(result.warnings),
    )
    return result


async def _cascade_plugins(
    db: Session,
    workspace_id: UUID,
    marketplace_agent,
    cloned_agent,
    result: CascadeResult,
    install_plugin_fn,
) -> None:
    """
    Copy plugin assignments from marketplace agent to cloned agent.
    Also enables each plugin at the workspace level (idempotent).
    """
    from core.models.marketplace_plugins import AgentAssignedPlugin

    assigned = getattr(marketplace_agent, 'assigned_plugins', None)
    if not assigned:
        assigned = (
            db.query(AgentAssignedPlugin)
            .filter(AgentAssignedPlugin.agent_id == marketplace_agent.id)
            .all()
        )
    if not assigned:
        return

    for ap in assigned:
        plugin = getattr(ap, 'plugin', None)
        plugin_id = ap.plugin_id
        plugin_name = plugin.name if plugin else str(plugin_id)

        try:
            plugin_result = await install_plugin_fn(db, workspace_id, {"plugin_id": str(plugin_id)})
            if not plugin_result.get("success"):
                logger.warning("Cascade: plugin install returned failure for %s: %s", plugin_name, plugin_result.get("error"))
        except Exception as e:
            logger.warning("Cascade: failed to install plugin %s: %s", plugin_name, e)
            result.installed_dependencies.append({"type": "plugin", "name": plugin_name, "status": "failed"})
            continue

        # Assign plugin to cloned agent
        existing = (
            db.query(AgentAssignedPlugin)
            .filter(
                AgentAssignedPlugin.agent_id == cloned_agent.id,
                AgentAssignedPlugin.plugin_id == plugin_id,
            )
            .first()
        )
        if not existing:
            db.add(AgentAssignedPlugin(
                agent_id=cloned_agent.id,
                plugin_id=plugin_id,
                priority=getattr(ap, 'priority', 0) or 0,
            ))

        result.installed_dependencies.append({
            "type": "plugin",
            "name": plugin_name,
            "status": "installed",
        })

    db.flush()


def _copy_tool_assignments(
    db: Session,
    source_agent_id: int,
    target_agent_id: int,
    workspace_id: UUID,
) -> List[str]:
    """
    Copy tool assignments from a marketplace agent to a cloned workspace agent.
    Writes to THREE tables:
      - agent_app_assignments (runtime — used by get_tools_for_agent)
      - agent_tool_assignments (display — used by marketplace browse)
      - composio_connections (workspace — used by Settings > Integrations)
    Returns list of tool_id strings that were copied.
    """
    from core.composio.entity_manager import EntityManager
    from core.models.composio_cache import AgentAppAssignment

    # Read from legacy display table (what marketplace agents use)
    rows = db.execute(text("""
        SELECT tool_id FROM agent_tool_assignments
        WHERE agent_id = :agent_id AND enabled = true
    """), {"agent_id": source_agent_id}).fetchall()

    tool_names = [row[0] for row in rows if row[0]]
    if not tool_names:
        return []

    # Ensure workspace has a Composio entity for tool connections
    entity_manager = EntityManager(db)
    entity = entity_manager.get_or_create_entity(workspace_id)
    entity_id = entity["id"]

    for tool_name in tool_names:
        upper_name = tool_name.upper()

        # --- Runtime table (AgentAppAssignment) ---
        existing = (
            db.query(AgentAppAssignment)
            .filter(
                AgentAppAssignment.agent_id == target_agent_id,
                AgentAppAssignment.app_name == upper_name,
            )
            .first()
        )
        if not existing:
            db.add(AgentAppAssignment(
                agent_id=target_agent_id,
                app_name=upper_name,
                app_type="EXTERNAL",
                is_active=True,
            ))

        # --- Legacy display table (agent_tool_assignments) ---
        legacy_exists = db.execute(text("""
            SELECT 1 FROM agent_tool_assignments
            WHERE agent_id = :agent_id AND tool_id = :tool_id
        """), {"agent_id": target_agent_id, "tool_id": tool_name}).first()

        if not legacy_exists:
            db.execute(text("""
                INSERT INTO agent_tool_assignments (agent_id, tool_id, enabled, created_at, updated_at)
                VALUES (:agent_id, :tool_id, true, NOW(), NOW())
            """), {"agent_id": target_agent_id, "tool_id": tool_name})

        # --- Workspace entity connection (composio_connections) ---
        # "added" status = tool registered but OAuth not yet connected
        ws_conn_exists = db.execute(text("""
            SELECT 1 FROM composio_connections
            WHERE entity_id = :entity_id AND app_name = :app_name
        """), {"entity_id": entity_id, "app_name": upper_name}).first()

        if not ws_conn_exists:
            db.execute(text("""
                INSERT INTO composio_connections
                    (entity_id, app_name, status, connected_at, updated_at)
                VALUES
                    (:entity_id, :app_name, 'added', NOW(), NOW())
            """), {"entity_id": entity_id, "app_name": upper_name})

    db.flush()
    return [t.upper() for t in tool_names]


# ---------------------------------------------------------------------------
# Recipe dependency cascade
# ---------------------------------------------------------------------------

async def cascade_recipe_dependencies(
    db: Session,
    workspace_id: UUID,
    marketplace_recipe,
    cloned_recipe,
    user_id_int: Optional[int] = None,
) -> CascadeResult:
    """
    After cloning a marketplace recipe, auto-install its dependencies:
      1. Clone all recommended agents from marketplace
      2. Cascade each agent's dependencies (model, skills, tools)
      3. Remap recipe steps to point to cloned agent IDs
      4. Warn about OAuth connections for required_tools
    """
    from core.models.core import Agent

    result = CascadeResult()

    # --- 1. Clone recommended agents ---
    agent_name_to_cloned_id: Dict[str, int] = {}
    marketplace_id_to_cloned_id: Dict[int, int] = {}
    recommended = marketplace_recipe.recommended_agents or []

    # Also check metadata for suggested_agents (seed data uses this)
    if not recommended and hasattr(marketplace_recipe, 'template_definition'):
        tmpl_def = marketplace_recipe.template_definition
        if isinstance(tmpl_def, dict):
            recommended = tmpl_def.get("suggested_agents", [])
            if not recommended:
                recommended = tmpl_def.get("recommended_agents", [])

    # Also check the metadata field on the recipe
    if not recommended:
        metadata = getattr(marketplace_recipe, 'metadata', None)
        if isinstance(metadata, dict):
            recommended = metadata.get("suggested_agents", []) or metadata.get("recommended_agents", [])

    all_cascaded_tools: List[str] = []

    for agent_name in recommended:
        if not agent_name or not isinstance(agent_name, str):
            continue

        # Find marketplace agent by exact name (case-insensitive)
        from sqlalchemy import func as sa_func
        marketplace_agent = (
            db.query(Agent)
            .filter(
                sa_func.lower(Agent.name) == agent_name.lower(),
                Agent.owner_type == 'marketplace',
                Agent.is_approved == True,
            )
            .first()
        )

        if not marketplace_agent:
            result.warnings.append(
                f"Recommended agent '{agent_name}' not found in marketplace — skipped."
            )
            continue

        try:
            cloned_agent, final_name = clone_agent_to_workspace(
                db, workspace_id, marketplace_agent, user_id_int,
            )
            agent_name_to_cloned_id[marketplace_agent.name] = cloned_agent.id
            marketplace_id_to_cloned_id[marketplace_agent.id] = cloned_agent.id

            result.cloned_items.append({
                "type": "agent",
                "name": final_name,
                "id": cloned_agent.id,
            })

            # Increment install count on marketplace agent
            marketplace_agent.install_count = (marketplace_agent.install_count or 0) + 1

            # Cascade agent's own dependencies
            agent_cascade = await cascade_agent_dependencies(
                db, workspace_id, marketplace_agent, cloned_agent,
            )
            result.merge(agent_cascade)

            # Track tools for OAuth dedup
            for dep in agent_cascade.installed_dependencies:
                if dep.get("type") == "tool":
                    all_cascaded_tools.append(dep["name"])

        except Exception as e:
            logger.warning("Cascade: failed to clone agent '%s': %s", agent_name, e)
            result.warnings.append(f"Failed to install agent '{agent_name}': {e}")

    # --- 2. Remap recipe steps ---
    if (agent_name_to_cloned_id or marketplace_id_to_cloned_id) and cloned_recipe.steps:
        _remap_recipe_steps(db, cloned_recipe, agent_name_to_cloned_id, marketplace_id_to_cloned_id)

    # --- 3. OAuth warnings for recipe-level required_tools not covered by agents ---
    recipe_tools = marketplace_recipe.required_tools or []

    # Also check metadata
    if not recipe_tools:
        metadata = getattr(marketplace_recipe, 'metadata', None)
        if isinstance(metadata, dict):
            recipe_tools = metadata.get("required_tools", [])

    uncovered_tools = [
        t.upper() for t in recipe_tools
        if t.upper() not in all_cascaded_tools
    ]
    if uncovered_tools:
        oauth_map = check_oauth_requirements(db, uncovered_tools)
        for tool_name, needs_oauth in oauth_map.items():
            if needs_oauth:
                # Only warn if not already warned by agent cascade
                existing_warnings = {w.split(" requires")[0] for w in result.warnings if "requires" in w}
                if tool_name not in existing_warnings:
                    result.warnings.append(
                        f"{tool_name} requires an OAuth connection. "
                        f"Connect it at Settings \u2192 Integrations."
                    )

    logger.info(
        "[CascadeInstaller] Recipe '%s' → %d agents cloned, %d deps, %d warnings",
        cloned_recipe.name, len(agent_name_to_cloned_id),
        len(result.installed_dependencies), len(result.warnings),
    )
    return result


def _remap_recipe_steps(
    db: Session,
    cloned_recipe,
    agent_name_to_id: Dict[str, int],
    marketplace_id_to_cloned_id: Optional[Dict[int, int]] = None,
) -> None:
    """
    Walk the recipe's steps and remap agent references to newly cloned IDs.

    Steps may reference agents by:
      - agent_name (string) — matched by name (case-insensitive)
      - agent_id (int) — matched by marketplace agent ID → cloned ID map
      - prompt_template text — fuzzy match agent name in prompt text
      - (none) — round-robin assign from cloned agents list
    """
    from sqlalchemy.orm.attributes import flag_modified

    steps = cloned_recipe.steps
    if not isinstance(steps, list):
        return

    id_map = marketplace_id_to_cloned_id or {}
    cloned_ids = list(agent_name_to_id.values())
    changed = False

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue

        matched = False

        # 1. Match by agent_name (exact, case-insensitive)
        step_agent_name = step.get("agent_name")
        if step_agent_name:
            for original_name, cloned_id in agent_name_to_id.items():
                if step_agent_name.lower() == original_name.lower():
                    step["agent_id"] = cloned_id
                    changed = True
                    matched = True
                    break

        # 2. Fallback: match by agent_id (marketplace ID → cloned ID)
        if not matched and id_map:
            step_agent_id = step.get("agent_id")
            if step_agent_id and step_agent_id in id_map:
                step["agent_id"] = id_map[step_agent_id]
                changed = True
                matched = True

        # 3. Fallback: fuzzy-match agent name in prompt_template text
        if not matched and not step.get("agent_id") and agent_name_to_id:
            prompt = (step.get("prompt_template") or "").lower()
            if prompt:
                for original_name, cloned_id in agent_name_to_id.items():
                    if original_name.lower() in prompt:
                        step["agent_id"] = cloned_id
                        changed = True
                        matched = True
                        break

        # 4. Last resort: round-robin assign from cloned agents
        if not matched and not step.get("agent_id") and cloned_ids:
            step["agent_id"] = cloned_ids[idx % len(cloned_ids)]
            changed = True

    if changed:
        cloned_recipe.steps = steps
        flag_modified(cloned_recipe, "steps")
        db.flush()
