"""
Platform Action Executor (PRD-64)
==================================

Executes platform actions by querying the database directly.
Each handler method corresponds to an ActionDefinition in platform_actions.py.

All queries are workspace-scoped for multi-tenant isolation.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

logger = logging.getLogger(__name__)


class PlatformActionExecutor:
    """
    Executes platform actions using direct database queries.
    Workspace-scoped for multi-tenant isolation.
    """

    def __init__(self, db: Session, workspace_id: UUID):
        self.db = db
        self.workspace_id = workspace_id
        self._handlers = {
            # Read actions
            "platform_list_agents": self._list_agents,
            "platform_get_agent": self._get_agent,
            "platform_list_recipes": self._list_recipes,
            "platform_get_recipe": self._get_recipe,
            "platform_get_llm_usage": self._get_llm_usage,
            "platform_get_cost_breakdown": self._get_cost_breakdown,
            "platform_list_documents": self._list_documents,
            "platform_get_workspace_info": self._get_workspace_info,
            "platform_get_memory_stats": self._get_memory_stats,
            "platform_list_connected_apps": self._list_connected_apps,
            # Write actions
            "platform_create_agent": self._create_agent,
            "platform_update_agent": self._update_agent,
            "platform_create_recipe": self._create_recipe,
            "platform_update_recipe": self._update_recipe,
            "platform_add_recipe_step": self._add_recipe_step,
            "platform_update_recipe_step": self._update_recipe_step,
            "platform_delete_recipe_step": self._delete_recipe_step,
            "platform_store_memory": self._store_memory,
            "platform_delete_agent": self._delete_agent,
            # Infrastructure / observability
            "platform_get_logs": self._get_logs,
            "platform_list_services": self._list_services,
            # Visibility / discovery
            "platform_list_tools": self._list_tools,
            "platform_list_llms": self._list_llms,
            "platform_list_datasources": self._list_datasources,
            "platform_workspace_stats": self._workspace_stats,
            # Self-management
            "platform_execute_recipe": self._execute_recipe,
            "platform_get_recipe_execution": self._get_recipe_execution,
            "platform_get_system_health": self._get_system_health,
            "platform_delete_document": self._delete_document,
            "platform_reprocess_document": self._reprocess_document,
            "platform_delete_recipe": self._delete_recipe,
            "platform_get_activity_feed": self._get_activity_feed,
        }

    async def execute(self, action_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a platform action by name with permission checking."""
        handler = self._handlers.get(action_name)
        if not handler:
            return {"success": False, "error": f"Unknown platform action: {action_name}"}

        # Permission check for write/destructive actions (fail-closed)
        try:
            from modules.tools.discovery import get_action_registry
            action_def = get_action_registry().get(action_name)
            if action_def and action_def.requires_confirmation:
                return {
                    "success": False,
                    "requires_confirmation": True,
                    "action": action_name,
                    "permission_level": action_def.permission_level,
                    "message": (
                        f"This action ({action_def.permission_level}) requires confirmation. "
                        f"Action: {action_name} — {action_def.description[:100]}"
                    ),
                    "params": params,
                }
        except Exception as e:
            # Fail-closed: if we can't verify permissions, require confirmation
            logger.warning(
                "[PlatformExecutor] Registry lookup failed for %s: %s — requiring confirmation",
                action_name, e,
            )
            return {
                "success": False,
                "requires_confirmation": True,
                "action": action_name,
                "permission_level": "unknown",
                "message": (
                    f"Could not verify permissions for '{action_name}'. "
                    "Confirmation required for safety."
                ),
                "params": params,
            }

        # Rate limit write/destructive actions
        if action_def and action_def.permission_level in ("write", "destructive"):
            try:
                from core.security.rate_limiter import check_rate_limit
                await check_rate_limit(str(self.workspace_id), "platform_write")
            except HTTPException as e:
                if e.status_code == 429:
                    return {
                        "success": False,
                        "rate_limited": True,
                        "error": "Rate limit exceeded: max 10 write actions per minute. Try again shortly.",
                    }
                raise
            except Exception:
                pass  # Fail open

        try:
            return await handler(params)
        except Exception as e:
            logger.error(f"[PlatformExecutor] {action_name} failed: {e}", exc_info=True)
            try:
                self.db.rollback()
            except Exception:
                pass
            return {"success": False, "error": f"Action '{action_name}' failed"}

    # ── Agent Handlers ──────────────────────────────────────────────

    async def _list_agents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models import Agent
        from core.models.composio_cache import AgentAppAssignment

        query = self.db.query(Agent).filter(Agent.workspace_id == self.workspace_id)

        status_filter = params.get("status_filter", "all")
        if status_filter != "all":
            query = query.filter(Agent.status == status_filter)

        agents = query.order_by(Agent.id).all()

        # Batch-load tool counts for all agents in one query
        tool_counts = {}
        try:
            rows = (
                self.db.query(
                    AgentAppAssignment.agent_id,
                    func.count(AgentAppAssignment.id).label("cnt"),
                )
                .filter(AgentAppAssignment.is_active == True)
                .group_by(AgentAppAssignment.agent_id)
                .all()
            )
            tool_counts = {r.agent_id: r.cnt for r in rows}
        except Exception:
            pass

        agent_list = []
        for a in agents:
            mc = a.model_config or {}
            cfg = a.configuration or {}
            agent_list.append({
                "id": a.id,
                "name": a.name,
                "type": a.agent_type,
                "status": a.status,
                "description": (a.description or "")[:200],
                "model_id": mc.get("model_id") or cfg.get("model") or cfg.get("llm_model"),
                "provider": mc.get("provider") or cfg.get("provider"),
                "temperature": mc.get("temperature"),
                "tools_count": tool_counts.get(a.id, 0),
                "has_persona": bool(a.custom_persona_prompt),
                "tags": a.tags or [],
                "created_at": a.created_at.isoformat() if a.created_at else None,
            })

        return {
            "success": True,
            "agents": agent_list,
            "count": len(agent_list),
        }

    async def _get_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models import Agent

        agent_id = params.get("agent_id")
        agent_name = params.get("agent_name")

        query = self.db.query(Agent).filter(Agent.workspace_id == self.workspace_id)
        if agent_id:
            query = query.filter(Agent.id == agent_id)
        elif agent_name:
            query = query.filter(Agent.name.ilike(f"%{agent_name}%"))
        else:
            return {"success": False, "error": "Provide agent_name or agent_id"}

        agent = query.first()
        if not agent:
            return {"success": False, "error": "Agent not found"}

        # Get assigned tools count
        tool_count = 0
        try:
            from core.models.composio_cache import AgentAppAssignment
            tool_count = (
                self.db.query(AgentAppAssignment)
                .filter(AgentAppAssignment.agent_id == agent.id, AgentAppAssignment.is_active == True)
                .count()
            )
        except Exception:
            pass

        mc = agent.model_config or {}
        config = agent.configuration or {}
        return {
            "success": True,
            "agent": {
                "id": agent.id,
                "name": agent.name,
                "type": agent.agent_type,
                "status": agent.status,
                "description": agent.description,
                "model_id": mc.get("model_id") or config.get("model") or config.get("llm_model"),
                "provider": mc.get("provider") or config.get("provider") or config.get("llm_provider"),
                "temperature": mc.get("temperature"),
                "has_system_prompt": bool(agent.custom_persona_prompt),
                "system_prompt_preview": (agent.custom_persona_prompt or "")[:200] or None,
                "assigned_tools": tool_count,
                "tags": agent.tags or [],
                "created_at": agent.created_at.isoformat() if agent.created_at else None,
                "updated_at": agent.updated_at.isoformat() if agent.updated_at else None,
            },
        }

    # ── Recipe Handlers ─────────────────────────────────────────────

    async def _list_recipes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models.core import WorkflowTemplate

        query = self.db.query(WorkflowTemplate).filter(
            WorkflowTemplate.workspace_id == self.workspace_id
        )

        status_filter = params.get("status_filter", "all")
        if status_filter != "all" and hasattr(WorkflowTemplate, "status"):
            query = query.filter(WorkflowTemplate.status == status_filter)

        recipes = query.order_by(WorkflowTemplate.id).all()

        return {
            "success": True,
            "recipes": [
                {
                    "id": r.id,
                    "name": r.name,
                    "template_id": r.template_id,
                    "description": (r.description or "")[:200],
                    "tags": r.tags or [],
                    "created_at": r.created_at.isoformat() if hasattr(r, "created_at") and r.created_at else None,
                }
                for r in recipes
            ],
            "count": len(recipes),
        }

    async def _get_recipe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models.core import WorkflowTemplate

        recipe_id = params.get("recipe_id")
        recipe_name = params.get("recipe_name")

        query = self.db.query(WorkflowTemplate).filter(
            WorkflowTemplate.workspace_id == self.workspace_id
        )
        if recipe_id:
            query = query.filter(WorkflowTemplate.id == recipe_id)
        elif recipe_name:
            query = query.filter(WorkflowTemplate.name.ilike(f"%{recipe_name}%"))
        else:
            return {"success": False, "error": "Provide recipe_name or recipe_id"}

        recipe = query.first()
        if not recipe:
            return {"success": False, "error": "Recipe not found"}

        # Count executions
        exec_count = 0
        try:
            from core.models.core import RecipeExecution
            exec_count = (
                self.db.query(RecipeExecution)
                .filter(RecipeExecution.recipe_id == recipe.id)
                .count()
            )
        except Exception:
            pass

        steps = recipe.steps or []

        return {
            "success": True,
            "recipe": {
                "id": recipe.id,
                "name": recipe.name,
                "template_id": recipe.template_id,
                "description": recipe.description,
                "tags": recipe.tags or [],
                "step_count": len(steps),
                "steps": [
                    {
                        "index": i,
                        "prompt_preview": (s.get("prompt_template", "") or "")[:120],
                        "agent_id": s.get("agent_id"),
                        "error_handling": s.get("error_handling", "stop"),
                        "output_key": s.get("output_key"),
                    }
                    for i, s in enumerate(steps[:10])
                    if isinstance(s, dict)
                ],
                "total_executions": exec_count,
            },
        }

    # ── Analytics Handlers ──────────────────────────────────────────

    async def _get_llm_usage(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models.core import LLMUsage

        days = params.get("days", 30)
        since = datetime.now(timezone.utc) - timedelta(days=days)

        rows = (
            self.db.query(
                LLMUsage.model_id,
                LLMUsage.provider,
                func.count(LLMUsage.id).label("request_count"),
                func.sum(LLMUsage.input_tokens).label("total_input_tokens"),
                func.sum(LLMUsage.output_tokens).label("total_output_tokens"),
                func.sum(LLMUsage.total_tokens).label("total_tokens"),
            )
            .filter(
                LLMUsage.workspace_id == self.workspace_id,
                LLMUsage.created_at >= since,
            )
            .group_by(LLMUsage.model_id, LLMUsage.provider)
            .all()
        )

        models = []
        total_requests = 0
        total_tokens = 0
        for row in rows:
            models.append({
                "model": row.model_id,
                "provider": row.provider,
                "requests": row.request_count,
                "input_tokens": row.total_input_tokens or 0,
                "output_tokens": row.total_output_tokens or 0,
                "total_tokens": row.total_tokens or 0,
            })
            total_requests += row.request_count
            total_tokens += (row.total_tokens or 0)

        return {
            "success": True,
            "period_days": days,
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "by_model": models,
        }

    async def _get_cost_breakdown(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models.core import LLMUsage

        days = params.get("days", 30)
        group_by = params.get("group_by", "model")
        since = datetime.now(timezone.utc) - timedelta(days=days)

        if group_by == "agent":
            group_col = LLMUsage.agent_id
        elif group_by == "day":
            group_col = func.date(LLMUsage.created_at)
        else:
            group_col = LLMUsage.model_id

        rows = (
            self.db.query(
                group_col.label("group_key"),
                func.sum(LLMUsage.total_cost).label("total_cost"),
                func.sum(LLMUsage.input_cost).label("input_cost"),
                func.sum(LLMUsage.output_cost).label("output_cost"),
                func.count(LLMUsage.id).label("request_count"),
            )
            .filter(
                LLMUsage.workspace_id == self.workspace_id,
                LLMUsage.created_at >= since,
            )
            .group_by(group_col)
            .order_by(func.sum(LLMUsage.total_cost).desc())
            .all()
        )

        breakdown = []
        total_cost = 0.0
        for row in rows:
            key = str(row.group_key) if row.group_key is not None else "unknown"
            cost = float(row.total_cost or 0)
            breakdown.append({
                group_by: key,
                "total_cost": round(cost, 6),
                "input_cost": round(float(row.input_cost or 0), 6),
                "output_cost": round(float(row.output_cost or 0), 6),
                "requests": row.request_count,
            })
            total_cost += cost

        return {
            "success": True,
            "period_days": days,
            "group_by": group_by,
            "total_cost": round(total_cost, 6),
            "breakdown": breakdown,
        }

    # ── Document Handlers ───────────────────────────────────────────

    async def _list_documents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models import Document

        limit = min(params.get("limit", 50), 200)

        docs = (
            self.db.query(Document)
            .filter(Document.workspace_id == self.workspace_id)
            .order_by(Document.upload_date.desc())
            .limit(limit)
            .all()
        )

        return {
            "success": True,
            "documents": [
                {
                    "id": d.id,
                    "filename": d.original_filename or d.filename,
                    "file_type": d.file_type,
                    "file_size": d.file_size,
                    "status": d.status,
                    "chunk_count": d.chunk_count or 0,
                    "uploaded_at": d.upload_date.isoformat() if d.upload_date else None,
                }
                for d in docs
            ],
            "count": len(docs),
        }

    # ── Workspace Handlers ──────────────────────────────────────────

    async def _get_workspace_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models.workspaces import Workspace

        ws = self.db.query(Workspace).filter(Workspace.id == self.workspace_id).first()
        if not ws:
            return {"success": False, "error": "Workspace not found"}

        # Count resources
        from core.models import Agent, Document
        agent_count = self.db.query(Agent).filter(Agent.workspace_id == self.workspace_id).count()
        doc_count = self.db.query(Document).filter(Document.workspace_id == self.workspace_id).count()

        return {
            "success": True,
            "workspace": {
                "id": str(ws.id),
                "name": ws.name,
                "plan": ws.plan,
                "is_personal": ws.is_personal,
                "agent_count": agent_count,
                "document_count": doc_count,
                "created_at": ws.created_at.isoformat() if ws.created_at else None,
            },
        }

    # ── Memory Handlers ─────────────────────────────────────────────

    async def _get_memory_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get memory stats. Queries mem0 API if available, otherwise returns basic info."""
        try:
            import httpx
            from config import config

            mem0_url = config.MEM0_API_URL
            if not mem0_url:
                return {"success": True, "message": "Memory service not configured", "total_memories": 0}

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{mem0_url}/v1/memories/",
                    params={"user_id": str(self.workspace_id)},
                    headers={"Authorization": f"Bearer {config.MEM0_API_KEY}"} if config.MEM0_API_KEY else {},
                )
                if resp.status_code == 200:
                    memories = resp.json()
                    mem_list = memories if isinstance(memories, list) else memories.get("results", [])
                    return {
                        "success": True,
                        "total_memories": len(mem_list),
                        "workspace_id": str(self.workspace_id),
                    }
        except Exception as e:
            logger.debug(f"[PlatformExecutor] Memory stats unavailable: {e}")

        return {
            "success": True,
            "total_memories": 0,
            "message": "Memory service unavailable or not configured",
        }

    # ── Integration Handlers ────────────────────────────────────────

    async def _list_connected_apps(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models import Agent
        from core.models.composio_cache import AgentAppAssignment

        assignments = (
            self.db.query(
                AgentAppAssignment.app_name,
                AgentAppAssignment.app_type,
                func.count(AgentAppAssignment.id).label("agent_count"),
            )
            .filter(AgentAppAssignment.is_active == True)
            .join(Agent, AgentAppAssignment.agent_id == Agent.id)
            .filter(Agent.workspace_id == self.workspace_id)
            .group_by(AgentAppAssignment.app_name, AgentAppAssignment.app_type)
            .all()
        )

        return {
            "success": True,
            "connected_apps": [
                {
                    "app_name": a.app_name,
                    "app_type": a.app_type,
                    "assigned_to_agents": a.agent_count,
                }
                for a in assignments
            ],
            "count": len(assignments),
        }

    # ══════════════════════════════════════════════════════════════════
    # WRITE ACTION HANDLERS (PRD-64 Phase 2)
    # ══════════════════════════════════════════════════════════════════

    async def _create_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models import Agent

        name = params.get("name")
        if not name:
            return {"success": False, "error": "Missing required parameter: name"}

        agent_type = params.get("agent_type", "chatbot")
        description = params.get("description", "")
        model_id = params.get("model_id") or params.get("model")  # back-compat
        system_prompt = params.get("system_prompt")
        temperature = params.get("temperature")
        tags = params.get("tags")

        # Build model_config (the field the chat service actually reads)
        model_config: Dict[str, Any] = {
            "provider": "openai",
            "model_id": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "fallback_model_id": None,
        }
        if model_id:
            model_config["model_id"] = model_id
            # Infer provider from model name
            if "claude" in model_id.lower() or "anthropic" in model_id.lower():
                model_config["provider"] = "anthropic"
            elif "gemini" in model_id.lower():
                model_config["provider"] = "google"
            elif "llama" in model_id.lower() or "mixtral" in model_id.lower():
                model_config["provider"] = "groq"
        if temperature is not None:
            model_config["temperature"] = max(0.0, min(2.0, float(temperature)))

        agent = Agent(
            name=name,
            agent_type=agent_type,
            description=description,
            status="active",
            configuration={},
            model_config=model_config,
            workspace_id=self.workspace_id,
            created_by="platform",
            owner_type="workspace",
            owner_id=str(self.workspace_id),
        )

        # System prompt → custom_persona_prompt
        if system_prompt:
            agent.custom_persona_prompt = system_prompt
            agent.use_custom_persona = True

        # Tags
        if tags:
            agent.tags = tags

        self.db.add(agent)
        self.db.flush()  # Get the ID without committing (caller commits)

        logger.info(f"[PlatformExecutor] Created agent '{name}' (id={agent.id}) in workspace {self.workspace_id}")

        return {
            "success": True,
            "agent": {
                "id": agent.id,
                "name": agent.name,
                "type": agent.agent_type,
                "status": agent.status,
                "description": agent.description,
                "model_id": model_config["model_id"],
                "provider": model_config["provider"],
                "temperature": model_config["temperature"],
                "has_system_prompt": bool(system_prompt),
                "tags": agent.tags or [],
            },
            "message": f"Agent '{name}' created successfully with ID {agent.id}.",
        }

    async def _update_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models import Agent

        agent_id = params.get("agent_id")
        agent_name = params.get("agent_name")

        query = self.db.query(Agent).filter(Agent.workspace_id == self.workspace_id)
        if agent_id:
            query = query.filter(Agent.id == agent_id)
        elif agent_name:
            query = query.filter(Agent.name.ilike(f"%{agent_name}%"))
        else:
            return {"success": False, "error": "Provide agent_name or agent_id"}

        agent = query.first()
        if not agent:
            return {"success": False, "error": "Agent not found"}

        changes = []

        # Basic fields
        if params.get("new_name"):
            agent.name = params["new_name"]
            changes.append(f"name → '{params['new_name']}'")
        if params.get("description") is not None:
            agent.description = params["description"]
            changes.append("description updated")
        if params.get("status"):
            agent.status = params["status"]
            changes.append(f"status → '{params['status']}'")

        # Model configuration
        model_id = params.get("model_id")
        temperature = params.get("temperature")
        if model_id or temperature is not None:
            mc = dict(agent.model_config or {})
            if model_id:
                mc["model_id"] = model_id
                # Infer provider
                if "claude" in model_id.lower() or "anthropic" in model_id.lower():
                    mc["provider"] = "anthropic"
                elif "gemini" in model_id.lower():
                    mc["provider"] = "google"
                elif "llama" in model_id.lower() or "mixtral" in model_id.lower():
                    mc["provider"] = "groq"
                else:
                    mc["provider"] = "openai"
                changes.append(f"model → '{model_id}'")
            if temperature is not None:
                mc["temperature"] = max(0.0, min(2.0, float(temperature)))
                changes.append(f"temperature → {mc['temperature']}")
            agent.model_config = mc

        # System prompt
        system_prompt = params.get("system_prompt")
        if system_prompt is not None:
            agent.custom_persona_prompt = system_prompt
            agent.use_custom_persona = True
            changes.append("system prompt updated")

        # Tags
        tags = params.get("tags")
        if tags is not None:
            agent.tags = tags
            changes.append(f"tags → {tags}")

        if not changes:
            return {"success": True, "message": "No changes specified", "agent_id": agent.id}

        self.db.flush()
        logger.info(f"[PlatformExecutor] Updated agent {agent.id}: {', '.join(changes)}")

        return {
            "success": True,
            "agent_id": agent.id,
            "changes": changes,
            "message": f"Agent '{agent.name}' updated: {', '.join(changes)}",
        }

    async def _create_recipe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models.core import WorkflowTemplate
        import uuid

        name = params.get("name")
        description = params.get("description")
        if not name or not description:
            return {"success": False, "error": "Missing required: name and description"}

        tags = params.get("tags", [])
        template_id = f"custom-{uuid.uuid4().hex[:8]}"

        recipe = WorkflowTemplate(
            name=name,
            template_id=template_id,
            description=description,
            workspace_id=self.workspace_id,
            owner_type="workspace",
            owner_id=str(self.workspace_id),
            created_by="platform",
            tags=tags,
            template_definition={"steps": [], "agents": [], "config": {}, "variables": []},
        )
        self.db.add(recipe)
        self.db.flush()

        logger.info(f"[PlatformExecutor] Created recipe '{name}' (id={recipe.id}) in workspace {self.workspace_id}")

        return {
            "success": True,
            "recipe": {
                "id": recipe.id,
                "name": recipe.name,
                "template_id": recipe.template_id,
                "description": recipe.description,
            },
            "message": f"Recipe '{name}' created successfully. Add steps via the recipe editor.",
        }

    async def _update_recipe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models.core import WorkflowTemplate

        recipe_id = params.get("recipe_id")
        if not recipe_id:
            return {"success": False, "error": "Missing required parameter: recipe_id"}

        recipe = (
            self.db.query(WorkflowTemplate)
            .filter(
                WorkflowTemplate.id == recipe_id,
                WorkflowTemplate.workspace_id == self.workspace_id,
            )
            .first()
        )
        if not recipe:
            return {"success": False, "error": "Recipe not found"}

        changes = []
        if params.get("name"):
            recipe.name = params["name"]
            changes.append(f"name → '{params['name']}'")
        if params.get("description") is not None:
            recipe.description = params["description"]
            changes.append("description updated")
        if params.get("tags") is not None:
            recipe.tags = params["tags"]
            changes.append(f"tags → {params['tags']}")
        if params.get("execution_config") is not None:
            recipe.execution_config = params["execution_config"]
            changes.append("execution_config updated")
        if params.get("schedule_config") is not None:
            recipe.schedule_config = params["schedule_config"]
            changes.append("schedule_config updated")

        if not changes:
            return {"success": True, "message": "No changes specified", "recipe_id": recipe.id}

        self.db.flush()
        logger.info(f"[PlatformExecutor] Updated recipe {recipe.id}: {', '.join(changes)}")

        return {
            "success": True,
            "recipe_id": recipe.id,
            "changes": changes,
            "message": f"Recipe '{recipe.name}' updated: {', '.join(changes)}",
        }

    async def _add_recipe_step(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models.core import WorkflowTemplate
        from sqlalchemy.orm.attributes import flag_modified
        import uuid

        recipe_id = params.get("recipe_id")
        prompt_template = params.get("prompt_template")
        if not recipe_id or not prompt_template:
            return {"success": False, "error": "Missing required: recipe_id and prompt_template"}

        recipe = (
            self.db.query(WorkflowTemplate)
            .filter(
                WorkflowTemplate.id == recipe_id,
                WorkflowTemplate.workspace_id == self.workspace_id,
            )
            .first()
        )
        if not recipe:
            return {"success": False, "error": "Recipe not found"}

        steps = list(recipe.steps or [])
        order = params.get("order", len(steps))

        step = {
            "step_id": uuid.uuid4().hex[:12],
            "step_number": order + 1,
            "prompt_template": prompt_template,
            "agent_id": params.get("agent_id"),
            "error_handling": params.get("error_handling", "stop"),
            "output_key": params.get("output_key"),
        }

        if order >= len(steps):
            steps.append(step)
        else:
            steps.insert(order, step)

        # Re-number all steps
        for i, s in enumerate(steps):
            s["step_number"] = i + 1

        recipe.steps = steps
        flag_modified(recipe, "steps")
        self.db.flush()

        logger.info(f"[PlatformExecutor] Added step to recipe {recipe.id} (now {len(steps)} steps)")

        return {
            "success": True,
            "recipe_id": recipe.id,
            "step_index": order if order < len(steps) else len(steps) - 1,
            "total_steps": len(steps),
            "message": f"Step added to recipe '{recipe.name}' (now {len(steps)} steps).",
        }

    async def _update_recipe_step(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models.core import WorkflowTemplate
        from sqlalchemy.orm.attributes import flag_modified

        recipe_id = params.get("recipe_id")
        step_index = params.get("step_index")
        if recipe_id is None or step_index is None:
            return {"success": False, "error": "Missing required: recipe_id and step_index"}

        recipe = (
            self.db.query(WorkflowTemplate)
            .filter(
                WorkflowTemplate.id == recipe_id,
                WorkflowTemplate.workspace_id == self.workspace_id,
            )
            .first()
        )
        if not recipe:
            return {"success": False, "error": "Recipe not found"}

        steps = list(recipe.steps or [])
        if step_index < 0 or step_index >= len(steps):
            return {"success": False, "error": f"step_index {step_index} out of range (0-{len(steps)-1})"}

        step = steps[step_index]
        changes = []

        for field in ("prompt_template", "agent_id", "order", "error_handling", "output_key"):
            if field in params and params[field] is not None:
                step[field] = params[field]
                changes.append(f"{field} updated")

        if not changes:
            return {"success": True, "message": "No changes specified", "recipe_id": recipe.id}

        recipe.steps = steps
        flag_modified(recipe, "steps")
        self.db.flush()

        logger.info(f"[PlatformExecutor] Updated step {step_index} of recipe {recipe.id}: {', '.join(changes)}")

        return {
            "success": True,
            "recipe_id": recipe.id,
            "step_index": step_index,
            "changes": changes,
            "message": f"Step {step_index} of '{recipe.name}' updated: {', '.join(changes)}",
        }

    async def _delete_recipe_step(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models.core import WorkflowTemplate
        from sqlalchemy.orm.attributes import flag_modified

        recipe_id = params.get("recipe_id")
        step_index = params.get("step_index")
        if recipe_id is None or step_index is None:
            return {"success": False, "error": "Missing required: recipe_id and step_index"}

        recipe = (
            self.db.query(WorkflowTemplate)
            .filter(
                WorkflowTemplate.id == recipe_id,
                WorkflowTemplate.workspace_id == self.workspace_id,
            )
            .first()
        )
        if not recipe:
            return {"success": False, "error": "Recipe not found"}

        steps = list(recipe.steps or [])
        if step_index < 0 or step_index >= len(steps):
            return {"success": False, "error": f"step_index {step_index} out of range (0-{len(steps)-1})"}

        removed = steps.pop(step_index)

        # Re-number remaining steps
        for i, s in enumerate(steps):
            s["step_number"] = i + 1

        recipe.steps = steps
        flag_modified(recipe, "steps")
        self.db.flush()

        logger.info(f"[PlatformExecutor] Deleted step {step_index} from recipe {recipe.id} (now {len(steps)} steps)")

        return {
            "success": True,
            "recipe_id": recipe.id,
            "deleted_step_index": step_index,
            "remaining_steps": len(steps),
            "message": f"Step {step_index} removed from '{recipe.name}' ({len(steps)} steps remaining).",
        }

    async def _store_memory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        content = params.get("content")
        if not content:
            return {"success": False, "error": "Missing required parameter: content"}

        try:
            import httpx
            from config import config

            mem0_url = config.MEM0_API_URL
            if not mem0_url:
                return {"success": False, "error": "Memory service not configured"}

            headers = {}
            if config.MEM0_API_KEY:
                headers["Authorization"] = f"Bearer {config.MEM0_API_KEY}"

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{mem0_url}/v1/memories/",
                    json={
                        "messages": [{"role": "user", "content": content}],
                        "user_id": str(self.workspace_id),
                    },
                    headers=headers,
                )
                if resp.status_code in (200, 201):
                    return {
                        "success": True,
                        "message": f"Stored in memory: '{content[:100]}...'",
                    }
                else:
                    return {"success": False, "error": f"Memory API returned {resp.status_code}"}
        except Exception as e:
            logger.warning(f"[PlatformExecutor] Memory store failed: {e}")
            return {"success": False, "error": f"Memory service error: {e}"}

    async def _delete_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete an agent. Requires confirmation (handled by execute())."""
        from core.models import Agent

        agent_id = params.get("agent_id")
        agent_name = params.get("agent_name")

        query = self.db.query(Agent).filter(Agent.workspace_id == self.workspace_id)
        if agent_id:
            query = query.filter(Agent.id == agent_id)
        elif agent_name:
            query = query.filter(Agent.name.ilike(f"%{agent_name}%"))
        else:
            return {"success": False, "error": "Provide agent_name or agent_id"}

        agent = query.first()
        if not agent:
            return {"success": False, "error": "Agent not found"}

        agent_info = {"id": agent.id, "name": agent.name}
        self.db.delete(agent)
        self.db.flush()

        logger.info(f"[PlatformExecutor] Deleted agent {agent_info}")

        return {
            "success": True,
            "deleted_agent": agent_info,
            "message": f"Agent '{agent_info['name']}' (ID {agent_info['id']}) has been deleted.",
        }

    # ── Infrastructure / Observability ─────────────────────────────────

    async def _get_logs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch deployment logs from a Railway service."""
        from core.railway_client import RailwayClient

        client = RailwayClient()
        if not client.is_configured:
            return {
                "success": False,
                "error": "Railway API not configured. Set RAILWAY_API_TOKEN and RAILWAY_PROJECT_ID env vars.",
            }

        service_name = params.get("service", "")
        if not service_name:
            return {"success": False, "error": "service parameter is required"}

        # Special case: "list" returns available services
        if service_name.lower() == "list":
            return await self._list_services(params)

        lines = min(params.get("lines", 200), 1000)
        filter_text = params.get("filter")

        result = await client.fetch_service_logs(
            service_name=service_name,
            lines=lines,
            filter_text=filter_text,
        )

        if not result.get("success"):
            return result

        # Format logs for LLM consumption — compact text format
        logs = result.get("logs", [])
        log_lines = []
        for entry in logs:
            ts = entry.get("timestamp", "")
            sev = entry.get("severity", "")
            msg = entry.get("message", "")
            prefix = f"[{sev}]" if sev else ""
            log_lines.append(f"{ts} {prefix} {msg}".strip())

        result["formatted_logs"] = "\n".join(log_lines)
        # Truncate formatted output for LLM context (keep under 8K chars)
        if len(result["formatted_logs"]) > 8000:
            result["formatted_logs"] = result["formatted_logs"][:8000] + "\n... (truncated)"
            result["truncated"] = True

        return result

    async def _list_services(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all Railway services in the project."""
        from core.railway_client import RailwayClient

        client = RailwayClient()
        if not client.is_configured:
            return {
                "success": False,
                "error": "Railway API not configured. Set RAILWAY_API_TOKEN and RAILWAY_PROJECT_ID env vars.",
            }

        try:
            services = await client.list_services()
            return {
                "success": True,
                "services": services,
                "count": len(services),
            }
        except Exception as exc:
            logger.error("[PlatformExecutor] list_services failed: %s", exc, exc_info=True)
            return {"success": False, "error": f"Failed to list services: {exc}"}

    # ══════════════════════════════════════════════════════════════════
    # VISIBILITY / DISCOVERY HANDLERS
    # ══════════════════════════════════════════════════════════════════

    async def _list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all available tools — platform actions + Composio integrations."""
        from core.models import Agent
        from core.models.composio_cache import ComposioAppCache, AgentAppAssignment

        category = params.get("category", "all")
        search = (params.get("search") or "").lower()
        connected_only = params.get("connected_only", False)

        results = []

        # 1. Platform actions (from ActionRegistry)
        if category in ("all", "platform"):
            from modules.tools.discovery import get_action_registry
            registry = get_action_registry()
            for action in registry.get_all():
                if search and search not in action.name.lower() and search not in (action.description or "").lower():
                    continue
                results.append({
                    "name": action.name,
                    "type": "platform",
                    "category": action.category,
                    "description": (action.description or "")[:200],
                    "permission": action.permission_level,
                })

        # 2. Composio integrations (from cache + assignments)
        if category in ("all", "composio"):
            apps = self.db.query(ComposioAppCache).all()
            # Get connected apps for this workspace
            connected = set()
            try:
                rows = (
                    self.db.query(AgentAppAssignment.app_name)
                    .join(Agent, AgentAppAssignment.agent_id == Agent.id)
                    .filter(
                        Agent.workspace_id == self.workspace_id,
                        AgentAppAssignment.is_active == True,
                    )
                    .distinct()
                    .all()
                )
                connected = {r.app_name for r in rows}
            except Exception:
                pass

            for app in apps:
                is_connected = app.app_name in connected
                if connected_only and not is_connected:
                    continue
                if search and search not in app.app_name.lower() and search not in (app.display_name or "").lower():
                    continue
                results.append({
                    "name": app.app_name,
                    "type": "composio",
                    "display_name": app.display_name,
                    "description": (app.description or "")[:200],
                    "action_count": app.action_count,
                    "connected": is_connected,
                    "categories": app.categories or [],
                })

        return {"success": True, "tools": results, "count": len(results)}

    async def _list_llms(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available LLM models from OpenRouter cache."""
        from core.models.openrouter_cache import OpenRouterModelCache

        query = self.db.query(OpenRouterModelCache).filter(
            OpenRouterModelCache.status == "active"
        )

        capability = params.get("capability")
        if capability == "tools":
            query = query.filter(OpenRouterModelCache.supports_tools == True)
        elif capability == "vision":
            query = query.filter(OpenRouterModelCache.supports_vision == True)
        elif capability == "reasoning":
            query = query.filter(OpenRouterModelCache.supports_reasoning == True)
        elif capability == "json_mode":
            query = query.filter(OpenRouterModelCache.supports_json_mode == True)

        tier = params.get("tier")
        if tier:
            query = query.filter(OpenRouterModelCache.tier == tier)

        sort_by = params.get("sort_by", "cost")
        if sort_by == "cost":
            query = query.order_by(OpenRouterModelCache.prompt_cost.asc())
        elif sort_by == "context_length":
            query = query.order_by(OpenRouterModelCache.context_length.desc())
        else:
            query = query.order_by(OpenRouterModelCache.display_name.asc())

        limit = min(params.get("limit", 20), 50)
        models = query.limit(limit).all()

        total_active = (
            self.db.query(func.count(OpenRouterModelCache.id))
            .filter(OpenRouterModelCache.status == "active")
            .scalar()
        ) or 0

        return {
            "success": True,
            "models": [
                {
                    "model_id": m.model_id,
                    "display_name": m.display_name,
                    "provider": m.provider,
                    "prompt_cost_per_1k": m.prompt_cost,
                    "completion_cost_per_1k": m.completion_cost,
                    "context_length": m.context_length,
                    "supports_tools": m.supports_tools,
                    "supports_vision": m.supports_vision,
                    "supports_reasoning": m.supports_reasoning,
                    "tier": m.tier,
                    "category": m.category,
                }
                for m in models
            ],
            "count": len(models),
            "total_available": total_active,
        }

    async def _list_datasources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all data sources — documents (RAG) and databases (NL2SQL)."""
        ds_type = params.get("type", "all")
        result: Dict[str, Any] = {"success": True}

        # RAG document collections
        if ds_type in ("all", "documents"):
            from core.models import Document

            docs = (
                self.db.query(
                    Document.file_type,
                    func.count(Document.id).label("count"),
                    func.sum(Document.file_size).label("total_size"),
                )
                .filter(
                    Document.workspace_id == self.workspace_id,
                    Document.status == "completed",
                )
                .group_by(Document.file_type)
                .all()
            )

            total_chunks = (
                self.db.query(func.sum(Document.chunk_count))
                .filter(
                    Document.workspace_id == self.workspace_id,
                    Document.status == "completed",
                )
                .scalar()
            ) or 0

            result["documents"] = {
                "total_files": sum(r.count for r in docs),
                "total_chunks": total_chunks,
                "by_type": [
                    {
                        "type": r.file_type,
                        "count": r.count,
                        "size_bytes": r.total_size or 0,
                    }
                    for r in docs
                ],
            }

        # NL2SQL database connections
        if ds_type in ("all", "databases"):
            from core.models.database_knowledge import DatabaseKnowledgeSource

            sources = (
                self.db.query(DatabaseKnowledgeSource)
                .filter(
                    DatabaseKnowledgeSource.workspace_id == self.workspace_id,
                    DatabaseKnowledgeSource.is_active == True,
                )
                .all()
            )

            result["databases"] = [
                {
                    "id": str(s.id),
                    "name": s.name,
                    "dialect": s.dialect,
                    "description": (s.description or "")[:200],
                    "tables_indexed": (
                        len(s.schema_metadata.get("tables", []))
                        if s.schema_metadata else 0
                    ),
                    "last_introspected": (
                        s.last_introspected.isoformat()
                        if s.last_introspected else None
                    ),
                }
                for s in sources
            ]

        return result

    async def _workspace_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get workspace usage stats — LLM usage, top models, top agents."""
        from core.models.core import LLMUsage
        from core.models import Agent, Document

        period = params.get("period", "7d")
        days = {"today": 1, "7d": 7, "30d": 30}.get(period, 7)
        since = datetime.now(timezone.utc) - timedelta(days=days)

        # LLM usage summary
        usage = (
            self.db.query(
                func.count(LLMUsage.id).label("total_requests"),
                func.sum(LLMUsage.total_tokens).label("total_tokens"),
                func.sum(LLMUsage.total_cost).label("total_cost"),
            )
            .filter(
                LLMUsage.workspace_id == self.workspace_id,
                LLMUsage.created_at >= since,
            )
            .first()
        )

        # Top models by usage
        top_models = (
            self.db.query(
                LLMUsage.model_id,
                func.count(LLMUsage.id).label("requests"),
                func.sum(LLMUsage.total_cost).label("cost"),
            )
            .filter(
                LLMUsage.workspace_id == self.workspace_id,
                LLMUsage.created_at >= since,
            )
            .group_by(LLMUsage.model_id)
            .order_by(func.count(LLMUsage.id).desc())
            .limit(5)
            .all()
        )

        # Top agents by cost
        top_agents = (
            self.db.query(
                LLMUsage.agent_id,
                func.count(LLMUsage.id).label("requests"),
                func.sum(LLMUsage.total_cost).label("cost"),
            )
            .filter(
                LLMUsage.workspace_id == self.workspace_id,
                LLMUsage.created_at >= since,
                LLMUsage.agent_id.isnot(None),
            )
            .group_by(LLMUsage.agent_id)
            .order_by(func.sum(LLMUsage.total_cost).desc())
            .limit(5)
            .all()
        )

        # Resource counts
        agent_count = (
            self.db.query(func.count(Agent.id))
            .filter(Agent.workspace_id == self.workspace_id, Agent.is_active == True)
            .scalar()
        ) or 0
        doc_count = (
            self.db.query(func.count(Document.id))
            .filter(Document.workspace_id == self.workspace_id)
            .scalar()
        ) or 0

        return {
            "success": True,
            "period": period,
            "usage": {
                "total_requests": usage.total_requests or 0,
                "total_tokens": usage.total_tokens or 0,
                "total_cost": round(float(usage.total_cost or 0), 6),
            },
            "top_models": [
                {
                    "model": r.model_id,
                    "requests": r.requests,
                    "cost": round(float(r.cost or 0), 6),
                }
                for r in top_models
            ],
            "top_agents": [
                {
                    "agent_id": r.agent_id,
                    "requests": r.requests,
                    "cost": round(float(r.cost or 0), 6),
                }
                for r in top_agents
            ],
            "resources": {
                "agents": agent_count,
                "documents": doc_count,
            },
        }

    # ══════════════════════════════════════════════════════════════════
    # SELF-MANAGEMENT HANDLERS
    # ══════════════════════════════════════════════════════════════════

    async def _execute_recipe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger a recipe run asynchronously. Returns execution_id immediately."""
        from core.models.core import WorkflowTemplate, RecipeExecution
        import uuid
        import asyncio

        recipe_id = params.get("recipe_id")
        recipe_name = params.get("recipe_name")
        input_data = params.get("input_data") or {}

        # Resolve recipe
        query = self.db.query(WorkflowTemplate).filter(
            WorkflowTemplate.workspace_id == self.workspace_id
        )
        if recipe_id:
            query = query.filter(WorkflowTemplate.id == recipe_id)
        elif recipe_name:
            query = query.filter(WorkflowTemplate.name.ilike(f"%{recipe_name}%"))
        else:
            return {"success": False, "error": "Provide recipe_id or recipe_name"}

        recipe = query.first()
        if not recipe:
            return {"success": False, "error": "Recipe not found"}

        # Create execution record
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        execution = RecipeExecution(
            execution_id=execution_id,
            recipe_id=recipe.id,
            workspace_id=self.workspace_id,
            status="pending",
            input_data=input_data,
            triggered_by="platform_action",
        )
        self.db.add(execution)
        self.db.commit()  # Must commit before async task (it opens its own session)

        # Launch async execution fire-and-forget
        try:
            from api.recipe_executor import launch_recipe_task
            launch_recipe_task(
                recipe_execution_id=execution_id,
                recipe_id=recipe.id,
                workspace_id=self.workspace_id,
                input_data=input_data,
            )
        except Exception as e:
            logger.warning("[PlatformExecutor] Failed to launch recipe task: %s", e)

        logger.info(
            "[PlatformExecutor] Triggered recipe '%s' (id=%d) — execution_id=%s",
            recipe.name, recipe.id, execution_id,
        )

        return {
            "success": True,
            "execution_id": execution_id,
            "recipe_id": recipe.id,
            "recipe_name": recipe.name,
            "status": "pending",
            "message": f"Recipe '{recipe.name}' triggered. Track with execution_id: {execution_id}",
        }

    async def _get_recipe_execution(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check status/results of a recipe execution."""
        from core.models.core import RecipeExecution

        execution_id = params.get("execution_id")
        recipe_id = params.get("recipe_id")

        if execution_id:
            execution = (
                self.db.query(RecipeExecution)
                .filter(
                    RecipeExecution.execution_id == execution_id,
                    RecipeExecution.workspace_id == self.workspace_id,
                )
                .first()
            )
            if not execution:
                return {"success": False, "error": f"Execution '{execution_id}' not found"}

            # Summarize step_results (200 char preview per step)
            step_summaries = []
            for i, step in enumerate(execution.step_results or []):
                if isinstance(step, dict):
                    output = str(step.get("output", step.get("result", "")))[:200]
                    step_summaries.append({
                        "step": i,
                        "status": step.get("status", "unknown"),
                        "output_preview": output,
                    })

            return {
                "success": True,
                "execution": {
                    "execution_id": execution.execution_id,
                    "recipe_id": execution.recipe_id,
                    "status": execution.status,
                    "started_at": execution.started_at.isoformat() if execution.started_at else None,
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "error_message": execution.error_message,
                    "step_results": step_summaries,
                    "current_step": execution.current_step,
                },
            }

        elif recipe_id:
            # List recent executions for this recipe
            executions = (
                self.db.query(RecipeExecution)
                .filter(
                    RecipeExecution.recipe_id == recipe_id,
                    RecipeExecution.workspace_id == self.workspace_id,
                )
                .order_by(RecipeExecution.started_at.desc())
                .limit(5)
                .all()
            )

            return {
                "success": True,
                "executions": [
                    {
                        "execution_id": e.execution_id,
                        "status": e.status,
                        "started_at": e.started_at.isoformat() if e.started_at else None,
                        "completed_at": e.completed_at.isoformat() if e.completed_at else None,
                        "error_message": e.error_message,
                    }
                    for e in executions
                ],
                "count": len(executions),
            }

        return {"success": False, "error": "Provide execution_id or recipe_id"}

    async def _get_system_health(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """System health check — database, Redis, API, RAG, server metrics."""
        components = {}

        # 1. Database
        try:
            self.db.execute(func.literal(1).select())
            components["database"] = {"status": "healthy"}
        except Exception as e:
            components["database"] = {"status": "unhealthy", "error": str(e)[:100]}

        # 2. Redis
        try:
            from core.redis.client import get_redis_client
            redis = get_redis_client()
            if redis:
                redis.ping()
                components["redis"] = {"status": "healthy"}
            else:
                components["redis"] = {"status": "unavailable"}
        except Exception as e:
            components["redis"] = {"status": "unhealthy", "error": str(e)[:100]}

        # 3. RAG pipeline
        try:
            from core.models import Document
            doc_count = (
                self.db.query(func.count(Document.id))
                .filter(Document.workspace_id == self.workspace_id)
                .scalar()
            ) or 0
            completed = (
                self.db.query(func.count(Document.id))
                .filter(
                    Document.workspace_id == self.workspace_id,
                    Document.status == "completed",
                )
                .scalar()
            ) or 0
            components["rag"] = {
                "status": "healthy",
                "total_documents": doc_count,
                "processed": completed,
            }
        except Exception as e:
            components["rag"] = {"status": "unhealthy", "error": str(e)[:100]}

        # 4. Server metrics (psutil)
        try:
            import psutil
            components["server"] = {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
            }
        except ImportError:
            components["server"] = {"status": "psutil not installed"}
        except Exception as e:
            components["server"] = {"status": "error", "error": str(e)[:100]}

        # Overall status
        unhealthy = [k for k, v in components.items() if v.get("status") == "unhealthy"]
        overall = "unhealthy" if unhealthy else "healthy"

        return {
            "success": True,
            "overall_status": overall,
            "components": components,
            "unhealthy": unhealthy or None,
        }

    async def _delete_document(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a document — S3 file + vector embeddings + DB record."""
        from core.models import Document

        document_id = params.get("document_id")
        if not document_id:
            return {"success": False, "error": "Missing required parameter: document_id"}

        doc = (
            self.db.query(Document)
            .filter(
                Document.id == document_id,
                Document.workspace_id == self.workspace_id,
            )
            .first()
        )
        if not doc:
            return {"success": False, "error": "Document not found"}

        doc_info = {
            "id": doc.id,
            "filename": doc.original_filename or doc.filename,
        }
        cleanup_notes = []

        # Phase 1: S3 file cleanup (non-fatal)
        file_path = doc.file_path or ""
        if file_path.startswith("s3://"):
            try:
                import boto3
                parts = file_path.replace("s3://", "").split("/", 1)
                bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
                s3 = boto3.client("s3")
                s3.delete_object(Bucket=bucket, Key=key)
                cleanup_notes.append("S3 file deleted")
            except Exception as e:
                logger.warning("[PlatformExecutor] S3 cleanup failed for doc %d: %s", doc.id, e)
                cleanup_notes.append(f"S3 cleanup failed: {e}")

        # Phase 2: Vector embedding cleanup (non-fatal)
        try:
            from modules.search.vector_store.backends.s3_vectors_backend import S3VectorsBackend
            backend = S3VectorsBackend()
            deleted = backend.delete_documents(str(doc.id))
            cleanup_notes.append(f"Vector embeddings deleted ({deleted} removed)")
        except Exception as e:
            logger.warning("[PlatformExecutor] Vector cleanup failed for doc %d: %s", doc.id, e)
            cleanup_notes.append(f"Vector cleanup failed: {e}")

        # Phase 3: DB record (cascades to document_chunks via FK)
        self.db.delete(doc)
        self.db.flush()
        cleanup_notes.append("Database record deleted")

        logger.info("[PlatformExecutor] Deleted document %s — %s", doc_info, ", ".join(cleanup_notes))

        return {
            "success": True,
            "deleted_document": doc_info,
            "cleanup": cleanup_notes,
            "message": f"Document '{doc_info['filename']}' (ID {doc_info['id']}) deleted.",
        }

    async def _reprocess_document(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Re-process a document — regenerate chunks and vector embeddings."""
        from core.models import Document

        document_id = params.get("document_id")
        if not document_id:
            return {"success": False, "error": "Missing required parameter: document_id"}

        doc = (
            self.db.query(Document)
            .filter(
                Document.id == document_id,
                Document.workspace_id == self.workspace_id,
            )
            .first()
        )
        if not doc:
            return {"success": False, "error": "Document not found"}

        file_path = doc.file_path or ""

        # Validate file exists
        if file_path.startswith("s3://"):
            try:
                import boto3
                parts = file_path.replace("s3://", "").split("/", 1)
                bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
                s3 = boto3.client("s3")
                s3.head_object(Bucket=bucket, Key=key)
            except Exception as e:
                return {"success": False, "error": f"S3 file not accessible: {e}"}
        elif file_path:
            import os
            if not os.path.exists(file_path):
                return {"success": False, "error": f"Local file not found: {file_path}"}
        else:
            return {"success": False, "error": "Document has no file_path"}

        # Set status to processing
        doc.status = "processing"
        self.db.flush()

        # Re-process via DocumentManager
        try:
            from api.documents import get_document_manager

            dm = get_document_manager(str(self.workspace_id))

            # For S3 files, download to temp first
            actual_path = file_path
            if file_path.startswith("s3://"):
                import tempfile
                import boto3
                parts = file_path.replace("s3://", "").split("/", 1)
                bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
                suffix = "." + key.rsplit(".", 1)[-1] if "." in key else ""
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                boto3.client("s3").download_file(bucket, key, tmp.name)
                actual_path = tmp.name

            new_doc_id = await dm.upload_document(
                file_path=actual_path,
                filename=doc.original_filename or doc.filename,
            )

            # Refresh doc from DB to get updated chunk_count
            self.db.refresh(doc)
            doc.status = "completed"
            self.db.flush()

            logger.info("[PlatformExecutor] Reprocessed document %d", doc.id)

            return {
                "success": True,
                "document_id": doc.id,
                "status": "completed",
                "chunk_count": doc.chunk_count or 0,
                "message": f"Document '{doc.original_filename or doc.filename}' reprocessed successfully.",
            }
        except Exception as e:
            doc.status = "failed"
            self.db.flush()
            logger.error("[PlatformExecutor] Reprocess failed for doc %d: %s", doc.id, e, exc_info=True)
            return {"success": False, "error": f"Reprocessing failed: {e}"}

    async def _delete_recipe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a recipe with full cleanup."""
        from core.models.core import WorkflowTemplate

        recipe_id = params.get("recipe_id")
        recipe_name = params.get("recipe_name")

        query = self.db.query(WorkflowTemplate).filter(
            WorkflowTemplate.workspace_id == self.workspace_id
        )
        if recipe_id:
            query = query.filter(WorkflowTemplate.id == recipe_id)
        elif recipe_name:
            query = query.filter(WorkflowTemplate.name.ilike(f"%{recipe_name}%"))
        else:
            return {"success": False, "error": "Provide recipe_id or recipe_name"}

        recipe = query.first()
        if not recipe:
            return {"success": False, "error": "Recipe not found"}

        # Guard against system recipes
        if getattr(recipe, "is_system", False):
            return {"success": False, "error": "System recipes cannot be deleted"}

        recipe_info = {"id": recipe.id, "name": recipe.name}
        cleanup_notes = []

        # Trigger subscription cleanup (non-fatal)
        try:
            from api.workflow_recipes import _cleanup_trigger_subscriptions
            _cleanup_trigger_subscriptions(recipe.id, self.db)
            cleanup_notes.append("Trigger subscriptions deactivated")
        except Exception as e:
            logger.warning("[PlatformExecutor] Trigger cleanup failed for recipe %d: %s", recipe.id, e)
            cleanup_notes.append(f"Trigger cleanup failed: {e}")

        # Mem0 memory cleanup (non-fatal)
        try:
            import httpx
            from config import config
            mem0_url = config.MEM0_API_URL
            if mem0_url:
                import asyncio
                async with httpx.AsyncClient(timeout=5.0) as client:
                    headers = {}
                    if config.MEM0_API_KEY:
                        headers["Authorization"] = f"Bearer {config.MEM0_API_KEY}"
                    await client.delete(
                        f"{mem0_url}/v1/memories/",
                        params={"user_id": f"recipe-{recipe.id}"},
                        headers=headers,
                    )
                cleanup_notes.append("Recipe memories cleaned up")
        except Exception as e:
            logger.debug("[PlatformExecutor] Mem0 cleanup skipped for recipe %d: %s", recipe.id, e)

        # Delete the recipe (cascades to executions via FK)
        self.db.delete(recipe)
        self.db.flush()
        cleanup_notes.append("Database record deleted")

        logger.info("[PlatformExecutor] Deleted recipe %s — %s", recipe_info, ", ".join(cleanup_notes))

        return {
            "success": True,
            "deleted_recipe": recipe_info,
            "cleanup": cleanup_notes,
            "message": f"Recipe '{recipe_info['name']}' (ID {recipe_info['id']}) deleted.",
        }

    async def _get_activity_feed(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Unified activity feed — chats, recipe runs, routines."""
        from services.activity_service import ActivityService

        period = params.get("period", "7d")
        type_csv = params.get("type", "")
        types = [t.strip() for t in type_csv.split(",") if t.strip()] if type_csv else None
        limit = min(params.get("limit", 20), 50)

        service = ActivityService(self.db, self.workspace_id)
        feed = service.get_feed(types=types, period=period, limit=limit)

        return {
            "success": True,
            "period": period,
            "items": feed.get("items", []),
            "total": feed.get("total", 0),
            "message": f"Showing {len(feed.get('items', []))} activities from the last {period}.",
        }
