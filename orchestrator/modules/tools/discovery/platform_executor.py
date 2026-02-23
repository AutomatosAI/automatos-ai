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
            "platform_store_memory": self._store_memory,
            "platform_delete_agent": self._delete_agent,
        }

    async def execute(self, action_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a platform action by name with permission checking."""
        handler = self._handlers.get(action_name)
        if not handler:
            return {"success": False, "error": f"Unknown platform action: {action_name}"}

        # Permission check for write/destructive actions
        try:
            from modules.tools.discovery import get_action_registry
            action_def = get_action_registry().get(action_name)
            if action_def and action_def.requires_confirmation:
                return {
                    "success": True,
                    "requires_confirmation": True,
                    "action": action_name,
                    "permission_level": action_def.permission_level,
                    "message": (
                        f"This action ({action_def.permission_level}) requires confirmation. "
                        f"Action: {action_name} — {action_def.description[:100]}"
                    ),
                    "params": params,
                }
        except Exception:
            pass  # If registry check fails, proceed (fail open for read actions)

        try:
            return await handler(params)
        except Exception as e:
            logger.error(f"[PlatformExecutor] {action_name} failed: {e}", exc_info=True)
            try:
                self.db.rollback()
            except Exception:
                pass
            return {"success": False, "error": str(e)}

    # ── Agent Handlers ──────────────────────────────────────────────

    async def _list_agents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.models import Agent

        query = self.db.query(Agent).filter(Agent.workspace_id == self.workspace_id)

        status_filter = params.get("status_filter", "all")
        if status_filter != "all":
            query = query.filter(Agent.status == status_filter)

        agents = query.order_by(Agent.id).all()

        return {
            "success": True,
            "agents": [
                {
                    "id": a.id,
                    "name": a.name,
                    "type": a.agent_type,
                    "status": a.status,
                    "description": (a.description or "")[:200],
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                }
                for a in agents
            ],
            "count": len(agents),
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

        config = agent.configuration or {}
        return {
            "success": True,
            "agent": {
                "id": agent.id,
                "name": agent.name,
                "type": agent.agent_type,
                "status": agent.status,
                "description": agent.description,
                "model": config.get("model") or config.get("llm_model"),
                "provider": config.get("provider") or config.get("llm_provider"),
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

        definition = recipe.template_definition or {}
        steps = definition.get("steps", [])

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
                    {"name": s.get("name", f"Step {i+1}"), "type": s.get("type", "unknown")}
                    for i, s in enumerate(steps[:10])
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
        model = params.get("model")

        config = {}
        if model:
            config["model"] = model

        agent = Agent(
            name=name,
            agent_type=agent_type,
            description=description,
            status="active",
            configuration=config,
            workspace_id=self.workspace_id,
            created_by="platform",
            owner_type="workspace",
            owner_id=str(self.workspace_id),
        )
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
        if params.get("new_name"):
            agent.name = params["new_name"]
            changes.append(f"name → '{params['new_name']}'")
        if params.get("description") is not None:
            agent.description = params["description"]
            changes.append("description updated")
        if params.get("status"):
            agent.status = params["status"]
            changes.append(f"status → '{params['status']}'")

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
