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
            # Chat & memory search
            "platform_search_chat_history": self._search_chat_history,
            "platform_search_memory": self._search_memory,
            # PRD-73: Monitoring (Loki, Prometheus, Alerts)
            "platform_query_loki_logs": self._query_loki_logs,
            "platform_query_prometheus": self._query_prometheus,
            "platform_get_alerts": self._get_alerts,
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
            # Marketplace discovery & workspace inventory (PRD-71)
            "platform_browse_marketplace_plugins": self._browse_marketplace_plugins,
            "platform_browse_marketplace_skills": self._browse_marketplace_skills,
            "platform_list_workspace_plugins": self._list_workspace_plugins,
            "platform_list_workspace_skills": self._list_workspace_skills,
            "platform_list_workspace_models": self._list_workspace_models,
            "platform_install_plugin": self._install_plugin,
            "platform_install_skill": self._install_skill,
            "platform_install_model": self._install_model,
            # Agent assignment (PRD-71)
            "platform_assign_tool_to_agent": self._assign_tool_to_agent,
            "platform_assign_skill_to_agent": self._assign_skill_to_agent,
            "platform_assign_plugin_to_agent": self._assign_plugin_to_agent,
            "platform_configure_agent_heartbeat": self._configure_agent_heartbeat,
            # PRD-76: Agent Reports
            "platform_submit_report": self._submit_report,
            "platform_get_latest_report": self._get_latest_report,
            # PRD-72: Board Tasks
            "platform_create_task": self._create_board_task,
            "platform_list_tasks": self._list_board_tasks,
            "platform_board_summary": self._board_summary,
            "platform_get_task": self._get_board_task,
            "platform_assign_task": self._assign_board_task,
            "platform_update_task_status": self._update_board_task_status,
            # PRD-77: Agent Self-Scheduling
            "platform_schedule_task": self._schedule_task,
            "platform_list_scheduled_tasks": self._list_scheduled_tasks,
            "platform_cancel_scheduled_task": self._cancel_scheduled_task,
            # PRD-77: Memory Browsing
            "platform_browse_memories": self._browse_memories,
            "platform_delete_memory": self._delete_memory,
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
        """Get memory stats from Mem0 — global + per-agent memories."""
        import asyncio
        from modules.memory.integrations.mem0_client import Mem0Client

        try:
            client = Mem0Client()
            if not client.api_url:
                return {"success": False, "error": "Memory service not configured (MEM0_API_URL empty)"}

            ws_id = str(self.workspace_id)
            global_user_id = f"ws_{ws_id}"

            loop = asyncio.get_event_loop()

            # Fetch global memories
            global_memories = await loop.run_in_executor(
                None, lambda: client.get_all(user_id=global_user_id, limit=200)
            )

            # Also check per-agent memories for workspace agents
            from core.models.core import Agent
            agents = (
                self.db.query(Agent.id, Agent.name)
                .filter(Agent.workspace_id == self.workspace_id)
                .all()
            )

            agent_scan_limit = 10
            scanned_agents = agents[:agent_scan_limit]
            partial = len(agents) > agent_scan_limit
            agent_stats = []
            agent_tasks = [
                loop.run_in_executor(
                    None,
                    lambda uid=f"ws_{ws_id}_agent_{agent_id}": client.get_all(user_id=uid, limit=200),
                )
                for agent_id, _agent_name in scanned_agents
            ]
            agent_results = await asyncio.gather(*agent_tasks) if agent_tasks else []
            for (agent_id, agent_name), agent_mems in zip(scanned_agents, agent_results):
                if agent_mems:
                    agent_stats.append({
                        "agent_id": agent_id,
                        "agent_name": agent_name,
                        "memory_count": len(agent_mems),
                        "sample": [(m.get("memory") or m.get("content", ""))[:80] for m in agent_mems[:3]],
                    })

            global_count = len(global_memories) if global_memories else 0
            total_agent = sum(a["memory_count"] for a in agent_stats)

            # Format for LLM
            lines = [f"Memory Stats for workspace {ws_id}:\n"]
            lines.append(f"Global memories: {global_count}")
            if global_memories:
                lines.append("Sample global memories:")
                for m in (global_memories or [])[:5]:
                    content = m.get("memory") or m.get("content", "")
                    lines.append(f"  - {content[:100]}")

            lines.append(f"\nAgent-specific memories: {total_agent} across {len(agent_stats)} agent(s)")
            for a in agent_stats:
                lines.append(f"  {a['agent_name']}: {a['memory_count']} memories")
                for s in a["sample"]:
                    lines.append(f"    - {s[:80]}")

            return {
                "success": True,
                "global_memories": global_count,
                "agent_memories": total_agent,
                "total_memories": global_count + total_agent,
                "agent_stats": agent_stats,
                "partial": partial,
                "scanned_agents": len(scanned_agents),
                "total_agents": len(agents),
                "formatted": "\n".join(lines),
            }
        except Exception as e:
            logger.warning(f"[PlatformExecutor] Memory stats failed: {e}", exc_info=True)
            return {"success": False, "error": f"Memory service error: {e}"}

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
            import asyncio
            from modules.memory.integrations.mem0_client import Mem0Client

            client = Mem0Client()
            if not client.api_url:
                return {"success": False, "error": "Memory service not configured (MEM0_API_URL empty)"}

            messages = [{"role": "user", "content": content}]
            # Use correct user_id format: ws_{workspace_id} for global memories
            user_id = f"ws_{self.workspace_id}"

            # If agent_id provided, store as agent-specific memory
            agent_id = params.get("agent_id")
            if agent_id:
                user_id = f"ws_{self.workspace_id}_agent_{agent_id}"

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: client.add(
                    messages=messages,
                    user_id=user_id,
                    metadata={"workspace_id": str(self.workspace_id), "source": "platform_tool"},
                )
            )

            if result.get("error"):
                return {"success": False, "error": result["error"]}

            facts = result.get("facts_extracted", "unknown")
            return {
                "success": True,
                "message": f"Stored in memory (user_id={user_id}): '{content[:100]}'",
                "facts_extracted": facts,
            }
        except Exception as e:
            logger.warning(f"[PlatformExecutor] Memory store failed: {e}", exc_info=True)
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
    # CHAT HISTORY SEARCH
    # ══════════════════════════════════════════════════════════════════

    async def _search_chat_history(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search across all chat messages by keyword."""
        from sqlalchemy import text

        query = params.get("query", "").strip()
        if not query:
            return {"success": False, "error": "query parameter is required"}

        days = min(params.get("days", 30), 365)
        limit = min(params.get("limit", 20), 100)
        search_term = f"%{query}%"

        try:
            rows = self.db.execute(
                text("""
                    SELECT m.id, m.chat_id, m.role, m.parts, m.created_at,
                           c.title AS chat_title
                    FROM messages m
                    JOIN chats c ON c.id = m.chat_id
                    WHERE c.user_id = (SELECT id FROM users LIMIT 1)
                      AND m.created_at >= NOW() - INTERVAL ':days days'
                      AND EXISTS (
                          SELECT 1 FROM jsonb_array_elements(m.parts) AS p
                          WHERE p->>'text' ILIKE :search
                      )
                    ORDER BY m.created_at DESC
                    LIMIT :lim
                """),
                {"days": days, "search": search_term, "lim": limit},
            ).fetchall()

            results = []
            for r in rows:
                parts = r.parts if isinstance(r.parts, list) else []
                text_content = " ".join(
                    p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")
                )
                results.append({
                    "chat_title": r.chat_title,
                    "role": r.role,
                    "content": text_content[:300],
                    "date": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else None,
                    "chat_id": str(r.chat_id),
                })

            # Format for LLM
            lines = [f"Found {len(results)} message(s) matching '{query}':\n"]
            for i, r in enumerate(results, 1):
                lines.append(
                    f"{i}. [{r['date']}] ({r['role']}) in \"{r['chat_title']}\":\n"
                    f"   {r['content']}\n"
                )

            return {
                "success": True,
                "query": query,
                "total": len(results),
                "results": results,
                "formatted": "\n".join(lines),
            }
        except Exception as exc:
            logger.error("[PlatformExecutor] Chat search failed: %s", exc, exc_info=True)
            return {"success": False, "error": f"Chat search failed: {exc}"}

    async def _search_memory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search Mem0 memories by query."""
        import asyncio
        from modules.memory.integrations.mem0_client import Mem0Client

        query = params.get("query", "").strip()
        if not query:
            return {"success": False, "error": "query parameter is required"}

        agent_id = params.get("agent_id")
        limit = min(params.get("limit", 10), 50)
        result_char_limit = 150

        try:
            client = Mem0Client()
            if not client.api_url:
                return {"success": False, "error": "Memory service not configured (MEM0_API_URL empty)"}

            ws_id = str(self.workspace_id)
            loop = asyncio.get_event_loop()

            # Search global memories
            global_user_id = f"ws_{ws_id}"
            global_results = await loop.run_in_executor(
                None, lambda: client.search(query=query, user_id=global_user_id, limit=limit)
            )

            # Search agent-specific if agent_id given, otherwise search all agents
            agent_results = []
            partial = False
            scanned_agents = 0
            total_agents = 0
            if agent_id:
                agent_user_id = f"ws_{ws_id}_agent_{agent_id}"
                agent_results = await loop.run_in_executor(
                    None, lambda: client.search(query=query, user_id=agent_user_id, limit=limit)
                )
                for m in agent_results:
                    m["_tier"] = f"agent-{agent_id}"
                scanned_agents = 1
                total_agents = 1
            else:
                # Search top agents
                from core.models.core import Agent
                agents = (
                    self.db.query(Agent.id)
                    .filter(Agent.workspace_id == self.workspace_id)
                    .limit(5)
                    .all()
                )
                total_agents_query = (
                    self.db.query(func.count(Agent.id))
                    .filter(Agent.workspace_id == self.workspace_id)
                    .scalar()
                ) or 0
                total_agents = int(total_agents_query)
                scanned_agents = len(agents)
                partial = total_agents > scanned_agents
                agent_tasks = [
                    loop.run_in_executor(
                        None, lambda u=f"ws_{ws_id}_agent_{aid}": client.search(query=query, user_id=u, limit=5)
                    )
                    for (aid,) in agents
                ]
                agent_batches = await asyncio.gather(*agent_tasks) if agent_tasks else []
                for (aid,), res in zip(agents, agent_batches):
                    for m in (res or []):
                        m["_tier"] = f"agent-{aid}"
                    agent_results.extend(res or [])

            # Mark global
            for m in (global_results or []):
                m["_tier"] = "global"

            all_results = (global_results or []) + agent_results

            # Format
            lines = [f"Memory search for '{query}': {len(all_results)} result(s)\n"]
            for i, m in enumerate(all_results[:limit], 1):
                content = (m.get("memory") or m.get("content", "") or "")[:result_char_limit]
                tier = m.get("_tier", "unknown")
                created = m.get("created_at", "")
                lines.append(f"{i}. [{tier}] {content}")
                if created:
                    lines.append(f"   Created: {created}")

            return {
                "success": True,
                "query": query,
                "total": len(all_results),
                "global_count": len(global_results or []),
                "agent_count": len(agent_results),
                "partial": partial,
                "scanned_agents": scanned_agents,
                "total_agents": total_agents,
                "results": [
                    {
                        "memory": (m.get("memory") or m.get("content", "") or "")[:result_char_limit],
                        "tier": m.get("_tier", "unknown"),
                        "created_at": m.get("created_at"),
                    }
                    for m in all_results[:limit]
                ],
                "formatted": "\n".join(lines),
            }
        except Exception as e:
            logger.warning(f"[PlatformExecutor] Memory search failed: {e}", exc_info=True)
            return {"success": False, "error": f"Memory search error: {e}"}

    # ══════════════════════════════════════════════════════════════════
    # PRD-73: MONITORING HANDLERS (Loki, Prometheus, Alerts)
    # ══════════════════════════════════════════════════════════════════

    async def _query_loki_logs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Query application logs from Loki."""
        import httpx
        from config import config

        loki_url = getattr(config, "LOKI_URL", None) or "http://loki.railway.internal:3100"
        minutes = min(params.get("minutes", 60), 10080)
        limit = min(params.get("limit", 100), 500)
        service = params.get("service")
        level = params.get("level")
        search = params.get("search")

        # Build LogQL query
        label_parts = []
        if service:
            label_parts.append(f'service="{service}"')
        if level:
            label_parts.append(f'level="{level}"')
        label_selector = "{" + ", ".join(label_parts) + "}" if label_parts else '{}'

        # Add line filter for search
        line_filter = ""
        if search:
            line_filter = f' |= `{search}`'

        logql = f"{label_selector}{line_filter}"

        import time as _time
        end_ns = int(_time.time() * 1e9)
        start_ns = int(((_time.time()) - minutes * 60) * 1e9)

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{loki_url}/loki/api/v1/query_range",
                    params={
                        "query": logql,
                        "start": str(start_ns),
                        "end": str(end_ns),
                        "limit": str(limit),
                        "direction": "backward",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("data", {}).get("result", [])
            log_lines = []
            for stream in results:
                labels = stream.get("stream", {})
                svc = labels.get("service", "unknown")
                lvl = labels.get("level", "")
                for ts_ns, msg in stream.get("values", []):
                    ts_sec = int(ts_ns) / 1e9
                    ts_str = datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%H:%M:%S")
                    log_lines.append(f"[{ts_str}] [{svc}] [{lvl.upper()}] {msg}")

            formatted = "\n".join(log_lines[:limit])
            if len(formatted) > 8000:
                formatted = formatted[:8000] + "\n... (truncated)"

            return {
                "success": True,
                "query": logql,
                "total_entries": len(log_lines),
                "time_range_minutes": minutes,
                "formatted_logs": formatted,
            }
        except httpx.ConnectError:
            return {
                "success": False,
                "error": (
                    f"Cannot reach Loki at {loki_url}. "
                    "Loki is only accessible within the Railway internal network."
                ),
            }
        except Exception as exc:
            logger.error("[PlatformExecutor] Loki query failed: %s", exc, exc_info=True)
            return {"success": False, "error": f"Loki query failed: {exc}"}

    async def _query_prometheus(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Query Prometheus metrics with presets or raw PromQL."""
        import httpx
        from config import config

        prom_url = getattr(config, "PROMETHEUS_URL", None) or "http://prometheus.railway.internal:9090"
        query_input = params.get("query", "health")
        range_minutes = min(params.get("range_minutes", 15), 1440)

        # Preset queries for common health checks
        presets = {
            "health": [
                ("Service Health", "up"),
            ],
            "error_rate": [
                ("HTTP 5xx Rate (5m)", 'rate(automatos_http_requests_total{status_code=~"5.."}[5m])'),
                ("HTTP Total Rate (5m)", "rate(automatos_http_requests_total[5m])"),
            ],
            "latency": [
                ("p95 Response Time", "histogram_quantile(0.95, rate(automatos_http_request_duration_seconds_bucket[5m]))"),
                ("p50 Response Time", "histogram_quantile(0.50, rate(automatos_http_request_duration_seconds_bucket[5m]))"),
            ],
            "postgres": [
                ("DB Connections", "pg_stat_activity_count"),
                ("Cache Hit Ratio", "pg_stat_database_blks_hit / (pg_stat_database_blks_hit + pg_stat_database_blks_read)"),
                ("Dead Tuples", "pg_stat_user_tables_n_dead_tup"),
            ],
            "redis": [
                ("Redis Memory (MB)", "redis_memory_used_bytes / 1024 / 1024"),
                ("Redis Clients", "redis_connected_clients"),
                ("Redis Evicted Keys (5m)", "rate(redis_evicted_keys_total[5m])"),
                ("Redis Command Latency", "redis_commands_duration_seconds_total"),
            ],
            "all": [],  # filled below
        }
        # "all" = union of all presets
        for k, v in presets.items():
            if k != "all":
                presets["all"].extend(v)

        query_lower = query_input.lower().strip()
        queries_to_run = presets.get(query_lower, [(query_input, query_input)])

        try:
            results = []
            async with httpx.AsyncClient(timeout=15.0) as client:
                for label, promql in queries_to_run:
                    resp = await client.get(
                        f"{prom_url}/api/v1/query",
                        params={"query": promql},
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    metric_results = data.get("data", {}).get("result", [])
                    formatted_values = []
                    for m in metric_results:
                        metric_labels = m.get("metric", {})
                        value = m.get("value", [None, None])
                        val = value[1] if len(value) > 1 else "N/A"

                        # Human-readable label
                        desc_parts = []
                        for k in ["job", "instance", "service", "datname", "status_code", "relname"]:
                            if k in metric_labels:
                                desc_parts.append(f"{k}={metric_labels[k]}")
                        desc = ", ".join(desc_parts) if desc_parts else "global"
                        formatted_values.append({"labels": desc, "value": val})

                    results.append({
                        "metric": label,
                        "query": promql,
                        "values": formatted_values,
                    })

            # Format for LLM consumption
            lines = []
            for r in results:
                lines.append(f"### {r['metric']}")
                if not r["values"]:
                    lines.append("  No data")
                for v in r["values"]:
                    lines.append(f"  {v['labels']}: {v['value']}")
                lines.append("")

            return {
                "success": True,
                "preset_used": query_lower if query_lower in presets else None,
                "results": results,
                "formatted": "\n".join(lines),
            }
        except httpx.ConnectError:
            return {
                "success": False,
                "error": (
                    f"Cannot reach Prometheus at {prom_url}. "
                    "Prometheus is only accessible within the Railway internal network."
                ),
            }
        except Exception as exc:
            logger.error("[PlatformExecutor] Prometheus query failed: %s", exc, exc_info=True)
            return {"success": False, "error": f"Prometheus query failed: {exc}"}

    async def _get_alerts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get infrastructure alerts from the database."""
        from sqlalchemy import text

        status_filter = params.get("status", "all")
        severity_filter = params.get("severity")
        hours = min(params.get("hours", 24), 168)

        try:
            conditions = ["created_at > NOW() - INTERVAL ':hours hours'"]
            bind_params: Dict[str, Any] = {"hours": hours}

            if status_filter and status_filter != "all":
                conditions.append("status = :status")
                bind_params["status"] = status_filter
            if severity_filter:
                conditions.append("severity = :severity")
                bind_params["severity"] = severity_filter

            where_clause = " AND ".join(conditions)

            rows = self.db.execute(
                text(f"""
                    SELECT alertname, severity, status, service,
                           annotations, agent_response, created_at, resolved_at
                    FROM infrastructure_alerts
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT 50
                """),
                bind_params,
            ).fetchall()

            alerts = []
            for r in rows:
                annotations = r.annotations if isinstance(r.annotations, dict) else {}
                alerts.append({
                    "alert": r.alertname,
                    "severity": r.severity,
                    "status": r.status,
                    "service": r.service,
                    "summary": annotations.get("summary", ""),
                    "description": annotations.get("description", ""),
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "resolved_at": r.resolved_at.isoformat() if r.resolved_at else None,
                    "investigated": bool(r.agent_response),
                })

            # Summary
            firing = [a for a in alerts if a["status"] == "firing"]
            critical = [a for a in firing if a["severity"] == "critical"]

            formatted_lines = []
            if not alerts:
                formatted_lines.append(f"No alerts found in the last {hours} hours.")
            else:
                if critical:
                    formatted_lines.append(f"🔴 {len(critical)} CRITICAL alert(s) firing!")
                if firing:
                    formatted_lines.append(f"⚠️ {len(firing)} alert(s) currently firing")
                formatted_lines.append(f"Total: {len(alerts)} alert(s) in last {hours}h\n")

                for a in alerts[:20]:
                    icon = "🔴" if a["severity"] == "critical" else "🟡" if a["severity"] == "warning" else "ℹ️"
                    status_icon = "🔥" if a["status"] == "firing" else "✅"
                    formatted_lines.append(
                        f"{icon}{status_icon} [{a['severity'].upper()}] {a['alert']} "
                        f"({a['service'] or 'unknown'}) — {a['summary']}"
                    )

            return {
                "success": True,
                "total": len(alerts),
                "firing_count": len(firing),
                "critical_count": len(critical),
                "alerts": alerts,
                "formatted": "\n".join(formatted_lines),
            }
        except Exception as exc:
            # Table might not exist yet
            if "infrastructure_alerts" in str(exc) and ("does not exist" in str(exc) or "UndefinedTable" in str(exc)):
                return {
                    "success": True,
                    "total": 0,
                    "firing_count": 0,
                    "critical_count": 0,
                    "alerts": [],
                    "formatted": "No alerts table found — monitoring alerts not yet configured.",
                }
            logger.error("[PlatformExecutor] get_alerts failed: %s", exc, exc_info=True)
            return {"success": False, "error": f"Alert query failed: {exc}"}

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

        # Concurrency guard — return error to agent if workspace is at capacity
        from services.concurrency_guard import check_concurrency
        concurrency = await check_concurrency(self.workspace_id, self.db)
        if not concurrency.allowed:
            logger.warning(
                "[PlatformExecutor] Concurrency limit reached for workspace %s: %s",
                self.workspace_id, concurrency.reason,
            )
            return {"status": "error", "error": concurrency.reason}

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
            logger.error("[PlatformExecutor] Failed to launch recipe task: %s", e)
            # Mark execution as failed so it doesn't stay "pending" forever
            execution.status = "failed"
            execution.error_message = f"Failed to enqueue: {str(e)[:500]}"
            self.db.commit()
            return {"success": False, "error": f"Recipe triggered but failed to launch: {str(e)[:200]}"}

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
            from sqlalchemy import select as sa_select
            self.db.execute(sa_select(1))
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

            logger.info("[PlatformExecutor] Reprocessed document %d → new doc %s", doc.id, new_doc_id)

            return {
                "success": True,
                "document_id": new_doc_id,
                "original_document_id": doc.id,
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

    # ══════════════════════════════════════════════════════════════════
    # MARKETPLACE DISCOVERY & WORKSPACE INVENTORY (PRD-71)
    # ══════════════════════════════════════════════════════════════════

    async def _browse_marketplace_plugins(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Browse/search approved marketplace plugins."""
        from core.models.marketplace_plugins import (
            MarketplacePlugin, PluginCategory, WorkspaceEnabledPlugin,
        )

        query = self.db.query(MarketplacePlugin).filter(
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
            cat = self.db.query(PluginCategory).filter(
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
                self.db.query(WorkspaceEnabledPlugin.plugin_id)
                .filter(WorkspaceEnabledPlugin.workspace_id == self.workspace_id)
                .all()
            )
            enabled_ids = {r.plugin_id for r in rows}
        except Exception:
            pass

        # Resolve category names in one query
        cat_ids = {p.category_id for p in plugins if p.category_id}
        cat_map = {}
        if cat_ids:
            cats = self.db.query(PluginCategory).filter(PluginCategory.id.in_(cat_ids)).all()
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

    async def _browse_marketplace_skills(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Browse/search global marketplace skills (workspace_id IS NULL)."""
        from core.models.core import Skill
        from core.models.marketplace_plugins import WorkspaceEnabledSkill

        query = self.db.query(Skill).filter(
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
                self.db.query(WorkspaceEnabledSkill.skill_id)
                .filter(WorkspaceEnabledSkill.workspace_id == self.workspace_id)
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

    async def _list_workspace_plugins(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List plugins enabled for this workspace."""
        from core.models.marketplace_plugins import (
            WorkspaceEnabledPlugin, MarketplacePlugin, PluginCategory,
        )

        rows = (
            self.db.query(WorkspaceEnabledPlugin, MarketplacePlugin)
            .join(MarketplacePlugin, WorkspaceEnabledPlugin.plugin_id == MarketplacePlugin.id)
            .filter(WorkspaceEnabledPlugin.workspace_id == self.workspace_id)
            .order_by(WorkspaceEnabledPlugin.enabled_at.desc())
            .all()
        )

        # Resolve category names
        cat_ids = {mp.category_id for _, mp in rows if mp.category_id}
        cat_map = {}
        if cat_ids:
            cats = self.db.query(PluginCategory).filter(PluginCategory.id.in_(cat_ids)).all()
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

    async def _list_workspace_skills(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List skills enabled for this workspace."""
        from core.models.core import Skill
        from core.models.marketplace_plugins import WorkspaceEnabledSkill

        rows = (
            self.db.query(WorkspaceEnabledSkill, Skill)
            .join(Skill, WorkspaceEnabledSkill.skill_id == Skill.id)
            .filter(WorkspaceEnabledSkill.workspace_id == self.workspace_id)
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

    async def _list_workspace_models(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List LLM models installed for this workspace + default models."""
        from core.models.core import LLMModel, WorkspaceModel

        # Workspace-installed models
        installed = (
            self.db.query(WorkspaceModel, LLMModel)
            .join(LLMModel, WorkspaceModel.model_id == LLMModel.id)
            .filter(
                WorkspaceModel.workspace_id == self.workspace_id,
                WorkspaceModel.is_active == True,
            )
            .all()
        )

        installed_llm_ids = {wm.model_id for wm, _ in installed}

        # Default models (available to all workspaces, not already in installed set)
        defaults = (
            self.db.query(LLMModel)
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

    async def _install_plugin(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enable a marketplace plugin for this workspace."""
        from core.models.marketplace_plugins import (
            MarketplacePlugin, WorkspaceEnabledPlugin,
        )

        plugin_id = params.get("plugin_id")
        plugin_slug = params.get("plugin_slug")

        if not plugin_id and not plugin_slug:
            return {"success": False, "error": "Provide plugin_id or plugin_slug"}

        # Resolve plugin
        query = self.db.query(MarketplacePlugin)
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
            self.db.query(WorkspaceEnabledPlugin)
            .filter(
                WorkspaceEnabledPlugin.workspace_id == self.workspace_id,
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
            workspace_id=self.workspace_id,
            plugin_id=plugin.id,
        )
        self.db.add(junction)

        # Increment enable_count
        plugin.enable_count = (plugin.enable_count or 0) + 1
        self.db.flush()

        logger.info(
            "[PlatformExecutor] Installed plugin '%s' (id=%s) for workspace %s",
            plugin.name, plugin.id, self.workspace_id,
        )

        return {
            "success": True,
            "plugin": {"id": str(plugin.id), "slug": plugin.slug, "name": plugin.name},
            "message": f"Plugin '{plugin.name}' enabled for this workspace.",
        }

    async def _install_skill(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enable a marketplace skill for this workspace."""
        from core.models.core import Skill
        from core.models.marketplace_plugins import WorkspaceEnabledSkill

        skill_id = params.get("skill_id")
        skill_name = params.get("skill_name")

        if not skill_id and not skill_name:
            return {"success": False, "error": "Provide skill_id or skill_name"}

        # Resolve skill
        query = self.db.query(Skill).filter(Skill.workspace_id.is_(None))
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
            self.db.query(WorkspaceEnabledSkill)
            .filter(
                WorkspaceEnabledSkill.workspace_id == self.workspace_id,
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
            workspace_id=self.workspace_id,
            skill_id=skill.id,
        )
        self.db.add(junction)
        self.db.flush()

        logger.info(
            "[PlatformExecutor] Installed skill '%s' (id=%d) for workspace %s",
            skill.name, skill.id, self.workspace_id,
        )

        return {
            "success": True,
            "skill": {"id": skill.id, "name": skill.name},
            "message": f"Skill '{skill.name}' enabled for this workspace.",
        }

    async def _install_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Install an LLM model for this workspace from the OpenRouter catalog."""
        from core.models.core import LLMModel, WorkspaceModel
        from core.models.openrouter_cache import OpenRouterModelCache

        model_id = params.get("model_id")
        if not model_id:
            return {"success": False, "error": "Missing required parameter: model_id"}

        # Find or create LLMModel from OpenRouter cache
        llm = self.db.query(LLMModel).filter(LLMModel.model_id == model_id).first()
        if not llm:
            cached = self.db.query(OpenRouterModelCache).filter(
                OpenRouterModelCache.model_id == model_id,
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
            self.db.add(llm)
            self.db.flush()
            logger.info("[PlatformExecutor] Auto-created LLMModel from cache: %s", model_id)

        # Check for existing workspace install
        existing = (
            self.db.query(WorkspaceModel)
            .filter(
                WorkspaceModel.workspace_id == self.workspace_id,
                WorkspaceModel.model_id == llm.id,
            )
            .first()
        )

        if existing:
            if existing.is_active:
                return {
                    "success": True,
                    "already_installed": True,
                    "model": {"model_id": llm.model_id, "display_name": llm.display_name},
                    "message": f"Model '{llm.display_name}' is already installed.",
                }
            # Re-activate
            existing.is_active = True
            self.db.flush()
            logger.info("[PlatformExecutor] Re-activated model '%s' for workspace %s", model_id, self.workspace_id)
            return {
                "success": True,
                "reactivated": True,
                "model": {"model_id": llm.model_id, "display_name": llm.display_name},
                "message": f"Model '{llm.display_name}' re-activated for this workspace.",
            }

        # Create new workspace install
        wm = WorkspaceModel(
            workspace_id=self.workspace_id,
            model_id=llm.id,
            source="marketplace",
        )
        self.db.add(wm)

        # Increment install_count
        llm.install_count = (llm.install_count or 0) + 1
        self.db.flush()

        logger.info(
            "[PlatformExecutor] Installed model '%s' for workspace %s",
            model_id, self.workspace_id,
        )

        return {
            "success": True,
            "model": {"model_id": llm.model_id, "display_name": llm.display_name},
            "message": f"Model '{llm.display_name}' installed for this workspace.",
        }

    # ══════════════════════════════════════════════════════════════════
    # AGENT ASSIGNMENT HANDLERS (PRD-71)
    # ══════════════════════════════════════════════════════════════════

    def _resolve_agent(self, params: Dict[str, Any]):
        """Resolve agent by ID or name within this workspace. Returns (agent, error_dict)."""
        from core.models import Agent

        agent_id = params.get("agent_id")
        agent_name = params.get("agent_name")

        if not agent_id and not agent_name:
            return None, {"success": False, "error": "Provide agent_id or agent_name"}

        query = self.db.query(Agent).filter(Agent.workspace_id == self.workspace_id)
        if agent_id:
            query = query.filter(Agent.id == agent_id)
        else:
            query = query.filter(Agent.name.ilike(f"%{agent_name}%"))

        agent = query.first()
        if not agent:
            return None, {"success": False, "error": "Agent not found in this workspace"}

        return agent, None

    async def _assign_tool_to_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assign a Composio tool/app to an agent."""
        from core.models.composio_cache import AgentAppAssignment

        agent, err = self._resolve_agent(params)
        if err:
            return err

        app_name = params.get("app_name")
        if not app_name:
            return {"success": False, "error": "Missing required parameter: app_name"}

        app_name = app_name.upper()

        # Idempotency: check existing assignment
        existing = (
            self.db.query(AgentAppAssignment)
            .filter(
                AgentAppAssignment.agent_id == agent.id,
                AgentAppAssignment.app_name == app_name,
            )
            .first()
        )

        if existing:
            if existing.is_active:
                return {
                    "success": True,
                    "already_assigned": True,
                    "agent": {"id": agent.id, "name": agent.name},
                    "app_name": app_name,
                    "message": f"Tool '{app_name}' is already assigned to agent '{agent.name}'.",
                }
            # Re-activate
            existing.is_active = True
            self.db.flush()
            logger.info("[PlatformExecutor] Re-activated tool '%s' for agent %d", app_name, agent.id)
            return {
                "success": True,
                "reactivated": True,
                "agent": {"id": agent.id, "name": agent.name},
                "app_name": app_name,
                "message": f"Tool '{app_name}' re-activated for agent '{agent.name}'.",
            }

        # Create assignment
        assignment = AgentAppAssignment(
            agent_id=agent.id,
            app_name=app_name,
            app_type="EXTERNAL",
            is_active=True,
        )
        self.db.add(assignment)
        self.db.flush()

        logger.info("[PlatformExecutor] Assigned tool '%s' to agent '%s' (id=%d)", app_name, agent.name, agent.id)

        return {
            "success": True,
            "agent": {"id": agent.id, "name": agent.name},
            "app_name": app_name,
            "message": f"Tool '{app_name}' assigned to agent '{agent.name}'.",
        }

    async def _assign_skill_to_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assign a skill to an agent via the agent_skills M2M table."""
        from core.models.core import Skill, agent_skills

        agent, err = self._resolve_agent(params)
        if err:
            return err

        skill_id = params.get("skill_id")
        skill_name = params.get("skill_name")

        if not skill_id and not skill_name:
            return {"success": False, "error": "Provide skill_id or skill_name"}

        # Resolve skill
        query = self.db.query(Skill)
        if skill_id:
            query = query.filter(Skill.id == skill_id)
        else:
            query = query.filter(Skill.name.ilike(f"%{skill_name}%"))

        skill = query.first()
        if not skill:
            return {"success": False, "error": "Skill not found"}

        # Idempotency: check if already assigned
        from sqlalchemy import select as sa_select
        existing = self.db.execute(
            sa_select(agent_skills).where(
                agent_skills.c.agent_id == agent.id,
                agent_skills.c.skill_id == skill.id,
            )
        ).first()

        if existing:
            return {
                "success": True,
                "already_assigned": True,
                "agent": {"id": agent.id, "name": agent.name},
                "skill": {"id": skill.id, "name": skill.name},
                "message": f"Skill '{skill.name}' is already assigned to agent '{agent.name}'.",
            }

        # Insert into M2M table
        self.db.execute(
            agent_skills.insert().values(agent_id=agent.id, skill_id=skill.id)
        )
        self.db.flush()

        logger.info("[PlatformExecutor] Assigned skill '%s' (id=%d) to agent '%s' (id=%d)",
                     skill.name, skill.id, agent.name, agent.id)

        return {
            "success": True,
            "agent": {"id": agent.id, "name": agent.name},
            "skill": {"id": skill.id, "name": skill.name},
            "message": f"Skill '{skill.name}' assigned to agent '{agent.name}'.",
        }

    async def _assign_plugin_to_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assign a marketplace plugin to an agent."""
        from core.models.marketplace_plugins import (
            MarketplacePlugin, WorkspaceEnabledPlugin, AgentAssignedPlugin,
        )

        agent, err = self._resolve_agent(params)
        if err:
            return err

        plugin_id = params.get("plugin_id")
        plugin_slug = params.get("plugin_slug")

        if not plugin_id and not plugin_slug:
            return {"success": False, "error": "Provide plugin_id or plugin_slug"}

        # Resolve plugin
        query = self.db.query(MarketplacePlugin)
        if plugin_id:
            from uuid import UUID as _UUID
            query = query.filter(MarketplacePlugin.id == _UUID(str(plugin_id)))
        else:
            query = query.filter(MarketplacePlugin.slug == plugin_slug)

        plugin = query.first()
        if not plugin:
            return {"success": False, "error": "Plugin not found"}

        # Verify plugin is enabled for this workspace
        ws_enabled = (
            self.db.query(WorkspaceEnabledPlugin)
            .filter(
                WorkspaceEnabledPlugin.workspace_id == self.workspace_id,
                WorkspaceEnabledPlugin.plugin_id == plugin.id,
            )
            .first()
        )
        if not ws_enabled:
            return {
                "success": False,
                "error": f"Plugin '{plugin.name}' is not enabled for this workspace. Install it first with platform_install_plugin.",
            }

        # Idempotency check
        existing = (
            self.db.query(AgentAssignedPlugin)
            .filter(
                AgentAssignedPlugin.agent_id == agent.id,
                AgentAssignedPlugin.plugin_id == plugin.id,
            )
            .first()
        )
        if existing:
            return {
                "success": True,
                "already_assigned": True,
                "agent": {"id": agent.id, "name": agent.name},
                "plugin": {"id": str(plugin.id), "slug": plugin.slug, "name": plugin.name},
                "message": f"Plugin '{plugin.name}' is already assigned to agent '{agent.name}'.",
            }

        # Create assignment
        assignment = AgentAssignedPlugin(
            agent_id=agent.id,
            plugin_id=plugin.id,
        )
        self.db.add(assignment)
        self.db.flush()

        logger.info("[PlatformExecutor] Assigned plugin '%s' to agent '%s' (id=%d)",
                     plugin.name, agent.name, agent.id)

        return {
            "success": True,
            "agent": {"id": agent.id, "name": agent.name},
            "plugin": {"id": str(plugin.id), "slug": plugin.slug, "name": plugin.name},
            "message": f"Plugin '{plugin.name}' assigned to agent '{agent.name}'.",
        }

    # ══════════════════════════════════════════════════════════════════
    # AGENT HEARTBEAT CONFIGURATION (PRD-71)
    # ══════════════════════════════════════════════════════════════════

    async def _configure_agent_heartbeat(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Configure or update the heartbeat schedule for an agent."""
        from sqlalchemy.orm.attributes import flag_modified

        agent, err = self._resolve_agent(params)
        if err:
            return err

        # Read current configuration (immutable pattern — build new dict)
        config = dict(agent.configuration or {})
        hb = dict(config.get("heartbeat", {}))

        changes = []

        # Apply each provided field
        if "enabled" in params:
            hb["enabled"] = bool(params["enabled"])
            changes.append(f"enabled → {hb['enabled']}")

        if "interval_minutes" in params:
            minutes = max(5, min(1440, int(params["interval_minutes"])))
            hb["interval_minutes"] = minutes
            changes.append(f"interval → {minutes}m")

        if "prompt" in params:
            hb["prompt"] = str(params["prompt"])[:2000]
            changes.append("prompt updated")

        if "auto_act" in params:
            hb["auto_act"] = bool(params["auto_act"])
            changes.append(f"auto_act → {hb['auto_act']}")

        if "active_hours_start" in params:
            hb["active_hours_start"] = str(params["active_hours_start"])
            changes.append(f"active_hours_start → {hb['active_hours_start']}")

        if "active_hours_end" in params:
            hb["active_hours_end"] = str(params["active_hours_end"])
            changes.append(f"active_hours_end → {hb['active_hours_end']}")

        if "proactive_level" in params:
            level = str(params["proactive_level"])
            if level in ("silent", "notify", "act_notify", "autonomous"):
                hb["proactive_level"] = level
                changes.append(f"proactive_level → {level}")

        if "notification_channel" in params:
            hb["notification_channel"] = str(params["notification_channel"])
            changes.append(f"notification_channel → {hb['notification_channel']}")

        if "checklist" in params:
            hb["checklist"] = str(params["checklist"])[:5000]
            changes.append("checklist updated")

        if not changes:
            return {
                "success": True,
                "message": "No changes specified",
                "current_heartbeat": hb,
                "agent_id": agent.id,
            }

        # Write back (immutable: new dict, not mutation)
        config["heartbeat"] = hb
        agent.configuration = config
        flag_modified(agent, "configuration")
        self.db.flush()

        logger.info(
            "[PlatformExecutor] Configured heartbeat for agent '%s' (id=%d): %s",
            agent.name, agent.id, ", ".join(changes),
        )

        # Note: heartbeat schedule will be picked up on next service reload.
        # Live rescheduling requires the HeartbeatService singleton (future enhancement).

        return {
            "success": True,
            "agent": {"id": agent.id, "name": agent.name},
            "heartbeat": hb,
            "changes": changes,
            "message": f"Heartbeat for agent '{agent.name}' configured: {', '.join(changes)}",
        }

    # ── PRD-76: Agent Reports ────────────────────────────────────────

    async def _submit_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a report: write file to workspace + insert DB row."""
        from services.report_service import ReportService

        title = params.get("title")
        content = params.get("content")
        report_type = params.get("report_type", "standup")
        status = params.get("status", "ok")

        if not title or not content:
            return {"success": False, "error": "title and content are required"}

        valid_types = {"standup", "research", "incident", "summary", "delivery", "audit"}
        if report_type not in valid_types:
            return {"success": False, "error": f"report_type must be one of: {', '.join(sorted(valid_types))}"}

        valid_statuses = {"ok", "warning", "critical", "info"}
        if status not in valid_statuses:
            return {"success": False, "error": f"status must be one of: {', '.join(sorted(valid_statuses))}"}

        # Resolve agent context — the calling agent's ID is passed via execution context
        agent_id = params.get("_agent_id")
        agent_name = params.get("_agent_name", "unknown")

        if not agent_id:
            # Fallback: try to find from params
            agent_id = params.get("agent_id")
            if not agent_id:
                return {"success": False, "error": "Could not determine calling agent"}

            from core.models import Agent
            agent = self.db.query(Agent).filter(
                Agent.id == agent_id,
                Agent.workspace_id == self.workspace_id,
            ).first()
            if not agent:
                return {"success": False, "error": f"Agent {agent_id} not found in workspace"}
            agent_name = agent.name

        svc = ReportService(self.db, self.workspace_id)
        return await svc.create_report(
            agent_id=agent_id,
            agent_name=agent_name,
            title=title,
            content=content,
            report_type=report_type,
            status=status,
            summary=params.get("summary"),
            metrics=params.get("metrics"),
            attachments=params.get("attachments"),
            heartbeat_result_id=params.get("_heartbeat_result_id"),
        )

    async def _get_latest_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get the most recent report from a specific agent."""
        from services.report_service import ReportService

        agent_name = params.get("agent_name")
        agent_id = params.get("agent_id")
        report_type = params.get("report_type")

        if not agent_name and not agent_id:
            return {"success": False, "error": "Provide agent_name or agent_id"}

        svc = ReportService(self.db, self.workspace_id)
        return await svc.get_latest_report(
            agent_name=agent_name,
            agent_id=agent_id,
            report_type=report_type,
        )

    # ------------------------------------------------------------------
    # PRD-72: Board Tasks
    # ------------------------------------------------------------------

    async def _create_board_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a board task (called by agents via platform_create_task)."""
        from core.models.core import BoardTask

        title = params.get("title")
        description = params.get("description")
        if not title or not description:
            return {"success": False, "error": "title and description are required"}

        # Resolve assigned agent by name
        assigned_agent_id = None
        agent_name = params.get("assigned_agent_name")
        if agent_name:
            from core.models import Agent
            from sqlalchemy import func as sa_func
            agent = self.db.query(Agent).filter(
                Agent.workspace_id == self.workspace_id,
                sa_func.lower(Agent.name) == agent_name.lower(),
            ).first()
            if agent:
                assigned_agent_id = agent.id

        task = BoardTask(
            workspace_id=self.workspace_id,
            title=title,
            description=description,
            priority=params.get("priority", "medium"),
            assigned_agent_id=assigned_agent_id,
            status="assigned" if assigned_agent_id else "inbox",
            created_by_type="agent",
            created_by_id=str(params.get("_agent_id", "")),
            parent_task_id=params.get("parent_task_id"),
            tags=params.get("tags", []),
        )
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)

        return {
            "success": True,
            "task_id": task.id,
            "status": task.status,
            "title": task.title,
        }

    async def _list_board_tasks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List board tasks with optional filters."""
        from core.models.core import BoardTask

        query = self.db.query(BoardTask).filter(
            BoardTask.workspace_id == self.workspace_id,
        )

        status = params.get("status")
        if status:
            query = query.filter(BoardTask.status == status)

        priority = params.get("priority")
        if priority:
            query = query.filter(BoardTask.priority == priority)

        agent_name = params.get("assigned_agent_name")
        if agent_name:
            from core.models import Agent
            from sqlalchemy import func as sa_func
            agent = self.db.query(Agent).filter(
                Agent.workspace_id == self.workspace_id,
                sa_func.lower(Agent.name) == agent_name.lower(),
            ).first()
            if agent:
                query = query.filter(BoardTask.assigned_agent_id == agent.id)
            else:
                return {"success": True, "tasks": [], "total": 0, "note": f"No agent named '{agent_name}' found"}

        limit = min(int(params.get("limit", 20)), 50)
        tasks = query.order_by(BoardTask.created_at.desc()).limit(limit).all()

        # Enrich with agent names
        agent_ids = {t.assigned_agent_id for t in tasks if t.assigned_agent_id}
        agents_map = {}
        if agent_ids:
            from core.models import Agent
            for a in self.db.query(Agent).filter(Agent.id.in_(agent_ids)).all():
                agents_map[a.id] = a.name

        result = []
        for t in tasks:
            result.append({
                "id": t.id,
                "title": t.title,
                "status": t.status,
                "priority": t.priority,
                "assigned_agent": agents_map.get(t.assigned_agent_id, "unassigned"),
                "created_at": str(t.created_at) if t.created_at else None,
                "started_at": str(t.started_at) if t.started_at else None,
                "completed_at": str(t.completed_at) if t.completed_at else None,
                "error_message": t.error_message,
            })

        return {"success": True, "tasks": result, "total": len(result)}

    async def _board_summary(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of the task board: counts, busiest agents, failures."""
        from core.models.core import BoardTask
        from core.models import Agent
        from sqlalchemy import func as sa_func

        all_tasks = self.db.query(BoardTask).filter(
            BoardTask.workspace_id == self.workspace_id,
        ).all()

        # Counts by status
        by_status: Dict[str, int] = {}
        by_priority: Dict[str, int] = {}
        agent_task_counts: Dict[int, int] = {}
        failed_tasks = []

        for t in all_tasks:
            by_status[t.status] = by_status.get(t.status, 0) + 1
            by_priority[t.priority] = by_priority.get(t.priority, 0) + 1
            if t.assigned_agent_id:
                agent_task_counts[t.assigned_agent_id] = agent_task_counts.get(t.assigned_agent_id, 0) + 1
            if t.error_message:
                failed_tasks.append({"id": t.id, "title": t.title, "error": t.error_message[:200]})

        # Resolve agent names for busiest
        busiest_agents = []
        if agent_task_counts:
            sorted_agents = sorted(agent_task_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            agent_ids = [a[0] for a in sorted_agents]
            agents_map = {
                a.id: a.name
                for a in self.db.query(Agent).filter(Agent.id.in_(agent_ids)).all()
            }
            busiest_agents = [
                {"agent": agents_map.get(aid, f"Agent {aid}"), "task_count": count}
                for aid, count in sorted_agents
            ]

        return {
            "success": True,
            "total_tasks": len(all_tasks),
            "by_status": by_status,
            "by_priority": by_priority,
            "busiest_agents": busiest_agents,
            "failed_tasks": failed_tasks[:5],
        }

    async def _get_board_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get full details of a single board task."""
        from core.models.core import BoardTask

        task_id = params.get("task_id")
        if not task_id:
            return {"success": False, "error": "task_id is required"}

        task = self.db.query(BoardTask).filter(
            BoardTask.id == int(task_id),
            BoardTask.workspace_id == self.workspace_id,
        ).first()

        if not task:
            return {"success": False, "error": f"Task {task_id} not found"}

        # Resolve agent name
        agent_name = None
        if task.assigned_agent_id:
            from core.models import Agent
            agent = self.db.query(Agent).get(task.assigned_agent_id)
            agent_name = agent.name if agent else None

        return {
            "success": True,
            "task": {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "raw_prompt": task.raw_prompt,
                "status": task.status,
                "priority": task.priority,
                "review_mode": task.review_mode,
                "assigned_agent": agent_name or "unassigned",
                "tags": task.tags or [],
                "result": str(task.result)[:2000] if task.result else None,
                "error_message": task.error_message,
                "created_at": str(task.created_at) if task.created_at else None,
                "started_at": str(task.started_at) if task.started_at else None,
                "completed_at": str(task.completed_at) if task.completed_at else None,
            },
        }

    async def _assign_board_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assign a board task to an agent by name."""
        from core.models.core import BoardTask
        from core.models import Agent
        from sqlalchemy import func as sa_func

        task_id = params.get("task_id")
        agent_name = params.get("agent_name")
        if not task_id or not agent_name:
            return {"success": False, "error": "task_id and agent_name are required"}

        task = self.db.query(BoardTask).filter(
            BoardTask.id == int(task_id),
            BoardTask.workspace_id == self.workspace_id,
        ).first()
        if not task:
            return {"success": False, "error": f"Task {task_id} not found"}

        agent = self.db.query(Agent).filter(
            Agent.workspace_id == self.workspace_id,
            sa_func.lower(Agent.name) == agent_name.lower(),
        ).first()
        if not agent:
            return {"success": False, "error": f"Agent '{agent_name}' not found"}

        task.assigned_agent_id = agent.id
        if task.status == "inbox":
            task.status = "assigned"
        self.db.commit()

        return {
            "success": True,
            "task_id": task.id,
            "assigned_agent": agent.name,
            "status": task.status,
        }

    async def _update_board_task_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update a board task's status. Moving to in_progress triggers execution."""
        from core.models.core import BoardTask
        from datetime import datetime, timezone

        task_id = params.get("task_id")
        new_status = params.get("status")
        if not task_id or not new_status:
            return {"success": False, "error": "task_id and status are required"}

        valid = {"inbox", "assigned", "in_progress", "review", "done"}
        if new_status not in valid:
            return {"success": False, "error": f"Invalid status: {new_status}. Must be one of {valid}"}

        task = self.db.query(BoardTask).filter(
            BoardTask.id == int(task_id),
            BoardTask.workspace_id == self.workspace_id,
        ).first()
        if not task:
            return {"success": False, "error": f"Task {task_id} not found"}

        task.status = new_status
        if new_status == "in_progress" and not task.started_at:
            task.started_at = datetime.now(timezone.utc)
        if new_status in ("done", "review") and not task.completed_at:
            task.completed_at = datetime.now(timezone.utc)

        self.db.commit()

        # Trigger agent execution if moved to in_progress with an assigned agent
        if new_status == "in_progress" and task.assigned_agent_id:
            from api.board_tasks import _launch_task_execution
            _launch_task_execution(
                task_id=task.id,
                agent_id=task.assigned_agent_id,
                workspace_id=str(self.workspace_id),
                prompt=task.raw_prompt or task.description or task.title,
                review_mode=task.review_mode or "auto",
            )

        return {
            "success": True,
            "task_id": task.id,
            "status": task.status,
            "triggered_execution": new_status == "in_progress" and task.assigned_agent_id is not None,
        }

    # ------------------------------------------------------------------
    # PRD-77: Agent Self-Scheduling
    # ------------------------------------------------------------------

    async def _schedule_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a follow-up task for self or another agent."""
        from services.scheduled_task_service import ScheduledTaskService

        task_type = params.get("task_type")
        description = params.get("description")
        schedule = params.get("schedule")

        if not task_type or not description or not schedule:
            return {"success": False, "error": "task_type, description, and schedule are required"}

        # Resolve calling agent
        created_by_agent_id = params.get("_agent_id")
        if not created_by_agent_id:
            return {"success": False, "error": "Could not determine calling agent"}

        # Resolve target agent (default: self)
        target_agent_id = created_by_agent_id
        target_name = params.get("target_agent_name")
        if target_name:
            from core.models import Agent
            target = self.db.query(Agent).filter(
                Agent.workspace_id == self.workspace_id,
                func.lower(Agent.name) == target_name.lower(),
            ).first()
            if not target:
                return {"success": False, "error": f"Agent '{target_name}' not found in workspace"}
            target_agent_id = target.id

        svc = ScheduledTaskService(self.db, self.workspace_id)
        return await svc.create_task(
            created_by_agent_id=created_by_agent_id,
            target_agent_id=target_agent_id,
            task_type=task_type,
            description=description,
            schedule=schedule,
            max_runs=params.get("max_runs"),
        )

    async def _list_scheduled_tasks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List scheduled tasks for the workspace."""
        from services.scheduled_task_service import ScheduledTaskService

        # Resolve optional agent_name to agent_id
        agent_id = None
        agent_name = params.get("agent_name")
        if agent_name:
            from core.models import Agent
            agent = self.db.query(Agent).filter(
                Agent.workspace_id == self.workspace_id,
                func.lower(Agent.name) == agent_name.lower(),
            ).first()
            if not agent:
                return {"success": False, "error": f"Agent '{agent_name}' not found in workspace"}
            agent_id = agent.id

        svc = ScheduledTaskService(self.db, self.workspace_id)
        return await svc.list_tasks(
            agent_id=agent_id,
            status=params.get("status"),
        )

    async def _cancel_scheduled_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel a scheduled task by ID."""
        from services.scheduled_task_service import ScheduledTaskService

        task_id = params.get("task_id")
        if not task_id:
            return {"success": False, "error": "task_id is required"}

        svc = ScheduledTaskService(self.db, self.workspace_id)
        return await svc.update_task_status(task_id, "cancelled")

    # ------------------------------------------------------------------
    # PRD-77: Memory Browsing
    # ------------------------------------------------------------------

    async def _browse_memories(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Browse/search memories via Mem0."""
        import asyncio

        try:
            from modules.memory.integrations.mem0_client import Mem0Client

            client = Mem0Client()
            user_id = f"ws_{self.workspace_id}"
            limit = params.get("limit", 20)
            query = params.get("query")

            loop = asyncio.get_event_loop()
            if query:
                results = await loop.run_in_executor(
                    None, lambda: client.search(query=query, user_id=user_id, limit=limit),
                )
            else:
                results = await loop.run_in_executor(
                    None, lambda: client.get_all(user_id=user_id, limit=limit),
                )

            # Normalise to consistent format
            memories = []
            for m in results:
                if isinstance(m, dict):
                    memories.append({
                        "id": m.get("id"),
                        "content": m.get("memory") or m.get("content", ""),
                        "score": m.get("score"),
                        "metadata": m.get("metadata") or m.get("metadata_"),
                        "created_at": m.get("created_at"),
                    })

            return {
                "success": True,
                "memories": memories,
                "total": len(memories),
                "source": "mem0",
                "search_query": query,
            }
        except Exception as e:
            logger.error("[PlatformExecutor] browse_memories failed: %s", e, exc_info=True)
            return {"success": False, "error": f"Memory service unavailable: {str(e)[:200]}"}

    async def _delete_memory(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a memory by ID with workspace ownership check."""
        import asyncio

        memory_id = params.get("memory_id")
        if not memory_id:
            return {"success": False, "error": "memory_id is required"}

        try:
            from modules.memory.integrations.mem0_client import Mem0Client

            client = Mem0Client()
            user_id = f"ws_{self.workspace_id}"
            loop = asyncio.get_event_loop()

            # Ownership check
            all_mems = await loop.run_in_executor(
                None, lambda: client.get_all(user_id=user_id, limit=500),
            )
            owned_ids = {str(m.get("id", "")) for m in (all_mems if isinstance(all_mems, list) else [])}
            if memory_id not in owned_ids:
                return {"success": False, "error": "Memory not found or not owned by this workspace"}

            deleted = await loop.run_in_executor(None, lambda: client.delete(memory_id))

            if deleted:
                return {"success": True, "message": f"Memory {memory_id} deleted"}
            return {"success": False, "error": f"Failed to delete memory {memory_id}"}
        except Exception as e:
            logger.error("[PlatformExecutor] delete_memory failed: %s", e, exc_info=True)
            return {"success": False, "error": f"Memory service unavailable: {str(e)[:200]}"}
