"""
Platform Action Executor (PRD-64)
==================================

Thin dispatcher that routes platform actions to domain-specific handler modules.
Each handler is a standalone async function in modules/tools/discovery/handlers_*.py.

All queries are workspace-scoped for multi-tenant isolation.
"""

import logging
from typing import Any, Callable, Dict
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.orm import Session

from modules.tools.discovery.handlers_agents import (
    list_agents,
    get_agent,
    create_agent,
    update_agent,
    delete_agent,
)
from modules.tools.discovery.handlers_playbooks import (
    list_playbooks,
    get_playbook,
    create_playbook,
    update_playbook,
    add_playbook_step,
    update_playbook_step,
    delete_playbook_step,
    execute_playbook,
    get_playbook_execution,
    delete_playbook,
)
from modules.tools.discovery.handlers_analytics import (
    get_llm_usage,
    get_cost_breakdown,
    workspace_stats,
    board_summary,
)
from modules.tools.discovery.handlers_documents import (
    list_documents,
    delete_document,
    reprocess_document,
)
from modules.tools.discovery.handlers_workspace import (
    get_workspace_info,
    get_memory_stats,
    list_connected_apps,
    store_memory,
)
from modules.tools.discovery.handlers_monitoring import (
    get_logs,
    list_services,
    query_loki_logs,
    query_prometheus,
    get_alerts,
    get_system_health,
)
from modules.tools.discovery.handlers_search import (
    search_chat_history,
    search_memory,
    browse_memories,
    delete_memory,
)
from modules.tools.discovery.handlers_tools_llms import (
    list_tools,
    list_llms,
    list_datasources,
)
from modules.tools.discovery.handlers_marketplace import (
    browse_marketplace_agents,
    browse_marketplace_plugins,
    browse_marketplace_skills,
    list_workspace_plugins,
    list_workspace_skills,
    list_workspace_models,
    install_plugin,
    install_skill,
    install_model,
)
from modules.tools.discovery.handlers_board_tasks import (
    create_board_task,
    list_board_tasks,
    get_board_task,
    assign_board_task,
    update_board_task_status,
)
from modules.tools.discovery.handlers_scheduling import (
    schedule_task,
    list_scheduled_tasks,
    cancel_scheduled_task,
    query_data,
)
from modules.tools.discovery.handlers_reports import (
    submit_report,
    get_latest_report,
)
from modules.tools.discovery.handlers_assignments import (
    assign_tool_to_agent,
    assign_skill_to_agent,
    assign_plugin_to_agent,
    configure_agent_heartbeat,
)
from modules.tools.discovery.handlers_activity import (
    get_activity_feed,
)
from modules.tools.discovery.handlers_field import (
    field_query,
    field_inject,
    field_stability,
)
from modules.tools.discovery.handlers_blog import (
    publish_blog_post,
    list_blog_posts,
    get_blog_post,
    update_blog_post,
)
from modules.tools.discovery.handlers_missions import (
    create_mission,
    list_missions,
    get_mission,
)
from modules.tools.discovery.handlers_governance import (
    list_blueprints,
    get_blueprint,
    create_blueprint,
    update_blueprint,
    validate_agent_handler,
    check_budget_handler,
)
from modules.tools.discovery.handlers_analytics_enhanced import (
    get_success_rate,
    get_completion_time,
    get_error_rates,
    get_queue_depth,
    get_efficiency_score,
    get_cost_per_execution,
    get_peak_hours,
    get_bottlenecks,
    get_predictive_alerts,
    get_agent_ranking,
    get_sla_compliance,
)

logger = logging.getLogger(__name__)


class PlatformActionExecutor:
    """
    Executes platform actions using direct database queries.
    Workspace-scoped for multi-tenant isolation.
    """

    def __init__(self, db: Session, workspace_id: UUID):
        self.db = db
        self.workspace_id = workspace_id
        self._handlers: Dict[str, Callable] = {
            # Read actions
            "platform_list_agents": list_agents,
            "platform_get_agent": get_agent,
            "platform_list_recipes": list_playbooks,
            "platform_get_recipe": get_playbook,
            "platform_get_llm_usage": get_llm_usage,
            "platform_get_cost_breakdown": get_cost_breakdown,
            "platform_list_documents": list_documents,
            "platform_get_workspace_info": get_workspace_info,
            "platform_get_memory_stats": get_memory_stats,
            "platform_list_connected_apps": list_connected_apps,
            # Write actions
            "platform_create_agent": create_agent,
            "platform_update_agent": update_agent,
            "platform_create_recipe": create_playbook,
            "platform_update_recipe": update_playbook,
            "platform_add_recipe_step": add_playbook_step,
            "platform_update_recipe_step": update_playbook_step,
            "platform_delete_recipe_step": delete_playbook_step,
            "platform_store_memory": store_memory,
            "platform_delete_agent": delete_agent,
            # Infrastructure / observability
            "platform_get_logs": get_logs,
            "platform_list_services": list_services,
            # Chat & memory search
            "platform_search_chat_history": search_chat_history,
            "platform_search_memory": search_memory,
            # PRD-73: Monitoring (Loki, Prometheus, Alerts)
            "platform_query_loki_logs": query_loki_logs,
            "platform_query_prometheus": query_prometheus,
            "platform_get_alerts": get_alerts,
            # Visibility / discovery
            "platform_list_tools": list_tools,
            "platform_list_llms": list_llms,
            "platform_list_datasources": list_datasources,
            "platform_workspace_stats": workspace_stats,
            # Self-management
            "platform_execute_recipe": execute_playbook,
            "platform_get_recipe_execution": get_playbook_execution,
            "platform_get_system_health": get_system_health,
            "platform_delete_document": delete_document,
            "platform_reprocess_document": reprocess_document,
            "platform_delete_recipe": delete_playbook,
            "platform_get_activity_feed": get_activity_feed,
            # Marketplace discovery & workspace inventory (PRD-71)
            "platform_browse_marketplace_agents": browse_marketplace_agents,
            "platform_browse_marketplace_plugins": browse_marketplace_plugins,
            "platform_browse_marketplace_skills": browse_marketplace_skills,
            "platform_list_workspace_plugins": list_workspace_plugins,
            "platform_list_workspace_skills": list_workspace_skills,
            "platform_list_workspace_models": list_workspace_models,
            "platform_install_plugin": install_plugin,
            "platform_install_skill": install_skill,
            "platform_install_model": install_model,
            # Agent assignment (PRD-71)
            "platform_assign_tool_to_agent": assign_tool_to_agent,
            "platform_assign_skill_to_agent": assign_skill_to_agent,
            "platform_assign_plugin_to_agent": assign_plugin_to_agent,
            "platform_configure_agent_heartbeat": configure_agent_heartbeat,
            # PRD-76: Agent Reports
            "platform_submit_report": submit_report,
            "platform_get_latest_report": get_latest_report,
            # PRD-72: Board Tasks
            "platform_create_task": create_board_task,
            "platform_list_tasks": list_board_tasks,
            "platform_board_summary": board_summary,
            "platform_get_task": get_board_task,
            "platform_assign_task": assign_board_task,
            "platform_update_task_status": update_board_task_status,
            # PRD-77: Agent Self-Scheduling
            "platform_schedule_task": schedule_task,
            "platform_list_scheduled_tasks": list_scheduled_tasks,
            "platform_cancel_scheduled_task": cancel_scheduled_task,
            # PRD-79: NL2SQL
            "platform_query_data": query_data,
            # PRD-77: Memory Browsing
            "platform_browse_memories": browse_memories,
            "platform_delete_memory": delete_memory,
            # PRD-108: Shared Mission Field
            "platform_field_query": field_query,
            "platform_field_inject": field_inject,
            "platform_field_stability": field_stability,
            # Blog Widget
            "platform_publish_blog_post": publish_blog_post,
            "platform_list_blog_posts": list_blog_posts,
            "platform_get_blog_post": get_blog_post,
            "platform_update_blog_post": update_blog_post,
            # PRD-82A: Missions
            "platform_create_mission": create_mission,
            "platform_list_missions": list_missions,
            "platform_get_mission": get_mission,
            # Governance & Blueprints
            "platform_list_blueprints": list_blueprints,
            "platform_get_blueprint": get_blueprint,
            "platform_create_blueprint": create_blueprint,
            "platform_update_blueprint": update_blueprint,
            "platform_validate_agent": validate_agent_handler,
            "platform_check_budget": check_budget_handler,
            # Enhanced Analytics (dashboard + performance)
            "platform_get_success_rate": get_success_rate,
            "platform_get_completion_time": get_completion_time,
            "platform_get_error_rates": get_error_rates,
            "platform_get_queue_depth": get_queue_depth,
            "platform_get_efficiency_score": get_efficiency_score,
            "platform_get_cost_per_execution": get_cost_per_execution,
            "platform_get_peak_hours": get_peak_hours,
            "platform_get_bottlenecks": get_bottlenecks,
            "platform_get_predictive_alerts": get_predictive_alerts,
            "platform_get_agent_ranking": get_agent_ranking,
            "platform_get_sla_compliance": get_sla_compliance,
        }

    async def execute(self, action_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a platform action by name with permission checking."""
        # LLMs sometimes send params as a JSON string instead of a dict
        if isinstance(params, str):
            try:
                import json
                params = json.loads(params)
            except (json.JSONDecodeError, TypeError):
                return {"success": False, "error": f"Invalid params format: expected dict, got string"}
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

        # PRD-108: Auto-inject field_id for field tools from active mission
        if action_name.startswith("platform_field_") and "field_id" not in params:
            try:
                from core.models.orchestration import OrchestrationRun
                active_run = (
                    self.db.query(OrchestrationRun)
                    .filter(
                        OrchestrationRun.workspace_id == self.workspace_id,
                        OrchestrationRun.state == "running",
                    )
                    .first()
                )
                if active_run:
                    field_id = (active_run.config or {}).get("field_id")
                    if field_id:
                        params = {**params, "field_id": field_id}
                        logger.info(
                            "[PRD-108] Auto-injected field_id %s for %s",
                            field_id, action_name,
                        )
            except Exception as e:
                logger.warning("[PRD-108] Failed to resolve field_id: %s", e)

        try:
            return await handler(self.db, self.workspace_id, params)
        except Exception as e:
            logger.error(f"[PlatformExecutor] {action_name} failed: {e}", exc_info=True)
            try:
                self.db.rollback()
            except Exception:
                pass
            return {"success": False, "error": f"Action '{action_name}' failed"}
