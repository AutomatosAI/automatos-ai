"""
Platform Action Executor (PRD-64)
==================================

Thin dispatcher that routes platform actions to domain-specific handler modules.
Each handler is a standalone async function in modules/tools/discovery/handlers_*.py.

All queries are workspace-scoped for multi-tenant isolation.
"""

import json
import logging
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Union
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.orm import Session

from modules.tools.discovery.handlers_agents import (
    list_agents,
    recommend_agent,
    get_agent,
    create_agent,
    update_agent,
    delete_agent,
    get_agent_heartbeat,
    unassign_skill_from_agent,
    unassign_tool_from_agent,
)
from modules.tools.discovery.handlers_playbooks import (
    list_playbooks,
    get_playbook,
    create_playbook,
    update_playbook,
    add_playbook_step,
    update_playbook_step,
    delete_playbook_step,
    schedule_playbook,
    execute_playbook,
    get_playbook_execution,
    delete_playbook,
)
from modules.tools.discovery.handlers_analytics import (
    get_llm_usage,
    get_cost_breakdown,
    workspace_stats,
    board_summary,
    board_snapshot,
)
from modules.tools.discovery.handlers_documents import (
    list_documents,
    delete_document,
    reprocess_document,
    upload_document,
    read_document,
    grep_documents,
    search_documents,
    list_templates,
    get_template_schema,
    get_brand_kit_tool,  # PRD-251 US-115
    update_brand_kit_tool,  # PRD-251 US-115
)
from modules.tools.discovery.handlers_channels import (  # PRD-143 S10
    list_channels,
    connect_channel,
    configure_channel,
    start_channel,
    stop_channel,
)
from modules.tools.discovery.handlers_widgets import (  # PRD-143 S10
    get_widget_config,
    update_widget_config,
)
from modules.tools.discovery.handlers_workspace import (
    get_workspace_info,
    get_memory_stats,
    list_connected_apps,
    store_memory,
    checkpoint_thread,  # PRD-206 S2
    resume_context,  # PRD-206 S3
    update_workspace_settings,  # PRD-143 S11
    list_system_settings,  # PRD-143 S11
    update_system_setting,  # PRD-143 S11
)
from modules.tools.discovery.handlers_members import (  # PRD-143 S11
    list_members,
    invite_member,
    set_member_role,
    remove_member,
)
from modules.tools.discovery.handlers_api_keys import (  # PRD-143 S11
    list_api_keys,
    revoke_api_key,
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
from modules.tools.discovery.handlers_capabilities import find_tools
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
    uninstall_plugin,  # PRD-143 S11
)
from modules.tools.discovery.handlers_packages import (  # PRD-230 US-006
    search_packages,
    install_package_tool,
    install_marketplace_agent_tool,
)
from modules.tools.discovery.handlers_skills import (
    get_skill_content,
    create_workspace_skill,
    update_skill,
    delete_workspace_skill,
)
from modules.tools.discovery.handlers_skill_runtime import (  # PRD-202 S2/S3/S4
    load_skill,
    run_skill_script,
    set_skill_script_execution,
)
from modules.tools.discovery.handlers_board_tasks import (
    wait_for_board_task,
    create_board_task,
    list_board_tasks,
    get_board_task,
    assign_board_task,
    update_board_task,
    update_board_task_status,
)
from modules.tools.discovery.handlers_scheduling import (
    schedule_task,
    list_scheduled_tasks,
    cancel_scheduled_task,
    get_schedule,
    query_data,
)
from modules.tools.discovery.handlers_reports import (
    submit_report,
    get_latest_report,
    browse_reports,
    acknowledge_report,
    link_report_to_task,
)
from modules.tools.discovery.handlers_deliverables import (  # PRD-164 S3
    list_deliverables as handle_list_deliverables,
    get_deliverable as handle_get_deliverable,
)
from modules.tools.discovery.handlers_harness import (
    harness_status,
    harness_trigger,
    harness_history,
)
from modules.tools.discovery.handlers_routing import create_routing_rule  # PRD-142 Wave 4 (W4-S6)
from modules.tools.discovery.handlers_power import set_power_mode, get_power_mode  # PRD-142 W4-S5 / PRD-143 S10
from modules.tools.discovery.handlers_auto_reporting import (
    get_auto_reporting_prefs,
    update_auto_reporting_prefs,
    send_notification,
)
from modules.tools.discovery.handlers_notifications import notify_owner
from modules.tools.discovery.handlers_autonomy import (
    get_autonomy_level as handle_get_autonomy_level,
    set_autonomy_level as handle_set_autonomy_level,
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
    create_blog_post_from_topic,
    generate_cover_image,
)
from modules.tools.discovery.handlers_watches import (  # PRD-204 S9
    create_watch,
    list_watches,
    get_watch,
    cancel_watch,
)
from modules.tools.discovery.handlers_fleet import fleet_status  # PRD-228
from modules.tools.discovery.handlers_missions import (
    create_mission,
    list_missions,
    get_mission,
    approve_mission,
    reject_mission,
    pause_mission,
    resume_mission,
    cancel_mission,
    replan_mission,
    update_mission_plan,
)
from modules.tools.discovery.handlers_governance import (
    list_blueprints,
    get_blueprint,
    create_blueprint,
    update_blueprint,
    validate_agent_handler,
    check_budget_handler,
)
from modules.tools.discovery.handlers_asks import ask_human  # PRD-225
from modules.tools.discovery.handlers_clarify import ask_orchestrator  # PRD-229
from modules.tools.discovery.handlers_graph import (
    handle_query_graph,
    handle_graph_neighbors,
    handle_graph_communities,
    handle_graph_impact,
    handle_graph_stats,
    handle_graph_path,
)
from modules.tools.discovery.handlers_shopify import (  # PRD-183 S3 (F088)
    shopify_sync_catalog,
    shopify_sync_status,
)
from modules.tools.discovery.handlers_codegraph import (
    codegraph_list_projects,
    codegraph_search,
    codegraph_get_symbol,
    codegraph_call_graph,
    codegraph_dependencies,
    codegraph_architecture,
    codegraph_index,          # PRD-183 S4 (F087)
    codegraph_reindex,        # PRD-183 S4 (F087)
    codegraph_set_auto_reindex,  # PRD-183 S4 (F022)
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
from modules.tools.discovery.handlers_onboarding import (  # PRD-222 W1S3
    update_onboarding,
)
from modules.tools.discovery.handlers_intake import (  # PRD-222 W1S8
    get_intake_status,
    scan_business_site,
)
from modules.tools.discovery.handlers_web import (  # PRD-240
    web_fetch,
    web_search,
)
from modules.tools.discovery.handlers_socials import (  # PRD-251 US-116
    create_social_post,
    update_social_post,
    submit_social_post,
    get_social_post,
    list_social_posts,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hierarchy permissions — PRD-140 Phase 1
# ---------------------------------------------------------------------------
# Maps mutating action_name → (target_type, param_key).
#
#   target_type — one of core.security.hierarchy_permissions.KNOWN_TARGETS
#   param_key   — name of the field on params that carries the target id
#                 (the `_agent_id` field is the *actor*; this is a different
#                 key — the thing being modified).
#
# Actions NOT in this map skip the hierarchy check entirely — they are
# either workspace-scoped (e.g. platform_store_memory) or already protected
# by admin_only / rate-limit gates. Add an entry here when introducing a new
# mutating action that targets a specific agent / playbook / task / skill /
# tool assignment / heartbeat. The CI audit-grep gate enforces this.
# ---------------------------------------------------------------------------

from core.security.hierarchy_permissions import (  # noqa: E402
    can_actor_modify,
    TARGET_AGENT,
    TARGET_HEARTBEAT,
    TARGET_PLAYBOOK,
    TARGET_TASK,
    TARGET_SKILL,
    TARGET_TOOL_ASSIGNMENT,
)

_HIERARCHY_TARGETS: Dict[str, tuple[str, Optional[str]]] = {
    # Agent edits — owner of the change is the target agent itself.
    "platform_update_agent":              (TARGET_AGENT, "agent_id"),
    "platform_delete_agent":              (TARGET_AGENT, "agent_id"),
    # Heartbeat edits live on the agent row.
    "platform_configure_agent_heartbeat": (TARGET_HEARTBEAT, "agent_id"),
    # Tool / skill / plugin assignments — target is the receiving agent.
    "platform_assign_plugin_to_agent":     (TARGET_TOOL_ASSIGNMENT, "agent_id"),
    "platform_assign_skill_to_agent":      (TARGET_TOOL_ASSIGNMENT, "agent_id"),
    "platform_assign_tool_to_agent":       (TARGET_TOOL_ASSIGNMENT, "agent_id"),
    "platform_unassign_skill_from_agent":  (TARGET_TOOL_ASSIGNMENT, "agent_id"),
    "platform_unassign_tool_from_agent":   (TARGET_TOOL_ASSIGNMENT, "agent_id"),
    # Skill content — always escalates for non-system actors regardless of id.
    "platform_create_workspace_skill":     (TARGET_SKILL, None),
    "platform_update_skill":               (TARGET_SKILL, "skill_id"),
    "platform_delete_workspace_skill":     (TARGET_SKILL, "skill_id"),
    # Playbook edits.
    "platform_update_playbook":            (TARGET_PLAYBOOK, "playbook_id"),
    "platform_delete_playbook":            (TARGET_PLAYBOOK, "playbook_id"),
    "platform_add_playbook_step":          (TARGET_PLAYBOOK, "playbook_id"),
    "platform_update_playbook_step":       (TARGET_PLAYBOOK, "playbook_id"),
    "platform_delete_playbook_step":       (TARGET_PLAYBOOK, "playbook_id"),
    # Tasks — target is the assigned agent (resolved via the task row).
    "platform_assign_task":                (TARGET_TASK, "task_id"),
    "platform_update_task":                (TARGET_TASK, "task_id"),
    "platform_update_task_status":         (TARGET_TASK, "task_id"),
}


# F133 / F148: the actions whose handlers are told who the call is made for
# (server-injected _driving_user_id / _driving_super_admin; see execute()).
_DRIVER_AWARE_ACTIONS = (
    "platform_create_playbook",
    "platform_invite_member",
    "platform_set_member_role",
)


# PRD-234 D16: tool calls that file, assign or re-queue a board ticket carry the
# driving human so the local edition can record consent before dispatch.
# platform_schedule_task joins them (calendar): a scheduled board ticket is
# filed at fire time, and the operator who scheduled it is that consent.
OPERATOR_CONSENT_ACTIONS = (
    "platform_create_task",
    "platform_assign_task",
    "platform_update_task",
    "platform_update_task_status",
    "platform_schedule_task",
)


def _person_for_mission(db, caller_context: Optional[Dict[str, Any]]) -> Optional[str]:
    """F166: who a mission action is made for, as the mission records it — the
    chat's Clerk id, or on the local edition (no Clerk id) the person's email, read
    from the server-threaded ``driving_user_id``. The email is what the local REST
    API records (``created_by = ctx.user.id``). Never the bare ``users.id``: a digit
    string in ``created_by`` is already an agent id wherever no person drove the
    action. A widget turn is made for nobody (F155); an unreadable user is nobody."""
    from core.security.driving_user import driving_user_id
    from core.security.surface import widget_turn

    if widget_turn():
        return None
    ctx = caller_context if isinstance(caller_context, dict) else {}
    if ctx.get("user_id"):
        return str(ctx["user_id"])
    user = driving_user_id(ctx)
    if user is None:
        return None
    try:
        from core.models.core import User

        with db.begin_nested():  # a failed lookup never poisons the caller's transaction
            row = db.query(User.email).filter(User.id == user).first()
    except Exception:  # noqa: BLE001 — fails closed: nobody, so F036 still leaves it to the owner
        logger.warning("[PlatformExecutor] no email for driving user %s — the call is made for nobody", user,
                       exc_info=True)
        return None
    email = str(row[0] or "").strip() if row else ""
    return email or None


def _workspace_role_for_clerk(db, workspace_id, clerk_user_id) -> Optional[str]:
    """Resolve the driving user's workspace role, fresh, at the gate.

    Keyed by the server-threaded clerk principal (``caller_context.user_id``)
    — never by anything the model can write. Any failure returns ``None`` and
    the caller falls closed to the ask.
    """
    try:
        from core.models.core import User
        from core.workspaces.models import WorkspaceMember

        row = (
            db.query(WorkspaceMember.role)
            .join(User, User.id == WorkspaceMember.user_id)
            .filter(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.is_active.is_(True),
                User.clerk_user_id == str(clerk_user_id),
            )
            .first()
        )
        return str(row[0]).strip().lower() if row and row[0] else None
    except Exception:
        logger.warning(
            "[PlatformExecutor] workspace-role resolve failed — falling closed to the ask",
            exc_info=True,
        )
        return None


def _human_directed_admin(db, workspace_id, caller_context) -> bool:
    """The instruction IS the approval (Gerard, 2026-08-06).

    The confirmation gate exists to stop the AGENT deciding to do something
    destructive on its own — not to make a human repeat an instruction they
    just gave. When the call originates from an interactive chat turn
    (``conversation_id`` is server-threaded by build_tool_caller_context —
    heartbeat/cadence/board/mission lanes never carry it) AND the driving
    user holds owner/admin in this workspace (resolved fresh from
    workspace_members, not from anything model-writable), the gate lets the
    instructed action run and stamps ``human_directed`` for the audit trail.

    Everything else — agent-initiated lanes, editors/viewers, a missing or
    unresolvable principal, any lookup error — keeps the ask. The su tier is
    untouched (its gate runs earlier and never consults this).

    F166's twin (refresh 5): the local edition has no Clerk id, so the owner's own
    instruction still got a card. There, the person typing is the chat's
    server-threaded ``driving_user_id``: an active owner/admin of this workspace, or
    the super admin the local operator is — the admin gate's own predicate (F145).
    A Clerk principal keeps the Clerk path, unchanged. On SaaS a turn with no Clerk
    id (a failed lookup, a row never linked to Clerk) keeps the ask, as before.
    """
    from core.security.surface import widget_turn

    if widget_turn():  # F155: a widget visitor's instruction is no approval
        return False
    ctx = caller_context if isinstance(caller_context, dict) else {}
    if not ctx.get("conversation_id"):
        return False
    clerk_id = ctx.get("user_id")
    if clerk_id:
        if not isinstance(clerk_id, str):
            return False
        role = _workspace_role_for_clerk(db, workspace_id, clerk_id)
        return role in ("owner", "admin")
    from config import config as app_config

    if not app_config.IS_LOCAL_EDITION:
        return False
    from core.security.driving_user import driver_is_workspace_admin

    return driver_is_workspace_admin(db, workspace_id, ctx)


def _caller_is_super_admin(caller_context: Optional[Dict[str, Any]]) -> bool:
    """PRD-143's super-admin predicate: only a literal system_role ==
    'super_admin' passes. There is no fallback, and no caller_context refuses.
    The gate and platform_list_tools' listing (F122) both read this one. A
    public widget turn never passes, whatever its context names (F155)."""
    from core.security.surface import widget_turn

    if widget_turn():
        return False
    return (caller_context or {}).get("system_role") == "super_admin"


def _bind_ask_orchestrator_context(
    params: Dict[str, Any],
    caller_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Bind ``platform_ask_orchestrator`` to the SERVER-threaded clarification
    subject, never a tool parameter (P229-RVW-2).

    STRIP any client/LLM-supplied ``_run_id`` / ``_task_id`` / ``_field_id`` FIRST
    — mirroring the ``_agent_id`` hardening in ``exec_platform`` — THEN inject the
    server ``field_context`` values. A prompt-injected tool call therefore cannot
    smuggle a FOREIGN task id past the binding: with an empty field_context (the
    non-mission lanes — channels, webhooks, board, scheduled — where
    agent_factory defaults the mode to TASK_EXECUTION but threads no run) the keys
    stay ABSENT and the handler fails closed to proceed-with-assumption; with a
    real field_context only the server values survive. Rebuild-don't-mutate.
    """
    bound = {
        k: v for k, v in params.items()
        if k not in ("_run_id", "_task_id", "_field_id")
    }
    fctx = (caller_context or {}).get("field_context") or {}
    for src, dst in (("run_id", "_run_id"), ("task_id", "_task_id"), ("field_id", "_field_id")):
        val = fctx.get(src)
        if val is not None:
            bound[dst] = val
    return bound


# Params that identify WHAT an action will act on, in the order a person reads
# them. An approval card that says only "cancel a scheduled task" gives the
# operator nothing to decide on (night 1, 2026-09-18: the card named neither the
# id nor the title of the schedule it would cancel).
_SUBJECT_PARAMS: Tuple[str, ...] = (
    "title", "name", "task_id", "agent_id", "document_id", "mission_id",
    "report_id", "id", "app_name", "query", "playbook_name",
)


def _subject_line(params: Dict[str, Any]) -> str:
    """" on <what this will act on>", or "" when nothing identifies it."""
    if not isinstance(params, dict):
        return ""
    named = [
        f"{key}={params[key]!r}"
        for key in _SUBJECT_PARAMS
        if params.get(key) not in (None, "", [], {})
    ]
    return f" on {', '.join(named[:3])}" if named else ""


# F179 (the TESTER's call, 2026-09-26; Gerard can reverse it): a floor below the
# full-autonomy dial. These actions ask on every lane nobody is instructing (a
# ticket, a mission, a playbook) even with the dial on, because what they do
# cannot be taken back: a public link, once shared, stays shared. An owner's or
# admin's own chat turn still runs them.
ASKS_EVEN_UNDER_FULL_AUTONOMY = frozenset({"workspace_get_public_url"})


def _dial_skips_the_card(action_def: Any, full_autonomy: bool) -> bool:
    """Whether the full-autonomy dial skips this action's confirmation card."""
    return bool(full_autonomy and action_def is not None
                and action_def.name not in ASKS_EVEN_UNDER_FULL_AUTONOMY)


class Cleared(NamedTuple):
    """How a call cleared PlatformActionExecutor.clear: its definition, the
    full-autonomy dial, the grant that said yes, and whether the instructing
    owner's or admin's request was the approval."""

    action_def: Any
    full_autonomy: bool
    approved_via_grant_id: Optional[int]
    human_directed: bool


def marked(result: Any, cleared: Cleared) -> Any:
    """``result`` marked with how its call cleared the confirmation gate; the
    universal telemetry hook persists each mark to tool_execution_logs."""
    if not isinstance(result, dict):
        return result
    action_def = cleared.action_def
    # PRD-143 S8: an invocation that ran only because the full-autonomy
    # dial skipped the confirmation gate is marked here, and the
    # universal telemetry hook persists it to tool_execution_logs
    # (router_decision->>'autonomous') — the Wave 4 audit trail
    # records autonomous actions distinctly and queryably.
    if _dial_skips_the_card(action_def, cleared.full_autonomy) and action_def.requires_confirmation:
        result = {**result, "autonomous": True}
    # PRD-193 S2: a grant-authorised execution records WHICH grant
    # said yes (router_decision->>'approved_via_grant_id' via the
    # same universal telemetry hook) — distinct from the dial-skip
    # marker above. Attribution must be honest: approved is not
    # autonomous.
    if cleared.approved_via_grant_id is not None:
        result = {**result, "approved_via_grant_id": cleared.approved_via_grant_id}
    # 2026-08-06: executed because the instructing human admin's
    # interactive request IS the approval — distinct from both the
    # dial-skip (autonomous) and a card-approved grant.
    if cleared.human_directed:
        result = {**result, "human_directed": True}
    return result


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
            "platform_recommend_agent": recommend_agent,  # PRD-234 S3
            "platform_get_agent": get_agent,
            "platform_list_playbooks": list_playbooks,
            "platform_get_playbook": get_playbook,
            "platform_get_llm_usage": get_llm_usage,
            "platform_get_cost_breakdown": get_cost_breakdown,
            "platform_list_documents": list_documents,
            "platform_read_document": read_document,
            "platform_grep_documents": grep_documents,
            "platform_search_documents": search_documents,
            "platform_list_templates": list_templates,
            "platform_get_template_schema": get_template_schema,
            # PRD-251 US-115: the brand kit (the REST routes' functions)
            "platform_get_brand_kit": get_brand_kit_tool,
            "platform_update_brand_kit": update_brand_kit_tool,
            # PRD-251 US-116 (S4.1): Socials drafts. No tool approves, schedules or publishes.
            "platform_create_social_post": create_social_post,
            "platform_update_social_post": update_social_post,
            "platform_submit_social_post": submit_social_post,
            "platform_get_social_post": get_social_post,
            "platform_list_social_posts": list_social_posts,
            "platform_get_workspace_info": get_workspace_info,
            "platform_get_memory_stats": get_memory_stats,
            "platform_list_connected_apps": list_connected_apps,
            # Write actions
            "platform_create_agent": create_agent,
            "platform_update_agent": update_agent,
            "platform_create_playbook": create_playbook,
            "platform_update_playbook": update_playbook,
            "platform_add_playbook_step": add_playbook_step,
            "platform_update_playbook_step": update_playbook_step,
            "platform_delete_playbook_step": delete_playbook_step,
            "platform_schedule_playbook": schedule_playbook,
            "platform_store_memory": store_memory,
            "platform_checkpoint_thread": checkpoint_thread,  # PRD-206 S2
            "platform_resume_context": resume_context,  # PRD-206 S3
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
            "platform_find_tools": find_tools,  # PR-B: search the action catalog itself
            "platform_list_tools": list_tools,
            "platform_list_llms": list_llms,
            "platform_list_datasources": list_datasources,
            "platform_workspace_stats": workspace_stats,
            # Self-management
            "platform_execute_playbook": execute_playbook,
            "platform_get_playbook_execution": get_playbook_execution,
            "platform_get_system_health": get_system_health,
            "platform_delete_document": delete_document,
            "platform_reprocess_document": reprocess_document,
            "platform_delete_playbook": delete_playbook,
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
            # PRD-230 US-006: package search/install (full-closure, workspace-owned)
            "platform_search_packages": search_packages,
            "platform_install_package": install_package_tool,
            "platform_install_marketplace_agent": install_marketplace_agent_tool,
            # Skill editing (read / create / update / delete)
            "platform_get_skill_content": get_skill_content,
            "platform_create_workspace_skill": create_workspace_skill,
            "platform_update_skill": update_skill,
            "platform_delete_workspace_skill": delete_workspace_skill,
            # Skill runtime (PRD-202): L2 trigger-load / L3 worker exec / L3 enablement
            "platform_load_skill": load_skill,
            "platform_run_skill_script": run_skill_script,
            "platform_set_skill_script_execution": set_skill_script_execution,
            # Agent assignment (PRD-71)
            "platform_assign_tool_to_agent": assign_tool_to_agent,
            "platform_assign_skill_to_agent": assign_skill_to_agent,
            "platform_assign_plugin_to_agent": assign_plugin_to_agent,
            "platform_configure_agent_heartbeat": configure_agent_heartbeat,
            "platform_get_agent_heartbeat": get_agent_heartbeat,
            "platform_unassign_skill_from_agent": unassign_skill_from_agent,
            "platform_unassign_tool_from_agent": unassign_tool_from_agent,
            # Owner escalation channel
            "platform_notify_owner": notify_owner,
            # Full-autonomy dial (per-workspace setting)
            "platform_get_autonomy_level": handle_get_autonomy_level,
            "platform_set_autonomy_level": handle_set_autonomy_level,
            # PRD-222 W1S3: Auto-led onboarding spine (the ONLY state writer)
            "platform_update_onboarding": update_onboarding,
            # PRD-222 W1S8: business-intake pipeline as Auto tools
            "platform_scan_business_site": scan_business_site,
            "platform_get_intake_status": get_intake_status,
            # PRD-240: the web as a platform capability (no app, no assignment)
            "platform_web_fetch": web_fetch,
            "platform_web_search": web_search,
            # PRD-76: Agent Reports
            "platform_submit_report": submit_report,
            "platform_get_latest_report": get_latest_report,
            "platform_browse_reports": browse_reports,
            # PRD-164 S3: Deliverables (agent outputs) discovery
            "platform_list_deliverables": handle_list_deliverables,
            "platform_get_deliverable": handle_get_deliverable,
            # Wave 3 — operating-signal lifecycle
            "platform_acknowledge_report": acknowledge_report,
            "platform_link_report_to_task": link_report_to_task,
            # PRD-72: Board Tasks
            "platform_create_task": create_board_task,
            "platform_list_tasks": list_board_tasks,
            "platform_board_snapshot": board_snapshot,
            "platform_board_summary": board_summary,
            "platform_get_task": get_board_task,
            "platform_wait_for_task": wait_for_board_task,  # PRD-238 S4
            "platform_assign_task": assign_board_task,
            "platform_update_task": update_board_task,
            "platform_update_task_status": update_board_task_status,
            # PRD-77: Agent Self-Scheduling
            "platform_schedule_task": schedule_task,
            "platform_list_scheduled_tasks": list_scheduled_tasks,
            "platform_cancel_scheduled_task": cancel_scheduled_task,
            "platform_get_schedule": get_schedule,
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
            "platform_create_blog_post": create_blog_post_from_topic,
            "platform_generate_cover_image": generate_cover_image,
            # PRD-82A: Missions
            "platform_create_mission": create_mission,
            "platform_list_missions": list_missions,
            "platform_get_mission": get_mission,
            # PRD-204 S9: Watches (supervision to a verdict)
            "platform_create_watch": create_watch,
            "platform_list_watches": list_watches,
            "platform_get_watch": get_watch,
            "platform_cancel_watch": cancel_watch,
            # PRD-228: live floor read-model in one call
            "platform_fleet_status": fleet_status,
            # PRD-163 S1: mission lifecycle control
            "platform_approve_mission": approve_mission,
            "platform_reject_mission": reject_mission,
            "platform_pause_mission": pause_mission,
            "platform_resume_mission": resume_mission,
            "platform_cancel_mission": cancel_mission,
            "platform_replan_mission": replan_mission,
            "platform_update_mission_plan": update_mission_plan,
            # Governance & Blueprints
            "platform_list_blueprints": list_blueprints,
            "platform_get_blueprint": get_blueprint,
            "platform_create_blueprint": create_blueprint,
            "platform_update_blueprint": update_blueprint,
            "platform_validate_agent": validate_agent_handler,
            "platform_check_budget": check_budget_handler,
            # PRD-225: agent → human question (park, notify, return)
            "platform_ask_human": ask_human,
            # PRD-229: agent → orchestrator clarification (answer inline / escalate).
            # Dispatch key carries the platform_ prefix (namespace invariant); the
            # handler function keeps its bare name.
            "platform_ask_orchestrator": ask_orchestrator,
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
            # PRD-121: HARNESS Self-Optimizing Loop
            "platform_harness_status": harness_status,
            "platform_harness_trigger": harness_trigger,
            "platform_harness_history": harness_history,
            # PRD-142 Wave 4 (W4-S6): routing-rule creation
            "platform_create_routing_rule": create_routing_rule,
            # PRD-142 Wave 4 (W4-S5): workspace power-mode knob
            "platform_set_power_mode": set_power_mode,
            # PRD-143 S10: setup-surface gap-fill (operator tier)
            "platform_get_power_mode": get_power_mode,
            "platform_list_channels": list_channels,
            "platform_connect_channel": connect_channel,
            "platform_configure_channel": configure_channel,
            "platform_start_channel": start_channel,
            "platform_stop_channel": stop_channel,
            "platform_get_widget_config": get_widget_config,
            "platform_update_widget_config": update_widget_config,
            "platform_upload_document": upload_document,
            # PRD-143 S11: administration surface (operator tier by design)
            "platform_list_members": list_members,
            "platform_invite_member": invite_member,
            "platform_set_member_role": set_member_role,
            "platform_remove_member": remove_member,
            "platform_update_workspace_settings": update_workspace_settings,
            "platform_list_system_settings": list_system_settings,
            "platform_update_system_setting": update_system_setting,
            "platform_list_api_keys": list_api_keys,
            "platform_revoke_api_key": revoke_api_key,
            "platform_uninstall_plugin": uninstall_plugin,
            # Wave 2: Auto reporting preferences + send-notification wrapper
            "platform_get_auto_reporting_prefs": get_auto_reporting_prefs,
            "platform_update_auto_reporting_prefs": update_auto_reporting_prefs,
            "platform_send_notification": send_notification,
            # PRD-126: Knowledge Graph
            "platform_query_graph": handle_query_graph,
            "platform_graph_neighbors": handle_graph_neighbors,
            "platform_graph_communities": handle_graph_communities,
            "platform_graph_impact": handle_graph_impact,
            "platform_graph_stats": handle_graph_stats,
            "platform_graph_path": handle_graph_path,
            # PRD-183 S3 (F088): Shopify sync + freshness as tools
            "platform_shopify_sync_catalog": shopify_sync_catalog,
            "platform_shopify_sync_status": shopify_sync_status,
            # PRD-165 S4: CodeGraph as an agent capability
            "platform_codegraph_list_projects": codegraph_list_projects,
            "platform_codegraph_search": codegraph_search,
            "platform_codegraph_get_symbol": codegraph_get_symbol,
            "platform_codegraph_call_graph": codegraph_call_graph,
            "platform_codegraph_dependencies": codegraph_dependencies,
            "platform_codegraph_architecture": codegraph_architecture,
            # PRD-183 S4: codegraph write tools (index / reindex / auto-reindex)
            "platform_codegraph_index": codegraph_index,
            "platform_codegraph_reindex": codegraph_reindex,
            "platform_codegraph_set_auto_reindex": codegraph_set_auto_reindex,
        }

    def _workspace_has_admin_owner(self) -> bool:
        """Check if the workspace owner has an admin/owner role.

        Read only under the policy plane's opt-in ``agents_inherit_admin``
        policy (see ``_agent_inherits_admin``): there is an admin to inherit from.
        Fail-closed: returns False on any error.
        """
        try:
            from core.workspaces.models import WorkspaceMember

            with self.db.begin_nested():  # a failed probe never poisons the caller's transaction
                member = (
                    self.db.query(WorkspaceMember)
                    .filter(
                        WorkspaceMember.workspace_id == self.workspace_id,
                        WorkspaceMember.role.in_(("owner", "admin")),
                        WorkspaceMember.is_active.is_(True),
                    )
                    .first()
                )
            if member:
                logger.debug(
                    "[PlatformExecutor] Workspace %s has admin/owner member — "
                    "granting admin_only access to agent",
                    self.workspace_id,
                )
                return True
            return False
        except Exception:
            logger.exception(
                "[PlatformExecutor] Failed to resolve workspace owner role for %s",
                self.workspace_id,
            )
            return False

    def _agent_inherits_admin(self) -> bool:
        """Whether an agent (no caller identity) may act as admin here.

        PRD-174 F014: closes the ``admin_only`` no-op. Historically a missing
        caller_context fell back to "does the workspace have *an* admin member"
        (true for every workspace) — so ``admin_only`` never actually gated an
        agent. Under the policy plane this fallback is opt-in: it applies only
        when the workspace's explicit, default-OFF ``agents_inherit_admin``
        policy is set (and there really is an admin/owner to inherit from).

        F145: that policy is the ONLY inheritance path. With the plane off, or
        when the policy cannot be read, an agent acting for nobody is not an
        admin; the old plane-off fallback made it one in every workspace that
        has an owner or admin member.
        """
        try:
            from modules.policy import policy_plane_enabled

            if not policy_plane_enabled():
                return False
            from modules.policy.policy_document import load_policy_document

            doc = load_policy_document(self.db, self.workspace_id)
            return bool(doc.agents_inherit_admin) and self._workspace_has_admin_owner()
        except Exception:
            logger.warning(
                "[PlatformExecutor] agents_inherit_admin policy read failed for %s — not admin",
                self.workspace_id, exc_info=True,
            )
            return False

    def _caller_is_admin(self, caller_context: Optional[Dict[str, Any]], full_autonomy: bool) -> bool:
        """US-003's admin predicate. The admin gate and platform_list_tools'
        listing (F122) both read this one.

        F145: a call is an admin's when it is made for a workspace owner/admin
        (the driving user's active membership, read fresh) or for the server-side
        super_admin role (core.security.driving_user). It used to read a
        ``workspace_role`` the chat's context never carries, so an owner read as
        non-admin. Full autonomy is the owner's explicit grant. With no caller
        context, only the plane's opt-in inheritance applies.
        """
        from core.security.surface import widget_turn

        if widget_turn():  # F155: a public widget's visitor is never an admin
            return False
        if full_autonomy:
            return True
        if caller_context is None:
            return self._agent_inherits_admin()
        from core.security.driving_user import driver_is_workspace_admin

        return driver_is_workspace_admin(self.db, self.workspace_id, caller_context)

    def _full_autonomy(self) -> bool:
        """True when this workspace is dialled to full autonomy.

        Reads ``workspace.settings.autonomy`` via the canonical service.
        Fail-safe: any error returns False (supervised), never True — the
        dial fails to the gated behaviour, never past it.
        """
        try:
            from core.services.auto_autonomy import is_full_autonomy

            return is_full_autonomy(self.db, self.workspace_id)
        except Exception:
            logger.warning(
                "[PlatformExecutor] autonomy-level lookup failed for %s — "
                "defaulting to standard (supervised)",
                self.workspace_id,
                exc_info=True,
            )
            return False

    def clear(
        self,
        action_name: str,
        params: Dict[str, Any],
        caller_context: Optional[Dict[str, Any]] = None,
        *,
        card_subject: str = "",
    ) -> Union["Cleared", Dict[str, Any]]:
        """The permission gates a call clears before it runs, fail closed: a
        registered definition, the super admin and admin gates, then the
        confirmation gate (the instructing owner or admin, a grant that said yes,
        or the card). Returns the refusal or card to hand back, or how the call
        cleared. The workspace tools clear these same gates (F179); a card whose
        parameters cannot name what it acts on is given ``card_subject``.
        """
        # PRD-193 S2 (P2-12): when a human grant authorises this exact call,
        # its id is recorded here so the execution is audit-marked as
        # grant-authorised — distinct from the full-autonomy dial skipping
        # the gate.
        approved_via_grant_id: Optional[int] = None
        human_directed: bool = False

        # Permission check for write/destructive actions (fail-closed)
        try:
            from modules.tools.discovery import get_action_registry
            action_def = get_action_registry().get(action_name)
            # Every gate below reads action_def, so a handler with no registered
            # ActionDefinition is refused as unknown (fail-closed).
            if action_def is None:
                logger.warning(
                    "[PlatformExecutor] '%s' has a handler but no registered action — refused",
                    action_name,
                )
                return {"success": False, "error": f"Unknown platform action: {action_name}"}

            # PRD-143: Super-admin gate — fail-closed, BEFORE and independent
            # of the admin gate below. The ONLY principal that passes is a
            # literal system_role == 'super_admin' in caller_context. The
            # full-autonomy dial, workspace roles, the workspace-owner
            # fallback and API keys (system_role='admin') NEVER satisfy it;
            # caller_context=None refuses (no identity resolution).
            if action_def and action_def.super_admin_only:
                if not _caller_is_super_admin(caller_context):
                    logger.warning(
                        "[PlatformExecutor] Super-admin-only action '%s' denied — "
                        "workspace_id=%s, caller_context=%s",
                        action_name,
                        self.workspace_id,
                        {k: v for k, v in (caller_context or {}).items() if k != "user_id"},
                    )
                    return {
                        "success": False,
                        "permission_denied": True,
                        "error": (
                            f"Action '{action_name}' is restricted to the platform "
                            "super admin (observability tier)."
                        ),
                    }

            # Full-autonomy dial (per-workspace setting). When on: Auto is
            # treated as admin and the confirmation gate is skipped, except for
            # ASKS_EVEN_UNDER_FULL_AUTONOMY. Everything else (hierarchy check,
            # rate limits, destructive backstop) stands.
            full_autonomy = self._full_autonomy()
            dial_skips_the_card = _dial_skips_the_card(action_def, full_autonomy)

            # US-003: Admin gate — deny admin_only actions for non-admin callers
            if action_def and action_def.admin_only:
                if not self._caller_is_admin(caller_context, full_autonomy):
                    logger.warning(
                        "[PlatformExecutor] Admin-only action '%s' denied — "
                        "workspace_id=%s, caller_context=%s",
                        action_name,
                        self.workspace_id,
                        {k: v for k, v in (caller_context or {}).items() if k != "user_id"},
                    )
                    return {
                        "success": False,
                        "permission_denied": True,
                        # F151: machine-readable, so a lane (HARNESS) can hold the
                        # change for an owner's or admin's approval.
                        "required_role": "owner_or_admin",
                        "error": (
                            f"Action '{action_name}' requires workspace admin or owner role."
                        ),
                    }

            # 2026-08-06 (Gerard): a destructive/write action INSTRUCTED by a
            # workspace owner/admin in an interactive chat turn does not ask
            # again — the instruction is the approval. Agent-initiated lanes
            # (heartbeat / cadence / board / mission) still get the card.
            human_directed = bool(
                action_def
                and action_def.requires_confirmation
                and not dial_skips_the_card
                and _human_directed_admin(self.db, self.workspace_id, caller_context)
            )
            if human_directed:
                logger.info(
                    "[PlatformExecutor] '%s' human-directed — instructing "
                    "workspace admin's request is the approval (workspace=%s)",
                    action_name, self.workspace_id,
                )

            if (
                action_def
                and action_def.requires_confirmation
                and not dial_skips_the_card
                and not human_directed
            ):
                # PRD-193 S1/S2 (P2-12): the ask is no longer a dead end.
                # S2 — consult FIRST: an authorising grant on this exact
                # subject key (GRANTED + unexpired + params-hash equality)
                # opens the gate; destructive grants are retired on use
                # (single-use). Anything else — pending, expired, revoked,
                # denied, params drift, or ANY error in the consult — falls
                # through to the ask (fail closed; the ask is the floor).
                # S1 — otherwise issue (or reuse) a PENDING tool_call
                # ApprovalGrant and return the ask WITH the grant attached,
                # so a human finally has something to say yes to.
                from modules.tools.execution import tool_grants

                _grant = tool_grants.consume_tool_grant(
                    self.db,
                    self.workspace_id,
                    action=action_name,
                    params=params,
                    permission_level=action_def.permission_level,
                )
                if _grant is not None:
                    approved_via_grant_id = getattr(_grant, "id", None)
                    logger.info(
                        "[PlatformExecutor] '%s' authorised by approval grant %s "
                        "— proceeding (workspace=%s)",
                        action_name, approved_via_grant_id, self.workspace_id,
                    )
                else:
                    # F091 (night 3): never ask about something that is not
                    # there, and name what is on the card.
                    from modules.tools.execution.subject_targets import (
                        missing_targets_error, named_subject, resolve_targets,
                    )

                    found, missing = resolve_targets(self.db, self.workspace_id, params, action_name)
                    if missing:
                        return missing_targets_error(action_name, missing)
                    subject = card_subject or named_subject(found) or _subject_line(params)
                    ask = {
                        "success": False,
                        "requires_confirmation": True,
                        "action": action_name,
                        "permission_level": action_def.permission_level,
                        "message": (
                            f"This action ({action_def.permission_level}) requires confirmation. "
                            f"Action: {action_name}{subject} — "
                            f"{action_def.description[:100]}"
                        ),
                        "params": params,
                    }
                    return tool_grants.attach_ask_grant(
                        self.db,
                        self.workspace_id,
                        action=action_name,
                        params=params,
                        ask=ask,
                        permission_level=action_def.permission_level,
                        description=action_def.description,
                        caller_context=caller_context,
                        subject=subject,
                    )
        except Exception as e:
            # Fail-closed: if we can't verify permissions, require confirmation
            logger.warning(
                "[PlatformExecutor] Registry lookup failed for %s: %s — requiring confirmation",
                action_name, e,
            )
            ask = {
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
            # PRD-193 S1: the fail-closed ask gets the grant loop too (best
            # effort — the same surface resolves it if/when the registry
            # heals). NOTE: no grant is CONSUMED on this path — the permission
            # stack could not be verified, so nothing may execute here.
            try:
                from modules.tools.execution import tool_grants

                return tool_grants.attach_ask_grant(
                    self.db,
                    self.workspace_id,
                    action=action_name,
                    params=params,
                    ask=ask,
                    permission_level=None,
                    description=None,
                    caller_context=caller_context,
                )
            except Exception:  # pragma: no cover - attach_ask_grant never raises
                return ask

        return Cleared(action_def, full_autonomy, approved_via_grant_id, human_directed)

    async def execute(
        self,
        action_name: str,
        params: Dict[str, Any],
        caller_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a platform action by name with permission checking.

        Args:
            action_name: Registered platform action name.
            params: Action parameters.
            caller_context: The chat's server-built context
                (``build_tool_caller_context``): ``user_id`` (the driving
                user's Clerk id), ``driving_user_id`` (their users.id),
                ``system_role`` (only ever the literal ``super_admin``) and,
                on an interactive turn, ``conversation_id``. The
                super_admin_only gate (PRD-143) reads ``system_role``; the
                admin_only gate (US-003, F145) reads the driving user's active
                owner/admin membership. With no caller context,
                super_admin_only is denied and admin_only passes only under the
                workspace's opt-in ``agents_inherit_admin`` policy.
        """
        # LLMs sometimes send params as a JSON string instead of a dict
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (json.JSONDecodeError, TypeError):
                return {"success": False, "error": f"Invalid params format: expected dict, got string"}
        handler = self._handlers.get(action_name)
        if not handler:
            return {"success": False, "error": f"Unknown platform action: {action_name}"}

        # F179: the gates a call clears are one method, so the workspace tools
        # clear the same ones.
        cleared = self.clear(action_name, params, caller_context)
        if not isinstance(cleared, Cleared):
            return cleared
        # F193: a single-use yes whose call did nothing (a refusal after the gates,
        # or a handler that failed) is given back, so the next call runs on it.
        result = await self._run_cleared(action_name, params, caller_context, cleared, handler)
        from modules.tools.execution.tool_grants import give_back_unused

        give_back_unused(self.db, cleared.approved_via_grant_id, result)
        return result

    async def _run_cleared(
        self,
        action_name: str,
        params: Dict[str, Any],
        caller_context: Optional[Dict[str, Any]],
        cleared: "Cleared",
        handler: Callable,
    ) -> Dict[str, Any]:
        """Everything after the permission gates (``clear``): the hierarchy check,
        the rate limit, the destructive backstop, the server-side params and the
        handler itself."""
        action_def, full_autonomy = cleared.action_def, cleared.full_autonomy

        # PRD-140 Phase 1 — hierarchy permission check. Runs before the
        # rate limiter so denied calls don't spend rate-limit budget.
        # Only mutating actions in _HIERARCHY_TARGETS are gated; everything
        # else (workspace-scoped, admin-only, read) falls through.
        if action_def and action_def.permission_level in ("write", "destructive"):
            target_spec = _HIERARCHY_TARGETS.get(action_name)
            if target_spec is not None:
                target_type, target_param = target_spec
                actor_id_raw = params.get("_agent_id") if isinstance(params, dict) else None
                # A bulk call carries its targets as ``<target_param>s`` (e.g.
                # platform_update_task_status task_ids) — EVERY id is gated, and
                # one denial denies the whole call. A missing/None target keeps
                # the pre-bulk behaviour (one check with target_id=None).
                bulk_key = f"{target_param}s" if target_param else None
                bulk_raw = (
                    params.get(bulk_key)
                    if (bulk_key and isinstance(params, dict))
                    else None
                )
                if isinstance(bulk_raw, (list, tuple)) and bulk_raw:
                    target_ids_raw = list(bulk_raw)
                else:
                    target_ids_raw = [
                        params.get(target_param)
                        if (target_param and isinstance(params, dict))
                        else None
                    ]
                try:
                    actor_id = int(actor_id_raw) if actor_id_raw is not None else None
                except (TypeError, ValueError):
                    actor_id = None
                n_targets = len(target_ids_raw)
                for idx, target_id_raw in enumerate(target_ids_raw, 1):
                    try:
                        target_id = int(target_id_raw) if target_id_raw is not None else None
                    except (TypeError, ValueError):
                        target_id = target_id_raw  # leave string IDs alone (UUIDs etc.)

                    try:
                        decision = can_actor_modify(
                            self.db,
                            actor_agent_id=actor_id,
                            target_type=target_type,
                            workspace_id=self.workspace_id,
                            target_id=target_id,
                            change_type="update" if action_def.permission_level == "write" else "delete",
                            source="platform_tool",
                            caller_context=caller_context,
                        )
                    except Exception as e:
                        # Fail closed: a permission check that errors must DENY, never
                        # fall through to execution. can_actor_modify's DB probes
                        # (_agent_row / _reports_to_id) aren't all savepoint-guarded,
                        # so a transient DB error would otherwise raise out of the gate
                        # and leave the write's fate to upstream handling. Deny locally
                        # and escalate to Auto — mirrors the registry-lookup fail-closed
                        # above.
                        # The target id itself is caller-supplied data: it is returned
                        # in the error below, never written to the log.
                        logger.warning(
                            "[PlatformExecutor] hierarchy_check_failed action=%s actor=%s "
                            "target=%s (#%d of %d) err=%s — denying (fail-closed)",
                            action_name, actor_id, target_type, idx, n_targets, e,
                        )
                        return {
                            "success": False,
                            "permission_denied": True,
                            "reason": "permission_check_failed",
                            "escalation_target": "auto",
                            "error": (
                                f"Action '{action_name}' denied — permission check could not be "
                                "completed. Route this through the auto for arbitration."
                            ),
                        }
                    if not decision.allowed:
                        logger.warning(
                            "[PlatformExecutor] hierarchy_denied action=%s actor=%s (%s) "
                            "target=%s (#%d of %d) reason=%s",
                            action_name, actor_id, decision.actor_name, target_type, idx, n_targets,
                            decision.reason,
                        )
                        return {
                            "success": False,
                            "permission_denied": True,
                            "reason": decision.reason,
                            "escalation_target": decision.escalation_target,
                            # F133: a refusal that says how to get it done, when it has one.
                            "error": (
                                f"Action '{action_name}' denied ({decision.reason}). {decision.message}"
                                if decision.message
                                else (
                                    f"Action '{action_name}' denied — {decision.reason}. "
                                    + (
                                        f"Route this through the {decision.escalation_target} "
                                        "for arbitration."
                                        if decision.escalation_target
                                        else ""
                                    )
                                ).strip()
                            ),
                        }

        # Rate limit write/destructive actions — scoped per (workspace, agent)
        # so a chatty Auto session doesn't starve mission tasks of headroom.
        if action_def and action_def.permission_level in ("write", "destructive"):
            try:
                from core.security.rate_limiter import check_rate_limit, DEFAULT_LIMITS
                _agent_id = params.get("_agent_id") if isinstance(params, dict) else None
                subject = str(_agent_id) if _agent_id else None
                await check_rate_limit(
                    str(self.workspace_id),
                    "platform_write",
                    subject_id=subject,
                )
            except HTTPException as e:
                if e.status_code == 429:
                    limit, window = DEFAULT_LIMITS.get("platform_write", (60, 60))
                    return {
                        "success": False,
                        "rate_limited": True,
                        "error": (
                            f"Rate limit exceeded: max {limit} write actions per "
                            f"{window}s. Try again shortly."
                        ),
                    }
                raise
            except Exception as exc:
                logger.warning("[PlatformExecutor] Rate limiter unavailable, failing open: %s", exc)

        # US-003: Destructive safety check — destructive actions must require confirmation
        if (
            action_def
            and action_def.permission_level == "destructive"
            and not action_def.requires_confirmation
        ):
            logger.error(
                "[PlatformExecutor] Destructive action '%s' missing requires_confirmation flag — "
                "rejecting as safety precaution",
                action_name,
            )
            return {
                "success": False,
                "error": (
                    f"Internal error: destructive action '{action_name}' is misconfigured "
                    "(missing confirmation requirement). Contact platform admin."
                ),
            }

        # PRD-124/126: Auto-inject _agent_id for graph tools (team scoping).
        # PRD-157 S2: the document-reading tools are team-scoped the same way.
        if (
            action_name.startswith("platform_graph")
            or action_name == "platform_query_graph"
            or action_name in ("platform_read_document", "platform_grep_documents", "platform_search_documents")
        ):
            if "_agent_id" not in params:
                # Resolve agent_id from the active mission or caller context
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
                        _aid = (active_run.config or {}).get("agent_id")
                        if _aid:
                            params = {**params, "_agent_id": int(_aid)}
                except Exception as e:
                    logger.debug("[PRD-124] Failed to resolve _agent_id for graph tool: %s", e)

        # PRD-108 / PRD-178 S1 (F020): Auto-inject field_id for field tools from
        # the CALLING task's run — threaded down via caller_context["field_context"]
        # by the agent runtime. The previous `.first()`-on-any-running-run lookup
        # bound an arbitrary concurrent mission's field (F020) and let a running
        # mission shadow workspace recall (F021); it is deleted, not shimmed.
        # No threaded context ⇒ no injection (an explicit field_id still wins) —
        # an ambient guess is exactly the bug we are removing.
        if action_name.startswith("platform_field_") and "field_id" not in params:
            field_id = ((caller_context or {}).get("field_context") or {}).get("field_id")
            if field_id:
                params = {**params, "field_id": field_id}
                logger.info(
                    "[PRD-178 S1] Bound field_id %s to calling task for %s",
                    field_id, action_name,
                )

        # PRD-229 / P229-RVW-2: bind ask_orchestrator to the CALLING task/run from
        # the server-threaded field_context (never a tool param). The binding
        # STRIPS any smuggled _run_id/_task_id/_field_id BEFORE injecting the
        # server values, so a prompt-injected call in a non-mission lane cannot
        # point Auto at a foreign task. _agent_id is already server-minted above
        # (exec_platform). Only the server context wins.
        if action_name == "platform_ask_orchestrator":
            params = _bind_ask_orchestrator_context(params, caller_context)

        # PRD-163 S1/Q56: attribute mission create + lifecycle to the chatting
        # user. The chat path threads the driving user's clerk id via
        # caller_context['user_id'] (on the local edition, which has no Clerk
        # id, only the internal driving_user_id: their email is recorded); inject
        # it as _created_by so the handler sets created_by / actor to the user,
        # not the agent.
        _MISSION_ATTRIBUTED = (
            "platform_create_mission",
            "platform_approve_mission",
            "platform_reject_mission",
            "platform_pause_mission",
            "platform_resume_mission",
            "platform_cancel_mission",
            "platform_replan_mission",
            "platform_update_mission_plan",
        )
        if action_name in _MISSION_ATTRIBUTED:
            # Hardened 2026-07-17 (Gerard's call, flagged in #563): STRIP any
            # caller-supplied _created_by first, then inject from context --
            # the original PRD-163 caller-preserving guard let a spoofed tool
            # arg claim attribution on headless paths (board dispatcher,
            # workflows) where caller_context carries no user_id. Grep-verified
            # no legitimate params-side producer exists.
            params = {k: v for k, v in params.items() if k != "_created_by"}
            # F166: with the Clerk id alone, every local chat approval arrived
            # with no person behind it and F036 refused it. A board ticket or a
            # workflow threads no driver, so F036 still refuses their approvals.
            _person = _person_for_mission(self.db, caller_context)
            if _person:
                params = {**params, "_created_by": _person}

        # F133 / F148: who the call is made for, for the handlers that record or
        # check it: a playbook's creator (created_by_user_id), an invitation's
        # sender, the owner a role change needs. From the server-built context
        # only: caller-supplied values are ALWAYS stripped first.
        if action_name in _DRIVER_AWARE_ACTIONS and isinstance(params, dict):
            from core.security.driving_user import driving_user_id

            params = {k: v for k, v in params.items() if k not in ("_driving_user_id", "_driving_super_admin")}
            _user = driving_user_id(caller_context)
            if _user is not None:
                params = {**params, "_driving_user_id": _user}
            if isinstance(caller_context, dict) and _caller_is_super_admin(caller_context):
                params = {**params, "_driving_super_admin": True}

        # PRD-205 S4: capture the originating conversation for watch-creating
        # actions (direct create + the launches whose handlers auto-create a
        # watch) so verdicts post back into that chat. Server-injected from
        # caller_context; a caller-supplied param of the same name is ALWAYS
        # stripped first -- inject-on-truthy alone would let a spoofed tool
        # arg survive the headless paths (board dispatcher, workflows) where
        # caller_context carries no conversation_id. The origin is never
        # spoofable via tool args.
        _WATCH_ORIGIN_ACTIONS = (
            "platform_create_watch",
            "platform_create_mission",
            "platform_execute_playbook",
            "platform_schedule_task",
            # PRD-224 US-005: the ASSIGN-lane ticket auto-attaches a watch whose
            # verdict must post back to THIS conversation.
            "platform_create_task",
        )
        if action_name in _WATCH_ORIGIN_ACTIONS:
            params = {k: v for k, v in params.items() if k != "_origin_chat_id"}
            _origin_chat = (caller_context or {}).get("conversation_id")
            if _origin_chat:
                params = {**params, "_origin_chat_id": str(_origin_chat)}

        # PRD-224 US-005: the ASSIGN-lane flag rides caller_context (set only when
        # Auto's turn routed ASSIGN). Strip-then-inject like the origin above, so a
        # tool arg can NEVER fake supervision on headless paths (board dispatcher,
        # heartbeat, recipes) where caller_context carries no lane. This is what
        # makes auto-supervision mechanical — the LLM cannot forget or spoof it.
        if action_name == "platform_create_task":
            params = {k: v for k, v in params.items() if k != "_assign_lane"}
            if (caller_context or {}).get("assign_lane"):
                params = {**params, "_assign_lane": True}

        # PRD-206 S1: memory writes carry their owner (drives the Q7 private/
        # workspace scope default) and their originating chat (the thread
        # link). Server-injected from caller_context; caller-supplied values
        # of the same names are ALWAYS stripped first (the #565 strip-then-
        # inject hardening) so neither is spoofable via tool args.
        _MEMORY_CONTEXT_ACTIONS = (
            "platform_store_memory",
            "platform_checkpoint_thread",
            "platform_resume_context",
        )
        if action_name in _MEMORY_CONTEXT_ACTIONS:
            params = {
                k: v for k, v in params.items()
                if k not in ("_user_id", "_origin_chat_id")
            }
            if action_name == "platform_store_memory":
                # F189: a memory is never filed as another agent (it recalls it as its own).
                params = {k: v for k, v in params.items() if k != "agent_id"}
            _mem_user = (caller_context or {}).get("user_id")
            if _mem_user:
                params = {**params, "_user_id": str(_mem_user)}
            _mem_chat = (caller_context or {}).get("conversation_id")
            if _mem_chat:
                params = {**params, "_origin_chat_id": str(_mem_chat)}

        # PRD-234 D16: a ticket the operator asks Auto to file, assign or re-queue
        # in a live chat turn carries the driving user, so the local edition can
        # record the board-task consent before dispatch. Strip-then-inject, like
        # the memory keys above — never caller-supplied.
        if action_name in OPERATOR_CONSENT_ACTIONS:
            from services.board_consent import driver_from_caller_context

            params = {k: v for k, v in params.items() if k != "_user_id"}
            _driver = driver_from_caller_context(caller_context)
            if _driver:
                params = {**params, "_user_id": _driver}

        # F122: platform_list_tools lists what THIS caller may run. Both flags
        # come from the predicates the gates above read. Strip-then-inject like
        # the keys around it, so a tool arg can never claim a role.
        if action_name == "platform_list_tools":
            params = {
                k: v for k, v in params.items()
                if k not in ("_caller_is_super_admin", "_caller_is_admin")
            }
            params = {
                **params,
                "_caller_is_super_admin": _caller_is_super_admin(caller_context),
                "_caller_is_admin": self._caller_is_admin(caller_context, full_autonomy),
            }

        # PRD-238 S4: every handler learns which chat turn (if any) it runs in,
        # so a long-running action can narrate progress through
        # services.turn_progress. Strip-then-inject like the keys above — the
        # turn id is server-side truth from caller_context, never a tool arg.
        params = {k: v for k, v in params.items() if k != "_turn_id"}
        _turn = (caller_context or {}).get("turn_id")
        if _turn:
            params = {**params, "_turn_id": str(_turn)}
        try:
            result = await handler(self.db, self.workspace_id, params)
            return marked(result, cleared)
        except Exception as e:
            logger.error(f"[PlatformExecutor] {action_name} failed: {e}", exc_info=True)
            try:
                self.db.rollback()
            except Exception:
                pass
            return {"success": False, "error": f"Action '{action_name}' failed"}
