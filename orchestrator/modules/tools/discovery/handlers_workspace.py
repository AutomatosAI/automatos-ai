"""Workspace handlers for PlatformActionExecutor — workspace info, memory stats, connected apps, store memory."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from sqlalchemy.orm import Session

from core.auth.clerk_person import clerk_person_row
from core.composio.pending_connections import CONNECTED, NOT_CONNECTED, PENDING
from modules.tools.discovery.memory_off import NOT_SAVED, memory_is_off, memory_off_answer, off_when_nothing_is_found
from modules.tools.discovery.memory_owner import owned_by_the_driver
from modules.tools.discovery.memory_source import defaults_the_source_type
from modules.tools.discovery.memory_write import memory_metadata, memory_params_refusal, owner_turns, rule_answer
from modules.tools.discovery.rule_replaces import retires_what_it_replaces

logger = logging.getLogger(__name__)

# platform_list_connected_apps: the statuses it lists (the ones the Tools page lists), and what the
# answer means. An agent's use of a connected app is gated by core/composio/tool_executor.py
# (assignment, then connection), with the rule in core/composio/agent_apps.py; the note says that rule.
LISTED_STATUSES = frozenset({CONNECTED, PENDING, NOT_CONNECTED})
EXTERNAL_APP = "EXTERNAL"  # the assignments agent_apps.assignment_state counts as an agent's apps
CONNECTED_APPS_NOTE = (
    "Apps with status active are connected for the whole workspace. pending is as recorded here, not yet "
    "checked with Composio; added means it is in the workspace but not connected, so the owner reconnects "
    "it on Tools & Integrations. assigned_to_agents counts the agents that have the app assigned and "
    "switched on. An agent with no apps assigned can use every active app through composio_execute; an "
    "agent with apps assigned can use only the active apps assigned and switched on for it."
)
CONNECTED_APPS_UNREAD = (
    "Could not read this workspace's connected apps. Tell the owner the connections could not be checked "
    "just now, never that an app is not connected."
)


def _resolve_owner_subject(db: Session, raw_user_id: Any) -> Optional[str]:
    """PRD-206 S1: turn a caller_context user id into the ``user:{users.id}``
    subject tag (the PRD-196 owner identity). The chat path threads the Clerk
    STRING id (never treat it as ``users.id`` — the #513 lesson); an already-
    internal integer id passes through. Unresolvable → None (no owner tag)."""
    if raw_user_id is None:
        return None
    s = str(raw_user_id)
    if s.isdigit():
        return f"user:{int(s)}"
    try:
        row = clerk_person_row(db, s)
    except Exception:
        logger.warning("[PlatformExecutor] owner-subject resolution failed", exc_info=True)
        return None
    return f"user:{row[0]}" if row else None


async def get_workspace_info(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.workspaces import Workspace

    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        return {"success": False, "error": "Workspace not found"}

    # Count resources
    from core.models import Agent, Document
    agent_count = db.query(Agent).filter(Agent.workspace_id == workspace_id).count()
    doc_count = db.query(Document).filter(Document.workspace_id == workspace_id).count()

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


@off_when_nothing_is_found("total_memories")  # no embedding provider: memory is off, not empty
async def get_memory_stats(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Get memory stats from Mem0 -- global + per-agent memories."""
    from modules.memory.unified_memory_service import get_unified_memory_service

    try:
        service = get_unified_memory_service()
        if not service.is_durable_configured:
            return {"success": False, "error": "Memory service not configured (QDRANT_URL empty)"}

        ws_id = str(workspace_id)

        # Fetch global memories
        global_memories = await service.get_all_memories(workspace_id=ws_id, limit=200)

        # Also check per-agent memories for workspace agents
        from core.models.core import Agent
        agents = (
            db.query(Agent.id, Agent.name)
            .filter(Agent.workspace_id == workspace_id)
            .all()
        )

        agent_scan_limit = 10
        scanned_agents = agents[:agent_scan_limit]
        partial = len(agents) > agent_scan_limit
        agent_stats = []
        agent_tasks = [
            service.get_all_memories(workspace_id=ws_id, agent_id=agent_id, limit=200)
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


def _assignment_counts(db: Session, workspace_id: UUID) -> Dict[str, int]:
    """How many of the workspace's agents have each app assigned and switched on: the
    EXTERNAL rows the executor's rule counts (``agent_apps.assignment_state``), each agent
    once per upper-cased app name (connections are upper-cased; assignments are not always)."""
    from core.models import Agent
    from core.models.composio_cache import AgentAppAssignment

    assignments = (
        db.query(AgentAppAssignment.app_name, AgentAppAssignment.agent_id)
        .join(Agent, AgentAppAssignment.agent_id == Agent.id)
        .filter(
            Agent.workspace_id == workspace_id,
            AgentAppAssignment.is_active == True,  # noqa: E712
            AgentAppAssignment.app_type == EXTERNAL_APP,
        )
        .all()
    )
    agents_by_app: Dict[str, Set[Any]] = {}
    for row in assignments:
        agents_by_app.setdefault((row.app_name or "").upper().strip(), set()).add(row.agent_id)
    return {app: len(agents) for app, agents in agents_by_app.items()}


def _roll_back(db: Session) -> None:
    """Leave the caller's session usable after a failed read: a failed statement aborts its transaction."""
    try:
        db.rollback()
    except Exception:
        logger.exception("[PlatformExecutor] rollback after a failed connected-apps read failed too")


def _app_name(connection: Dict[str, Any]) -> str:
    return (connection.get("app_name") or "").upper().strip()


def _is_listed(connection: Dict[str, Any]) -> bool:
    """The statuses the Tools page lists: active, pending and added (disconnected). Unlike that
    page, a pending row is not checked with Composio here; it is read as recorded, as
    ``composio_execute`` reads it."""
    return bool(_app_name(connection)) and (connection.get("status") or "").lower() in LISTED_STATUSES


def _listed_app(connection: Dict[str, Any], connected: Set[str], counts: Dict[str, int]) -> Dict[str, Any]:
    """One connection: active when ``get_connected_apps`` counts it connected, else its own status.
    ``connected_at`` is given only for a connected app; an added one's is from before it was disconnected."""
    app = _app_name(connection)
    is_connected = app in connected
    connected_at = connection.get("connected_at") if is_connected else None
    return {
        "app_name": app,
        "status": CONNECTED if is_connected else (connection.get("status") or "").lower(),
        "connected_at": connected_at.isoformat() if connected_at else None,
        "assigned_to_agents": counts.get(app, 0),
    }


def _connected_apps_answer(listed: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "success": True,
        "connected_apps": listed,
        "count": len(listed),
        "connected_count": sum(1 for app in listed if app["status"] == CONNECTED),
        "note": CONNECTED_APPS_NOTE,
    }


async def list_connected_apps(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """The workspace's Composio connections, read through ``EntityManager`` as Socials and
    ``composio_execute`` read them. Before, this listed the apps assigned to some agent, so an
    app connected for the workspace but assigned to no agent was missing and Auto told the
    owner it was not connected. ``get_connected_apps`` decides which are connected, and
    settles a pending row that holds its Composio account id (nothing asks Composio); each
    app also says how many agents have it assigned. A failed read rolls the session back."""
    from core.composio.entity_manager import EntityManager

    try:
        manager = EntityManager(db)
        entity = manager.get_entity_by_workspace(workspace_id)
        if not entity:
            return _connected_apps_answer([])
        connected = {app.upper().strip() for app in manager.get_connected_apps(workspace_id)}
        connections = manager.get_entity_connections(entity["id"])  # read after the settle above
        counts = _assignment_counts(db, workspace_id)
    except Exception:
        # Never an empty list here: that would read as "nothing is connected".
        logger.exception("[PlatformExecutor] could not read the connected apps of workspace %s", workspace_id)
        _roll_back(db)
        return {"success": False, "error": CONNECTED_APPS_UNREAD}
    return _connected_apps_answer([_listed_app(c, connected, counts) for c in connections if _is_listed(c)])


@defaults_the_source_type  # PRD-256 FX-013: source_type is optional; the turn's driver sets its default
@owned_by_the_driver  # PRD-256 FX-015: no Clerk id (the local edition), the driving person owns it
@retires_what_it_replaces  # P256-FIX-N15-9 (F429): the owner's new rule retires the one it replaces
async def store_memory(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Store a curated fact in the workspace's long-term memory (``memory_write`` checks the
    params and builds the metadata; P256-FIX-N14-4: in the owner's own words, it is their
    standing rule, and the answer says which it is)."""
    content = params.get("content")
    if not content:
        return {"success": False, "error": "Missing required parameter: content"}
    # PRD-206 S1 (Q3 silent-everything): refuse loudly so the model can tell the user.
    refusal = memory_params_refusal(params)
    if refusal:
        return {"success": False, "error": refusal}

    try:
        from modules.memory.unified_memory_service import get_unified_memory_service

        service = get_unified_memory_service()
        if not service.is_durable_configured:
            return {"success": False, "error": "Memory service not configured (QDRANT_URL empty)"}

        ws_id = str(workspace_id)
        # F189: a model-chosen agent_id filed the memory in THAT agent's namespace,
        # recalled as its own, and a widget agent recalls only its own: injected
        # text could reach public visitors. Memories go to the workspace, as this
        # tool says; nothing files for another agent (the executor strips the key).
        agent_id_int = None
        # ``_user_id`` is server-injected by the executor (strip-then-inject), never caller-supplied.
        # N17-4: the owner and their messages are read once, for the rule check and the answer.
        owner = _resolve_owner_subject(db, params.get("_user_id"))
        said = owner_turns(db, ws_id, params, owner)
        metadata = memory_metadata(db, ws_id, params, owner, said=said)
        result = await service.store_long_term(
            workspace_id=ws_id,
            content=content,
            agent_id=agent_id_int,
            metadata=metadata,
        )

        if result.get("error"):
            if await memory_is_off(service):  # no embedding provider: say memory is off, not a raw error
                return memory_off_answer(NOT_SAVED)
            # F103: the model must not tell the owner "noted" over a lost write.
            return {"success": False, "error": f"Memory NOT saved — {result['error']}. Tell the owner it was not stored."}
        if result.get("skipped"):  # PRD-274 §5.5, F103: a held write saved nothing, so it is never "Stored"
            return {"success": False, "error": f"Memory NOT saved — {result['skipped']}. Tell them it was not stored."}

        user_id = service.namespace(ws_id).resolve(agent_id_int)
        return {
            "success": True,
            "message": f"Stored in memory (user_id={user_id}): '{content[:100]}'",
            "facts_extracted": result.get("facts_extracted", "unknown"),
            **rule_answer(metadata, content, said),
        }
    except Exception as e:
        logger.warning(f"[PlatformExecutor] Memory store failed: {e}", exc_info=True)
        return {"success": False, "error": f"Memory NOT saved — memory service error: {e}. Tell the owner it was not stored."}


@owned_by_the_driver  # PRD-256 FX-015: on the local edition the driving person is the viewer
async def resume_context(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """PRD-206 S3: 'where did we leave off?' from ANY chat.

    The viewer comes from the server-injected ``_user_id`` (strip-then-inject
    — never spoofable); without one (headless caller) the payload carries
    workspace decisions/open loops but no personal threads.
    """
    from modules.memory.resume_context import build_resume_payload, format_resume_for_llm

    owner = _resolve_owner_subject(db, params.get("_user_id"))
    viewer_user_id = int(owner.split(":", 1)[1]) if owner else None

    payload = await build_resume_payload(
        db,
        workspace_id=str(workspace_id),
        viewer_user_id=viewer_user_id,
    )
    return {
        "success": True,
        **payload,
        "formatted": format_resume_for_llm(payload),
    }


async def checkpoint_thread(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """PRD-206 S2: on-demand thread checkpoint ("save where we are").

    Defaults to the CURRENT conversation via the server-injected
    ``_origin_chat_id`` (never spoofable); an explicit ``chat_id`` param may
    target another thread in the same workspace (the checkpoint runner
    enforces the workspace match).
    """
    chat_id = params.get("chat_id") or params.get("_origin_chat_id")
    if not chat_id:
        return {
            "success": False,
            "error": "No conversation to checkpoint — this tool needs a chat context or an explicit chat_id.",
        }

    from modules.memory.thread_checkpoint import run_thread_checkpoint

    result = await run_thread_checkpoint(
        db,
        workspace_id=str(workspace_id),
        chat_id=str(chat_id),
        trigger="on_demand",
        # An explicit "save where we are" should work even on a short thread.
        min_messages=2,
    )
    if not result.get("success"):
        return {"success": False, "error": result.get("error", "checkpoint failed")}
    summary = result.get("summary") or {}
    return {
        "success": True,
        "message": (
            f"Checkpointed '{summary.get('topic') or 'this thread'}' — "
            f"{len(summary.get('decisions') or [])} decision(s), "
            f"{len(summary.get('open_questions') or [])} open loop(s) on record."
        ),
        "topic": summary.get("topic"),
        "next_step": summary.get("next_step"),
        "stored_memories": result.get("stored_memories", 0),
    }


# ---------------------------------------------------------------------------
# PRD-143 S11 — administration surface: workspace + system settings.
# Operator tier by design (the Rev 2 inversion); safety is the fail-closed
# key whitelist, the executor's confirmation gate and the audit trail.
# ---------------------------------------------------------------------------

# The only workspace.settings keys this tool may write. Other slices have
# their own dedicated tools (power_mode, widget config, autonomy,
# auto-reporting) or are deliberately excluded (integrations carries raw
# tokens; orchestrator has its own seeded-agent PUT flow).
OPERATOR_WORKSPACE_SETTINGS_KEYS = (
    "byok_overrides",
    "default_notification_channel",
    "voice_live",  # PRD-207 S4: {enabled, monthly_cap_minutes, retell_voice_id}
)


async def update_workspace_settings(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Set one whitelisted workspace-settings key. Fail-closed on any other key."""
    key = params.get("key")
    value = params.get("value")

    try:
        if key not in OPERATOR_WORKSPACE_SETTINGS_KEYS:
            return {
                "success": False,
                "error": f"key must be one of {list(OPERATOR_WORKSPACE_SETTINGS_KEYS)}, got {key!r}",
            }

        from core.models.workspaces import Workspace

        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if ws is None:
            return {"success": False, "error": "Workspace not found"}

        settings = dict(ws.settings or {})

        if key == "byok_overrides":
            # Same semantics as PUT /api/workspaces/current/byok-preferences:
            # merge, provider-whitelisted, booleans only.
            from api.workspaces import _ALLOWED_PROVIDERS

            if not isinstance(value, dict):
                return {"success": False, "error": "byok_overrides value must be an object of provider -> bool"}
            overrides = dict(settings.get("byok_overrides", {}))
            ignored = [p for p in value if p not in _ALLOWED_PROVIDERS]
            for provider, enabled in value.items():
                if provider in _ALLOWED_PROVIDERS:
                    overrides[provider] = bool(enabled)
            settings["byok_overrides"] = overrides
            applied: Any = overrides
        elif key == "voice_live":
            # PRD-207 S4: same fail-closed validation as PUT
            # /api/workspaces/current/voice-live — merge, never replace-blind.
            from modules.voice.live_settings import validate_voice_live_update

            try:
                normalized = validate_voice_live_update(value)
            except ValueError as ve:
                return {"success": False, "error": str(ve)}
            merged = dict(settings.get("voice_live", {}))
            merged.update(normalized)
            settings["voice_live"] = merged
            applied = merged
            ignored = []
        else:  # default_notification_channel
            from api.workspaces import _VALID_NOTIFICATION_CHANNELS

            channel = str(value or "").strip().lower()
            if channel not in _VALID_NOTIFICATION_CHANNELS:
                return {
                    "success": False,
                    "error": f"default_notification_channel must be one of {sorted(_VALID_NOTIFICATION_CHANNELS)}, got {value!r}",
                }
            settings["default_notification_channel"] = channel
            applied = channel
            ignored = []

        ws.settings = settings
        db.commit()

        result = {"success": True, "key": key, "value": applied}
        if ignored:
            result["ignored"] = ignored
        return result
    except Exception as exc:
        db.rollback()
        logger.error("[workspace] update_workspace_settings failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


def _masked_setting(s) -> Dict[str, Any]:
    return {
        "id": s.id,
        "category": s.category,
        "key": s.key,
        "value": "****" if s.is_sensitive else s.value,
        "is_sensitive": bool(s.is_sensitive),
        "is_required": bool(s.is_required),
        "description": s.description,
    }


async def list_system_settings(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List system settings (optionally by category). Sensitive values are
    ALWAYS masked — secrets never reach the LLM context (the REST router
    returns them raw to admin sessions; the tool deliberately does not)."""
    try:
        from core.models.system_settings import SystemSetting
        from services.mobile.edition_change import RESERVED_SETTING_CATEGORIES

        query = db.query(SystemSetting).filter(SystemSetting.category.notin_(RESERVED_SETTING_CATEGORIES))
        category = params.get("category")
        if category:
            query = query.filter(SystemSetting.category == category)

        rows = query.order_by(SystemSetting.category, SystemSetting.key).all()
        settings = [_masked_setting(s) for s in rows]
        return {"success": True, "settings": settings, "count": len(settings)}
    except Exception as exc:
        logger.error("[workspace] list_system_settings failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


def _restamp_auto_models(db: Session, setting, old_value: Optional[str]) -> Dict[str, Any]:
    """A change of the platform default for Auto (orchestrator_llm provider/model)
    moves the Auto rows still on it, as the System Settings API does
    (core/llm/auto_model_sync): ``{"auto_models": counts}``, or nothing for any other
    setting. A failed move puts the setting back and raises, so the tool reports it."""
    from core.llm import auto_model_sync

    if not auto_model_sync.changes_platform_default(setting.category, setting.key, old_value, setting.value):
        return {}
    return {"auto_models": auto_model_sync.restamp_after_settings_change(db, [(setting, old_value)])}


async def update_system_setting(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Update one system setting by category + key (platform-wide — the
    action carries requires_confirmation=True for exactly that reason)."""
    category = params.get("category")
    key = params.get("key")
    value = params.get("value")
    if not category or not key or value is None:
        return {"success": False, "error": "category, key and value are required"}

    try:
        from datetime import datetime

        from core.models.system_settings import SystemSetting
        from services.mobile.edition_change import RESERVED_SETTING_CATEGORIES

        setting = (
            db.query(SystemSetting)
            .filter(SystemSetting.category == category, SystemSetting.key == key,
                    SystemSetting.category.notin_(RESERVED_SETTING_CATEGORIES))
            .first()
        )
        if not setting:
            return {"success": False, "error": f"Setting not found: {category}.{key}"}

        old_value = setting.value
        setting.value = str(value)
        setting.updated_at = datetime.utcnow()
        db.commit()

        logger.info(
            "[workspace] system setting %s.%s updated via platform tool (workspace %s)",
            category, key, workspace_id,
        )
        # The restamp locks and writes every Auto row: off the event loop, as the
        # module's other blocking work runs (the session is not used meanwhile).
        restamped = await asyncio.to_thread(_restamp_auto_models, db, setting, old_value)
        return {"success": True, "setting": _masked_setting(setting), **restamped}
    except Exception as exc:
        db.rollback()
        logger.error("[workspace] update_system_setting failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}
