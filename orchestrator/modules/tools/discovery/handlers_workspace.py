"""Workspace handlers for PlatformActionExecutor — workspace info, memory stats, connected apps, store memory."""

import asyncio
import logging
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import func, text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


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
        row = db.execute(
            text("SELECT id FROM users WHERE clerk_user_id = :cid LIMIT 1"),
            {"cid": s},
        ).fetchone()
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


async def list_connected_apps(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models import Agent
    from core.models.composio_cache import AgentAppAssignment

    assignments = (
        db.query(
            AgentAppAssignment.app_name,
            AgentAppAssignment.app_type,
            func.count(AgentAppAssignment.id).label("agent_count"),
        )
        .filter(AgentAppAssignment.is_active == True)
        .join(Agent, AgentAppAssignment.agent_id == Agent.id)
        .filter(Agent.workspace_id == workspace_id)
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


async def store_memory(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from modules.memory.write_contract import (
        MEMORY_FACT_TYPES,
        MEMORY_SCOPES,
        build_memory_metadata,
        violates_exclusions,
    )

    content = params.get("content")
    if not content:
        return {"success": False, "error": "Missing required parameter: content"}

    valid_source_types = {"platform_verified", "claude_reports", "current_status", "inference"}
    source_type = params.get("source_type", "inference")
    if source_type not in valid_source_types:
        return {
            "success": False,
            "error": f"source_type must be one of: {', '.join(sorted(valid_source_types))}",
        }

    confidence = params.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError
        except (TypeError, ValueError):
            return {"success": False, "error": "confidence must be a number between 0 and 1"}

    fact_type = params.get("type")
    if fact_type is not None and fact_type not in MEMORY_FACT_TYPES:
        return {
            "success": False,
            "error": f"type must be one of: {', '.join(sorted(MEMORY_FACT_TYPES))}",
        }

    scope = params.get("scope")
    if scope is not None and scope not in MEMORY_SCOPES:
        return {
            "success": False,
            "error": f"scope must be one of: {', '.join(sorted(MEMORY_SCOPES))}",
        }

    # PRD-206 S1 (Q3 silent-everything): the exclusion validator IS the consent
    # gate — refuse loudly so the model can tell the user, and never echo the
    # content back into the refusal.
    violation = violates_exclusions(str(content))
    if violation:
        return {
            "success": False,
            "error": (
                f"Refused by the memory exclusion policy (rule: {violation}). "
                "Secrets, credentials, payment and similar sensitive data are "
                "never stored as memory."
            ),
        }

    try:
        from datetime import datetime, timezone
        from modules.memory.unified_memory_service import get_unified_memory_service

        service = get_unified_memory_service()
        if not service.is_durable_configured:
            return {"success": False, "error": "Memory service not configured (QDRANT_URL empty)"}

        ws_id = str(workspace_id)
        agent_id = params.get("agent_id")
        # Cast to int if provided (may come as string from tool params)
        agent_id_int = int(agent_id) if agent_id else None

        # PRD-206 S1: both write paths build the SAME canonical metadata via
        # the write contract — type + scope (Q7 split default) + provenance +
        # owner + chat link. ``_user_id`` / ``_origin_chat_id`` are server-
        # injected by the executor (strip-then-inject), never caller-supplied.
        extra: Dict[str, Any] = {
            "workspace_id": ws_id,
            "source": "platform_tool",
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        if params.get("evidence_uri"):
            extra["evidence_uri"] = params["evidence_uri"]

        metadata = build_memory_metadata(
            fact_type=fact_type or "business_fact",
            importance=params.get("importance"),
            confidence=confidence,
            source_type=source_type,
            scope=scope,
            owner=_resolve_owner_subject(db, params.get("_user_id")),
            chat_id=params.get("_origin_chat_id"),
            pinned=bool(params["pinned"]) if params.get("pinned") is not None else None,
            extra=extra,
        )

        result = await service.store_long_term(
            workspace_id=ws_id,
            content=content,
            agent_id=agent_id_int,
            metadata=metadata,
        )

        if result.get("error"):
            return {"success": False, "error": result["error"]}

        ns = service.namespace(ws_id)
        user_id = ns.resolve(agent_id_int)
        facts = result.get("facts_extracted", "unknown")
        return {
            "success": True,
            "message": f"Stored in memory (user_id={user_id}): '{content[:100]}'",
            "facts_extracted": facts,
        }
    except Exception as e:
        logger.warning(f"[PlatformExecutor] Memory store failed: {e}", exc_info=True)
        return {"success": False, "error": f"Memory service error: {e}"}


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
OPERATOR_WORKSPACE_SETTINGS_KEYS = ("byok_overrides", "default_notification_channel")


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

        query = db.query(SystemSetting)
        category = params.get("category")
        if category:
            query = query.filter(SystemSetting.category == category)

        rows = query.order_by(SystemSetting.category, SystemSetting.key).all()
        settings = [_masked_setting(s) for s in rows]
        return {"success": True, "settings": settings, "count": len(settings)}
    except Exception as exc:
        logger.error("[workspace] list_system_settings failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


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

        setting = (
            db.query(SystemSetting)
            .filter(SystemSetting.category == category, SystemSetting.key == key)
            .first()
        )
        if not setting:
            return {"success": False, "error": f"Setting not found: {category}.{key}"}

        setting.value = str(value)
        setting.updated_at = datetime.utcnow()
        db.commit()

        logger.info(
            "[workspace] system setting %s.%s updated via platform tool (workspace %s)",
            category, key, workspace_id,
        )
        return {"success": True, "setting": _masked_setting(setting)}
    except Exception as exc:
        db.rollback()
        logger.error("[workspace] update_system_setting failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}
