"""Workspace handlers for PlatformActionExecutor — workspace info, memory stats, connected apps, store memory."""

import asyncio
import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


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
        if not service.is_mem0_configured:
            return {"success": False, "error": "Memory service not configured (MEM0_API_URL empty)"}

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

    try:
        from datetime import datetime, timezone
        from modules.memory.unified_memory_service import get_unified_memory_service

        service = get_unified_memory_service()
        if not service.is_mem0_configured:
            return {"success": False, "error": "Memory service not configured (MEM0_API_URL empty)"}

        ws_id = str(workspace_id)
        agent_id = params.get("agent_id")
        # Cast to int if provided (may come as string from tool params)
        agent_id_int = int(agent_id) if agent_id else None

        # Wave 3 — provenance keys travel with the memory metadata so
        # future readers can tell verified facts from inference.
        metadata: Dict[str, Any] = {
            "workspace_id": ws_id,
            "source": "platform_tool",
            "source_type": source_type,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        if confidence is not None:
            metadata["confidence"] = confidence
        if params.get("evidence_uri"):
            metadata["evidence_uri"] = params["evidence_uri"]

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
