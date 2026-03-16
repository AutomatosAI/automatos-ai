"""Tools, LLMs, and datasources discovery handlers for PlatformActionExecutor."""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def list_tools(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List all available tools -- platform actions + Composio integrations."""
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
        apps = db.query(ComposioAppCache).all()
        # Get connected apps for this workspace
        connected = set()
        try:
            rows = (
                db.query(AgentAppAssignment.app_name)
                .join(Agent, AgentAppAssignment.agent_id == Agent.id)
                .filter(
                    Agent.workspace_id == workspace_id,
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


async def list_llms(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List available LLM models from OpenRouter cache."""
    from core.models.openrouter_cache import OpenRouterModelCache

    query = db.query(OpenRouterModelCache).filter(
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
        db.query(func.count(OpenRouterModelCache.id))
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


async def list_datasources(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List all data sources -- documents (RAG) and databases (NL2SQL)."""
    ds_type = params.get("type", "all")
    result: Dict[str, Any] = {"success": True}

    # RAG document collections
    if ds_type in ("all", "documents"):
        from core.models import Document

        docs = (
            db.query(
                Document.file_type,
                func.count(Document.id).label("count"),
                func.sum(Document.file_size).label("total_size"),
            )
            .filter(
                Document.workspace_id == workspace_id,
                Document.status == "completed",
            )
            .group_by(Document.file_type)
            .all()
        )

        total_chunks = (
            db.query(func.sum(Document.chunk_count))
            .filter(
                Document.workspace_id == workspace_id,
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
            db.query(DatabaseKnowledgeSource)
            .filter(
                DatabaseKnowledgeSource.workspace_id == workspace_id,
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
