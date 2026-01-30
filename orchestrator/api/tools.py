

"""
Tools API (Rewrite)

This router is the new source-of-truth for the Tools page.
It serves marketplace metadata from local cache tables for fast loads.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.composio.client import get_composio_client
from core.composio.entity_manager import EntityManager
from core.database.database import get_db
from core.models.composio_cache import ComposioActionCache, ComposioAppCache, ComposioStatsCache
from services.metadata_sync_service import MetadataSyncService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tools", tags=["Tools"])

INTERNAL_APP_NAMES = {"RAG", "MEMORY", "NL2SQL", "CODEGRAPH"}


class AppOut(BaseModel):
    id: int
    app_name: str
    display_name: str
    description: Optional[str] = None
    logo_url: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    auth_schemes: List[str] = Field(default_factory=list)
    action_count: int = 0
    trigger_count: int = 0
    status: str = "ACTIVE"
    is_connected: bool = False
    triggers: List[Dict[str, Any]] = Field(default_factory=list)  # Include triggers in response


class MarketplaceOut(BaseModel):
    apps: List[AppOut]
    total_apps: int
    total_actions: int
    categories: Dict[str, Any]
    last_synced: Optional[str] = None


class StatsOut(BaseModel):
    total_apps: int
    total_actions: int
    connected_apps: int
    categories: Dict[str, Any]
    last_synced: Optional[str] = None


class ConnectIn(BaseModel):
    app_name: str
    callback_url: Optional[str] = None


@router.get("/marketplace", response_model=MarketplaceOut)
async def marketplace(
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    # Connected apps for this workspace (from DB)
    # NOTE: Some Composio hosted-auth callbacks don't always include `connection_id`.
    # To avoid "connected in Composio but not showing in UI", opportunistically
    # sync any pending connections to ACTIVE on read.
    # OPTIMIZATION: Only sync pending connections that haven't been checked recently (within last 30s)
    # to reduce redundant Composio API calls (which also reduces telemetry volume).
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    connected_set = set()
    if entity:
        from datetime import datetime, timedelta
        connections = entity_manager.get_entity_connections(entity["id"])
        pending_to_sync = []
        for conn in connections:
            status = (conn.get("status") or "").lower()
            if status == "pending":
                # Only sync if last_synced_at is older than 30 seconds or doesn't exist
                last_synced = conn.get("last_synced_at")
                if not last_synced or (isinstance(last_synced, datetime) and (datetime.utcnow() - last_synced) > timedelta(seconds=30)):
                    pending_to_sync.append(conn)
            elif status == "active":
                connected_set.add((conn.get("app_name") or "").upper())
        
        # Batch sync pending connections (only if needed)
        if pending_to_sync:
            client = get_composio_client()
            for conn in pending_to_sync:
                try:
                    composio_status = client.get_connection_status(
                        entity_id=entity["composio_entity_id"],
                        app=(conn.get("app_name") or "").upper(),
                    )
                    if composio_status and composio_status.get("status") == "ACTIVE":
                        entity_manager.update_connection_status(
                            entity_id=entity["id"],
                            app_name=conn.get("app_name") or "",
                            status="active",
                            connection_id=composio_status.get("id"),
                        )
                        connected_set.add((conn.get("app_name") or "").upper())
                except Exception:
                    # best-effort sync only - mark as checked even on failure to avoid retry storms
                    try:
                        entity_manager.update_connection_status(
                            entity_id=entity["id"],
                            app_name=conn.get("app_name") or "",
                            status="pending",  # Keep as pending but update last_synced_at
                        )
                    except Exception:
                        pass
        else:
            # No pending connections to sync - just read active ones
            connected_set = {
                (c.get("app_name") or "").upper()
                for c in connections
                if (c.get("status") or "").lower() == "active"
            }

    # Read stats from composio_stats_cache (populated by /api/tools/sync)
    stats_rows = db.query(ComposioStatsCache).all()
    stats = {r.stat_key: r.stat_value for r in stats_rows}
    last_synced = (stats.get("last_full_sync") or {}).get("timestamp")

    # Cache-only marketplace (no live fallback).
    # Internal tools should never appear here.
    q = db.query(ComposioAppCache).filter(
        ComposioAppCache.status == "ACTIVE",
        ~ComposioAppCache.app_name.in_(INTERNAL_APP_NAMES),
    )
    if category:
        q = q.filter(ComposioAppCache.categories.contains([category]))
    if search:
        st = f"%{search}%"
        q = q.filter(
            (ComposioAppCache.display_name.ilike(st))
            | (ComposioAppCache.app_name.ilike(st))
            | (ComposioAppCache.description.ilike(st))
        )

    total_apps = q.count()
    apps = (
        q.order_by(ComposioAppCache.action_count.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    # Build response with triggers from app_metadata
    apps_out = []
    for a in apps:
        # Extract triggers from app_metadata JSONB field
        meta = a.app_metadata or {}
        triggers = meta.get("triggers") or []
        
        # If triggers are missing but trigger_count > 0, log a warning
        # This indicates triggers weren't synced properly
        if not triggers and a.trigger_count > 0:
            logger.warning(
                f"App {a.app_name} has trigger_count={a.trigger_count} but no triggers in metadata. "
                f"Re-sync may be needed: POST /api/tools/sync"
            )
        
        apps_out.append(
            AppOut(
                id=a.id,
                app_name=a.app_name,
                display_name=a.display_name,
                description=a.description,
                logo_url=a.logo_url,
                categories=a.categories or [],
                auth_schemes=a.auth_schemes or [],
                action_count=a.action_count or 0,
                trigger_count=a.trigger_count or 0,
                status=a.status,
                is_connected=a.app_name.upper() in connected_set,
                triggers=triggers,  # Include triggers array
            )
        )
    
    return MarketplaceOut(
        apps=apps_out,
        total_apps=total_apps,
        total_actions=(stats.get("total_actions") or {}).get("count", 0),
        categories=stats.get("categories") or {},
        last_synced=last_synced,
    )


@router.get("/stats", response_model=StatsOut)
async def stats(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    stats_rows = db.query(ComposioStatsCache).all()
    stats = {r.stat_key: r.stat_value for r in stats_rows}

    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    connected = 0
    if entity:
        connected = sum(
            1
            for c in entity_manager.get_entity_connections(entity["id"])
            if c.get("status") == "active"
        )

    return StatsOut(
        total_apps=(stats.get("total_apps") or {}).get("count", 0),
        total_actions=(stats.get("total_actions") or {}).get("count", 0),
        connected_apps=connected,
        categories=stats.get("categories") or {},
        last_synced=(stats.get("last_full_sync") or {}).get("timestamp"),
    )


@router.get("/connected")
async def connected(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    if not entity:
        return {"apps": [], "total": 0}

    connections = entity_manager.get_entity_connections(entity["id"])
    
    # OPTIMIZATION: Only sync pending connections that haven't been checked recently (within last 30s)
    # to reduce redundant Composio API calls (which also reduces telemetry volume).
    from datetime import datetime, timedelta
    pending_to_sync = []
    for conn in connections:
        status = (conn.get("status") or "").lower()
        if status == "pending":
            last_synced = conn.get("last_synced_at")
            if not last_synced or (isinstance(last_synced, datetime) and (datetime.utcnow() - last_synced) > timedelta(seconds=30)):
                pending_to_sync.append(conn)
    
    # Batch sync pending connections (only if needed)
    if pending_to_sync:
        client = get_composio_client()
        for conn in pending_to_sync:
            try:
                composio_status = client.get_connection_status(
                    entity_id=entity["composio_entity_id"],
                    app=(conn.get("app_name") or "").upper(),
                )
                if composio_status and composio_status.get("status") == "ACTIVE":
                    entity_manager.update_connection_status(
                        entity_id=entity["id"],
                        app_name=conn.get("app_name") or "",
                        status="active",
                        connection_id=composio_status.get("id"),
                    )
            except Exception:
                # Mark as checked even on failure to avoid retry storms
                try:
                    entity_manager.update_connection_status(
                        entity_id=entity["id"],
                        app_name=conn.get("app_name") or "",
                        status="pending",
                    )
                except Exception:
                    pass
        # Re-read after potential updates
        connections = entity_manager.get_entity_connections(entity["id"])
    
    active = [c for c in connections if (c.get("status") or "").lower() == "active"]

    # Enrich with cached metadata if present
    conn_app_names = [c.get("app_name") for c in active if c.get("app_name")]
    app_names_upper = [(a or "").upper() for a in conn_app_names]
    cache = {
        a.app_name: a
        for a in db.query(ComposioAppCache).filter(
            ComposioAppCache.app_name.in_(list(set(app_names_upper)))
        ).all()
    }
    out = []
    for c in active:
        app_name = (c.get("app_name") or "").upper()
        cached = cache.get(app_name)
        meta = (cached.app_metadata or {}) if cached else {}
        triggers = meta.get("triggers") or []
        action_count = cached.action_count if cached else 0
        if action_count == 0:
            n = (
                db.query(ComposioActionCache)
                .filter(ComposioActionCache.app_name == app_name)
                .count()
            )
            action_count = n
        out.append(
            {
                "id": cached.id if cached else None,
                "app_name": app_name,
                "status": c.get("status"),
                "connected_at": c.get("connected_at"),
                "connection_id": c.get("connection_id"),
                "display_name": cached.display_name if cached else app_name,
                "description": cached.description if cached else None,
                "logo_url": cached.logo_url if cached else None,
                "categories": cached.categories if cached else [],
                "action_count": action_count,
                "trigger_count": cached.trigger_count if cached else 0,
                "triggers": triggers if isinstance(triggers, list) else [],
            }
        )
    return {"apps": out, "total": len(out)}


@router.get("/{app_name}/actions")
async def app_actions(
    app_name: str,
    search: Optional[str] = Query(None),
    # NOTE: Many apps (e.g., GitHub) exceed 500 actions.
    # Default higher so UI can show "all actions" without implementing pagination first.
    limit: int = Query(5000, ge=1, le=20000),
    offset: int = Query(0, ge=0),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get actions for an app with enabled state for this workspace."""
    from core.models.tool_assignments import WorkspaceToolConfig

    # DB-backed: read from composio_actions_cache (populated by /api/tools/sync)
    q = db.query(ComposioActionCache).filter(ComposioActionCache.app_name == app_name.upper())
    if search:
        st = f"%{search}%"
        q = q.filter(
            (ComposioActionCache.action_name.ilike(st))
            | (ComposioActionCache.display_name.ilike(st))
            | (ComposioActionCache.description.ilike(st))
        )

    clean_tool_id = app_name.lower()
    config = (
        db.query(WorkspaceToolConfig)
        .filter(WorkspaceToolConfig.workspace_id == ctx.workspace_id, WorkspaceToolConfig.tool_id == clean_tool_id)
        .first()
    )
    enabled_actions = set()
    if config and config.configuration:
        enabled_actions = set(config.configuration.get("enabled_actions", []))

    rows = q.order_by(ComposioActionCache.action_name.asc()).offset(offset).limit(limit).all()
    out: List[Dict[str, Any]] = []
    for a in rows:
        out.append(
            {
                "name": a.action_name,
                "display_name": a.display_name,
                "description": a.description,
                "app_name": a.app_name,
                "parameters": a.parameters or {},
                "enabled": a.action_name in enabled_actions,
            }
        )
    return out


@router.get("/{app_name}/triggers")
async def app_triggers(
    app_name: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Get triggers for an app.

    Triggers are synced into `composio_apps_cache.metadata["triggers"]` during `/api/tools/sync`.
    This avoids schema changes while still exposing all Composio trigger types in the UI.
    """
    cached = db.query(ComposioAppCache).filter(ComposioAppCache.app_name == app_name.upper()).first()
    if not cached:
        return []
    meta = cached.app_metadata or {}
    triggers = meta.get("triggers") or []
    return triggers if isinstance(triggers, list) else []


@router.post("/{app_name}/actions")
async def save_app_actions(
    app_name: str,
    payload: Dict[str, Any],
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Persist enabled actions for an app in this workspace."""
    from core.models.tool_assignments import WorkspaceToolConfig

    enabled_actions = payload.get("actions", []) or []
    clean_tool_id = app_name.lower()

    config = (
        db.query(WorkspaceToolConfig)
        .filter(WorkspaceToolConfig.workspace_id == ctx.workspace_id, WorkspaceToolConfig.tool_id == clean_tool_id)
        .first()
    )

    if not config:
        config = WorkspaceToolConfig(
            workspace_id=ctx.workspace_id,
            tool_id=clean_tool_id,
            display_name=app_name,
            enabled=True,
            configuration={"enabled_actions": enabled_actions},
        )
        db.add(config)
    else:
        current_config = dict(config.configuration or {})
        current_config["enabled_actions"] = enabled_actions
        config.configuration = current_config
        config.enabled = True

    db.commit()
    return {"status": "success", "enabled_count": len(enabled_actions), "app_name": app_name.upper()}


@router.post("/connect")
async def connect_app(
    payload: ConnectIn,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    # Delegate to existing Composio client + entity manager logic
    client = get_composio_client()
    entity_manager = EntityManager(db)
    entity = entity_manager.get_or_create_entity(ctx.workspace_id)

    app_name = payload.app_name.upper()
    try:
        redirect_url = client.initiate_connection(
            entity_id=entity["composio_entity_id"],
            app=app_name,
            callback_url=payload.callback_url,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to initiate OAuth: {str(e)}")

    # Store pending connection in DB
    entity_manager.add_connection(entity_id=entity["id"], app_name=app_name, status="pending")
    return {"redirect_url": redirect_url, "app_name": app_name}


@router.post("/sync")
async def sync(
    sync_type: str = Query("full", description="full or incremental"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    logger.info(f"Sync requested: type={sync_type}, workspace_id={ctx.workspace_id}")
    service = MetadataSyncService(db)
    if sync_type == "incremental":
        result = service.run_incremental_sync()
    else:
        result = service.run_full_sync()
    logger.info(f"Sync completed: {result}")
    return result


@router.get("/sync/history")
async def sync_history(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    service = MetadataSyncService(db)
    return {"jobs": service.get_sync_history(limit), "count": limit}


# PRD-40/41: Dynamic Tool Suggestions (Phase 1 + Phase 2: Context-Aware)
class SuggestionsOut(BaseModel):
    """Response model for tool suggestions"""
    app: str
    suggestions: List[str]
    source: str  # "curated" or "generated"
    has_context: bool = False  # PRD-41: Whether context suggestions were included


@router.get("/{app_name}/suggestions", response_model=SuggestionsOut)
async def get_tool_suggestions(
    app_name: str,
    user_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    request: Request = None,
    db: Session = Depends(get_db),
):
    """
    Get suggestion prompts for a specific tool/app.

    Phase 1: Returns curated suggestions if available, otherwise generates
    basic suggestions from action schemas.

    Phase 2 (PRD-41): If user_id and session_id provided, merges context-aware
    suggestions based on recent tool results from Mem0.

    Args:
        app_name: The Composio app name (case-insensitive)
        user_id: Optional user/workspace ID for context retrieval
        session_id: Optional session ID for context filtering

    Returns:
        SuggestionsOut with app name, suggestions list, source type, and has_context flag
    """
    # Normalize app name to uppercase (Composio convention)
    app_name_upper = app_name.upper()

    # PRD-41: Check for context-aware suggestions if user_id provided
    context_suggestions = []
    has_context = False

    # Get workspace_id from headers (frontend sends via X-Workspace-ID)
    workspace_id = request.headers.get("x-workspace-id") if request else None

    if workspace_id and session_id:
        try:
            from modules.memory.context_manager import get_recent_tool_context

            # Format user_id with ws_ prefix for consistency with SmartMemory
            context_user_id = f"ws_{workspace_id}"

            logger.info(f"[ContextSuggestions] Fetching for workspace_id={workspace_id}, user_id={context_user_id}, app={app_name_upper}")

            # Query Redis for recent tool context (10-minute TTL)
            contexts = get_recent_tool_context(
                user_id=context_user_id,
                tool_name=app_name_upper,
                max_age_minutes=10,
                limit=5
            )

            logger.info(f"[ContextSuggestions] Retrieved {len(contexts) if contexts else 0} contexts from Redis for user_id={context_user_id}")

            if contexts:
                # Generate context-aware suggestions from entities
                for context in contexts:
                    entities = context.get("entities", {})
                    logger.info(f"[ContextSuggestions] Context entities: {entities}")
                    if entities:
                        context_sugg = generate_context_suggestions(
                            app_name_upper,
                            entities,
                            limit=2
                        )
                        logger.info(f"[ContextSuggestions] Generated {len(context_sugg)} context suggestions: {context_sugg}")
                        context_suggestions.extend(context_sugg)
                        if context_sugg:
                            has_context = True
                        break  # Use most recent context only

        except Exception as e:
            logger.warning(f"Failed to retrieve context suggestions for {app_name_upper}: {e}")
            # Continue with Phase 1 behavior

    # Get curated or generated suggestions (Phase 1)
    base_suggestions = []
    source = "curated"

    # Try to get curated suggestions from cache
    cached = db.query(ComposioActionCache).filter(
        ComposioActionCache.app_name == app_name_upper
    ).first()

    if cached and cached.app_suggestions and len(cached.app_suggestions) > 0:
        base_suggestions = cached.app_suggestions
        source = "curated"
    else:
        # Fallback: Generate from action schemas
        base_suggestions = _generate_suggestions_from_schema(app_name_upper, db)
        source = "generated"

    # Merge context + base suggestions
    # Context suggestions appear first (top 2 positions)
    # Then fill with base suggestions to reach 4 total
    if context_suggestions:
        final_suggestions = context_suggestions[:2]  # Top 2 context suggestions
        remaining = 4 - len(final_suggestions)
        final_suggestions.extend(base_suggestions[:remaining])
        logger.info(f"[ContextSuggestions] Final result: {len(context_suggestions)} context + {remaining} base = {len(final_suggestions)} total")
    else:
        final_suggestions = base_suggestions[:4]
        logger.info(f"[ContextSuggestions] No context suggestions, using {len(final_suggestions)} base suggestions only")

    return SuggestionsOut(
        app=app_name_upper,
        suggestions=final_suggestions,
        source=source,
        has_context=has_context
    )


def generate_context_suggestions(
    tool_name: str,
    entities: Dict[str, List[Any]],
    limit: int = 2
) -> List[str]:
    """
    Generate context-aware suggestions from extracted entities (PRD-41: US-008).

    Works with ALL 850+ tools. Specialized logic for Gmail/Slack/GitHub,
    generic fallback for everything else (Linear, Jira, Notion, Asana, etc.).

    Creates natural language suggestions that reference specific entities:
    - Gmail: "Reply to Sarah's email" (specialized)
    - Slack: "Send message to #general" (specialized)
    - GitHub: "Review PR #123" (specialized)
    - Linear/Jira/ANY tool: "View details for '[item]'" (generic)

    Args:
        tool_name: The tool/app name (e.g., "GMAIL", "LINEAR", "JIRA", etc.)
        entities: Dictionary of extracted entities from tool result
        limit: Maximum number of suggestions to generate (default: 2)

    Returns:
        List of context-specific suggestion strings (always returns suggestions)

    Example:
        >>> entities = {"senders": ["Sarah Johnson"], "labels": ["IMPORTANT"]}
        >>> generate_context_suggestions("GMAIL", entities, limit=2)
        ["Reply to Sarah Johnson's email", "Reply to the urgent email"]

        >>> entities = {"names": ["Bug fix"], "ids": ["ISS-123"]}
        >>> generate_context_suggestions("JIRA", entities, limit=2)
        ["View details for 'Bug fix'", "Open item #ISS-123"]
    """
    suggestions = []

    if tool_name == "GMAIL":
        # Reference specific senders
        senders = entities.get("senders", [])
        if senders:
            # Use first sender, truncate long names
            sender = senders[0]
            if len(sender) > 25:
                sender = sender[:22] + "..."
            suggestions.append(f"Reply to {sender}'s email")

        # Reference urgent/important emails
        labels = entities.get("labels", [])
        if "IMPORTANT" in labels or "urgent" in str(labels).lower():
            suggestions.append("Reply to the urgent email")

        # Reference subjects (truncated)
        subjects = entities.get("subjects", [])
        if subjects and len(suggestions) < limit:
            subject = subjects[0]
            if len(subject) > 30:
                subject = subject[:27] + "..."
            suggestions.append(f"Show email about '{subject}'")

    elif tool_name == "SLACK":
        # Reference channels
        channels = entities.get("channels", [])
        if channels:
            channel = channels[0]
            suggestions.append(f"Send message to {channel}")

        # Reference mentions
        mentions = entities.get("mentions", [])
        if mentions and len(suggestions) < limit:
            mention = mentions[0]
            suggestions.append(f"Reply to {mention}")

    elif tool_name == "GITHUB":
        # Reference PRs
        pr_numbers = entities.get("pr_numbers", [])
        if pr_numbers:
            pr_num = pr_numbers[0]
            suggestions.append(f"Review PR #{pr_num}")

        # Reference issues
        issue_numbers = entities.get("issue_numbers", [])
        if issue_numbers and len(suggestions) < limit:
            issue_num = issue_numbers[0]
            suggestions.append(f"Comment on issue #{issue_num}")

        # Reference repos
        repos = entities.get("repos", [])
        if repos and len(suggestions) < limit:
            repo = repos[0]
            if len(repo) > 30:
                repo = repo[:27] + "..."
            suggestions.append(f"Show activity in {repo}")

    # GENERIC FALLBACK: Handle all other tools (850+ tools)
    # If we don't have specialized suggestions yet, generate from generic entities
    if not suggestions and entities:
        # Try names (most common - titles, subjects, labels, etc.)
        names = entities.get("names", [])
        if names:
            name = names[0]
            if len(name) > 30:
                name = name[:27] + "..."
            suggestions.append(f"View details for '{name}'")

        # Try IDs (second most common)
        ids = entities.get("ids", [])
        if ids and len(suggestions) < limit:
            item_id = str(ids[0])
            if len(item_id) > 20:
                item_id = item_id[:17] + "..."
            suggestions.append(f"Open item #{item_id}")

        # Try key_values (status, type, etc.)
        key_values = entities.get("key_values", [])
        if key_values and len(suggestions) < limit:
            kv = key_values[0]
            if len(kv) > 40:
                kv = kv[:37] + "..."
            suggestions.append(f"Show items with {kv}")

        # Try emails (if email-based tool)
        emails = entities.get("emails", [])
        if emails and len(suggestions) < limit:
            email = emails[0]
            if len(email) > 30:
                email = email[:27] + "..."
            suggestions.append(f"Contact {email}")

    # Return up to limit
    return suggestions[:limit]


def _generate_suggestions_from_schema(app_name: str, db: Session) -> List[str]:
    """
    Generate basic suggestions from action descriptions.

    This is a fallback for apps without curated suggestions.
    Analyzes action descriptions to create natural language prompts.

    Args:
        app_name: The app name to generate suggestions for
        db: Database session

    Returns:
        List of generated suggestion strings (3-4 suggestions)
    """
    # Get top 10 actions for this app
    actions = db.query(ComposioActionCache).filter(
        ComposioActionCache.app_name == app_name.upper()
    ).limit(10).all()

    if not actions:
        return [f"Get started with {app_name}"]

    suggestions = set()

    for action in actions:
        desc = (action.description or "").lower()
        action_display = action.display_name or action.action_name

        # Pattern matching to generate prompts
        if any(verb in desc for verb in ["list", "fetch", "get", "retrieve", "show"]):
            suggestions.add(f"Show my {app_name.lower()} items")
        if any(verb in desc for verb in ["send", "create", "post", "add"]):
            suggestions.add(f"Create a new {app_name.lower()} item")
        if "search" in desc or "find" in desc:
            suggestions.add(f"Search {app_name.lower()} for...")
        if any(verb in desc for verb in ["update", "edit", "modify", "change"]):
            suggestions.add(f"Update a {app_name.lower()} item")
        if any(verb in desc for verb in ["delete", "remove", "revoke"]):
            suggestions.add(f"Delete a {app_name.lower()} item")

    # Return up to 4 suggestions
    result = list(suggestions)[:4]

    # If we have fewer than 3, add generic ones
    if len(result) < 3:
        result.extend([
            f"Connect to {app_name}",
            f"Explore {app_name} features",
            f"Get help with {app_name}"
        ])

    return result[:4]
