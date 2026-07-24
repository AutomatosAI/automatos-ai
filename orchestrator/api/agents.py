from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload, subqueryload
from sqlalchemy import and_, or_, func, text
from sqlalchemy.exc import SQLAlchemyError
import time
import logging

from core.database.database import get_db
from core.models import PriorityLevel
from core.models import Agent, Skill, Pattern, agent_skills
# New cache tables (rewrite)
from core.models.composio_cache import AgentAppAssignment, ComposioAppCache
# Plugin assignment models
from core.models.marketplace_plugins import AgentAssignedPlugin, MarketplacePlugin
# Composio connection manager (used to restrict assignments to connected apps)
from core.composio.entity_manager import EntityManager
# Import Pydantic models from database.models (not models.py)
from core.models import (
    AgentCreate, AgentUpdate, AgentResponse,
    SkillCreate, SkillUpdate, SkillResponse,
    PatternCreate, PatternResponse,
    AgentStatus, AgentType
)
# Import hybrid auth (supports both Clerk JWT and API key)
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"])


# ------------------------------------------------------------------
# PRD-64: Semantic embedding helper (fire-and-forget)
# ------------------------------------------------------------------

def _reindex_agent_embedding(agent: Agent, db: Session) -> None:
    """Trigger semantic re-embedding for a single agent in background.

    Creates its own DB session to avoid lifecycle issues with the request session.
    Non-blocking: failures are logged but never bubble up to the API caller.
    """
    import asyncio

    agent_id = agent.id  # Capture before session closes

    async def _do_embed():
        from core.database.database import SessionLocal
        _db = SessionLocal()
        try:
            from core.routing.semantic_indexer import embed_agent
            _agent = _db.query(Agent).get(agent_id)
            if _agent:
                await embed_agent(_agent, _db)
        except Exception:
            logger.warning("[semantic] Background embed failed for agent %d", agent_id, exc_info=True)
        finally:
            _db.close()

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_do_embed())
    except RuntimeError:
        pass


def _stable_tool_id(name: str) -> int:
    """Match frontend stableId() hash (negative int)."""
    h = 0
    for ch in (name or ""):
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        # convert to signed 32-bit
        if h & 0x80000000:
            h = -((~h + 1) & 0xFFFFFFFF)
    if h == 0:
        return -1
    return -abs(int(h))


def _fetch_attachable_skills(db: Session, skill_ids: List[int], ctx: RequestContext) -> List["Skill"]:
    """Resolve skill ids to rows the caller may attach (PRD-191 S5, Sec §3.2.a).

    Visibility parity with api/skills.py: global (workspace_id IS NULL) or
    own-workspace skills only — a foreign workspace's private skill is
    reported exactly like a nonexistent id, never silently attached (and thus
    never prompt-injected into the caller's agent).
    """
    from api.skills import _skill_visible_to

    rows = db.query(Skill).filter(
        Skill.id.in_(skill_ids),
        Skill.is_active == True  # noqa: E712
    ).all()
    visible = [sk for sk in rows if _skill_visible_to(sk, ctx)]
    if len(visible) != len(skill_ids):
        found_ids = {sk.id for sk in visible}
        missing_ids = [sid for sid in skill_ids if sid not in found_ids]
        raise HTTPException(status_code=404, detail=f"Skills not found: {missing_ids}")
    return visible


def _assigned_by_user_id(ctx: RequestContext) -> Optional[int]:
    """
    `agent_app_assignments.assigned_by` is an INTEGER in Postgres.
    Our RequestContext `user.id` is currently a Clerk user id string (e.g. "user_..."),
    so we must never write it into this column.
    """
    try:
        raw = getattr(getattr(ctx, "user", None), "id", None)
        if raw is None:
            return None
        # Only accept numeric values
        return int(raw)
    except Exception:
        return None


def _resolve_tool_ids_to_app_names(db: Session, ctx: RequestContext, tool_ids: List[int]) -> List[str]:
    """Resolve incoming tool IDs (ComposioAppCache.id or frontend stable hash) into app_name strings.

    The Agent UI uses the *connected tools* list, so we only allow tools that are
    connected for the current workspace.
    """
    if not tool_ids:
        return []

    # Connected apps for this workspace
    entity_manager = EntityManager(db)
    entity = entity_manager.get_entity_by_workspace(ctx.workspace_id)
    connected_app_names: List[str] = []
    if entity:
        # Match the same status set as /api/tools/connected endpoint
        allowed_statuses = {"active", "added", "pending"}
        connected_app_names = [
            (c.get("app_name") or "").upper()
            for c in entity_manager.get_entity_connections(entity["id"])
            if (c.get("status") or "").lower() in allowed_statuses
        ]

    connected_set = {a for a in connected_app_names if a}
    if not connected_set:
        return []

    id_to_app: Dict[int, str] = {}

    # Map stable negative IDs for connected apps
    for app_name in connected_set:
        id_to_app[_stable_tool_id(app_name)] = app_name

    # Map DB IDs for connected apps if cached
    cached_apps = (
        db.query(ComposioAppCache)
        .filter(ComposioAppCache.app_name.in_(list(connected_set)))
        .all()
    )
    for a in cached_apps:
        id_to_app[int(a.id)] = a.app_name

    resolved: List[str] = []
    for tid in tool_ids:
        app_name = id_to_app.get(int(tid))
        if app_name and app_name not in resolved:
            resolved.append(app_name)
    return resolved


def _normalize_tags(raw_tags) -> List[str]:
    """Normalize incoming tags into a list of unique, lower-trimmed strings."""
    if raw_tags is None:
        return []
    items: List[str] = []
    if isinstance(raw_tags, str):
        items = [segment.strip() for segment in raw_tags.split(',')]
    elif isinstance(raw_tags, (list, tuple, set)):
        for value in raw_tags:
            if isinstance(value, str):
                items.extend([segment.strip() for segment in value.split(',')])
    else:
        return []

    # Deduplicate while preserving order
    seen = set()
    normalized = []
    for item in items:
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return normalized


def _build_agent_response(agent: Agent, db: Session) -> AgentResponse:
    """Build agent response with skills, tools, and plugins"""
    # PRD-15: Debug logging for model_config
    model_cfg = getattr(agent, 'model_config', None)
    logger.debug(f"Agent {agent.id} model_config: {model_cfg}")
    
    # Build tools list from the NEW assignment table (agent_app_assignments).
    tools: List[Dict[str, Any]] = []
    assignments = (
        db.query(AgentAppAssignment)
        .filter(AgentAppAssignment.agent_id == agent.id, AgentAppAssignment.is_active == True)
        .all()
    )
    if assignments:
        app_names = [a.app_name.upper() for a in assignments if a.app_name]
        cache = {
            a.app_name: a
            for a in db.query(ComposioAppCache).filter(ComposioAppCache.app_name.in_(app_names)).all()
        }
        for assignment in assignments:
            app_name = (assignment.app_name or "").upper()
            cached = cache.get(app_name)
            tools.append(
                {
                    "id": cached.id if cached else None,
                    "assignment_id": assignment.id,
                    "name": app_name,
                    "description": (cached.description if cached else "") or "",
                    "provider": "Composio" if cached else None,
                    "category": ((cached.categories or [None])[0] if cached else None),
                    "icon": cached.logo_url if cached else None,
                    "permissions": {},
                    "configuration": assignment.config or {},
                    "assigned_at": assignment.assigned_at,
                }
            )
    
    # Build plugins list from assigned_plugins relationship (eager-loaded or queried)
    plugins: List[Dict[str, Any]] = []
    assigned_plugins = getattr(agent, 'assigned_plugins', None)
    if assigned_plugins is None:
        # Fallback: query if relationship wasn't eager-loaded
        assigned_plugins = (
            db.query(AgentAssignedPlugin)
            .filter(AgentAssignedPlugin.agent_id == agent.id)
            .all()
        )
    if assigned_plugins:
        for ap in assigned_plugins:
            mp = getattr(ap, 'plugin', None)
            if mp is None:
                # Relationship not loaded — single fallback query
                mp = db.query(MarketplacePlugin).filter(MarketplacePlugin.id == ap.plugin_id).first()
            if mp:
                plugins.append({
                    "plugin_id": str(mp.id),
                    "slug": mp.slug,
                    "name": mp.name,
                    "version": mp.version,
                    "description": mp.description or "",
                    "skills_count": mp.skills_count or 0,
                    "commands_count": mp.commands_count or 0,
                })

    # Read-time legacy cleanup: remove tags from configuration if present
    # agent.tags is the single source of truth, configuration should not contain tags
    configuration = agent.configuration.copy() if agent.configuration else {}
    if "tags" in configuration:
        configuration.pop("tags", None)
        logger.debug(f"Removed legacy tags from configuration for agent {agent.id}")

    return AgentResponse(
        id=agent.id,
        public_id=str(agent.public_id) if getattr(agent, 'public_id', None) else None,
        name=agent.name,
        description=agent.description,
        job_title=getattr(agent, 'job_title', None),
        agent_type=agent.agent_type,
        status=agent.status,
        configuration=configuration,
        skills=[SkillResponse(
            id=skill.id,
            name=skill.name,
            description=skill.description,
            skill_type=skill.skill_type,
            category=getattr(skill, 'category', None) or skill.skill_type,  # Fallback to skill_type if category missing
            is_active=skill.is_active,
            created_at=skill.created_at,
            updated_at=skill.updated_at
        ).model_dump() for skill in agent.skills] if agent.skills else [],
        tools=tools,
        plugins=plugins,
        priority_level=getattr(agent, 'priority_level', 'medium') or 'medium',
        max_concurrent_tasks=getattr(agent, 'max_concurrent_tasks', 5) or 5,
        auto_start=getattr(agent, 'auto_start', False) or False,
        tags=_normalize_tags(agent.tags) if getattr(agent, 'tags', None) else [],
        created_at=agent.created_at,
        updated_at=agent.updated_at or agent.created_at,
        performance_metrics=agent.performance_metrics or {},
        created_by=agent.created_by,
        agent_model_config=getattr(agent, 'model_config', None),  # PRD-15: Include model config (field renamed to agent_model_config)
        model_usage_stats=getattr(agent, 'model_usage_stats', None),  # PRD-54: LLM usage stats
        # PRD-67: System agent fields
        is_system_agent=getattr(agent, 'is_system_agent', False) or False,
        slug=getattr(agent, 'slug', None),
        required_role=getattr(agent, 'required_role', None),
        marketplace_category=getattr(agent, 'marketplace_category', None),
        voice_profile_id=str(agent.voice_profile_id) if getattr(agent, 'voice_profile_id', None) else None,
)

# SPECIFIC ROUTES FIRST (before {agent_id})
@router.get("/types")
async def get_agent_types(ctx: RequestContext = Depends(get_request_context_hybrid)):
    """Get available agent types"""
    return {
        "data": [
            "code_architect", 
            "security_expert", 
            "performance_optimizer",
            "data_analyst", 
            "infrastructure_manager", 
            "custom", 
            "system", 
            "specialized"
        ],
        "descriptions": {
            "code_architect": "Designs and reviews code architecture",
            "security_expert": "Performs security analysis and audits", 
            "performance_optimizer": "Optimizes system performance",
            "data_analyst": "Analyzes data and generates insights",
            "infrastructure_manager": "Manages infrastructure and deployments",
            "custom": "Custom agent configuration",
            "system": "System-level operations",
            "specialized": "Specialized domain expertise"
        }
    }

@router.get("/stats")
async def get_agent_stats(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """Get comprehensive agent statistics"""
    try:
        # Filter by workspace
        total_agents = db.query(func.count(Agent.id)).filter(Agent.workspace_id == ctx.workspace_id).scalar() or 0
        active_agents = db.query(func.count(Agent.id)).filter(Agent.workspace_id == ctx.workspace_id, Agent.status == "active").scalar() or 0
        inactive_agents = db.query(func.count(Agent.id)).filter(Agent.workspace_id == ctx.workspace_id, Agent.status == "inactive").scalar() or 0
        
        # Get agent counts by type (filtered by workspace)
        agent_types = {}
        for agent_type in AgentType:
            count = db.query(func.count(Agent.id)).filter(Agent.workspace_id == ctx.workspace_id, Agent.agent_type == agent_type.value).scalar() or 0
            agent_types[agent_type.value] = count
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "inactive_agents": inactive_agents,
            "agents_by_type": agent_types,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting agent stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/bulk", response_model=List[AgentResponse], dependencies=[Depends(require_workspace_permission("agents:create"))])
async def create_agents_bulk(agents: List[AgentCreate], ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """Create multiple agents at once"""
    try:
        created_agents = []
        
        for agent_data in agents:
            tags = _normalize_tags(getattr(agent_data, 'tags', None))
            # Check if agent with this name already exists in workspace
            existing = db.query(Agent).filter(Agent.workspace_id == ctx.workspace_id, Agent.name == agent_data.name).first()
            if existing:
                raise HTTPException(status_code=400, detail=f"Agent with name '{agent_data.name}' already exists")
            
            # Create agent with workspace
            agent = Agent(
                name=agent_data.name,
                description=agent_data.description,
                agent_type=agent_data.agent_type,
                configuration=agent_data.configuration or {},
                tags=tags,
                workspace_id=ctx.workspace_id,
                created_by="api"
            )

            db.add(agent)
            db.flush()  # Get the ID

            # Add skills if provided (PRD-191 S5: visibility-gated)
            if agent_data.skill_ids:
                skills = _fetch_attachable_skills(db, agent_data.skill_ids, ctx)
                agent.skills.extend(skills)
            
            # Note: agent.tags is the single source of truth for tags.
            # Tags are NOT stored in agent.configuration to avoid duplicate state.
            # Legacy clients reading tags from configuration should migrate to use agent.tags.
            
            created_agents.append(agent)
        
        db.commit()

        # Refresh and build responses
        result = []
        for agent in created_agents:
            db.refresh(agent)
            # PRD-64: Trigger semantic embedding (non-blocking)
            _reindex_agent_embedding(agent, db)
            agent_with_skills = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == agent.id).first()
            result.append(_build_agent_response(agent_with_skills, db))

        return result
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating bulk agents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/", response_model=AgentResponse, dependencies=[Depends(require_workspace_permission("agents:create"))])
async def create_agent(agent_data: AgentCreate, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """Create a new agent with enhanced fields"""
    print("🚀 API CALL: create_agent function called!")
    try:
        logger.info(f"🔧 Creating agent: {agent_data.name} in workspace {ctx.workspace_id}, tool_ids: {agent_data.tool_ids}")
        
        # Check if agent name already exists in workspace
        existing = db.query(Agent).filter(Agent.workspace_id == ctx.workspace_id, Agent.name == agent_data.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Agent with this name already exists")
        
        tags = _normalize_tags(agent_data.tags if hasattr(agent_data, 'tags') else None)
        
        # Create agent with workspace
        from uuid import uuid4 as _uuid4
        agent = Agent(
            public_id=_uuid4(),
            name=agent_data.name,
            description=agent_data.description,
            job_title=getattr(agent_data, 'job_title', None),
            agent_type=agent_data.agent_type,
            configuration=agent_data.configuration or {},
            marketplace_category=getattr(agent_data, 'marketplace_category', None),
            tags=tags,
            workspace_id=ctx.workspace_id,
            created_by="api"
        )
        
        db.add(agent)
        db.flush()  # Get the ID
        
        # Add skills if provided (PRD-191 S5: visibility-gated)
        if agent_data.skill_ids:
            skills = _fetch_attachable_skills(db, agent_data.skill_ids, ctx)
            agent.skills.extend(skills)

        # Note: agent.tags is the single source of truth for tags.
        # Tags are NOT stored in agent.configuration to avoid duplicate state.
        # Legacy clients reading tags from configuration should migrate to use agent.tags.
        
        # Add tools (NEW: agent_app_assignments)
        if agent_data.tool_ids:
            desired_apps = _resolve_tool_ids_to_app_names(db, ctx, agent_data.tool_ids)
            for app_name in desired_apps:
                db.add(
                    AgentAppAssignment(
                        agent_id=agent.id,
                        app_name=app_name,
                        app_type="EXTERNAL",
                        assigned_by=_assigned_by_user_id(ctx),
                        is_active=True,
                        priority=0,
                        config={},
                    )
                )
        
        db.commit()
        db.refresh(agent)

        # PRD-64: Trigger semantic embedding (non-blocking)
        _reindex_agent_embedding(agent, db)

        # Load skills and tools for response
        agent_with_skills_and_tools = db.query(Agent).options(
            joinedload(Agent.skills),
        ).filter(Agent.id == agent.id).first()

        return _build_agent_response(agent_with_skills_and_tools, db)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/", response_model=List[AgentResponse])
async def list_agents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[AgentStatus] = None,
    agent_type: Optional[AgentType] = None,
    priority_level: Optional[PriorityLevel] = None,
    search: Optional[str] = None,
    include_workspace_system: bool = Query(
        False,
        description=(
            "When true, include the workspace's system agents (e.g. Auto) in "
            "the result set. Used by task assignee pickers so Auto can be "
            "assigned work without being shown in the main Roster."
        ),
    ),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """List agents with enhanced filtering and pagination"""
    try:
        # PRD-67: Build a single unified query covering workspace agents
        # AND visible system agents, so filters and pagination apply to all.
        user_role = getattr(ctx.user, "system_role", "user") if ctx.user else "user"
        _ROLE_HIERARCHY = {"super_admin": {"super_admin", "admin"}, "admin": {"admin"}}
        visible_roles = _ROLE_HIERARCHY.get(user_role, set())

        # Base scope: workspace agents OR visible system agents
        workspace_predicate = Agent.workspace_id == ctx.workspace_id
        if visible_roles:
            system_predicate = and_(
                Agent.is_system_agent.is_(True),
                Agent.status == "active",
                or_(Agent.required_role.is_(None), Agent.required_role.in_(visible_roles)),
            )
            scope_filter = or_(workspace_predicate, system_predicate)
        else:
            scope_filter = workspace_predicate

        query = (
            db.query(Agent)
            .options(joinedload(Agent.skills), subqueryload(Agent.assigned_plugins))
            .filter(scope_filter)
            .filter(Agent.agent_type != "ephemeral")  # Hide Mission Zero ephemeral clones
        )

        # By default hide per-workspace system agents (Auto) from the Roster —
        # they are managed in Settings > Orchestrator. Assignee pickers pass
        # ``include_workspace_system=true`` so Auto is selectable as a task
        # owner without being listed alongside regular agents.
        if not include_workspace_system:
            query = query.filter(
                ~and_(Agent.is_system_agent.is_(True), Agent.workspace_id.isnot(None))
            )

        # Apply filters uniformly to all agents (workspace + system)
        if status:
            query = query.filter(Agent.status == status.value)

        if agent_type:
            query = query.filter(Agent.agent_type == agent_type.value)

        if priority_level:
            query = query.filter(Agent.priority_level == priority_level.value)

        if search:
            search_filter = or_(
                Agent.name.ilike(f"%{search}%"),
                Agent.description.ilike(f"%{search}%")
            )
            query = query.filter(search_filter)

        # Deduplicate by id (system agents with workspace_id=NULL won't overlap,
        # but guard against edge cases) and apply pagination
        query = query.distinct(Agent.id)
        agents = query.offset(skip).limit(limit).all()

        return [_build_agent_response(agent, db) for agent in agents]
        
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/org-chart")
async def get_org_chart(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get agents organized as an org-chart hierarchy tree.

    Returns nodes (one per agent), edges (reports_to relationships),
    and a distinct team list for grouping in the UI.
    """
    try:
        from core.models.composio_cache import AgentAppAssignment
        from core.team_access import list_teams

        # Fetch active workspace agents — the workspace Auto (slug=auto-{ws_id})
        # is included so the chart has a single root. Excludes the global
        # Auto CTO defensively (slug=auto-cto is supposed to be workspace_id=NULL
        # but stale data has leaked in the past — keep this filter even if
        # the data is correct, so a future drift doesn't surface it again).
        # NOTE: most workspace agents have slug=NULL, and `slug != 'auto-cto'`
        # excludes NULLs in SQL three-valued logic. The OR clause keeps them in.
        agents = (
            db.query(Agent)
            .filter(Agent.workspace_id == ctx.workspace_id)
            .filter(Agent.status == "active")
            .filter(or_(Agent.slug.is_(None), Agent.slug != "auto-cto"))
            .all()
        )

        if not agents:
            return {"success": True, "nodes": [], "edges": [], "teams": []}

        agent_ids = [a.id for a in agents]

        # Batch-load tool counts per agent
        tool_counts_rows = (
            db.query(
                AgentAppAssignment.agent_id,
                func.count(AgentAppAssignment.id).label("cnt"),
            )
            .filter(AgentAppAssignment.agent_id.in_(agent_ids))
            .group_by(AgentAppAssignment.agent_id)
            .all()
        )
        tool_counts = {row.agent_id: row.cnt for row in tool_counts_rows}

        # Build lookup for direct_reports count
        direct_reports_count: Dict[int, int] = {}
        for a in agents:
            if a.reports_to_id and a.reports_to_id in {ag.id for ag in agents}:
                direct_reports_count[a.reports_to_id] = direct_reports_count.get(a.reports_to_id, 0) + 1

        # Detect system/CTO agent for orphan assignment
        system_agent = next(
            (a for a in agents if a.is_system_agent),
            None,
        )

        nodes = []
        edges = []

        for a in agents:
            skill_names = [s.name for s in a.skills] if a.skills else []
            model_id = (a.model_config or {}).get("model_id")

            parent_id = a.reports_to_id
            # If the agent has no explicit parent and isn't the system agent,
            # assign it under the system agent (CTO).
            if parent_id is None and system_agent and a.id != system_agent.id:
                parent_id = system_agent.id

            nodes.append({
                "id": a.id,
                "name": a.name,
                "job_title": a.job_title,
                "team": a.team,
                "status": a.status or "active",
                "model": model_id,
                "skills": skill_names,
                "tools_count": tool_counts.get(a.id, 0),
                "reports_to_id": parent_id,
                "direct_reports_count": direct_reports_count.get(a.id, 0),
                "is_system_agent": a.is_system_agent or False,
            })

            if parent_id is not None:
                edges.append({"from": parent_id, "to": a.id})

        # PRD-158 S1: teams come from the Teams table (same source as /api/teams)
        # so the org-chart and the team API never disagree.
        teams = [t.name for t in list_teams(db, ctx.workspace_id)]

        return {
            "success": True,
            "nodes": nodes,
            "edges": edges,
            "teams": teams,
        }

    except Exception as e:
        logger.error(f"Org chart query error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{agent_id}/status")
async def get_agent_status(agent_id: int, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """Get current status of a specific agent"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id, Agent.workspace_id == ctx.workspace_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
            
        return {
            "agent_id": agent_id,
            "name": agent.name,
            "status": agent.status,
            "agent_type": agent.agent_type,
            "priority_level": getattr(agent, 'priority_level', 'medium'),
            "max_concurrent_tasks": getattr(agent, 'max_concurrent_tasks', 5),
            "auto_start": getattr(agent, 'auto_start', False),
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
            "updated_at": agent.updated_at.isoformat() if agent.updated_at else None,
            "configuration": agent.configuration or {}
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{agent_id}/execute", dependencies=[Depends(require_workspace_permission("agents:execute"))])
async def execute_agent(agent_id: int, execution_data: dict = {}, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """Execute an agent with given parameters"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id, Agent.workspace_id == ctx.workspace_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
            
        if agent.status != "active":
            raise HTTPException(status_code=400, detail="Agent must be active to execute")
            
        # Generate execution ID and simulate execution start
        execution_id = f"exec_{agent_id}_{int(time.time())}"
        
        return {
            "execution_id": execution_id,
            "agent_id": agent_id,
            "agent_name": agent.name,
            "status": "started",
            "parameters": execution_data,
            "started_at": "2025-08-01T12:57:03Z",
            "estimated_duration": "5-10 minutes",
            "message": f"Execution started for agent {agent.name}"
        }
    except HTTPException:
        raise  
    except Exception as e:
        logger.error(f"Error executing agent: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{agent_id:int}", response_model=AgentResponse)
async def get_agent(agent_id: int, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """Get a specific agent by ID with skills and tools"""
    try:
        agent = (
            db.query(Agent)
            .options(joinedload(Agent.skills), subqueryload(Agent.assigned_plugins))
            .filter(Agent.id == agent_id, Agent.workspace_id == ctx.workspace_id)
            .first()
        )
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        return _build_agent_response(agent, db)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{agent_id}/skills")
async def get_agent_skills(agent_id: int, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """Get skills for a specific agent"""
    try:
        agent = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == agent_id, Agent.workspace_id == ctx.workspace_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        skills = [SkillResponse(
            id=skill.id,
            name=skill.name,
            description=skill.description,
            skill_type=skill.skill_type,
            category=getattr(skill, 'category', None) or skill.skill_type,  # Fallback to skill_type if category missing
            is_active=skill.is_active,
            created_at=skill.created_at,
            updated_at=skill.updated_at
        ).model_dump() for skill in agent.skills] if agent.skills else []
        
        return {"data": skills}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent skills: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{agent_id}/skills", dependencies=[Depends(require_workspace_permission("agents:update"))])
async def add_agent_skills(agent_id: int, skill_ids: List[int], ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """Add skills to an agent"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id, Agent.workspace_id == ctx.workspace_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        skills = _fetch_attachable_skills(db, skill_ids, ctx)  # PRD-191 S5
        agent.skills.extend(skills)
        db.commit()

        # PRD-64: Trigger semantic embedding (non-blocking)
        _reindex_agent_embedding(agent, db)

        return {"data": {"message": "Skills added successfully", "agent_id": agent_id, "skill_ids": skill_ids}}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error adding agent skills: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{agent_id}/skills/{skill_id}", dependencies=[Depends(require_workspace_permission("agents:update"))])
async def remove_agent_skill(agent_id: int, skill_id: int, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """Remove a single skill from an agent"""
    try:
        agent = db.query(Agent).options(joinedload(Agent.skills)).filter(
            Agent.id == agent_id, Agent.workspace_id == ctx.workspace_id
        ).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        agent.skills = [s for s in agent.skills if s.id != skill_id]
        db.commit()

        _reindex_agent_embedding(agent, db)

        return {"data": {"message": "Skill removed", "agent_id": agent_id, "skill_id": skill_id}}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error removing skill from agent: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/{agent_id}", response_model=AgentResponse, dependencies=[Depends(require_workspace_permission("agents:update"))])
async def update_agent(agent_id: int, agent_update: AgentUpdate, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """Update an existing agent"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id, Agent.workspace_id == ctx.workspace_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Update fields if provided
        if agent_update.name is not None:
            # Check for name conflicts in workspace
            existing = db.query(Agent).filter(Agent.workspace_id == ctx.workspace_id, Agent.name == agent_update.name, Agent.id != agent_id).first()
            if existing:
                raise HTTPException(status_code=400, detail="Agent with this name already exists")
            agent.name = agent_update.name
        
        if agent_update.description is not None:
            agent.description = agent_update.description

        if agent_update.job_title is not None:
            # Empty string clears the job title; non-empty stores the trimmed value.
            trimmed = agent_update.job_title.strip()
            agent.job_title = trimmed if trimmed else None

        if agent_update.status is not None:
            agent.status = agent_update.status.value

        if agent_update.tags is not None:
            tags = _normalize_tags(agent_update.tags)
            agent.tags = tags
            # Remove tags from configuration if present (cleanup legacy data)
            if agent.configuration and "tags" in agent.configuration:
                config = agent.configuration.copy()
                config.pop("tags", None)
                agent.configuration = config

        # Update agent_type if provided
        if agent_update.agent_type is not None:
            agent.agent_type = agent_update.agent_type

        # Update marketplace_category if provided (used for icon mapping)
        if agent_update.marketplace_category is not None:
            agent.marketplace_category = agent_update.marketplace_category

        # Update configuration if provided
        if agent_update.configuration is not None:
            # Merge with existing configuration
            if agent.configuration:
                agent.configuration = {**agent.configuration, **agent_update.configuration}
            else:
                agent.configuration = agent_update.configuration

        # Handle tool updates (NEW: agent_app_assignments)
        if agent_update.tool_ids is not None:
            desired_apps = _resolve_tool_ids_to_app_names(db, ctx, agent_update.tool_ids)
            desired_set = {a.upper() for a in desired_apps}

            current = (
                db.query(AgentAppAssignment)
                .filter(AgentAppAssignment.agent_id == agent.id)
                .all()
            )
            current_map = {c.app_name.upper(): c for c in current if c.app_name}

            # Disable anything no longer selected
            for app_name, row in current_map.items():
                if app_name not in desired_set:
                    row.is_active = False

            # Add or re-enable selected apps
            for app_name in desired_set:
                if app_name in current_map:
                    current_map[app_name].is_active = True
                else:
                    db.add(
                        AgentAppAssignment(
                            agent_id=agent.id,
                            app_name=app_name,
                            app_type="EXTERNAL",
                            assigned_by=_assigned_by_user_id(ctx),
                            is_active=True,
                            priority=0,
                            config={},
                        )
                    )
        
        # PRD-74: Voice profile assignment
        if agent_update.voice_profile_id is not None:
            from uuid import UUID as _UUID
            try:
                agent.voice_profile_id = _UUID(agent_update.voice_profile_id) if agent_update.voice_profile_id else None
            except (ValueError, AttributeError):
                agent.voice_profile_id = None

        db.commit()
        db.refresh(agent)

        # PRD-64: Trigger semantic embedding (non-blocking)
        _reindex_agent_embedding(agent, db)

        # Load with skills for response
        agent_with_skills = db.query(Agent).options(joinedload(Agent.skills)).filter(Agent.id == agent.id).first()

        return _build_agent_response(agent_with_skills, db)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating agent: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{agent_id}", dependencies=[Depends(require_workspace_permission("agents:delete"))])
async def delete_agent(agent_id: int, ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """Delete an agent and all related records"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id, Agent.workspace_id == ctx.workspace_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Block delete if any Playbook references this agent in its steps JSON.
        # Playbook steps store agent_id in opaque JSONB (no FK) — without this check,
        # the Playbook will crash with FK violation next time it runs.
        agent_ref = f'[{{"agent_id": {agent_id}}}]'
        playbooks_using_agent = db.execute(
            text(
                "SELECT id, name FROM workflow_recipes "
                "WHERE workspace_id = :workspace_id "
                "AND steps @> CAST(:agent_ref AS jsonb)"
            ),
            {"workspace_id": ctx.workspace_id, "agent_ref": agent_ref},
        ).fetchall()
        if playbooks_using_agent:
            names = [r.name for r in playbooks_using_agent]
            raise HTTPException(
                status_code=409,
                detail={
                    "message": f"Agent is used by {len(names)} Playbook(s). Reassign or remove the Playbook step(s) first.",
                    "playbooks": [{"id": r.id, "name": r.name} for r in playbooks_using_agent],
                },
            )

        # Delete related records first (tables without CASCADE) - use savepoints to handle errors
        # Order matters: delete in correct order to avoid FK violations

        deletions = [
            ("agent_skills", "DELETE FROM agent_skills WHERE agent_id = :agent_id", True),
            ("workflow_agents", "DELETE FROM workflow_agents WHERE agent_id = :agent_id", True),
            ("memory_items", "DELETE FROM memory_items WHERE agent_id = :agent_id", True),
            ("workflow_executions", "DELETE FROM workflow_executions WHERE agent_id = :agent_id", False),
        ]
        
        for table_name, sql_stmt, required in deletions:
            savepoint = db.begin_nested()  # Create savepoint for this deletion
            try:
                db.execute(text(sql_stmt), {"agent_id": agent_id})
                savepoint.commit()  # Commit savepoint
            except Exception as e:
                savepoint.rollback()  # Rollback savepoint, but keep main transaction
                if required:
                    logger.error(f"Error deleting {table_name} for agent {agent_id}: {e}")
                    raise HTTPException(status_code=500, detail="Internal server error")
                else:
                    logger.warning(f"Error deleting {table_name} for agent {agent_id}: {e}")
                    # Continue for optional tables
        
        # Now delete the agent (other relationships have CASCADE)
        db.delete(agent)
        db.commit()
        
        return {"message": f"Agent {agent_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting agent: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ------------------------------------------------------------------
# PRD-64: Bulk re-index semantic embeddings
# ------------------------------------------------------------------

@router.post("/reindex-embeddings", dependencies=[Depends(require_workspace_permission("agents:update"))])
async def reindex_embeddings(
    force: bool = Query(False, description="Force re-embed even if text unchanged"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Re-embed all active agents in the workspace for semantic routing.

    Use ``force=true`` to regenerate even when the semantic text hash has not changed.
    """
    try:
        from core.routing.semantic_indexer import embed_workspace_agents

        count = await embed_workspace_agents(ctx.workspace_id, db, force=force)
        total = db.query(Agent).filter(
            Agent.workspace_id == ctx.workspace_id, Agent.status == "active"
        ).count()

        return {
            "embedded": count,
            "total_active": total,
            "force": force,
            "workspace_id": str(ctx.workspace_id),
        }
    except Exception as e:
        logger.error(f"Error reindexing embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to reindex embeddings")
