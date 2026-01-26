"""AgentToolService - Manage agent app/tool assignments.

Part of the tools integration rewrite. Manages assignments of:
- External tools (Composio apps)
- Internal tools (platform-native capabilities: RAG, CodeGraph, Memory, NL2SQL)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from core.models.composio_cache import AgentAppAssignment, ComposioActionCache, ComposioAppCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal tool catalog (platform-native capabilities)
# ---------------------------------------------------------------------------

INTERNAL_TOOLS: Dict[str, Dict[str, Any]] = {
    "RAG": {
        "display_name": "Knowledge Base (RAG)",
        "description": (
            "Search and query your knowledge bases using semantic search. "
            "Backed by internal RAG modules."
        ),
        "category": "knowledge",
        "actions": ["RAG_SEARCH", "RAG_RETRIEVE", "RAG_INDEX", "RAG_DELETE"],
        "always_available": True,
    },
    "CODEGRAPH": {
        "display_name": "Code Analysis (CodeGraph)",
        "description": "Analyze code structure and dependencies.",
        "category": "code",
        "actions": [
            "CODEGRAPH_SEARCH",
            "CODEGRAPH_ANALYZE",
            "CODEGRAPH_DEPENDENCIES",
            "CODEGRAPH_IMPACT",
            "CODEGRAPH_METRICS",
        ],
        "always_available": True,
    },
    "MEMORY": {
        "display_name": "Conversation Memory",
        "description": "Store and retrieve conversational context.",
        "category": "memory",
        "actions": ["MEMORY_RECALL", "MEMORY_STORE", "MEMORY_SEARCH", "MEMORY_UPDATE", "MEMORY_DELETE"],
        "always_available": True,
    },
    "NL2SQL": {
        "display_name": "Natural Language to SQL",
        "description": "Convert natural language to SQL and query configured databases.",
        "category": "database",
        "actions": ["NL2SQL_QUERY", "NL2SQL_EXPLAIN", "NL2SQL_SCHEMA", "NL2SQL_ANALYZE"],
        "always_available": True,
    },
}


def get_always_available_tools() -> List[str]:
    return [name for name, cfg in INTERNAL_TOOLS.items() if cfg.get("always_available")]


class AgentToolService:
    def __init__(self, db_session: Session):
        self.db = db_session

    async def assign_app_to_agent(
        self,
        agent_id: int,
        app_name: str,
        assigned_by: Optional[int] = None,
        priority: int = 0,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        app_name = (app_name or "").strip().upper()
        if not app_name:
            raise ValueError("app_name is required")

        app_type = "INTERNAL" if app_name in INTERNAL_TOOLS else "EXTERNAL"

        existing = (
            self.db.query(AgentAppAssignment)
            .filter(AgentAppAssignment.agent_id == agent_id, AgentAppAssignment.app_name == app_name)
            .first()
        )
        if existing:
            existing.is_active = True
            existing.priority = priority
            existing.config = config or existing.config
            existing.assigned_by = assigned_by
            self.db.commit()
            return existing.to_dict()

        assignment = AgentAppAssignment(
            agent_id=agent_id,
            app_name=app_name,
            app_type=app_type,
            assigned_by=assigned_by,
            priority=priority,
            config=config or {},
        )
        self.db.add(assignment)
        self.db.commit()
        self.db.refresh(assignment)
        return assignment.to_dict()

    async def unassign_app_from_agent(self, agent_id: int, app_name: str, hard_delete: bool = False) -> bool:
        app_name = (app_name or "").strip().upper()
        if not app_name:
            return False

        assignment = (
            self.db.query(AgentAppAssignment)
            .filter(AgentAppAssignment.agent_id == agent_id, AgentAppAssignment.app_name == app_name)
            .first()
        )
        if not assignment:
            return False

        if hard_delete:
            self.db.delete(assignment)
        else:
            assignment.is_active = False
        self.db.commit()
        return True

    async def update_app_priority(self, agent_id: int, app_name: str, priority: int) -> bool:
        app_name = (app_name or "").strip().upper()
        assignment = (
            self.db.query(AgentAppAssignment)
            .filter(
                AgentAppAssignment.agent_id == agent_id,
                AgentAppAssignment.app_name == app_name,
                AgentAppAssignment.is_active.is_(True),
            )
            .first()
        )
        if not assignment:
            return False
        assignment.priority = priority
        self.db.commit()
        return True

    async def get_agent_tools(
        self,
        agent_id: int,
        include_inactive: bool = False,
        app_type: Optional[str] = None,
        include_always_available: bool = True,
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        explicitly_assigned: set[str] = set()

        q = self.db.query(AgentAppAssignment).filter(AgentAppAssignment.agent_id == agent_id)
        if not include_inactive:
            q = q.filter(AgentAppAssignment.is_active.is_(True))
        if app_type:
            q = q.filter(AgentAppAssignment.app_type == app_type.upper())

        assignments = q.order_by(AgentAppAssignment.priority.desc()).all()

        for a in assignments:
            tool_data = a.to_dict()
            explicitly_assigned.add(a.app_name)

            if a.app_type == "INTERNAL":
                cfg = INTERNAL_TOOLS.get(a.app_name, {})
                tool_data.update(
                    {
                        "display_name": cfg.get("display_name", a.app_name),
                        "description": cfg.get("description", ""),
                        "category": cfg.get("category", "internal"),
                        "actions": cfg.get("actions", []),
                        "always_available": bool(cfg.get("always_available", False)),
                    }
                )
            else:
                app_cache = self.db.query(ComposioAppCache).filter(ComposioAppCache.app_name == a.app_name).first()
                if app_cache:
                    tool_data.update(
                        {
                            "display_name": app_cache.display_name,
                            "description": app_cache.description,
                            "logo_url": app_cache.logo_url,
                            "categories": app_cache.categories,
                            "action_count": app_cache.action_count,
                        }
                    )

            result.append(tool_data)

        if include_always_available and (app_type is None or app_type.upper() == "INTERNAL"):
            for name, cfg in INTERNAL_TOOLS.items():
                if not cfg.get("always_available"):
                    continue
                if name in explicitly_assigned:
                    continue
                result.append(
                    {
                        "app_name": name,
                        "app_type": "INTERNAL",
                        "display_name": cfg.get("display_name", name),
                        "description": cfg.get("description", ""),
                        "category": cfg.get("category", "internal"),
                        "actions": cfg.get("actions", []),
                        "always_available": True,
                        "is_active": True,
                        "priority": 100,
                        "assigned_at": None,
                        "implicit": True,
                    }
                )

        result.sort(
            key=lambda x: (
                -1 if x.get("always_available") else 0,
                -int(x.get("priority", 0) or 0),
            )
        )
        return result

    async def get_agent_tool_names(self, agent_id: int, include_always_available: bool = True) -> List[str]:
        names = {
            row[0]
            for row in (
                self.db.query(AgentAppAssignment.app_name)
                .filter(AgentAppAssignment.agent_id == agent_id, AgentAppAssignment.is_active.is_(True))
                .all()
            )
        }
        if include_always_available:
            names.update(get_always_available_tools())
        return sorted(names)

    async def is_app_assigned(self, agent_id: int, app_name: str) -> bool:
        app_name = (app_name or "").strip().upper()
        if app_name in INTERNAL_TOOLS and INTERNAL_TOOLS[app_name].get("always_available"):
            return True
        return (
            self.db.query(AgentAppAssignment)
            .filter(
                AgentAppAssignment.agent_id == agent_id,
                AgentAppAssignment.app_name == app_name,
                AgentAppAssignment.is_active.is_(True),
            )
            .first()
            is not None
        )

    async def get_agent_actions(self, agent_id: int, app_name: Optional[str] = None) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []

        q = self.db.query(AgentAppAssignment).filter(
            AgentAppAssignment.agent_id == agent_id, AgentAppAssignment.is_active.is_(True)
        )
        if app_name:
            q = q.filter(AgentAppAssignment.app_name == app_name.upper())

        assignments = q.all()
        for a in assignments:
            if a.app_type == "INTERNAL":
                cfg = INTERNAL_TOOLS.get(a.app_name, {})
                for action in cfg.get("actions", []):
                    actions.append(
                        {
                            "app_name": a.app_name,
                            "action_name": action,
                            "display_name": action.replace("_", " ").title(),
                            "app_type": "INTERNAL",
                        }
                    )
            else:
                cached_actions = (
                    self.db.query(ComposioActionCache).filter(ComposioActionCache.app_name == a.app_name).all()
                )
                for ca in cached_actions:
                    actions.append(
                        {
                            "app_name": a.app_name,
                            "action_name": ca.action_name,
                            "display_name": ca.display_name,
                            "description": ca.description,
                            "category": ca.category,
                            "app_type": "EXTERNAL",
                        }
                    )
        return actions

    async def bulk_assign_apps(self, agent_id: int, app_names: List[str], assigned_by: Optional[int] = None) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for name in app_names or []:
            results.append(await self.assign_app_to_agent(agent_id=agent_id, app_name=name, assigned_by=assigned_by))
        return results

    async def sync_agent_apps(self, agent_id: int, app_names: List[str], assigned_by: Optional[int] = None) -> Dict[str, Any]:
        desired = {(n or "").strip().upper() for n in (app_names or []) if (n or "").strip()}
        current = {
            a.app_name
            for a in self.db.query(AgentAppAssignment)
            .filter(AgentAppAssignment.agent_id == agent_id, AgentAppAssignment.is_active.is_(True))
            .all()
        }

        to_add = desired - current
        to_remove = current - desired

        added: List[Dict[str, Any]] = []
        for name in sorted(to_add):
            added.append(await self.assign_app_to_agent(agent_id=agent_id, app_name=name, assigned_by=assigned_by))

        removed: List[str] = []
        for name in sorted(to_remove):
            await self.unassign_app_from_agent(agent_id=agent_id, app_name=name)
            removed.append(name)

        return {"added": added, "removed": removed, "unchanged": sorted(current & desired), "total_assigned": len(desired)}

    async def get_internal_tools(self) -> List[Dict[str, Any]]:
        return [{"app_name": name, "app_type": "INTERNAL", **cfg} for name, cfg in INTERNAL_TOOLS.items()]

