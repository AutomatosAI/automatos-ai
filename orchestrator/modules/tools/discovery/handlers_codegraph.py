"""CodeGraph platform-tool handlers (PRD-165 S4).

Promotes the implemented-but-unregistered CodeGraph executors to agent tools:
list projects, find a symbol, call graph, dependency/impact, and architecture
overview. Thin wrappers over CodeGraphService — all follow the standard
``(db, workspace_id, params)`` signature used by PlatformActionExecutor.

Every result carries the project's ``last_indexed`` staleness stamp so an agent
(and the UI) knows whether the answer is fresh. The default result limit is
config-driven (D11), not hardcoded.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _service(db: Session):
    from modules.codegraph.codegraph_service import CodeGraphService
    return CodeGraphService(db)


def _result_limit(params: Dict[str, Any]) -> int:
    if params.get("limit") is not None:
        try:
            return max(1, min(100, int(params["limit"])))
        except (TypeError, ValueError):
            pass
    from core.llm.manager import get_system_setting
    return int(get_system_setting("codegraph", "result_limit", "10"))


def _resolve_project(service, name: str, workspace_id: str) -> Optional[Dict[str, Any]]:
    """Resolve a project by name within the workspace → its row (id, name,
    last_indexed, status, …) or None."""
    try:
        for p in service.list_projects(workspace_id=workspace_id):
            if p.get("name") == name:
                return p
    except Exception:
        logger.debug("codegraph: list_projects failed resolving '%s'", name)
    return None


def _stale(project: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Staleness stamp for a result from a project row."""
    if not project:
        return {}
    return {
        "last_indexed": project.get("last_indexed"),
        "index_status": project.get("status"),
    }


# ------------------------------------------------------------------
# 1. list_projects
# ------------------------------------------------------------------


async def codegraph_list_projects(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """List the code repositories indexed in this workspace."""
    try:
        projects = _service(db).list_projects(workspace_id=str(workspace_id))
        return {"success": True, "project_count": len(projects), "projects": projects}
    except Exception as e:
        logger.error("codegraph_list_projects failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------
# 2. get_symbol (fuzzy name search)
# ------------------------------------------------------------------


async def codegraph_get_symbol(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Find a code symbol (function/class/method) by name in a project."""
    project = (params.get("project") or "").strip()
    query = (params.get("symbol") or params.get("query") or "").strip()
    if not project or not query:
        return {"success": False, "error": "project and symbol are required"}

    try:
        service = _service(db)
        result = await service.search_symbols(
            project, query,
            symbol_type=params.get("symbol_type"),
            limit=_result_limit(params),
            workspace_id=str(workspace_id),
        )
        return {"success": True, **result, **_stale(_resolve_project(service, project, str(workspace_id)))}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error("codegraph_get_symbol failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------
# 3. call_graph
# ------------------------------------------------------------------


async def codegraph_call_graph(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """What calls X / what does X call — the call graph for a symbol."""
    project = (params.get("project") or "").strip()
    symbol = (params.get("symbol") or "").strip()
    if not project or not symbol:
        return {"success": False, "error": "project and symbol are required"}

    direction = (params.get("direction") or "outgoing").strip().lower()
    try:
        depth = max(1, min(5, int(params.get("depth", 1))))
    except (TypeError, ValueError):
        depth = 1

    try:
        service = _service(db)
        result = await service.get_call_graph(
            project, symbol, depth=depth, direction=direction,
            workspace_id=str(workspace_id),
        )
        return {"success": True, **result, **_stale(_resolve_project(service, project, str(workspace_id)))}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error("codegraph_call_graph failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------
# 4. dependencies / impact
# ------------------------------------------------------------------


async def codegraph_dependencies(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """What breaks if I change X — symbols that depend on it (or it depends on)."""
    project = (params.get("project") or "").strip()
    symbol = (params.get("symbol") or "").strip()
    if not project or not symbol:
        return {"success": False, "error": "project and symbol are required"}

    direction = (params.get("direction") or "both").strip().lower()
    try:
        service = _service(db)
        proj = _resolve_project(service, project, str(workspace_id))
        if proj is None:
            return {"success": False, "error": f"Project '{project}' not found"}
        result = await service.find_dependencies(
            proj["id"], symbol, direction=direction, workspace_id=str(workspace_id),
        )
        return {"success": True, **result, **_stale(proj)}
    except Exception as e:
        logger.error("codegraph_dependencies failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------
# 5. architecture
# ------------------------------------------------------------------


async def codegraph_architecture(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """High-level architecture: modules, key symbols, dependency patterns."""
    project = (params.get("project") or "").strip()
    if not project:
        return {"success": False, "error": "project is required"}

    try:
        service = _service(db)
        proj = _resolve_project(service, project, str(workspace_id))
        if proj is None:
            return {"success": False, "error": f"Project '{project}' not found"}
        result = await service.analyze_architecture(
            proj["id"], workspace_id=str(workspace_id),
            focus_path=params.get("focus_path"),
        )
        return {"success": True, **result, **_stale(proj)}
    except Exception as e:
        logger.error("codegraph_architecture failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------
# 6. search_codebase (semantic routing)
# ------------------------------------------------------------------


async def codegraph_search(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Search code by meaning (semantic, default) or by name (fuzzy)."""
    project = (params.get("project") or "").strip()
    query = (params.get("query") or params.get("symbol") or "").strip()
    if not project or not query:
        return {"success": False, "error": "project and query are required"}

    mode = (params.get("mode") or "semantic").strip().lower()
    try:
        service = _service(db)
        if mode == "fuzzy":
            result = await service.search_symbols(
                project, query, limit=_result_limit(params), workspace_id=str(workspace_id),
            )
        else:
            mode = "semantic"
            result = await service.semantic_search(
                project, query, limit=_result_limit(params), workspace_id=str(workspace_id),
            )
        return {"success": True, "mode": mode, **result, **_stale(_resolve_project(service, project, str(workspace_id)))}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error("codegraph_search failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}
