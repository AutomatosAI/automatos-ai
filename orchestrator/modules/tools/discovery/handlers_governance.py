"""Governance handlers — blueprints CRUD, agent validation, budget checks."""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def list_blueprints(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List all blueprints for the workspace."""
    from core.models.blueprints import AgentBlueprint

    try:
        rows = (
            db.query(AgentBlueprint)
            .filter(AgentBlueprint.workspace_id == workspace_id)
            .order_by(AgentBlueprint.created_at.desc())
            .all()
        )

        return {
            "success": True,
            "blueprints": [
                {
                    "id": str(bp.id),
                    "name": bp.name,
                    "description": bp.description,
                    "rules": bp.rules or {},
                    "is_default": bp.is_default,
                    "created_at": bp.created_at.isoformat() if bp.created_at else None,
                }
                for bp in rows
            ],
            "total": len(rows),
        }
    except Exception as e:
        logger.error("list_blueprints failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def get_blueprint(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Get a single blueprint by ID."""
    from core.models.blueprints import AgentBlueprint

    try:
        bp_id = params.get("blueprint_id")
        if not bp_id:
            return {"success": False, "error": "blueprint_id is required"}

        bp = (
            db.query(AgentBlueprint)
            .filter(
                AgentBlueprint.id == bp_id,
                AgentBlueprint.workspace_id == workspace_id,
            )
            .first()
        )

        if not bp:
            return {"success": False, "error": "Blueprint not found"}

        return {
            "success": True,
            "blueprint": {
                "id": str(bp.id),
                "name": bp.name,
                "description": bp.description,
                "rules": bp.rules or {},
                "is_default": bp.is_default,
                "created_at": bp.created_at.isoformat() if bp.created_at else None,
                "updated_at": bp.updated_at.isoformat() if bp.updated_at else None,
            },
        }
    except Exception as e:
        logger.error("get_blueprint failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def create_blueprint(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new blueprint."""
    from core.models.blueprints import AgentBlueprint

    try:
        name = params.get("name")
        if not name:
            return {"success": False, "error": "name is required"}

        rules = params.get("rules", {})
        is_default = params.get("is_default", False)

        # If setting as default, unset existing default
        if is_default:
            db.query(AgentBlueprint).filter(
                AgentBlueprint.workspace_id == workspace_id,
                AgentBlueprint.is_default == True,  # noqa: E712
            ).update({"is_default": False})

        bp = AgentBlueprint(
            workspace_id=workspace_id,
            name=name,
            description=params.get("description"),
            rules=rules,
            is_default=is_default,
        )
        db.add(bp)
        db.commit()
        db.refresh(bp)

        return {
            "success": True,
            "blueprint": {
                "id": str(bp.id),
                "name": bp.name,
                "rules": bp.rules,
                "is_default": bp.is_default,
            },
        }
    except Exception as e:
        db.rollback()
        logger.error("create_blueprint failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def update_blueprint(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing blueprint."""
    from core.models.blueprints import AgentBlueprint

    try:
        bp_id = params.get("blueprint_id")
        if not bp_id:
            return {"success": False, "error": "blueprint_id is required"}

        bp = (
            db.query(AgentBlueprint)
            .filter(
                AgentBlueprint.id == bp_id,
                AgentBlueprint.workspace_id == workspace_id,
            )
            .first()
        )

        if not bp:
            return {"success": False, "error": "Blueprint not found"}

        if "name" in params:
            bp.name = params["name"]
        if "description" in params:
            bp.description = params["description"]
        if "rules" in params:
            bp.rules = params["rules"]
        if "is_default" in params:
            if params["is_default"]:
                # Unset other defaults
                db.query(AgentBlueprint).filter(
                    AgentBlueprint.workspace_id == workspace_id,
                    AgentBlueprint.id != bp.id,
                    AgentBlueprint.is_default == True,  # noqa: E712
                ).update({"is_default": False})
            bp.is_default = params["is_default"]

        db.commit()
        db.refresh(bp)

        return {
            "success": True,
            "blueprint": {
                "id": str(bp.id),
                "name": bp.name,
                "rules": bp.rules,
                "is_default": bp.is_default,
            },
        }
    except Exception as e:
        db.rollback()
        logger.error("update_blueprint failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def validate_agent_handler(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate an agent against a blueprint."""
    from services.blueprint_validator import validate_agent

    try:
        agent_id = params.get("agent_id")
        if not agent_id:
            return {"success": False, "error": "agent_id is required"}

        blueprint_id = params.get("blueprint_id")

        result = validate_agent(
            db=db,
            workspace_id=workspace_id,
            agent_id=int(agent_id),
            blueprint_id=UUID(blueprint_id) if blueprint_id else None,
        )

        return {"success": True, **result}
    except Exception as e:
        logger.error("validate_agent_handler failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def check_budget_handler(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Check mission budget status."""
    from services.blueprint_validator import check_mission_budget

    try:
        run_id = params.get("run_id")
        if not run_id:
            return {"success": False, "error": "run_id is required"}

        result = check_mission_budget(
            db=db,
            workspace_id=workspace_id,
            run_id=UUID(run_id),
        )

        return {"success": True, **result}
    except Exception as e:
        logger.error("check_budget_handler failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}
