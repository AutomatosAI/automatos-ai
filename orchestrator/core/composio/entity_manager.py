"""
Composio Entity Manager.

Persists Composio entities + connections in the database so the UI can show
"connected tools" and drive OAuth connect/disconnect flows.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.composio import ComposioConnection, ComposioEntity


class EntityManager:
    def __init__(self, db: Session):
        self.db = db

    # ---------------------------------------------------------------------
    # Workspace-scoped entity helpers
    # ---------------------------------------------------------------------
    def get_entity_by_workspace(self, workspace_id: UUID) -> Optional[Dict[str, Any]]:
        row = (
            self.db.query(ComposioEntity)
            .filter(ComposioEntity.workspace_id == workspace_id)
            .first()
        )
        if not row:
            return None
        return {
            "id": row.id,
            "workspace_id": row.workspace_id,
            "composio_entity_id": row.composio_entity_id,
            "created_at": row.created_at,
        }

    def get_or_create_entity(self, workspace_id: UUID) -> Dict[str, Any]:
        existing = (
            self.db.query(ComposioEntity)
            .filter(ComposioEntity.workspace_id == workspace_id)
            .first()
        )
        if existing:
            return {
                "id": existing.id,
                "workspace_id": existing.workspace_id,
                "composio_entity_id": existing.composio_entity_id,
                "created_at": existing.created_at,
            }

        # In the current Composio SDK shape, "entity_id" can just be a stable string.
        # Use the workspace UUID string for deterministic mapping.
        row = ComposioEntity(
            workspace_id=workspace_id,
            composio_entity_id=str(workspace_id),
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return {
            "id": row.id,
            "workspace_id": row.workspace_id,
            "composio_entity_id": row.composio_entity_id,
            "created_at": row.created_at,
        }

    # ---------------------------------------------------------------------
    # Connection helpers
    # ---------------------------------------------------------------------
    def get_entity_connections(self, entity_id: str) -> List[Dict[str, Any]]:
        try:
            entity_pk = int(entity_id)
        except Exception:
            return []

        rows = (
            self.db.query(ComposioConnection)
            .filter(ComposioConnection.entity_id == entity_pk)
            .order_by(ComposioConnection.updated_at.desc())
            .all()
        )
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": r.id,
                    "entity_id": r.entity_id,
                    "app_name": (r.app_name or "").upper(),
                    "status": r.status,
                    "connected_at": r.connected_at,
                    "connection_id": r.connection_id,
                    "last_synced_at": r.last_synced_at,
                }
            )
        return out

    def get_connected_apps(self, workspace_id: UUID) -> List[str]:
        entity = self.get_entity_by_workspace(workspace_id)
        if not entity:
            return []
        conns = self.get_entity_connections(str(entity["id"]))
        return [
            (c.get("app_name") or "").upper()
            for c in conns
            if (c.get("status") or "").lower() == "active"
        ]

    def add_connection(
        self,
        entity_id: str,
        app_name: str,
        status: str = "pending",
        connection_id: Optional[str] = None,
        **_: Any,
    ) -> None:
        try:
            entity_pk = int(entity_id)
        except Exception:
            return None

        app = (app_name or "").upper()
        row = (
            self.db.query(ComposioConnection)
            .filter(ComposioConnection.entity_id == entity_pk, ComposioConnection.app_name == app)
            .first()
        )
        if not row:
            row = ComposioConnection(entity_id=entity_pk, app_name=app, status=status)
            self.db.add(row)
        else:
            row.status = status or row.status

        # Store connection_id when provided
        if connection_id is not None:
            row.connection_id = connection_id

        # Set connected_at for active connections
        if status == "active":
            row.connected_at = row.connected_at or datetime.utcnow()

        row.last_synced_at = datetime.utcnow()
        self.db.commit()
        return None

    def update_connection_status(self, entity_id: str, app_name: str, status: str, **_: Any) -> bool:
        try:
            entity_pk = int(entity_id)
        except Exception:
            return False

        app = (app_name or "").upper()
        row = (
            self.db.query(ComposioConnection)
            .filter(ComposioConnection.entity_id == entity_pk, ComposioConnection.app_name == app)
            .first()
        )
        if not row:
            return False

        row.status = status
        if status == "active":
            row.connected_at = row.connected_at or datetime.utcnow()
        if "connection_id" in _:
            row.connection_id = _["connection_id"]
        row.last_synced_at = datetime.utcnow()
        self.db.commit()
        return True

    def remove_connection(self, entity_id: str, app_name: str) -> bool:
        try:
            entity_pk = int(entity_id)
        except Exception:
            return False

        app = (app_name or "").upper()
        row = (
            self.db.query(ComposioConnection)
            .filter(ComposioConnection.entity_id == entity_pk, ComposioConnection.app_name == app)
            .first()
        )
        if not row:
            return False
        self.db.delete(row)
        self.db.commit()
        return True

