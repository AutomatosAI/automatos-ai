"""
Jira trigger ingestor (PRD-50: Universal Orchestrator Router).

Normalises a Jira webhook event into a RequestEnvelope so Jira trigger
events flow through the universal router.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.composio import ComposioEntity
from core.models.routing import (
    ChannelSource,
    RequestEnvelope,
    RequestUser,
)
from core.routing.ingestors.base import BaseIngestor


class JiraTriggerIngestor(BaseIngestor):
    """Transform a Jira webhook event into a RequestEnvelope."""

    def ingest(
        self,
        *,
        trigger_name: str,
        entity_id: str,
        event_data: Dict[str, Any],
        db: Session,
    ) -> RequestEnvelope:
        """Build a RequestEnvelope from a Jira trigger webhook payload.

        Parameters
        ----------
        trigger_name:
            The Composio trigger name (e.g. ``JIRA_NEW_ISSUE_TRIGGER``).
        entity_id:
            The Composio entity ID from the webhook payload.  Used to
            resolve the workspace via the ``composio_entities`` table.
        event_data:
            The raw Jira event data from the webhook payload.
        db:
            SQLAlchemy session for database lookups.
        """
        # --- Extract Jira fields from event_data ------------------------------
        issue_key: str = event_data.get("issue_key", "") or event_data.get("key", "")
        summary: str = event_data.get("summary", "")
        description: str = event_data.get("description", "") or ""
        issue_type: str = event_data.get("issue_type", "") or event_data.get("issuetype", "")
        priority: str = event_data.get("priority", "")
        project: str = event_data.get("project", "") or event_data.get("project_key", "")

        # Reporter may be nested or flat
        reporter_data = event_data.get("reporter", {})
        if isinstance(reporter_data, dict):
            reporter_email: Optional[str] = reporter_data.get("emailAddress") or reporter_data.get("email")
            reporter_name: Optional[str] = reporter_data.get("displayName") or reporter_data.get("name")
        else:
            reporter_email = None
            reporter_name = str(reporter_data) if reporter_data else None

        # --- Build content string ---------------------------------------------
        content_parts = []
        if issue_key:
            content_parts.append(f"[{issue_key}] {summary}")
        else:
            content_parts.append(summary)
        if description:
            content_parts.append(f"\n\n{description}")
        content = "".join(content_parts) if content_parts else "Jira trigger event"

        # --- Build metadata dict with all Jira fields -------------------------
        metadata: Dict[str, Any] = {
            "trigger_name": trigger_name,
            "issue_key": issue_key,
            "summary": summary,
            "description": description,
            "issue_type": issue_type,
            "priority": priority,
            "project": project,
            "reporter_email": reporter_email,
            "reporter_name": reporter_name,
        }

        # --- Build RequestUser from reporter ----------------------------------
        user = RequestUser(
            email=reporter_email,
            name=reporter_name,
            auth_type="system",
        )

        # --- Resolve workspace_id from entity_id via composio_entities --------
        workspace_id = self._resolve_workspace_id(entity_id, db)

        # --- Build envelope ---------------------------------------------------
        return RequestEnvelope(
            source=ChannelSource.JIRA_TRIGGER,
            content=content,
            raw_payload=event_data,
            user=user,
            workspace_id=workspace_id,
            metadata=metadata,
        )

    @staticmethod
    def _resolve_workspace_id(entity_id: str, db: Session) -> UUID:
        """Look up workspace_id from the composio_entities table.

        Falls back to a nil UUID if the entity is not found (should not
        happen in practice — the webhook is only fired for known entities).
        """
        entity = (
            db.query(ComposioEntity)
            .filter(ComposioEntity.composio_entity_id == entity_id)
            .first()
        )
        if entity is not None:
            return entity.workspace_id  # type: ignore[return-value]
        return UUID(int=0)
