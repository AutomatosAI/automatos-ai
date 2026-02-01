"""
Chatbot ingestor (PRD-50: Universal Orchestrator Router).

Normalises a ChatRequest + RequestContext into a RequestEnvelope so the
chatbot channel flows through the universal router.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from core.auth.dependencies import RequestContext
from core.models.routing import (
    ChannelSource,
    RequestEnvelope,
    RequestUser,
)
from core.routing.ingestors.base import BaseIngestor


class ChatbotIngestor(BaseIngestor):
    """Transform chatbot request parameters into a RequestEnvelope."""

    def ingest(
        self,
        *,
        message: str,
        agent_id: Optional[int] = None,
        session_id: Optional[str] = None,
        request_context: Optional[RequestContext] = None,
    ) -> RequestEnvelope:
        """Build a RequestEnvelope from chatbot inputs.

        Parameters
        ----------
        message:
            The user's chat message text.
        agent_id:
            Explicit agent selected by the user.  When provided the router
            will treat this as a Tier-0 override.  When ``None`` the router
            decides automatically.
        session_id:
            The chat/session ID (used as ``conversation_id``).
        request_context:
            The FastAPI ``RequestContext`` carrying auth and workspace info.
        """
        # --- Map RequestContext → RequestUser ---------------------------------
        user = RequestUser()
        workspace_id: Optional[UUID] = None

        if request_context is not None:
            workspace_id = request_context.workspace_id
            user = RequestUser(
                id=request_context.user.id,
                email=request_context.user.email,
                name=None,
                auth_type=request_context.auth_type,
                clerk_user_id=request_context.user.clerk_user_id,
            )

        # workspace_id is required by RequestEnvelope — fall back to a nil UUID
        # when context is missing (should not happen in practice).
        if workspace_id is None:
            workspace_id = UUID(int=0)

        # --- Conversation ID --------------------------------------------------
        conversation_id: Optional[UUID] = None
        if session_id is not None:
            try:
                conversation_id = UUID(session_id)
            except ValueError:
                # session_id is not a valid UUID — store in metadata instead
                pass

        # --- Build envelope ---------------------------------------------------
        return RequestEnvelope(
            source=ChannelSource.CHATBOT,
            content=message,
            raw_payload={"message": message, "agent_id": agent_id, "session_id": session_id},
            user=user,
            workspace_id=workspace_id,
            metadata={},
            override_agent_id=agent_id if agent_id is not None else None,
            conversation_id=conversation_id,
        )
