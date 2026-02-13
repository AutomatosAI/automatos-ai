"""
Webhook ingestor (general workspace webhook).

Normalises an incoming webhook POST body into a RequestEnvelope
so general workspace webhooks flow through the universal router.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from uuid import UUID

from core.models.routing import (
    ChannelSource,
    RequestEnvelope,
    RequestUser,
)
from core.routing.ingestors.base import BaseIngestor


class WebhookIngestor(BaseIngestor):
    """Transform a generic webhook payload into a RequestEnvelope."""

    def ingest(
        self,
        *,
        body: Dict[str, Any],
        workspace_id: UUID,
    ) -> RequestEnvelope:
        """Build a RequestEnvelope from a webhook POST body.

        Extracts message content from common webhook payload formats:
        - body.message / body.text / body.content  -> content string
        - Falls back to JSON stringification of the body
        - body.source / body.channel -> stored in metadata for routing rules
        - body.agent_id -> optional Tier-0 override
        """
        # --- Extract content text ------------------------------------------------
        # Supports: plain fields, Telegram, Slack, WhatsApp/Twilio, generic
        content: Optional[str] = None

        # 1. Direct string fields (simple webhooks, curl)
        for key in ("message", "text", "content", "body"):
            val = body.get(key)
            if isinstance(val, str) and val.strip():
                content = val.strip()
                break

        # 2. Telegram: { "message": { "text": "hello" } }
        if content is None:
            tg_msg = body.get("message")
            if isinstance(tg_msg, dict):
                content = tg_msg.get("text") or tg_msg.get("caption")

        # 3. Slack: { "event": { "text": "hello" } }
        if content is None:
            slack_event = body.get("event")
            if isinstance(slack_event, dict):
                content = slack_event.get("text")

        # 4. WhatsApp/Twilio: { "Body": "hello" } or { "entry": [{ "changes": [...] }] }
        if content is None:
            content = body.get("Body")  # Twilio SMS/WhatsApp
        if content is None:
            entries = body.get("entry")
            if isinstance(entries, list) and entries:
                changes = entries[0].get("changes", [])
                if changes:
                    value = changes[0].get("value", {})
                    messages = value.get("messages", [])
                    if messages:
                        content = messages[0].get("text", {}).get("body")

        # 5. Fallback: stringify the full body
        if not content:
            content = json.dumps(body, default=str)

        content = str(content)

        # --- Optional Tier-0 override -------------------------------------------
        override_agent_id: Optional[int] = None
        raw_agent_id = body.get("agent_id")
        if raw_agent_id is not None:
            try:
                override_agent_id = int(raw_agent_id)
            except (ValueError, TypeError):
                pass

        # --- Metadata for routing rules ------------------------------------------
        metadata: Dict[str, Any] = {}
        for key in ("source", "channel", "event_type", "service"):
            if key in body:
                metadata[key] = body[key]

        # --- Build envelope ------------------------------------------------------
        return RequestEnvelope(
            source=ChannelSource.WEBHOOK,
            content=content,
            raw_payload=body,
            user=RequestUser(auth_type="webhook"),
            workspace_id=workspace_id,
            metadata=metadata,
            override_agent_id=override_agent_id,
        )
