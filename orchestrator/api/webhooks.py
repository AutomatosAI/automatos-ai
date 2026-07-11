"""
General Webhook Endpoints
=========================

Two webhook paths:
1. POST /api/webhooks/ws/{workspace_key}  — General workspace webhook
   Routes incoming requests through UniversalRouter to the right agent.
   The workspace_key in the URL is the credential floor; when a webhook
   secret (or Slack signing secret) is configured, a valid signature is
   additionally mandatory.

2. POST /api/webhooks/recipe/{webhook_id} — Recipe-specific webhook
   (Defined in workflow_recipes.py, registered separately.)
"""

import asyncio
import hashlib
import hmac
import logging
import time
from typing import Any, Dict, Optional, Set
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models.workspaces import Workspace
from core.routing.cache import get_routing_cache
from core.routing.engine import UniversalRouter
from core.routing.ingestors.webhook import WebhookIngestor
from services import webhook_dedup
from config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])

# Track background tasks to prevent GC collection
_background_tasks: Set[asyncio.Task] = set()


# =============================================================================
# HMAC-SHA256 Signature Verification
# =============================================================================

async def _verify_webhook_signature(
    request: Request,
    secret: Optional[str],
) -> None:
    """
    Verify HMAC-SHA256 signature on an incoming webhook request.

    Checks headers: X-Hub-Signature-256 (GitHub), X-Composio-Signature,
    X-Webhook-Signature.

    When a secret is configured, a valid signature is mandatory: a missing
    header, a mismatch, or a verification error all reject with 401 (P2-13).
    If no secret is configured, verification is skipped (URL-as-secret
    pattern still applies).
    """
    if not secret:
        return

    # Look for signature in common headers
    sig_header = (
        request.headers.get("x-hub-signature-256")
        or request.headers.get("x-composio-signature")
        or request.headers.get("x-webhook-signature")
    )
    if not sig_header:
        logger.warning("[webhook] Rejected: secret configured but no signature header present")
        raise HTTPException(status_code=401, detail="Missing webhook signature")

    raw_body = await request.body()

    # Strip optional "sha256=" prefix (GitHub format)
    expected_sig = sig_header.removeprefix("sha256=")

    computed = hmac.new(
        secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(computed, expected_sig):
        logger.warning("[webhook] HMAC signature mismatch")
        raise HTTPException(status_code=401, detail="Invalid webhook signature")


# Slack's documented v0 replay window: reject requests whose signing
# timestamp is further than 5 minutes from now.
_SLACK_TS_SKEW_SECONDS = 60 * 5


def _resolve_slack_signing_secret(db: Session, workspace: Workspace) -> Optional[str]:
    """The signing secret collected for this workspace's Slack channel, if any.

    Prefers an active ``channel_connections`` row; falls back to any Slack row
    that carries a secret. Returns ``None`` when Slack was never configured
    with a signing secret (verification is then skipped — URL-as-secret floor).
    """
    from core.models.channels import ChannelConnection

    rows = (
        db.query(ChannelConnection)
        .filter(
            ChannelConnection.workspace_id == workspace.id,
            ChannelConnection.platform == "slack",
        )
        .all()
    )
    fallback: Optional[str] = None
    for row in rows:
        secret = (row.config or {}).get("signing_secret")
        if not secret:
            continue
        if row.status == "active":
            return str(secret)
        fallback = fallback or str(secret)
    return fallback


async def _verify_slack_signature(request: Request, signing_secret: str) -> None:
    """Verify Slack's v0 signing scheme.

    Slack signs ``v0:{timestamp}:{raw_body}`` with the app's signing secret and
    sends ``X-Slack-Signature: v0=<hex>`` + ``X-Slack-Request-Timestamp``.
    Mandatory once a signing secret is collected: bad timestamp, stale
    timestamp, or mismatch ⇒ 401 (P2-13).
    """
    sig_header = request.headers.get("x-slack-signature", "")
    ts = request.headers.get("x-slack-request-timestamp", "")
    raw_body = await request.body()

    try:
        ts_int = int(ts)
    except (TypeError, ValueError):
        logger.warning("[webhook] Slack signature rejected: bad timestamp header")
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    if abs(time.time() - ts_int) > _SLACK_TS_SKEW_SECONDS:
        logger.warning("[webhook] Slack signature rejected: stale timestamp")
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    base = f"v0:{ts}:".encode("utf-8") + raw_body
    computed = "v0=" + hmac.new(signing_secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(computed, sig_header):
        logger.warning("[webhook] Slack signature mismatch")
        raise HTTPException(status_code=401, detail="Invalid webhook signature")


# =============================================================================
# Platform Detection
# =============================================================================

def _detect_platform(body: Dict[str, Any]) -> Optional[str]:
    """Detect which messaging platform sent this webhook payload."""
    # Telegram: has update_id or message.chat.id
    if "update_id" in body:
        return "telegram"
    msg = body.get("message")
    if isinstance(msg, dict) and "chat" in msg:
        return "telegram"

    # Slack: has team_id or event.type
    if "team_id" in body or "api_app_id" in body:
        return "slack"
    if isinstance(body.get("event"), dict) and "type" in body["event"]:
        return "slack"

    # WhatsApp (Meta): has entry[].changes[].value.messages
    entries = body.get("entry")
    if isinstance(entries, list) and entries:
        changes = entries[0].get("changes", [])
        if changes and "value" in changes[0]:
            return "whatsapp"

    # Twilio SMS/WhatsApp: has Body + From + To
    if "Body" in body and "From" in body:
        return "twilio"

    return None


def _extract_reply_context(body: Dict[str, Any], platform: str) -> Dict[str, Any]:
    """Extract the info needed to reply back to the originating chat."""
    ctx: Dict[str, Any] = {"platform": platform}

    if platform == "telegram":
        msg = body.get("message", {})
        ctx["chat_id"] = msg.get("chat", {}).get("id")
        ctx["from_user"] = msg.get("from", {}).get("first_name", "")

    elif platform == "slack":
        event = body.get("event", {})
        ctx["channel"] = event.get("channel")
        ctx["thread_ts"] = event.get("thread_ts") or event.get("ts")
        ctx["user"] = event.get("user")

    elif platform == "whatsapp":
        entries = body.get("entry", [{}])
        changes = entries[0].get("changes", [{}])
        value = changes[0].get("value", {})
        messages = value.get("messages", [{}])
        ctx["from_phone"] = messages[0].get("from") if messages else None
        ctx["phone_number_id"] = value.get("metadata", {}).get("phone_number_id")

    elif platform == "twilio":
        ctx["from_phone"] = body.get("From")
        ctx["to_phone"] = body.get("To")

    return ctx


# =============================================================================
# Platform Reply — single entry point
#
# All outbound platform messages flow through ``channels.sender.send_to_channel``
# which delegates to the per-platform driver. The driver reads creds from
# ``channel_connections`` (with a fallback to the legacy
# ``workspace.settings.integrations`` bag for not-yet-migrated workspaces).
#
# The legacy ``_send_telegram_reply`` / ``_send_slack_reply`` /
# ``_send_whatsapp_reply`` helpers that used to live here have been removed —
# nothing should import them. ``notification_service.send_workspace_notification``
# now calls the sender directly.
# =============================================================================


async def _deliver_reply(
    reply_text: str,
    reply_ctx: Dict[str, Any],
    integrations: Dict[str, str],
    *,
    workspace_id: Optional[Any] = None,
) -> bool:
    """Deliver agent response back to the originating platform via the
    unified ``channels.sender`` — which routes through the per-platform
    driver and reads creds from ``channel_connections`` (with legacy
    ``integrations`` fallback for pre-migration workspaces).

    Opens its own DB session so it's safe to fire from a background
    task after the request session has been closed. ``integrations``
    is accepted for backward compat but unused — the sender reads
    everything it needs from the DB.
    """
    platform = reply_ctx.get("platform")
    if not platform:
        logger.debug("[reply] No platform in reply_ctx")
        return False
    if workspace_id is None:
        logger.warning("[reply] _deliver_reply: workspace_id is required")
        return False

    target = (
        reply_ctx.get("chat_id")
        or reply_ctx.get("channel")
        or reply_ctx.get("from_phone")
    )
    if target is not None:
        target = str(target)

    from channels.sender import send_to_channel
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        result = await send_to_channel(
            db=db,
            workspace_id=workspace_id,
            platform=platform,
            text=reply_text,
            target=target,
        )
    finally:
        try:
            db.close()
        except Exception:
            pass

    if not result.ok:
        logger.warning(
            "[reply] platform=%s send failed: %s", platform, result.error,
        )
    return result.ok


# =============================================================================
# Extract agent response text from result dict
# =============================================================================

def _extract_response_text(result: Any) -> str:
    """Extract a human-readable text response from the agent result."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        # Common patterns from AgentFactory
        for key in ("response", "output", "result", "message", "text", "answer"):
            if key in result and isinstance(result[key], str):
                return result[key]
        # Nested under "data"
        data = result.get("data", {})
        if isinstance(data, dict):
            for key in ("response", "output", "result"):
                if key in data and isinstance(data[key], str):
                    return data[key]
    return str(result)[:2000]


# =============================================================================
# Integration defaults persistence
# =============================================================================

def _persist_integration_default(db: Session, workspace, key: str, value: str):
    """Store a platform default (e.g. telegram_default_chat_id) in workspace
    settings.integrations if not already set or changed."""
    try:
        settings = dict(workspace.settings or {})
        integrations = dict(settings.get("integrations", {}))
        if integrations.get(key) == value:
            return  # already correct
        integrations[key] = value
        settings["integrations"] = integrations
        workspace.settings = settings
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(workspace, "settings")
        db.commit()
        logger.info("[webhook] Persisted %s=%s for workspace %s", key, value, workspace.id)
    except Exception as e:
        logger.debug("[webhook] Failed to persist %s: %s", key, e)


# =============================================================================
# Webhook Endpoints
# =============================================================================

@router.get("/ws/{workspace_key}")
async def workspace_webhook_verify(
    workspace_key: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """Verification endpoint — handles GET validation from external services.

    Supports:
    - Meta/WhatsApp webhook verification (hub.mode, hub.verify_token, hub.challenge)
    - Generic URL validation (Jira, GitHub, etc.)
    """
    workspace = db.query(Workspace).filter(
        Workspace.webhook_key == workspace_key,
        Workspace.is_active == True,
    ).first()

    if not workspace:
        raise HTTPException(status_code=404, detail="Unknown webhook")

    # Meta/WhatsApp webhook verification
    params = request.query_params
    hub_mode = params.get("hub.mode")
    hub_challenge = params.get("hub.challenge")
    hub_verify_token = params.get("hub.verify_token")

    if hub_mode == "subscribe" and hub_challenge:
        # Meta sends a verify_token — we accept any token since the URL itself is the secret
        # (The workspace_key in the URL is the credential)
        logger.info("[webhook/ws] Meta webhook verification for workspace %s", workspace.id)
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(hub_challenge)

    return {"status": "ok"}


@router.post("/ws/{workspace_key}")
async def general_workspace_webhook(
    workspace_key: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    General workspace webhook — routes incoming requests to the right agent.

    The workspace_key in the URL is the credential floor (URL-as-secret
    pattern). When a webhook secret is configured — or a Slack signing
    secret has been collected — a valid signature is mandatory on top.

    Body (JSON):
    - message / text / content: The message to route
    - agent_id: Optional explicit agent override (Tier-0)
    - source / channel: Optional metadata for routing rules
    - Any other JSON fields are preserved in metadata

    Platform auto-detection:
    - Telegram, Slack, WhatsApp payloads are auto-detected
    - If a bot token is configured in workspace settings, the agent's
      response is delivered back to the originating chat
    """
    import json as _json

    # Parse body from any content type
    content_type = request.headers.get("content-type", "")
    try:
        if "application/json" in content_type:
            body = await request.json()
        elif "form" in content_type:
            form = await request.form()
            payload_str = form.get("payload", "{}")
            body = _json.loads(payload_str) if isinstance(payload_str, str) else {}
        else:
            raw = await request.body()
            try:
                body = _json.loads(raw) if raw else {}
            except (ValueError, _json.JSONDecodeError):
                body = {"raw": raw.decode("utf-8", errors="replace")}
    except Exception:
        body = {}

    # Slack url_verification challenge — echo back immediately
    if isinstance(body, dict) and body.get("type") == "url_verification":
        return {"challenge": body.get("challenge", "")}

    # 1. Look up workspace by webhook_key
    workspace = db.query(Workspace).filter(
        Workspace.webhook_key == workspace_key,
        Workspace.is_active == True,
    ).first()

    if not workspace:
        raise HTTPException(status_code=404, detail="Unknown webhook")

    # 1b. Verify inbound authenticity. Slack signs with its own v0 scheme
    # (X-Slack-Signature), so when a Slack signing secret has been collected
    # the request verifies against it. Every other request goes through the
    # generic HMAC, which is mandatory once a webhook secret is configured —
    # including requests that *carry* an x-slack-signature header when no
    # Slack secret was ever collected. Branching on the header alone would
    # let any caller bypass the mandatory generic check by adding a garbage
    # Slack header (P2-13, fail closed).
    slack_secret = (
        _resolve_slack_signing_secret(db, workspace)
        if request.headers.get("x-slack-signature")
        else None
    )
    if slack_secret:
        await _verify_slack_signature(request, slack_secret)
    else:
        webhook_secret = (workspace.settings or {}).get("webhook_secret") or config.WEBHOOK_SECRET
        await _verify_webhook_signature(request, webhook_secret)

    # 1c. Replay guard + event dedup (PRD-194 S2, P2-13). This lane executes
    # the agent synchronously inside the HTTP request; a slow run triggers
    # provider redelivery (Slack/Telegram retry on slow ack) and, until this
    # guard, the SAME event ran again — burning tokens and re-firing
    # side-effects. Keyed per-workspace on the platform's own event id
    # (Telegram update_id / Slack event_id / webhook-id header); a
    # redelivery is a fast no-op ack before any routing. The Shopify
    # /events debounce (PRD-189 S3) is a different endpoint — untouched.
    if webhook_dedup.timestamp_is_stale(request.headers.get("webhook-timestamp")):
        logger.warning("[webhook/ws] Rejected: stale webhook-timestamp (replay guard)")
        raise HTTPException(status_code=401, detail="Stale webhook timestamp")
    dedup_event_id = None
    if isinstance(body, dict):
        dedup_event_id = body.get("update_id") or body.get("event_id")
    dedup_event_id = dedup_event_id or request.headers.get("webhook-id")
    if await webhook_dedup.seen_before(f"ws:{workspace.id}", dedup_event_id):
        logger.info(
            "[webhook/ws] Duplicate event %s for workspace %s — no-op ack",
            dedup_event_id, workspace.id,
        )
        return {"status": "duplicate_ignored", "event_id": str(dedup_event_id)}

    # 2. Detect platform and extract reply context
    platform = _detect_platform(body) if isinstance(body, dict) else None
    reply_ctx = _extract_reply_context(body, platform) if platform else {}
    integrations = (workspace.settings or {}).get("integrations", {})

    logger.info(
        "[webhook/ws] workspace=%s platform=%s has_integrations=%s",
        workspace.id, platform, bool(integrations),
    )

    # 2b. Persist platform-specific IDs for reuse (heartbeat, notifications)
    if platform == "telegram" and reply_ctx.get("chat_id"):
        _persist_integration_default(db, workspace, "telegram_default_chat_id", str(reply_ctx["chat_id"]))
    elif platform == "slack" and reply_ctx.get("channel"):
        _persist_integration_default(db, workspace, "slack_default_channel", reply_ctx["channel"])

    # 3. Build RequestEnvelope via WebhookIngestor
    ingestor = WebhookIngestor()
    envelope = ingestor.ingest(
        body=body,
        workspace_id=workspace.id,
    )

    # 3b. Platform tool interception — if the message matches a platform keyword,
    # route to Auto (CTO agent) which has all platform tools, instead of going
    # through UniversalRouter. This lets Telegram/Slack users trigger missions,
    # create tasks, check stats, etc. just like they can from the chat UI.
    try:
        from consumers.chatbot.auto import AutoBrain
        platform_tool = AutoBrain._match_platform_query(envelope.content.lower())
        if platform_tool:
            logger.info(
                "[webhook/ws] Platform tool detected: %s — routing to Auto",
                platform_tool,
            )
            # Find the CTO/Auto agent
            from core.models.core import Agent as AgentModel
            auto_agent = db.query(AgentModel).filter(
                AgentModel.slug == "auto-cto",
                AgentModel.is_system_agent.is_(True),
                AgentModel.status == "active",
            ).first()
            if auto_agent:
                result = await _execute_agent_sync(
                    agent_id=auto_agent.id,
                    content=envelope.content,
                    metadata=envelope.metadata,
                    workspace_id=workspace.id,
                )
                if platform and integrations:
                    reply_text = _extract_response_text(result)
                    task = asyncio.create_task(
                        _deliver_reply(reply_text, reply_ctx, integrations, workspace_id=workspace.id)
                    )
                    _background_tasks.add(task)
                    task.add_done_callback(_background_tasks.discard)

                return {
                    "status": "completed",
                    "routed": True,
                    "route_type": "platform_tool",
                    "tool": platform_tool,
                    "platform": platform,
                    "reply_delivered": platform is not None and bool(integrations),
                    "result": result,
                }
    except Exception:
        logger.debug("[webhook/ws] Platform tool interception failed, continuing to router", exc_info=True)

    # 4. Route through UniversalRouter
    universal_router = UniversalRouter(db, cache=get_routing_cache())
    try:
        decision = await universal_router.route(envelope)
    except Exception:
        logger.exception("[webhook/ws] Router failed for workspace %s", workspace.id)
        decision = None

    if decision is None:
        # Try to let the platform know there's no route configured
        if platform and integrations:
            task = asyncio.create_task(
                _deliver_reply(
                    "I received your message but no routing rules are configured yet. "
                    "Please set up agents and routing in the Automatos dashboard.",
                    reply_ctx,
                    integrations,
                    workspace_id=workspace.id,
                )
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

        return {
            "status": "received",
            "routed": False,
            "reason": "No route found — configure routing rules or add agents to your workspace.",
        }

    # 5. Dispatch based on routing decision
    if decision.route_type == "agent" and decision.agent_id is not None:
        # Execute agent synchronously for webhook callers who want a response
        try:
            result = await _execute_agent_sync(
                agent_id=decision.agent_id,
                content=envelope.content,
                metadata=envelope.metadata,
                workspace_id=workspace.id,
            )

            # Deliver reply back to platform (async, don't block response)
            if platform and integrations:
                reply_text = _extract_response_text(result)
                task = asyncio.create_task(
                    _deliver_reply(reply_text, reply_ctx, integrations, workspace_id=workspace.id)
                )
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)

            return {
                "status": "completed",
                "routed": True,
                "route_type": "agent",
                "agent_id": decision.agent_id,
                "confidence": decision.confidence,
                "platform": platform,
                "reply_delivered": platform is not None and bool(integrations),
                "result": result,
            }
        except Exception:
            logger.exception("[webhook/ws] Agent %d execution failed", decision.agent_id)

            if platform and integrations:
                task = asyncio.create_task(
                    _deliver_reply(
                        "Sorry, I encountered an error processing your request. Please try again.",
                        reply_ctx,
                        integrations,
                        workspace_id=workspace.id,
                    )
                )
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)

            return {
                "status": "error",
                "routed": True,
                "route_type": "agent",
                "agent_id": decision.agent_id,
                "error": "Agent execution failed",
            }

    elif decision.route_type == "workflow" and decision.workflow_id is not None:
        # Dispatch workflow/recipe async, return execution_id
        execution_id = await _dispatch_workflow_async(
            workflow_id=decision.workflow_id,
            envelope=envelope,
            db=db,
        )

        # Let user know the workflow is running
        if platform and integrations:
            task = asyncio.create_task(
                _deliver_reply(
                    f"Your request has been dispatched to a workflow (ID: {execution_id}). "
                    "I'll process it in the background.",
                    reply_ctx,
                    integrations,
                    workspace_id=workspace.id,
                )
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

        return {
            "status": "dispatched",
            "routed": True,
            "route_type": "workflow",
            "workflow_id": decision.workflow_id,
            "execution_id": execution_id,
            "confidence": decision.confidence,
        }

    # Orchestrate / unknown route_type — find default agent and execute
    if decision.route_type == "orchestrate":
        from api.chat import get_default_agent_id

        default_agent_id = get_default_agent_id(db, workspace.id)
        logger.info(
            "[webhook/ws] orchestrate → default agent %s for workspace %s",
            default_agent_id, workspace.id,
        )
        try:
            result = await _execute_agent_sync(
                agent_id=default_agent_id,
                content=envelope.content,
                metadata=envelope.metadata,
                workspace_id=workspace.id,
            )

            if platform and integrations:
                reply_text = _extract_response_text(result)
                task = asyncio.create_task(
                    _deliver_reply(reply_text, reply_ctx, integrations, workspace_id=workspace.id)
                )
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)

            return {
                "status": "completed",
                "routed": True,
                "route_type": "orchestrate",
                "agent_id": default_agent_id,
                "confidence": decision.confidence,
                "platform": platform,
                "reply_delivered": platform is not None and bool(integrations),
                "result": result,
            }
        except Exception:
            logger.exception(
                "[webhook/ws] Orchestrate execution failed (agent=%s)", default_agent_id
            )

            if platform and integrations:
                task = asyncio.create_task(
                    _deliver_reply(
                        "Sorry, I encountered an error processing your request. Please try again.",
                        reply_ctx,
                        integrations,
                        workspace_id=workspace.id,
                    )
                )
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)

            return {
                "status": "error",
                "routed": True,
                "route_type": "orchestrate",
                "agent_id": default_agent_id,
                "error": "Orchestrate execution failed",
            }

    # Truly unknown route_type — acknowledge only
    return {
        "status": "received",
        "routed": True,
        "route_type": decision.route_type,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning[:200] if decision.reasoning else "",
    }


# =============================================================================
# Dispatch Helpers
# =============================================================================

async def _execute_agent_sync(
    agent_id: int,
    content: str,
    metadata: Dict[str, Any],
    workspace_id: UUID,
) -> Dict[str, Any]:
    """Execute an agent synchronously and return the result."""
    from modules.agents.factory.agent_factory import AgentFactory

    db = next(get_db())
    try:
        factory = AgentFactory(db_session=db)
        result = await factory.execute_with_prompt(
            agent=agent_id,
            prompt=content,
            context=metadata,
        )
        return result
    finally:
        db.close()


async def _dispatch_workflow_async(
    workflow_id: int,
    envelope,
    db: Session,
) -> str:
    """Dispatch a workflow/recipe execution asynchronously, return execution_id."""
    from core.models.core import RecipeExecution
    from core.models import WorkflowTemplate as WorkflowRecipe
    # PRD-142 W3-S12: workspace webhook dispatch goes via the engine seam.
    from services.playbook_engine import get_playbook_engine
    from datetime import datetime, timezone

    recipe = db.query(WorkflowRecipe).filter(
        WorkflowRecipe.id == workflow_id
    ).first()

    if not recipe or not recipe.steps:
        return "no_recipe_found"

    execution_id = f"ws-webhook-{uuid4().hex[:12]}"
    execution = RecipeExecution(
        execution_id=execution_id,
        recipe_id=recipe.id,
        workspace_id=envelope.workspace_id,
        status="pending",
        input_data={"content": envelope.content, "metadata": envelope.metadata},
        triggered_by="workspace_webhook",
        execution_metadata={
            "execution_type": "workspace_webhook",
            "total_steps": len(recipe.steps),
        },
    )
    db.add(execution)
    recipe.use_count += 1
    recipe.last_used_at = datetime.now(timezone.utc)
    db.commit()

    task = asyncio.create_task(
        get_playbook_engine().execute_direct(
            recipe_execution_id=execution_id,
            recipe_id=recipe.id,
            workspace_id=envelope.workspace_id,
            input_data=execution.input_data,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return execution_id
