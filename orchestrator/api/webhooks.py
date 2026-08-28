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
import re
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
        # P225-RVW-1: the STABLE numeric sender id (Telegram-assigned, not the
        # self-chosen first_name) — used for answer attribution, never for auth.
        ctx["from_id"] = msg.get("from", {}).get("id")
        # PRD-225: reply correlation — the id of THIS message and the id of the
        # message it replies to (a reply to a correlated question answers it).
        ctx["message_id"] = msg.get("message_id")
        ctx["reply_to_message_id"] = (msg.get("reply_to_message") or {}).get("message_id")

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
# PRD-225 US-005 — Telegram answer bridge (reply / /answer correlation)
# =============================================================================

_ANSWER_CMD = re.compile(r"^/answer\s+(\d+)\s+(.+)$", re.IGNORECASE | re.DOTALL)


def _pending_questions(db: Session, workspace_id: Any) -> list:
    """The workspace's open (pending) question-kind grants."""
    from core.models.approval_grants import ApprovalGrant, GrantStatus, KIND_QUESTION

    return (
        db.query(ApprovalGrant)
        .filter(
            ApprovalGrant.workspace_id == workspace_id,
            ApprovalGrant.kind == KIND_QUESTION,
            ApprovalGrant.status == GrantStatus.PENDING.value,
        )
        .all()
    )


def _find_pending_question(db: Session, workspace_id: Any, ask_id: int):
    """A pending question by id, scoped to the workspace. None if absent —
    wrong-workspace / already-answered targets simply aren't found (no leak)."""
    for g in _pending_questions(db, workspace_id):
        if g.id == ask_id:
            return g
    return None


def _telegram_ref(grant: Any) -> Optional[Dict[str, Any]]:
    """The grant's stored Telegram delivery ref ``{chat_id, message_id}`` (the
    chat the question was delivered to), or None. That chat is the ONLY one
    authorized to answer the question (P225-RVW-1)."""
    refs = grant.channel_refs if isinstance(grant.channel_refs, dict) else {}
    tg = refs.get("telegram") if isinstance(refs, dict) else None
    return tg if isinstance(tg, dict) else None


def _find_answerable_question(
    db: Session, workspace_id: Any, ask_id: int, reply_chat_id: Any,
):
    """A pending question the replying chat is AUTHORIZED to answer by id.

    Telegram routes EVERY bot update to the one workspace webhook, so scoping the
    lookup to (workspace, id) alone lets any reachable chat inject an answer
    against the global auto-increment grant id. Authorization additionally binds
    the sending chat to the question's stored delivery chat
    (``channel_refs.telegram.chat_id``) (P225-RVW-1).

    Every failure — wrong id, wrong-workspace id, already-answered, no delivery
    ref, or an unauthorized chat — returns None so the caller emits ONE identical
    'isn't open' reply (no existence leak)."""
    grant = _find_pending_question(db, workspace_id, ask_id)
    if grant is None:
        return None
    tg = _telegram_ref(grant)
    if tg is None or str(tg.get("chat_id")) != str(reply_chat_id):
        return None
    return grant


def _find_question_by_telegram_message(
    db: Session, workspace_id: Any, message_id: str, reply_chat_id: Any,
):
    """The pending question whose stored Telegram delivery matches BOTH the target
    ``message_id`` AND the replying ``chat_id``. Binding the chat stops a
    same-workspace but unauthorized chat from reply-colliding on a per-chat
    sequential ``reply_to_message_id`` (P225-RVW-1). Python-side filter over the
    few open asks (JSONB-portable)."""
    for g in _pending_questions(db, workspace_id):
        tg = _telegram_ref(g)
        if (
            tg is not None
            and str(tg.get("message_id")) == str(message_id)
            and str(tg.get("chat_id")) == str(reply_chat_id)
        ):
            return g
    return None


async def _maybe_answer_question(
    db: Session,
    workspace: Any,
    body: Dict[str, Any],
    reply_ctx: Dict[str, Any],
    integrations: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """If this inbound Telegram message answers a pending question — a reply to a
    correlated message, or ``/answer <id> <text>`` — apply it through the SHARED
    answer service (never an HTTP self-call) and confirm into the same thread.

    Returns a response dict when handled (short-circuits routing AND the trust
    gate — an answer is a response, not a directive), or None to fall through.
    A wrong-workspace / already-answered target gets a safe reply and changes
    nothing (no information leak).
    """
    if reply_ctx.get("platform") != "telegram":
        return None
    msg = body.get("message") if isinstance(body, dict) else None
    text = (msg.get("text") or "").strip() if isinstance(msg, dict) else ""
    if not text:
        return None

    # 1. Explicit fallback: /answer <id> <text> — applied ONLY from the chat the
    #    question was delivered to (P225-RVW-1: the id is a global auto-increment,
    #    so id-scope alone lets any reachable chat inject an answer).
    m = _ANSWER_CMD.match(text)
    if m:
        ask_id = int(m.group(1))
        answer_text = m.group(2).strip()
        grant = _find_answerable_question(db, workspace.id, ask_id, reply_ctx.get("chat_id"))
        if grant is None:
            await _deliver_reply(
                f"Question #{ask_id} isn't open in this workspace.",
                reply_ctx, integrations, workspace_id=workspace.id,
            )
            return {"status": "received", "routed": False, "reason": "answer_target_not_found"}
        return await _apply_telegram_answer(db, workspace, grant, answer_text, reply_ctx, integrations)

    # 2. A reply to a correlated question message answers it — the reply must come
    #    from the delivery chat too (P225-RVW-1), not merely target the message id.
    reply_to = reply_ctx.get("reply_to_message_id")
    if reply_to is not None:
        grant = _find_question_by_telegram_message(
            db, workspace.id, str(reply_to), reply_ctx.get("chat_id"),
        )
        if grant is not None:
            return await _apply_telegram_answer(db, workspace, grant, text, reply_ctx, integrations)

    return None


async def _apply_telegram_answer(
    db: Session,
    workspace: Any,
    grant: Any,
    answer_text: str,
    reply_ctx: Dict[str, Any],
    integrations: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply a Telegram-sourced answer via the shared service, then confirm back
    into the same thread through the existing ``_deliver_reply`` path."""
    from api.approval_grants import apply_question_answer

    # P225-RVW-1 AC3: attribute to the STABLE numeric telegram id, never the
    # self-chosen first_name (which is trivially spoofable, e.g. 'telegram:CEO').
    answered_by = f"telegram:{reply_ctx.get('from_id') or reply_ctx.get('chat_id') or 'user'}"
    await apply_question_answer(db, grant, answer_text=answer_text, answered_by=answered_by)
    await _deliver_reply(
        f"Answered #{grant.id} — the agent is resuming.",
        reply_ctx, integrations, workspace_id=workspace.id,
    )
    return {
        "status": "completed",
        "routed": True,
        "route_type": "question_answer",
        "ask_id": grant.id,
    }


# =============================================================================
# PRD-225 US-006 — the per-channel ingress trust gate
# =============================================================================

def _channel_for_platform(db: Session, workspace_id: Any, platform: str):
    """The workspace's channel connection for this platform (None if none)."""
    from core.models.channels import ChannelConnection

    return (
        db.query(ChannelConnection)
        .filter(
            ChannelConnection.workspace_id == workspace_id,
            ChannelConnection.platform == platform,
        )
        .first()
    )


def _telegram_message_content(m: Any) -> str:
    """Operator-visible text carried by a Telegram message / edited_message.

    Beyond ``text``/``caption``, a media or service message carries
    attacker-controllable text in a sub-object: a document/audio/video/voice
    ``file_name``, a ``poll`` question, a ``contact`` name, a ``venue``
    title/address, a ``sticker`` emoji. The ``WebhookIngestor`` has no explicit
    branch for these, so it serialises the whole update via ``json.dumps`` — and
    AutoBrain's UNANCHORED platform-keyword regex (consumers/chatbot/auto.py
    ``_match_platform_query``) then matches a platform keyword ANYWHERE in that
    blob, e.g. a caption-less document named "run the recipe.pdf" triggering
    ``platform_execute_recipe`` (P225-RVW-9). Extracting the same subfields here
    lets the gate score — and hold — them under a strict / communication_only
    channel, before the ingestor or the platform-tool interception ever runs.
    """
    if not isinstance(m, dict):
        return ""

    v = m.get("text") or m.get("caption")
    if isinstance(v, str) and v.strip():
        return v

    poll = m.get("poll")
    if isinstance(poll, dict) and isinstance(poll.get("question"), str) and poll["question"].strip():
        return poll["question"]

    contact = m.get("contact")
    if isinstance(contact, dict):
        name = " ".join(
            p.strip() for p in (contact.get("first_name"), contact.get("last_name"))
            if isinstance(p, str) and p.strip()
        )
        if name:
            return name

    venue = m.get("venue")
    if isinstance(venue, dict):
        place = " ".join(
            p.strip() for p in (venue.get("title"), venue.get("address"))
            if isinstance(p, str) and p.strip()
        )
        if place:
            return place

    for fkey in ("document", "audio", "video", "voice"):
        f = m.get(fkey)
        if isinstance(f, dict) and isinstance(f.get("file_name"), str) and f["file_name"].strip():
            return f["file_name"]

    sticker = m.get("sticker")
    if isinstance(sticker, dict) and isinstance(sticker.get("emoji"), str) and sticker["emoji"].strip():
        return sticker["emoji"]

    return ""


def _inbound_text(body: Dict[str, Any]) -> str:
    """The inbound message text the router would act on, across platforms.

    MUST stay aligned with ``WebhookIngestor`` content extraction
    (core/routing/ingestors/webhook.py): the gate scores exactly the text the
    router would route, so a directive can NEVER be scored empty here yet reach
    the router as content (P225-RVW-2). Covers Telegram text+caption including
    ``edited_message`` AND the text-bearing subfield of a media / service message
    (file_name, poll question, contact, venue, sticker — P225-RVW-9), Slack
    ``event.text``, Meta-WhatsApp ``messages[].text.body``, Twilio ``Body``, and
    top-level string fields.

    The ingestor's ``json.dumps(body)`` blanket fallback is deliberately NOT
    mirrored: a genuinely user-contentless update (a status / delivery-receipt
    callback with none of the above subfields) scores empty and is left to route
    as today, not held as a question.
    """
    if not isinstance(body, dict):
        return ""

    # 1. Direct string fields (simple webhooks / curl) — ingestor step 1.
    for key in ("message", "text", "content", "body"):
        v = body.get(key)
        if isinstance(v, str) and v.strip():
            return v

    # 2. Telegram message / edited_message content: text, caption, OR the
    #    text-bearing subfield of a media / service message (file_name, poll
    #    question, contact name, venue, sticker emoji). The ingestor has no branch
    #    for the media subfields, so it json.dumps the whole update and AutoBrain's
    #    unanchored keyword regex matches a platform keyword anywhere in it
    #    (P225-RVW-9) — extract them so a strict/communication_only channel holds.
    for mkey in ("message", "edited_message"):
        content = _telegram_message_content(body.get(mkey))
        if content.strip():
            return content

    # 3. Slack: event.text — ingestor step 3.
    event = body.get("event")
    if isinstance(event, dict) and isinstance(event.get("text"), str):
        return event["text"]

    # 4. Twilio `Body`, then Meta-WhatsApp entry[].changes[].value.messages[].text.body.
    twilio = body.get("Body")
    if isinstance(twilio, str) and twilio.strip():
        return twilio
    entries = body.get("entry")
    if isinstance(entries, list) and entries and isinstance(entries[0], dict):
        changes = entries[0].get("changes", [])
        if changes and isinstance(changes[0], dict):
            value = changes[0].get("value", {})
            messages = value.get("messages", []) if isinstance(value, dict) else []
            if messages and isinstance(messages[0], dict):
                text_obj = messages[0].get("text", {})
                if isinstance(text_obj, dict) and isinstance(text_obj.get("body"), str):
                    return text_obj["body"]

    return ""


def _fence_untrusted(text: str) -> str:
    """Wrap untrusted inbound text as an inert fenced code block (P225-RVW-6).

    Channel-sourced directive text is shown in the admin Questions tab, which
    renders ``question_md`` as GFM markdown. Interpolated raw, an attacker's
    ``[click me](https://evil.example)`` — or a bare URL that GFM autolinks —
    becomes a CLICKABLE anchor next to copy priming the operator to act: a
    phishing vector inside a trusted surface. Inside a code fence nothing renders
    as markdown (no links, no autolinks, no emphasis) and the operator still
    reads the literal text. The fence is one backtick longer than the longest
    backtick run in the body, so the content cannot break out of the fence.
    """
    body = text.strip()
    longest = run = 0
    for ch in body:
        if ch == "`":
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{body}\n{fence}"


async def _apply_trust_gate(
    db: Session,
    workspace: Any,
    platform: str,
    body: Dict[str, Any],
    reply_ctx: Dict[str, Any],
    integrations: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Hold inbound directives per the channel's ``trigger_mode``.

    Returns a response dict when the message is HELD (short-circuits ALL routing),
    or None to let it route. No channel row ⇒ no gate (legacy-integration inbound
    is unchanged). Correlated answers already returned upstream, so they bypass
    the gate in every mode.
    """
    from services.ingress_gate import (
        should_hold, trigger_mode_of, TRIGGER_MODE_ALLOW_ALL,
    )

    channel = _channel_for_platform(db, workspace.id, platform)
    if channel is None:
        return None  # no connected channel → the gate does not apply

    # A channel exists, so this inbound IS gated. From here ANY failure must
    # fail CLOSED — return a hold, never fall through to routing — so an internal
    # gate error can't silently open a strict / communication_only channel
    # (P225-RVW-5). ``allow_all`` is resolved first: it legitimately routes
    # everything, so an error on that path is a harmless no-op, not a bypass.
    ask_id = None
    try:
        mode = trigger_mode_of(channel.config)
        if mode == TRIGGER_MODE_ALLOW_ALL:
            return None

        text_in = _inbound_text(body)
        if not text_in.strip():
            return None  # nothing to hold (non-text update)
        if not should_hold(mode, text_in):
            # NOTE: never log the message body — only the gate decision.
            logger.info(
                "[trust-gate] ws=%s channel=%s platform=%s mode=%s verdict=route",
                workspace.id, channel.id, platform, mode,
            )
            return None

        # HELD — record a question-kind row against the channel; nothing executes.
        from core.models.approval_grants import KIND_QUESTION
        from core.services.approval_grants import create_grant

        grant = create_grant(
            db, workspace.id,
            subject_type="channel", subject_id=str(channel.id),
            kind=KIND_QUESTION,
            question_md=(
                "**Inbound directive awaiting approval**\n\n"
                f"{_fence_untrusted(text_in)}\n\n"
                '_Answer "route it" to let it proceed, or dismiss to keep it held._'
            ),
            reason="Inbound directive held by the channel trust gate",
        )
        db.commit()
        ask_id = grant.id
        logger.info(
            "[trust-gate] ws=%s channel=%s platform=%s mode=%s verdict=hold ask=%s",
            workspace.id, channel.id, platform, mode, ask_id,
        )
    except Exception:
        # Fail CLOSED: a gated channel errored — refuse to route. No message body
        # in the log (only the gate decision), consistent with the hold path.
        logger.error(
            "[trust-gate] ws=%s channel=%s platform=%s verdict=hold-on-error — failing closed",
            workspace.id, channel.id, platform, exc_info=True,
        )

    try:
        await _deliver_reply(
            "Received — this needs an operator's approval before I act on it.",
            reply_ctx, integrations, workspace_id=workspace.id,
        )
    except Exception:  # noqa: BLE001 — the ack is best-effort
        logger.debug("[trust-gate] ack reply failed", exc_info=True)
    return {
        "status": "held", "routed": False,
        "reason": "trust_gate_hold" if ask_id is not None else "trust_gate_error",
        "ask_id": ask_id,
    }


def _channel_is_allow_all(db: Session, workspace_id: Any, platform: str) -> bool:
    """True only when the channel is provably ``allow_all``.

    Any lookup failure — or no channel row — returns False, so a trust-gate error
    fails CLOSED (P225-RVW-5): the outer handler holds rather than silently
    opening a strict / communication_only channel. ``allow_all`` routes
    everything anyway, so this is the one mode where a gate error is a safe no-op.
    """
    try:
        from services.ingress_gate import trigger_mode_of, TRIGGER_MODE_ALLOW_ALL

        channel = _channel_for_platform(db, workspace_id, platform)
        if channel is None:
            return False
        return trigger_mode_of(channel.config) == TRIGGER_MODE_ALLOW_ALL
    except Exception:  # noqa: BLE001 — unknown mode ⇒ treat as gated (fail closed)
        logger.warning(
            "[trust-gate] allow_all resolution failed for ws=%s — treating as gated",
            workspace_id, exc_info=True,
        )
        return False


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
    """Seed a platform delivery default (e.g. telegram_default_chat_id) in
    workspace settings.integrations — SET ONCE, never silently retargeted.

    This value is the delivery target for agent-initiated questions
    (``platform_ask_human`` → ``channels.sender._resolve_target``) AND the chat
    the answer path binds a reply to (P225-RVW-1). Telegram/Slack deliver every
    inbound update for the bot to the one workspace webhook, so overwriting the
    anchor from arbitrary inbound senders let any user who can message the bot
    repoint the operator's questions to their own chat and answer them —
    re-opening the RVW-1 answer-injection class through the mutable anchor
    (P225-RVW-10). We therefore write the default only when it is UNSET:
    first-inbound seeds it as a convenience, later senders cannot move it. An
    operator changes it explicitly via Settings→Integrations
    (api/workspaces.save_integrations), never inbound traffic.
    """
    try:
        settings = dict(workspace.settings or {})
        integrations = dict(settings.get("integrations", {}))
        existing = integrations.get(key)
        if existing:
            # Already anchored — never silently retarget from inbound traffic.
            if str(existing) != str(value):
                logger.info(
                    "[webhook] %s already anchored for ws=%s — ignoring inbound retarget",
                    key, workspace.id,
                )
            return
        integrations[key] = value
        settings["integrations"] = integrations
        workspace.settings = settings
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(workspace, "settings")
        db.commit()
        logger.info("[webhook] Seeded %s for workspace %s (set-once)", key, workspace.id)
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

    # 2c. PRD-225 US-005: a reply / `/answer` to a pending question is a
    # RESPONSE, not a directive — correlate it BEFORE any routing (this also
    # bypasses the ingress trust gate, since answers are always allowed).
    if platform == "telegram":
        try:
            answered = await _maybe_answer_question(db, workspace, body, reply_ctx, integrations)
            if answered is not None:
                return answered
        except Exception:
            logger.exception(
                "[webhook/ws] question-answer correlation failed for ws=%s — "
                "falling through to routing", workspace.id,
            )

    # 2d. PRD-225 US-006: the ingress trust gate. Per the channel's trigger_mode,
    # hold inbound directives as pending questions instead of routing them.
    # Runs AFTER correlation (answers already returned) and BEFORE any routing or
    # platform-tool interception. Fail CLOSED on gate error (P225-RVW-5): a
    # strict / communication_only channel must never be opened by an internal
    # gate error — only a provably-allow_all channel falls through to routing.
    if platform:
        try:
            held = await _apply_trust_gate(db, workspace, platform, body, reply_ctx, integrations)
            if held is not None:
                return held
        except Exception:
            logger.error(
                "[webhook/ws] trust gate errored for ws=%s — failing closed",
                workspace.id, exc_info=True,
            )
            if not _channel_is_allow_all(db, workspace.id, platform):
                try:
                    await _deliver_reply(
                        "Received — an internal check must clear before I can act on it.",
                        reply_ctx, integrations, workspace_id=workspace.id,
                    )
                except Exception:  # noqa: BLE001 — best-effort ack
                    logger.debug("[trust-gate] fail-closed ack failed", exc_info=True)
                return {"status": "held", "routed": False, "reason": "trust_gate_error"}

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
