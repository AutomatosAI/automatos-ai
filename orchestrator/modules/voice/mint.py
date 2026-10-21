"""
PRD-266 US-008a (PRD-207 S1): the web-call mint's steps
=========================================================

``POST /api/voice/web-call`` (``api/voice_retell.mint_web_call``) is a short route over
these helpers, in its order: the plan (a route dependency, SaaS only, D34), the arming
gate (the platform switch, then the three Retell credentials), the strictly resolved
caller (in the route), the workspace's own toggle and the cap formula (the meter), the
chat binding, the vendor call, superseding the caller's older calls, and the
``voice_calls`` row born at mint.

Every refusal is a ``MintRefused`` carrying the status and the honest reason the route
has always given; the route maps it to its HTTP error. What the route's tests patch on
``api.voice_retell`` (``live_settings.*``, ``voice_meter.*``,
``retell_api.create_web_call``) reaches these helpers through ``MintPorts``, which the
route builds late-bound from its own names (the ``_socket_ports`` pattern), so a patch
where the route lives still takes effect. The socket's AI notice is the route's
``_socket_ports``, untouched by the mint.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from sqlalchemy.orm import Session

from config import config
from modules.voice import retell_api

logger = logging.getLogger(__name__)

# D34: Auto Live is the Business plan's on SaaS. The Automatos Go app keys on the code,
# never the sentence (the repo's coded-refusal shape: {"code", "message"}).
VOICE_FAMILY = "voice"
VOICE_LIVE_PLAN_REQUIRED = "voice_live_plan_required"
VOICE_LIVE_PLAN_REQUIRED_MESSAGE = "Auto Live is on the Business plan"

PLATFORM_DISABLED = "Auto Live is switched off platform-wide"
NOT_ARMED = "Auto Live is not armed — Retell credentials are missing in Settings"
WORKSPACE_NOT_FOUND = "Workspace not found"
WORKSPACE_DISABLED = "Auto Live is not enabled for this workspace"
CHAT_NOT_VALID = "chat_id is not a valid chat"
CHAT_NOT_IN_WORKSPACE = "chat_id is not a chat in this workspace"
CHAT_NOT_YOURS = "You can only bind a live call to your own chat"
VENDOR_REFUSED_PREFIX = "Could not start the call: "
# The caller's one voice thread is found by its title's prefix (#585: the title is
# timestamped, so a fixed title cannot collide on the second call).
VOICE_THREAD_TITLE_LIKE = "Voice call%"
ACTIVE_CALL_STATUSES = ("minted", "started")
SUPERSEDED_STATUS = "superseded"
SUPERSEDED_REASON = "superseded_by_newer_call"
MINTED_STATUS = "minted"
RETELL_PROVIDER = "retell"
SUPER_ADMIN_ROLE = "super_admin"


class MintRefused(Exception):
    """The mint refused, with the HTTP status and the caller-honest reason."""

    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class MintPorts:
    """What a mint borrows from the route, late-bound through the route module's names."""

    voice_live_enabled: Callable[[], bool]
    retell_credentials: Callable[[], Any]
    workspace_voice_live: Callable[[Any], Any]
    monthly_meter: Callable[[Session, Any], Any]
    cap_allows_mint: Callable[[Any, int], Tuple[bool, str]]
    create_web_call: Callable[[str, Dict[str, Any]], Awaitable[retell_api.RetellWebCall]]


@dataclass(frozen=True)
class MintTarget:
    """Who a call is for and where it lands: its dynamic variables and its row."""

    workspace_id: Any
    user_id: int
    chat_id: str
    agent_id: Optional[int] = None


def require_voice_live_plan(db: Session, workspace_id: Any) -> None:
    """D34: on SaaS, refuse (403, coded) a workspace whose plan has no voice family.

    ``services.plan_tiers.family_enabled`` reads it by the exposure profile's rules: a known tier
    that declares no families (``enterprise`` today) is unrestricted, as the nav and tool
    layers treat it; a workspace without a plan is the entry tier. The local edition is
    never plan-gated.
    """
    if config.IS_LOCAL_EDITION:
        return
    from core.models.workspaces import Workspace
    from services.plan_tiers import family_enabled

    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    plan = getattr(workspace, "plan", None)
    if family_enabled(plan, VOICE_FAMILY):
        return
    logger.info("voice_live_mint_denied reason=plan workspace=%s plan=%s", workspace_id, plan)
    raise MintRefused(
        403, {"code": VOICE_LIVE_PLAN_REQUIRED, "message": VOICE_LIVE_PLAN_REQUIRED_MESSAGE}
    )


def armed_credentials(ports: MintPorts, workspace_id: Any) -> Any:
    """Gates 1–2: the platform master switch (a Settings toggle, never an env var), then
    all three Retell credentials present in system settings. Returns the credentials."""
    if not ports.voice_live_enabled():
        logger.info("voice_live_mint_denied reason=platform_disabled workspace=%s", workspace_id)
        raise MintRefused(503, PLATFORM_DISABLED)
    creds = ports.retell_credentials()
    if not creds.armed:
        logger.warning("voice_live_mint_denied reason=not_armed workspace=%s", workspace_id)
        raise MintRefused(503, NOT_ARMED)
    return creds


def workspace_voice_within_cap(db: Session, ports: MintPorts, workspace_id: Any) -> Any:
    """Gates 3–4: the workspace's own toggle, then the cap formula (ended + active×reserve
    ≥ cap refuses, 429 with the honest budget line). Returns the workspace's voice view."""
    from core.models.workspaces import Workspace

    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if workspace is None:
        raise MintRefused(404, WORKSPACE_NOT_FOUND)
    ws_voice = ports.workspace_voice_live(workspace.settings)
    if not ws_voice.enabled:
        logger.info("voice_live_mint_denied reason=workspace_disabled workspace=%s", workspace_id)
        raise MintRefused(403, WORKSPACE_DISABLED)
    reading = ports.monthly_meter(db, workspace_id)
    allowed, reason = ports.cap_allows_mint(reading, ws_voice.monthly_cap_minutes)
    if not allowed:
        logger.warning("voice_live_cap_exceeded workspace=%s %s", workspace_id, reason)
        raise MintRefused(429, reason)
    return ws_voice


def bind_chat(db: Session, workspace_id: Any, user_id: int, chat_id: Optional[str]) -> str:
    """The thread this call lands in: voice and text are the SAME conversation.

    An on-screen ``chat_id`` must be the caller's own thread in THIS workspace (refused
    honestly rather than minting a lie the webhook would then reject, §7.5-1/2). With no
    chat on screen the caller's one voice thread is continued, or created (#585).
    """
    if chat_id:
        return _own_chat(db, workspace_id, user_id, chat_id)
    return _voice_thread(db, workspace_id, user_id)


def _own_chat(db: Session, workspace_id: Any, user_id: int, chat_id: str) -> str:
    """The on-screen chat's id, when it is the caller's own in this workspace."""
    from core.models.core import Chat

    try:
        chat_uuid = uuid.UUID(str(chat_id))
    except ValueError as exc:
        raise MintRefused(400, CHAT_NOT_VALID) from exc
    chat = db.query(Chat).filter(Chat.id == chat_uuid, Chat.workspace_id == workspace_id).first()
    if chat is None:
        raise MintRefused(400, CHAT_NOT_IN_WORKSPACE)
    if int(chat.user_id) != int(user_id):
        raise MintRefused(403, CHAT_NOT_YOURS)
    return str(chat_uuid)


def _voice_thread(db: Session, workspace_id: Any, user_id: int) -> str:
    """The caller's latest voice thread in this workspace, else a new one (a welcome-screen
    mint used to spin up a fresh thread per call, fragmenting the conversation, #585)."""
    from core.models.core import Chat
    from modules.voice.call_binding import create_voice_chat

    existing = (
        db.query(Chat)
        .filter(
            Chat.user_id == int(user_id),
            Chat.workspace_id == workspace_id,
            Chat.title.like(VOICE_THREAD_TITLE_LIKE),
        )
        .order_by(Chat.created_at.desc())
        .first()
    )
    if existing is not None:
        return str(existing.id)
    return str(create_voice_chat(db, user_id=int(user_id), workspace_id=workspace_id).id)


async def create_call(
    ports: MintPorts, creds: Any, ws_voice: Any, target: MintTarget
) -> retell_api.RetellWebCall:
    """Ask Retell for the web call (the key never reaches the browser); a vendor refusal is
    a 502 with Retell's operator-honest reason (never the key)."""
    payload = retell_api.build_web_call_payload(
        agent_id=creds.agent_id,
        dynamic_variables={
            "workspace_id": str(target.workspace_id),
            "user_id": target.user_id,
            "chat_id": target.chat_id,
            "agent_id": target.agent_id,
        },
        voice_id=ws_voice.retell_voice_id,
        max_call_minutes=int(config.VOICE_LIVE_MAX_CALL_MINUTES),
    )
    logger.info(
        "voice_live_mint_requested workspace=%s user=%s chat=%s",
        target.workspace_id, target.user_id, target.chat_id,
    )
    try:
        return await ports.create_web_call(creds.api_key, payload)
    except retell_api.RetellApiError as exc:
        logger.error("voice_live_mint_denied reason=vendor_error workspace=%s err=%s", target.workspace_id, exc)
        raise MintRefused(502, f"{VENDOR_REFUSED_PREFIX}{exc}") from exc


def caller_is_super_admin(ctx: Any) -> bool:
    """PRD-143: su derives from the session's ``system_role`` only, captured at mint (the
    one place the session exists) for the webhook and the turns."""
    return getattr(getattr(ctx, "user", None), "system_role", "user") == SUPER_ADMIN_ROLE


def _supersede_prior_calls(db: Session, user_id: int, new_call_id: str) -> None:
    """ONE LIVE CALL PER USER: mark the caller's active calls superseded, in the mint's
    transaction. A client that mints again (device switch, a re-press, two LiveVoiceMode
    instances) would otherwise leave several calls on the same mic, each streaming Auto
    over herself; a superseded call's socket closes on its next turn."""
    from core.models.voice_calls import VoiceCall

    superseded = (
        db.query(VoiceCall)
        .filter(VoiceCall.user_id == user_id, VoiceCall.status.in_(ACTIVE_CALL_STATUSES))
        .update(
            {VoiceCall.status: SUPERSEDED_STATUS, VoiceCall.disconnect_reason: SUPERSEDED_REASON},
            synchronize_session=False,
        )
    )
    if superseded:
        logger.info(
            "voice_live_superseded_prior count=%s user=%s (new call=%s)", superseded, user_id, new_call_id
        )


def record_minted_call(
    db: Session, web_call: retell_api.RetellWebCall, target: MintTarget, *, is_super_admin: bool
) -> None:
    """Supersede the caller's older calls, then the ``voice_calls`` row is BORN (status
    ``minted``) so the webhook's trust boundary has something to validate against before
    any event arrives (S2/S3); one commit for both."""
    from core.models.voice_calls import VoiceCall

    _supersede_prior_calls(db, target.user_id, web_call.call_id)
    db.add(
        VoiceCall(
            call_id=web_call.call_id,
            provider=RETELL_PROVIDER,
            workspace_id=target.workspace_id,
            user_id=target.user_id,
            chat_id=target.chat_id,
            status=MINTED_STATUS,
            is_super_admin=is_super_admin,
        )
    )
    db.commit()
    logger.info(
        "voice_live_minted call=%s workspace=%s user=%s chat=%s",
        web_call.call_id, target.workspace_id, target.user_id, target.chat_id,
    )


def mint_response(web_call: retell_api.RetellWebCall, chat_id: str) -> Dict[str, Any]:
    """The mint's answer: the call, its ~30s token, the bound thread (one conversation,
    spoken and typed) and Retell's transport fields when it returned them (D32)."""
    return {
        "call_id": web_call.call_id,
        "access_token": web_call.access_token,
        "chat_id": chat_id,
        **web_call.transport_fields(),
    }


__all__ = [
    "MintPorts",
    "MintRefused",
    "MintTarget",
    "VOICE_LIVE_PLAN_REQUIRED",
    "VOICE_LIVE_PLAN_REQUIRED_MESSAGE",
    "armed_credentials",
    "bind_chat",
    "caller_is_super_admin",
    "create_call",
    "mint_response",
    "record_minted_call",
    "require_voice_live_plan",
    "workspace_voice_within_cap",
]
