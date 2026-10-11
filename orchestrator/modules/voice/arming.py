"""
PRD-207 S7 / PRD-258 US-003: one-click arming of Auto Live
===========================================================

The body of ``POST /api/voice/arm`` (the route keeps the admin gate). Paste
the Retell API key, click Arm: a first arm creates the custom-LLM agent in
the customer's Retell account; a re-arm keeps the agent and sends ONE PATCH
with the tuning AND the endpoints built from the current
``config.PUBLIC_API_HOST``, so an agent created against one host (prod, an
old tunnel) is moved to the current one. Moved out of ``api/voice_retell.py``
by PRD-258 US-003 (D6: that file never grows).

The local edition names a new agent 'Auto Live (local)', so it is never
confused with prod's agent in a shared Retell dashboard.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from config import config
from modules.voice import live_settings, retell_api

logger = logging.getLogger(__name__)

AGENT_NAME_SAAS = "Auto Live"
AGENT_NAME_LOCAL = "Auto Live (local)"
_LOCAL_EDITION = "local"
# The voice a first arm gives the agent when the card names none.
DEFAULT_AGENT_VOICE_ID = "retell-Cimo"
# A Retell API key mis-filed into a workspace's voice field starts with this.
_RETELL_KEY_PREFIX = "key_"

HTTP_BAD_REQUEST = 400
HTTP_SERVER_ERROR = 500
HTTP_BAD_GATEWAY = 502


class ArmRefused(Exception):
    """Arming refused, with the HTTP status and an admin-honest reason."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def agent_name() -> str:
    """The name a newly created agent gets on this edition."""
    if config.AUTH_EDITION == _LOCAL_EDITION:
        return AGENT_NAME_LOCAL
    return AGENT_NAME_SAAS


def _current_endpoints() -> Dict[str, Any]:
    """The agent endpoints for ``config.PUBLIC_API_HOST``.

    Raises ``ValueError`` (logged) when the host is not a bare host name.
    """
    try:
        return retell_api.build_agent_endpoints(str(config.PUBLIC_API_HOST))
    except ValueError as exc:
        logger.error("voice_live_arm_bad_public_host err=%s", exc)
        raise


async def _create_agent(api_key: str, voice_id: Optional[str]) -> str:
    """First arm: create the custom-LLM agent.

    A bad ``PUBLIC_API_HOST`` is a 500 (no agent is born pointing nowhere);
    Retell's refusal is a 502.
    """
    try:
        endpoints = _current_endpoints()
    except ValueError as exc:
        raise ArmRefused(HTTP_SERVER_ERROR, str(exc)) from exc
    try:
        return await retell_api.create_custom_llm_agent(
            api_key,
            agent_name=agent_name(),
            endpoints=endpoints,
            voice_id=(voice_id or "").strip() or DEFAULT_AGENT_VOICE_ID,
        )
    except retell_api.RetellApiError as exc:
        raise ArmRefused(HTTP_BAD_GATEWAY, str(exc)) from exc


async def _repoint_agent(api_key: str, agent_id: str) -> None:
    """Re-arm: ONE PATCH carrying the tuning and the current endpoints.

    Non-fatal, as the re-tune always was: a bad ``PUBLIC_API_HOST`` sends the
    tuning alone (the agent keeps its endpoints), and a PATCH failure is
    logged — arming goes on either way (the agent works with its prior
    settings).
    """
    settings = retell_api.build_agent_tuning()
    try:
        settings = {**settings, **_current_endpoints()}
        logger.info("voice_live_arm_repoint agent=%s host=%s", agent_id, config.PUBLIC_API_HOST)
    except ValueError:
        logger.warning("voice_live_arm_repoint_skipped agent=%s: re-tuning only", agent_id)
    try:
        await retell_api.update_agent(api_key, agent_id, settings)
    except retell_api.RetellApiError as exc:
        logger.warning("voice_live_arm_tune_failed agent=%s err=%s", agent_id, exc)


def _store_credentials(
    db: Session, creds: live_settings.RetellCredentials, api_key: str, agent_id: str
) -> None:
    """File key, signing key and agent id, and flip the master toggle on."""
    live_settings.set_voice_setting(db, live_settings.KEY_RETELL_API_KEY, api_key)
    # Retell signs webhooks with the API key unless a distinct secret is
    # configured — only fill the slot when it is empty, never overwrite one.
    if not creds.webhook_secret:
        live_settings.set_voice_setting(db, live_settings.KEY_RETELL_WEBHOOK_SECRET, api_key)
    live_settings.set_voice_setting(db, live_settings.KEY_RETELL_AGENT_ID, agent_id)
    live_settings.set_voice_setting(db, live_settings.KEY_LIVE_ENABLED, "true")


def _sweep_misfiled_key(db: Session, workspace_id: Optional[uuid.UUID]) -> None:
    """Drop a Retell API key mis-filed into the workspace's voice field."""
    from core.models.workspaces import Workspace

    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if workspace is None:
        return
    settings = dict(workspace.settings or {})
    voice_live = dict(settings.get("voice_live") or {})
    stray = str(voice_live.get("retell_voice_id") or "")
    if not stray.lower().startswith(_RETELL_KEY_PREFIX):
        return
    voice_live.pop("retell_voice_id", None)
    settings["voice_live"] = voice_live
    workspace.settings = settings
    from sqlalchemy.orm.attributes import flag_modified

    flag_modified(workspace, "settings")
    logger.info("voice_live_arm swept a mis-filed API key out of workspace %s", workspace.id)


async def arm(
    db: Session,
    *,
    enabled: bool,
    api_key: Optional[str],
    voice_id: Optional[str],
    workspace_id: Optional[uuid.UUID],
    actor: Any,
) -> dict:
    """Arm (or disarm) Auto Live platform-wide; the caller is a platform admin.

    ``enabled=False`` flips the master toggle only (creds stay, instant kill
    preserved). Raises ``ArmRefused`` when there is no key or Retell refuses
    the first-arm agent creation.
    """
    if not enabled:
        live_settings.set_voice_setting(db, live_settings.KEY_LIVE_ENABLED, "false")
        db.commit()
        logger.warning("voice_live_disarmed by=%s", actor)
        return {"armed": live_settings.retell_credentials().armed, "platform_enabled": False}

    creds = live_settings.retell_credentials()
    key = (api_key or creds.api_key or "").strip()
    if not key:
        raise ArmRefused(HTTP_BAD_REQUEST, "Paste your Retell API key to arm Auto Live")

    agent_id = creds.agent_id
    if agent_id:
        await _repoint_agent(key, agent_id)
    else:
        agent_id = await _create_agent(key, voice_id)

    _store_credentials(db, creds, key, agent_id)
    _sweep_misfiled_key(db, workspace_id)
    db.commit()
    logger.info("voice_live_armed agent=%s by=%s", agent_id, actor)
    return {"armed": True, "platform_enabled": True, "agent_id": agent_id}
