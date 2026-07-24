"""
PRD-207 S1: Retell server-side REST client
===========================================

The one place the platform talks TO Retell (create-web-call). Split pure/IO:
``build_web_call_payload`` is pure (unit-tested — the ``agent_override``
nesting and string-typed dynamic variables are exactly what Retell's schema
demands), ``create_web_call`` is a thin httpx POST that never logs the key.

The API key arrives from masked system settings (``live_settings``) — never
config, never the browser. CI mocks this module at ``create_web_call``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

RETELL_CREATE_WEB_CALL_URL = "https://api.retellai.com/v2/create-web-call"
RETELL_CREATE_AGENT_URL = "https://api.retellai.com/create-agent"
RETELL_UPDATE_AGENT_URL = "https://api.retellai.com/update-agent"
_TIMEOUT_SECONDS = 10.0
_AGENT_TIMEOUT_SECONDS = 25.0


def build_agent_tuning(include_voice: bool = True) -> Dict[str, Any]:
    """Everything about HOW Auto hears and sounds — applied at agent creation
    AND re-applied on every one-click re-arm.

    Hearing: pinning ``language`` stops multilingual STT hallucinating
    confident nonsense; ``noise-cancellation`` strips steady room noise while
    leaving the speaker audible (the aggressive
    ``noise-and-background-speech-cancellation`` mode cancelled him too — a
    whole measured call logged zero turns until he shouted); ``accurate`` STT
    trades a little latency for far fewer garbage transcripts.

    Sounding: the voice itself, its pace and expressiveness, and Retell's
    ``normalize_for_speech`` so prices, dates and hostnames are spoken as
    words. ``include_voice=False`` re-tunes hearing only, leaving a voice a
    human picked in the dashboard untouched.

    All config-driven dials — no literals here.
    """
    from config import config

    tuning: Dict[str, Any] = {
        "interruption_sensitivity": float(config.VOICE_LIVE_INTERRUPTION_SENSITIVITY),
        "responsiveness": float(config.VOICE_LIVE_RESPONSIVENESS),
        "enable_backchannel": False,
        "normalize_for_speech": bool(config.VOICE_LIVE_NORMALIZE_FOR_SPEECH),
        "reminder_trigger_ms": int(config.VOICE_LIVE_REMINDER_TRIGGER_MS),
        "reminder_max_count": int(config.VOICE_LIVE_REMINDER_MAX_COUNT),
    }
    language = str(config.VOICE_LIVE_LANGUAGE or "").strip()
    if language:
        tuning["language"] = language
    denoising = str(config.VOICE_LIVE_DENOISING_MODE or "").strip()
    if denoising:
        tuning["denoising_mode"] = denoising
    stt_mode = str(config.VOICE_LIVE_STT_MODE or "").strip()
    if stt_mode:
        tuning["stt_mode"] = stt_mode
    if include_voice:
        voice_id = str(config.VOICE_LIVE_VOICE_ID or "").strip()
        if voice_id:
            tuning["voice_id"] = voice_id
        tuning["voice_speed"] = float(config.VOICE_LIVE_VOICE_SPEED)
        tuning["voice_temperature"] = float(config.VOICE_LIVE_VOICE_TEMPERATURE)
    return tuning


class RetellApiError(Exception):
    """Retell refused or failed the call-creation request."""


@dataclass(frozen=True)
class RetellWebCall:
    call_id: str
    access_token: str


def build_web_call_payload(
    *,
    agent_id: str,
    dynamic_variables: Dict[str, Any],
    voice_id: Optional[str] = None,
    max_call_minutes: Optional[int] = None,
) -> Dict[str, Any]:
    """Pure: the create-web-call body.

    * ``retell_llm_dynamic_variables`` values must be STRINGS (Retell injects
      them into prompts/tools as text) — everything is coerced, Nones dropped.
    * Per-call voice + max duration ride ``agent_override.agent`` (verified
      against Retell's schema: nested ``{"agent_override": {"agent": {...}}}``,
      ``max_call_duration_ms`` bounded 60s–2h vendor-side).
    """
    payload: Dict[str, Any] = {
        "agent_id": agent_id,
        "retell_llm_dynamic_variables": {
            str(k): str(v) for k, v in dynamic_variables.items() if v is not None
        },
    }

    override: Dict[str, Any] = {}
    if voice_id:
        override["voice_id"] = voice_id
    if max_call_minutes and max_call_minutes > 0:
        override["max_call_duration_ms"] = int(max_call_minutes) * 60_000
    if override:
        payload["agent_override"] = {"agent": override}

    return payload


async def create_custom_llm_agent(
    api_key: str,
    *,
    agent_name: str,
    llm_websocket_url: str,
    webhook_url: str,
    voice_id: str,
) -> str:
    """Create the Retell agent that fronts Auto Live (one-click arming, S7).

    Verified request shape: ``response_engine {type: custom-llm,
    llm_websocket_url}``, ``voice_id``, ``agent_name``, ``webhook_url``;
    the response carries ``agent_id``. Raises ``RetellApiError`` with the
    vendor's own words (never the key) so the settings card can show an
    honest reason.
    """
    payload = {
        "agent_name": agent_name,
        "voice_id": voice_id,
        "response_engine": {"type": "custom-llm", "llm_websocket_url": llm_websocket_url},
        "webhook_url": webhook_url,
        # STT / turn-taking tuning baked in at birth (fresh agents are honest
        # in a noisy room from the first call, not just after a re-arm).
        **build_agent_tuning(),
    }
    try:
        async with httpx.AsyncClient(timeout=_AGENT_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                RETELL_CREATE_AGENT_URL,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.HTTPError as exc:
        raise RetellApiError(f"Retell unreachable: {type(exc).__name__}") from exc

    if resp.status_code not in (200, 201):
        logger.error(
            "voice_live_create_agent_failed status=%s body=%s",
            resp.status_code,
            resp.text[:300],
        )
        raise RetellApiError(
            f"Retell refused the API key or agent creation (HTTP {resp.status_code}): {resp.text[:200]}"
        )

    agent_id = resp.json().get("agent_id")
    if not agent_id:
        raise RetellApiError("Retell response missing agent_id")
    return str(agent_id)


async def update_agent(api_key: str, agent_id: str, settings: Dict[str, Any]) -> None:
    """PATCH agent-level settings onto an EXISTING Retell agent (S7 re-tune).

    The already-armed agent predates the STT/turn-taking tuning, so a
    one-click re-arm re-applies ``build_agent_tuning()`` to it — the fix for
    'confident nonsense' transcripts on a live agent nobody wants to recreate.
    No-op on empty settings. Raises ``RetellApiError`` (never the key) so the
    arm route can log-and-continue rather than fail the whole arm.
    """
    if not settings:
        return
    url = f"{RETELL_UPDATE_AGENT_URL}/{agent_id}"
    try:
        async with httpx.AsyncClient(timeout=_AGENT_TIMEOUT_SECONDS) as client:
            resp = await client.patch(
                url, json=settings, headers={"Authorization": f"Bearer {api_key}"}
            )
    except httpx.HTTPError as exc:
        raise RetellApiError(f"Retell unreachable: {type(exc).__name__}") from exc

    if resp.status_code not in (200, 201):
        logger.error(
            "voice_live_update_agent_failed status=%s body=%s",
            resp.status_code,
            resp.text[:300],
        )
        raise RetellApiError(
            f"Retell refused the agent update (HTTP {resp.status_code}): {resp.text[:200]}"
        )


async def create_web_call(api_key: str, payload: Dict[str, Any]) -> RetellWebCall:
    """POST create-web-call; returns the short-lived token + call id.

    Raises ``RetellApiError`` with an operator-honest (never key-leaking)
    message on any refusal — the mint route turns that into a 502.
    """
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                RETELL_CREATE_WEB_CALL_URL,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.HTTPError as exc:
        raise RetellApiError(f"Retell unreachable: {type(exc).__name__}") from exc

    if resp.status_code not in (200, 201):
        logger.error(
            "voice_live_retell_create_failed status=%s body=%s",
            resp.status_code,
            resp.text[:300],
        )
        raise RetellApiError(f"Retell create-web-call failed (HTTP {resp.status_code})")

    data = resp.json()
    call_id = data.get("call_id")
    access_token = data.get("access_token")
    if not call_id or not access_token:
        raise RetellApiError("Retell response missing call_id/access_token")
    return RetellWebCall(call_id=str(call_id), access_token=str(access_token))
