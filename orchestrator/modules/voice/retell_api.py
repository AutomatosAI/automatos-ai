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


def build_agent_tuning() -> Dict[str, Any]:
    """The STT / turn-taking settings that keep transcription honest in a real
    (noisy, non-headset) room — applied at agent creation AND re-applied on
    every one-click re-arm.

    Pinning ``language`` stops multilingual STT from hallucinating confident
    nonsense; ``noise-and-background-speech-cancellation`` strips TV / room
    bleed (the "multiple voices" corruptor); ``accurate`` STT trades a little
    latency for far fewer garbage transcripts; low ``interruption_sensitivity``
    stops every tiny sound from grabbing the turn. All config-driven dials.
    """
    from config import config

    tuning: Dict[str, Any] = {
        "interruption_sensitivity": float(config.VOICE_LIVE_INTERRUPTION_SENSITIVITY),
        "responsiveness": float(config.VOICE_LIVE_RESPONSIVENESS),
        "enable_backchannel": False,
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
