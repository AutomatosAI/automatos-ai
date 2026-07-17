"""
PRD-207 S4: Auto Live settings plane
=====================================

The platform kill-switch and the Retell credentials are **DB system settings**
(category ``voice``), not env vars — PRD-143 replaced ``.env`` with
``system_settings`` and arming/disarming must be a super-admin Settings-page
act (no redeploy, no SSH). ``config.py`` keeps only the numeric tuning
constants (cap/reserve/max-call); the ON-switch never lives in config.

Reads go through ``core.llm.manager.get_system_setting`` — the canonical
runtime accessor (the coordinator's ``mission_budget_alert_usd`` precedent):
per-request DB read, fail-soft to the safe default (OFF / empty).

The workspace half (``workspace.settings.voice_live``) is parsed here too so
the mint gate, the tools whitelist handler and the settings API share ONE
shape: ``{enabled: bool, monthly_cap_minutes?: int, retell_voice_id?: str}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import config
from core.llm.manager import get_system_setting

VOICE_SETTINGS_CATEGORY = "voice"
KEY_LIVE_ENABLED = "live_enabled"
KEY_RETELL_API_KEY = "retell_api_key"
KEY_RETELL_WEBHOOK_SECRET = "retell_webhook_secret"
KEY_RETELL_AGENT_ID = "retell_agent_id"

# The only keys a workspace's voice_live settings object may carry.
WORKSPACE_VOICE_LIVE_KEYS = ("enabled", "monthly_cap_minutes", "retell_voice_id")


@dataclass(frozen=True)
class RetellCredentials:
    api_key: str
    webhook_secret: str
    agent_id: str

    @property
    def armed(self) -> bool:
        """All three present — the platform can mint and verify webhooks."""
        return bool(self.api_key and self.webhook_secret and self.agent_id)


def voice_live_enabled() -> bool:
    """The platform master switch — ``voice.live_enabled`` system setting.

    Default OFF everywhere: merging code never turns on microphones or
    billing; the super-admin flips the toggle in Settings (§7.5-3).
    """
    value = get_system_setting(VOICE_SETTINGS_CATEGORY, KEY_LIVE_ENABLED, "false")
    return str(value).strip().lower() == "true"


def retell_credentials() -> RetellCredentials:
    """Retell platform credentials from masked system settings (never env)."""
    return RetellCredentials(
        api_key=str(get_system_setting(VOICE_SETTINGS_CATEGORY, KEY_RETELL_API_KEY, "") or "").strip(),
        webhook_secret=str(
            get_system_setting(VOICE_SETTINGS_CATEGORY, KEY_RETELL_WEBHOOK_SECRET, "") or ""
        ).strip(),
        agent_id=str(get_system_setting(VOICE_SETTINGS_CATEGORY, KEY_RETELL_AGENT_ID, "") or "").strip(),
    )


@dataclass(frozen=True)
class WorkspaceVoiceLive:
    enabled: bool
    monthly_cap_minutes: int
    retell_voice_id: Optional[str]


def parse_workspace_voice_live(settings: Optional[Dict[str, Any]]) -> WorkspaceVoiceLive:
    """Pure: ``workspace.settings`` → the effective voice_live view.

    Missing/malformed → disabled with the config default cap (fail-closed:
    a workspace is never live by accident).
    """
    raw = (settings or {}).get("voice_live") or {}
    if not isinstance(raw, dict):
        raw = {}

    cap = raw.get("monthly_cap_minutes")
    try:
        cap_minutes = int(cap) if cap is not None else int(config.VOICE_LIVE_DEFAULT_MONTHLY_CAP_MINUTES)
    except (TypeError, ValueError):
        cap_minutes = int(config.VOICE_LIVE_DEFAULT_MONTHLY_CAP_MINUTES)
    if cap_minutes <= 0:
        cap_minutes = int(config.VOICE_LIVE_DEFAULT_MONTHLY_CAP_MINUTES)

    voice_id = raw.get("retell_voice_id")
    voice_id = str(voice_id).strip() if isinstance(voice_id, str) and voice_id.strip() else None

    return WorkspaceVoiceLive(
        enabled=bool(raw.get("enabled", False)),
        monthly_cap_minutes=cap_minutes,
        retell_voice_id=voice_id,
    )


def validate_voice_live_update(value: Any) -> Dict[str, Any]:
    """Pure, fail-closed validation for a voice_live settings write.

    Used by BOTH write surfaces (the PUT route and the platform tool) so a
    malformed value can never reach ``workspace.settings``. Returns the
    normalized object; raises ``ValueError`` with an honest reason otherwise.
    """
    if not isinstance(value, dict):
        raise ValueError("voice_live must be an object")

    unknown = [k for k in value if k not in WORKSPACE_VOICE_LIVE_KEYS]
    if unknown:
        raise ValueError(
            f"voice_live keys must be a subset of {list(WORKSPACE_VOICE_LIVE_KEYS)}, got {unknown!r}"
        )

    normalized: Dict[str, Any] = {}
    if "enabled" in value:
        if not isinstance(value["enabled"], bool):
            raise ValueError("voice_live.enabled must be a boolean")
        normalized["enabled"] = value["enabled"]

    if "monthly_cap_minutes" in value and value["monthly_cap_minutes"] is not None:
        cap = value["monthly_cap_minutes"]
        if isinstance(cap, bool) or not isinstance(cap, int):
            raise ValueError("voice_live.monthly_cap_minutes must be an integer")
        if cap <= 0 or cap > 100_000:
            raise ValueError("voice_live.monthly_cap_minutes must be between 1 and 100000")
        normalized["monthly_cap_minutes"] = cap

    if "retell_voice_id" in value and value["retell_voice_id"] is not None:
        vid = value["retell_voice_id"]
        if not isinstance(vid, str) or len(vid) > 64:
            raise ValueError("voice_live.retell_voice_id must be a string of at most 64 characters")
        normalized["retell_voice_id"] = vid.strip()

    return normalized
