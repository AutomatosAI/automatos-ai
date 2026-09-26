"""
PRD-251 S0.1: the two Socials switches
======================================

Socials is gated two ways (D1), on the Auto Live pattern
(``modules/voice/live_settings.py``):

* the platform master switch, the ``socials.enabled`` DB system setting. The
  super-admin flips it in Settings → System Settings, with no redeploy. Its
  default is ``config.SOCIALS_ENABLED_DEFAULT``: the ``prd251_socials``
  migration seeds the row with it, and it applies wherever no row exists.
* the workspace switch, ``workspace.settings['socials'].enabled``, which a
  workspace owner or admin sets through ``PUT /api/workspaces/current/socials``.

The same object carries the workspace's monthly media cap (D13, S1.8),
``media_monthly_cap_usd``: the most the workspace's connected media tools may
spend for Socials in a calendar month, in dollars (``modules/socials/media_caps.py``).
An owner or admin sets it on the same route; without it the cap is
``config.SOCIALS_MEDIA_MONTHLY_CAP_USD``. A stored value that is not a number of
dollars spends nothing until it is fixed: a money guard that cannot read its
limit must deny.

Every plan gets Socials (owner, 2026-09-23), so there is no plan exposure key.

``require_socials_enabled`` is the one route gate: every ``/api/socials/*``
route depends on it, and it answers 404 unless BOTH switches are on. A
workspace that can't use Socials never learns the routes exist. The agent
tools (US-116) ask ``socials_off_reason`` instead, which reads the same two
switches and says which one is off and who turns it on.

The master switch is read on every request, so flipping it takes effect on
the next one. The read is strict (``core.llm.manager.read_system_setting``): a
read that cannot complete (an exhausted pool, a timeout, a dropped connection)
switches Socials OFF, whatever ``SOCIALS_ENABLED_DEFAULT`` says, and is logged
at ERROR. A gate that can't decide must deny. Never ``get_system_setting``: its
catch-all returns the default on any failure, so with the default on, "could
not read" would read as ON (P251-RVW-8).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.llm.manager import read_system_setting
from core.models.system_settings import SettingCategory
from core.models.workspaces import Workspace

logger = logging.getLogger(__name__)

SOCIALS_SETTINGS_CATEGORY = SettingCategory.SOCIALS.value
KEY_ENABLED = "enabled"

MASTER_READ_FAILED_LOG = (
    "[Socials] system setting %s.%s could not be read; the Socials master switch is OFF until it can be"
)

# What the agent tools answer while a switch is off (US-116): nothing is read or written.
SOCIALS_OFF_FOR_PLATFORM = (
    "Socials is not switched on for this platform, so nothing was read or saved. "
    "The platform's super-admin turns it on in Settings → System Settings."
)
SOCIALS_OFF_FOR_WORKSPACE = (
    "Socials is off for this workspace, so nothing was read or saved. "
    "A workspace owner or admin turns it on in the Socials tab under Deliverables."
)

# The workspace settings key, and the only keys its object may carry.
WORKSPACE_SOCIALS_SETTINGS_KEY = "socials"
KEY_MEDIA_MONTHLY_CAP = "media_monthly_cap_usd"
WORKSPACE_SOCIALS_KEYS = ("enabled", KEY_MEDIA_MONTHLY_CAP)


def socials_master_default() -> str:
    """The master switch's default as the system-settings string."""
    return "true" if config.SOCIALS_ENABLED_DEFAULT else "false"


def socials_master_switch() -> bool:
    """The platform master switch, read strictly: raises when the read cannot
    complete.

    A readable row decides: ``"true"`` is on, any other value is off. No row, or
    an empty value, takes ``config.SOCIALS_ENABLED_DEFAULT``. For a guard that
    must tell "off" from "could not read": the Socials post gate
    (``core/composio/post_gate.py``) refuses when it cannot tell, where Socials
    OFF would let an agent post.
    """
    value = read_system_setting(SOCIALS_SETTINGS_CATEGORY, KEY_ENABLED)
    return str(value or socials_master_default()).strip().lower() == "true"


def socials_master_enabled() -> bool:
    """The platform master switch (``socials_master_switch``), where a read that
    cannot complete is off, whatever the default, and is logged at ERROR. Never
    raises.
    """
    try:
        return socials_master_switch()
    except Exception:  # noqa: BLE001 — any read that did not complete fails closed
        logger.error(MASTER_READ_FAILED_LOG, SOCIALS_SETTINGS_CATEGORY, KEY_ENABLED, exc_info=True)
        return False


@dataclass(frozen=True)
class WorkspaceSocials:
    enabled: bool


def parse_workspace_socials(settings: Optional[Dict[str, Any]]) -> WorkspaceSocials:
    """Pure: ``workspace.settings`` → the workspace's Socials switch.

    Missing or malformed → off (fail-closed). Only a real ``True`` turns it on,
    so a stray string such as ``"false"`` never reads as on.
    """
    raw = (settings or {}).get(WORKSPACE_SOCIALS_SETTINGS_KEY) or {}
    if not isinstance(raw, dict):
        raw = {}
    return WorkspaceSocials(enabled=raw.get(KEY_ENABLED) is True)


def _dollars(value: Any) -> Optional[float]:
    """A non-negative, finite number of dollars, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def validate_socials_update(value: Any) -> Dict[str, Any]:
    """Pure, fail-closed validation for a workspace Socials write.

    The object carries ``enabled`` (a boolean), ``media_monthly_cap_usd`` (a
    non-negative number of dollars), or both. Returns the normalized object;
    raises ``ValueError`` with the reason otherwise.
    """
    if not isinstance(value, dict):
        raise ValueError("socials must be an object")

    unknown = [k for k in value if k not in WORKSPACE_SOCIALS_KEYS]
    if unknown:
        raise ValueError(
            f"socials keys must be a subset of {list(WORKSPACE_SOCIALS_KEYS)}, got {unknown!r}"
        )
    if not value:
        raise ValueError(f"socials.enabled or socials.{KEY_MEDIA_MONTHLY_CAP} is required")
    normalized: Dict[str, Any] = {}
    if KEY_ENABLED in value:
        if not isinstance(value[KEY_ENABLED], bool):
            raise ValueError("socials.enabled must be a boolean")
        normalized[KEY_ENABLED] = value[KEY_ENABLED]
    if KEY_MEDIA_MONTHLY_CAP in value:
        cap = _dollars(value[KEY_MEDIA_MONTHLY_CAP])
        if cap is None:
            raise ValueError(f"socials.{KEY_MEDIA_MONTHLY_CAP} must be a number of dollars, 0 or more")
        normalized[KEY_MEDIA_MONTHLY_CAP] = cap
    return normalized


def media_monthly_cap_usd(settings: Optional[Dict[str, Any]]) -> Tuple[float, Optional[str]]:
    """The workspace's monthly media cap in dollars (D13), and why it spends
    nothing when its stored value is not a number of dollars.

    No value → ``config.SOCIALS_MEDIA_MONTHLY_CAP_USD``. A value that is not a
    non-negative, finite number → ``(0.0, why)``: fail closed.
    """
    raw = (settings or {}).get(WORKSPACE_SOCIALS_SETTINGS_KEY)
    raw = raw if isinstance(raw, dict) else {}
    if raw.get(KEY_MEDIA_MONTHLY_CAP) is None:
        return float(config.SOCIALS_MEDIA_MONTHLY_CAP_USD), None
    cap = _dollars(raw[KEY_MEDIA_MONTHLY_CAP])
    if cap is None:
        why = (
            f"the workspace's monthly media cap ({raw[KEY_MEDIA_MONTHLY_CAP]!r}) is not a number of dollars, "
            "so nothing is spent until an owner or admin sets it"
        )
        logger.error("[Socials] %s", why)
        return 0.0, why
    return cap, None


def socials_off_reason(workspace: Optional[Workspace]) -> Optional[str]:
    """Why Socials is off for ``workspace`` (D1), or ``None`` when both switches
    are on: the check behind the agent tools (US-116), on the route gate's two
    switches. A missing workspace is off; a master switch that cannot be read is
    off (``socials_master_enabled``)."""
    if not socials_master_enabled():
        return SOCIALS_OFF_FOR_PLATFORM
    if workspace is None or not parse_workspace_socials(workspace.settings).enabled:
        return SOCIALS_OFF_FOR_WORKSPACE
    return None


def socials_state(settings: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    """The ``socials`` block of ``GET /api/workspaces/current``.

    ``available`` is the master switch; ``enabled`` is the workspace switch.
    The frontend gate reads only this.
    """
    return {
        "available": socials_master_enabled(),
        "enabled": parse_workspace_socials(settings).enabled,
    }


async def require_socials_enabled(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> RequestContext:
    """Route gate: 404 unless the master switch AND the caller's workspace switch are on."""
    if not socials_master_enabled():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    workspace = db.get(Workspace, ctx.workspace_id)
    if workspace is None or not parse_workspace_socials(workspace.settings).enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    return ctx
