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

Every plan gets Socials (owner, 2026-09-23), so there is no plan exposure key.

``require_socials_enabled`` is the one route gate: every ``/api/socials/*``
route depends on it, and it answers 404 unless BOTH switches are on. A
workspace that can't use Socials never learns the routes exist.

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
from dataclasses import dataclass
from typing import Any, Dict, Optional

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

# The workspace settings key, and the only keys its object may carry.
WORKSPACE_SOCIALS_SETTINGS_KEY = "socials"
WORKSPACE_SOCIALS_KEYS = ("enabled",)


def socials_master_default() -> str:
    """The master switch's default as the system-settings string."""
    return "true" if config.SOCIALS_ENABLED_DEFAULT else "false"


def socials_master_enabled() -> bool:
    """The platform master switch: the ``socials.enabled`` system setting.

    A readable row decides: ``"true"`` is on, any other value is off. No row, or
    an empty value, takes ``config.SOCIALS_ENABLED_DEFAULT``. A read that cannot
    complete is off, whatever the default, and is logged at ERROR. Never raises.
    """
    try:
        value = read_system_setting(SOCIALS_SETTINGS_CATEGORY, KEY_ENABLED)
    except Exception:  # noqa: BLE001 — any read that did not complete fails closed
        logger.error(MASTER_READ_FAILED_LOG, SOCIALS_SETTINGS_CATEGORY, KEY_ENABLED, exc_info=True)
        return False
    return str(value or socials_master_default()).strip().lower() == "true"


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


def validate_socials_update(value: Any) -> Dict[str, Any]:
    """Pure, fail-closed validation for a workspace Socials write.

    The object carries exactly one key, ``enabled``, a boolean. Returns the
    normalized object; raises ``ValueError`` with the reason otherwise.
    """
    if not isinstance(value, dict):
        raise ValueError("socials must be an object")

    unknown = [k for k in value if k not in WORKSPACE_SOCIALS_KEYS]
    if unknown:
        raise ValueError(
            f"socials keys must be a subset of {list(WORKSPACE_SOCIALS_KEYS)}, got {unknown!r}"
        )
    if KEY_ENABLED not in value:
        raise ValueError("socials.enabled is required")
    if not isinstance(value[KEY_ENABLED], bool):
        raise ValueError("socials.enabled must be a boolean")
    return {KEY_ENABLED: value[KEY_ENABLED]}


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
