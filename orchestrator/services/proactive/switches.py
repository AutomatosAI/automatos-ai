"""PRD-268 §6.9, FR-13, K7 — the switches a check-in reads at its start, each failing closed.

* **The platform mode**: ``off``, ``shadow`` or ``live``, the system setting ``proactive.mode``
  (a super admin's), read strictly (``core.llm.manager.read_system_setting``). No row takes
  ``config.PROACTIVE_MODE_DEFAULT`` ('off'). A value that is none of the three, or a read
  that cannot complete, is off, as Socials reads its master switch
  (``modules.socials.settings.socials_master_enabled``).
* **The workspace switch**: ``settings.proactive.enabled`` on the workspace, the owner's or
  an admin's. Off unless it is ``true``.
* **The person's pause**: ``proactive_person_settings.paused`` for the person a suggestion
  is addressed to. A pause that cannot be read is a pause.

``effective_mode`` is a workspace's mode: the platform's where the workspace switch is on,
else off. In ``shadow`` today's heartbeats run unchanged and check-ins run beside them; in
``live`` check-ins run instead (the bridge, US-C04b).

The runner (US-C03d) reads these at every run start. The settings routes that write the
workspace switch and the pause are US-C07a's.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from config import config

logger = logging.getLogger(__name__)

__all__ = [
    "KEY_ENABLED",
    "KEY_MODE",
    "MODES",
    "MODE_LIVE",
    "MODE_OFF",
    "MODE_SHADOW",
    "SETTINGS_CATEGORY",
    "WORKSPACE_KEY",
    "effective_mode",
    "person_paused",
    "platform_mode",
    "read_mode",
    "workspace_enabled",
]

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_LIVE = "live"
MODES = (MODE_OFF, MODE_SHADOW, MODE_LIVE)

# The platform setting: system_settings (category, key).
SETTINGS_CATEGORY = "proactive"
KEY_MODE = "mode"
# The workspace switch: workspaces.settings[WORKSPACE_KEY][KEY_ENABLED].
WORKSPACE_KEY = "proactive"
KEY_ENABLED = "enabled"


def read_mode(value: Any) -> str:
    """A stored mode as one of ``MODES``: anything else is off."""
    mode = str(value or "").strip().lower()
    return mode if mode in MODES else MODE_OFF


def platform_mode() -> str:
    """The platform mode. No row is ``config.PROACTIVE_MODE_DEFAULT``; a read that cannot
    complete is off, and logged."""
    from core.llm.manager import read_system_setting

    try:
        stored = read_system_setting(SETTINGS_CATEGORY, KEY_MODE)
    except Exception:  # noqa: BLE001 — a read that did not complete fails closed
        logger.exception("[proactive] the %s.%s setting could not be read: check-ins are off",
                         SETTINGS_CATEGORY, KEY_MODE)
        return MODE_OFF
    return read_mode(stored if stored not in (None, "") else config.PROACTIVE_MODE_DEFAULT)


def workspace_enabled(db: Session, workspace_id: Any) -> bool:
    """Whether the workspace switched check-ins on (``settings.proactive.enabled`` is true).
    A read that fails is off, and logged; the caller's transaction is left usable."""
    from core.models.workspaces import Workspace

    try:
        with db.begin_nested():
            row = db.query(Workspace.settings).filter(Workspace.id == workspace_id).first()
    except Exception:  # noqa: BLE001 — a read that did not complete fails closed
        logger.exception("[proactive] the workspace switch of %s could not be read: off", workspace_id)
        return False
    settings = row[0] if row is not None else None
    proactive = settings.get(WORKSPACE_KEY) if isinstance(settings, dict) else None
    return isinstance(proactive, dict) and proactive.get(KEY_ENABLED) is True


def effective_mode(db: Session, workspace_id: Any) -> str:
    """The workspace's mode: the platform's where the workspace switch is on, else off."""
    mode = platform_mode()
    if mode == MODE_OFF:
        return MODE_OFF
    return mode if workspace_enabled(db, workspace_id) else MODE_OFF


def person_paused(db: Session, workspace_id: Any, user_id: int) -> bool:
    """Whether the person paused suggestions in this workspace. A read that fails is a
    pause, and logged; the caller's transaction is left usable."""
    from core.models.proactive import ProactivePersonSettings

    try:
        with db.begin_nested():
            row = db.query(ProactivePersonSettings.paused).filter(
                ProactivePersonSettings.workspace_id == workspace_id,
                ProactivePersonSettings.user_id == user_id,
            ).first()
    except Exception:  # noqa: BLE001 — a read that did not complete fails closed
        logger.exception("[proactive] the pause of user %s in %s could not be read: paused", user_id, workspace_id)
        return True
    return bool(row is not None and row[0])
