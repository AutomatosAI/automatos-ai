"""Auto reporting preferences — Wave 2.

Single canonical reader/writer for ``workspace.settings.auto_reporting`` and
the helpers the NotificationDispatcher uses to honour those preferences.

Settings shape (``workspace.settings.auto_reporting``):

    {
      "enabled": bool,
      "primary_channel":   "telegram" | "slack" | "in_app" | "webhook",
      "fallback_channel":  "in_app" | "webhook" | null,
      "quiet_hours": {
        "enabled":  bool,
        "start":    "22:00",
        "end":      "08:00",
        "timezone": "Europe/Dublin"
      },
      "digest_frequency": "immediate" | "daily" | "weekly",
      "digest_time":      "09:00",
      "routes": {
        "<event_type>": "<destination>"   # destination overrides workspace prefs
      }
    }

Defaults are conservative — when the field is absent, behaviour falls back
to the existing ``notification_preferences`` table (PRD-128).
"""

from __future__ import annotations

import logging
from datetime import datetime, time as dtime
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# Sensible defaults — workspaces inherit these unless they explicitly override.
DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "primary_channel": "in_app",
    "fallback_channel": "in_app",
    "quiet_hours": {
        "enabled": False,
        "start": "22:00",
        "end": "08:00",
        "timezone": "UTC",
    },
    "digest_frequency": "immediate",
    "digest_time": "09:00",
    "routes": {},
}


def load_auto_reporting_settings(
    db: Session, workspace_id: UUID | str
) -> Dict[str, Any]:
    """Return ``auto_reporting`` settings merged onto defaults."""
    from core.models.workspaces import Workspace

    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if ws is None:
        return dict(DEFAULTS)

    settings = ws.settings or {}
    user_cfg = settings.get("auto_reporting", {}) or {}
    return _merge_defaults(DEFAULTS, user_cfg)


def update_auto_reporting_settings(
    db: Session,
    workspace_id: UUID | str,
    partial: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge ``partial`` into ``workspace.settings.auto_reporting`` and persist.

    Caller owns the transaction — this only stages the update and flushes.
    Returns the resulting effective settings (defaults + stored override).
    """
    from core.models.workspaces import Workspace

    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if ws is None:
        raise ValueError(f"workspace {workspace_id} not found")

    settings = dict(ws.settings or {})
    current = dict(settings.get("auto_reporting") or {})
    merged = _merge_dicts(current, partial)
    settings["auto_reporting"] = merged
    ws.settings = settings
    db.flush()

    return _merge_defaults(DEFAULTS, merged)


def is_quiet_hours(
    settings: Dict[str, Any], now: Optional[datetime] = None
) -> bool:
    """Return True when ``now`` falls inside the workspace's quiet window."""
    qh = settings.get("quiet_hours") or {}
    if not qh.get("enabled"):
        return False

    tz_name = qh.get("timezone") or "UTC"
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:
        from datetime import timezone as _tz
        tz = _tz.utc

    moment = (now or datetime.now(tz)).astimezone(tz).time()
    start = _parse_hhmm(qh.get("start") or "22:00")
    end = _parse_hhmm(qh.get("end") or "08:00")

    if start == end:
        return False
    if start < end:
        return start <= moment < end
    # Window crosses midnight (e.g. 22:00 → 08:00)
    return moment >= start or moment < end


def route_for_event(
    settings: Dict[str, Any],
    event_type: str,
    severity: Optional[str] = None,
) -> Optional[str]:
    """Resolve the destination for an event from auto_reporting.routes.

    Lookup order:
      1. ``routes[f"{event_type}:{severity}"]`` — most specific
      2. ``routes[event_type]``
      3. ``routes[severity]`` — severity-only fallback
      4. None (caller falls back to workspace ``notification_preferences``)

    Special destination ``"primary"`` resolves to ``primary_channel``;
    ``"fallback"`` resolves to ``fallback_channel``.
    """
    routes = settings.get("routes") or {}
    candidates = []
    if severity:
        candidates.append(f"{event_type}:{severity}")
    candidates.append(event_type)
    if severity:
        candidates.append(severity)

    for key in candidates:
        if key in routes:
            return _resolve_alias(settings, routes[key])
    return None


# ----------------------------------------------------------------- internals


# Concrete destinations the dispatcher knows how to deliver.
_VALID_DESTINATIONS: frozenset[str] = frozenset(
    {"in_app", "telegram", "slack", "webhook", "silent"}
)
_VALID_CHANNELS: frozenset[str] = frozenset({"telegram", "slack", "in_app", "webhook"})


def _resolve_alias(settings: Dict[str, Any], destination: str) -> Optional[str]:
    """Map ``primary``/``fallback`` aliases onto the configured channels.

    Unknown destinations return ``None`` so a malformed ``routes`` entry
    falls back to the workspace's existing notification_preferences rather
    than letting an agent inject an arbitrary channel name.
    """
    if not isinstance(destination, str):
        return None
    if destination == "primary":
        primary = settings.get("primary_channel")
        return primary if primary in _VALID_CHANNELS else "in_app"
    if destination == "fallback":
        fallback = settings.get("fallback_channel")
        return fallback if fallback in _VALID_CHANNELS else "in_app"
    if destination in _VALID_DESTINATIONS:
        return destination
    return None


def _parse_hhmm(value: str) -> dtime:
    try:
        hh, mm = value.split(":", 1)
        return dtime(hour=int(hh), minute=int(mm))
    except Exception:
        logger.warning("[auto_reporting] could not parse time %r — defaulting to 00:00", value)
        return dtime(0, 0)


def _merge_defaults(defaults: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-merged copy of defaults+override (override wins)."""
    return _merge_dicts(defaults, override)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out
