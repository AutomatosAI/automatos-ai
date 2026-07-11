"""PRD-128 US-003: Unified NotificationDispatcher.

Single entry point every caller (heartbeat, task, mission, playbook, report,
trigger, agent error) uses to raise a platform event. The dispatcher reads
``notification_preferences`` for the current workspace / user, fans the
event out to every enabled destination, and returns a summary of what was
dispatched.

Design notes:

* **Never commits.** In-app rows are inserted via ``db.execute(...)`` but
  the caller owns the transaction — notification writes roll back if the
  main work fails.
* **Multi-destination fan-out.** The preference table has no unique
  constraint on ``(workspace_id, user_id, event_type)`` so a single event
  may have multiple rows (e.g. one ``in_app`` row *and* one ``telegram``
  row). Every enabled row fires; ``silent`` rows are skipped.
* **User override.** When both a user-specific row and a workspace default
  (``user_id IS NULL``) exist for the same destination, the user-specific
  row wins. Destinations only the workspace default mentions are still
  honoured.
* **No preferences → in_app default.** If the workspace has no rows for an
  event_type at all, dispatch a single ``in_app`` notification so we never
  silently drop a completion event.
* **External delivery** is delegated to
  ``core.services.notification_service.send_workspace_notification`` which
  already knows how to talk to Telegram / Slack / webhooks.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import text

from core.services.notification_service import send_workspace_notification

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- constants

VALID_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "heartbeat_complete",
        "task_complete",
        "task_failed",
        "task_sla_breach",
        "approval_pending",
        "mission_plan_ready",
        "mission_step_complete",
        "mission_complete",
        "playbook_step_complete",
        "playbook_complete",
        "playbook_failed",
        "trigger_fired",
        "report_submitted",
        "agent_error",
    }
)

_STATUS_ICON: dict[str, str] = {
    "ok": "✅",
    "success": "✅",
    "warn": "⚠️",
    "warning": "⚠️",
    "error": "❌",
    "failed": "❌",
}

_EXTERNAL_DESTINATIONS: frozenset[str] = frozenset({"telegram", "slack", "webhook"})

_MESSAGE_TRUNCATE = 200


# ------------------------------------------------------------------- class


class NotificationDispatcher:
    """Fans a single platform event out across a workspace's preferences."""

    def __init__(self, db: Any, workspace_id: str | UUID) -> None:
        self.db = db
        # Always store as string — SQLAlchemy handles UUID binding either way
        # but tests and log lines are easier to read with a stable form.
        self.workspace_id = str(workspace_id)

    # ----------------------------------------------------------- public API

    async def dispatch(
        self,
        event_type: str,
        title: str,
        message: Optional[str] = None,
        link_type: Optional[str] = None,
        link_id: Optional[str] = None,
        agent_id: Optional[int] = None,
        agent_name: Optional[str] = None,
        status: str = "ok",
        user_id: Optional[int] = None,
        severity: Optional[str] = None,
    ) -> dict[str, Any]:
        """Raise ``event_type`` against the configured preferences.

        Wave 2: when ``workspace.settings.auto_reporting`` is configured,
        ``routes`` overrides the per-event prefs and ``quiet_hours`` forces
        non-urgent traffic onto in_app. ``security`` and ``urgent`` severities
        always go through, even during quiet hours.

        Returns a dict with a ``dispatched_to`` list naming every
        destination that was actually fired (``in_app``, ``telegram``,
        ``slack``, ``webhook``, ``channel:<uuid>``). Silent and failed
        destinations are not included.
        """
        # Wave 2 — auto_reporting overrides.
        # If settings load fails, _load_auto_reporting returns {}; treat that
        # as disabled rather than enabled, otherwise a transient load error
        # would silently engage routing logic with empty state.
        ar_settings = self._load_auto_reporting()
        ar_enabled = bool(ar_settings) and bool(ar_settings.get("enabled", False))

        override_destination: Optional[str] = None
        if ar_enabled:
            override_destination = self._auto_reporting_destination(
                ar_settings, event_type, severity
            )

        prefs = self._get_preferences(event_type, user_id)

        if override_destination:
            prefs = [
                {
                    "destination": override_destination,
                    "enabled": True,
                    "channel_connection_id": None,
                    "user_id": user_id,
                }
            ]
        elif not prefs:
            # No preferences configured at all → default to a single in_app row.
            prefs = [
                {
                    "destination": "in_app",
                    "enabled": True,
                    "channel_connection_id": None,
                    "user_id": user_id,
                }
            ]

        # Quiet hours — funnel non-urgent traffic to in_app
        if (
            ar_enabled
            and severity not in {"urgent", "security"}
            and self._is_quiet_hours(ar_settings)
        ):
            prefs = [
                {
                    "destination": "in_app",
                    "enabled": True,
                    "channel_connection_id": None,
                    "user_id": user_id,
                }
            ]

        dispatched: list[str] = []

        for pref in prefs:
            if not pref.get("enabled", True):
                continue

            destination = pref.get("destination") or "in_app"

            if destination == "silent":
                continue

            try:
                if destination == "in_app":
                    self._insert_in_app(
                        user_id=pref.get("user_id", user_id),
                        event_type=event_type,
                        title=title,
                        message=message,
                        link_type=link_type,
                        link_id=link_id,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        status=status,
                    )
                    dispatched.append("in_app")

                elif destination in _EXTERNAL_DESTINATIONS:
                    formatted = self._format_external_message(
                        title=title,
                        status=status,
                        agent_name=agent_name,
                        message=message,
                    )
                    sent = await send_workspace_notification(
                        self.workspace_id, formatted, channel=destination
                    )
                    if sent:
                        dispatched.append(destination)

                elif destination == "channel":
                    connection_id = pref.get("channel_connection_id")
                    if not connection_id:
                        logger.warning(
                            "[Dispatcher] 'channel' destination missing "
                            "channel_connection_id (ws=%s event=%s)",
                            self.workspace_id,
                            event_type,
                        )
                        continue
                    formatted = self._format_external_message(
                        title=title,
                        status=status,
                        agent_name=agent_name,
                        message=message,
                    )
                    if await self._send_to_channel_connection(
                        str(connection_id), formatted
                    ):
                        dispatched.append(f"channel:{connection_id}")

                else:
                    logger.debug(
                        "[Dispatcher] Unknown destination '%s' (ws=%s event=%s) — skipping",
                        destination,
                        self.workspace_id,
                        event_type,
                    )
            except Exception:  # noqa: BLE001 — dispatcher must never raise
                logger.error(
                    "[Dispatcher] Failed to deliver event_type=%s destination=%s ws=%s",
                    event_type,
                    destination,
                    self.workspace_id,
                    exc_info=True,
                )

        return {"dispatched_to": dispatched}

    # -------------------------------------------------------- auto_reporting

    def _load_auto_reporting(self) -> dict[str, Any]:
        """Return the workspace's effective auto_reporting settings (Wave 2)."""
        try:
            from core.services.auto_reporting import load_auto_reporting_settings
            return load_auto_reporting_settings(self.db, self.workspace_id)
        except Exception:
            logger.warning(
                "[Dispatcher] Failed to load auto_reporting for ws=%s — falling back to prefs only",
                self.workspace_id, exc_info=True,
            )
            return {}

    @staticmethod
    def _auto_reporting_destination(
        settings: dict[str, Any], event_type: str, severity: Optional[str]
    ) -> Optional[str]:
        from core.services.auto_reporting import route_for_event
        return route_for_event(settings, event_type, severity)

    @staticmethod
    def _is_quiet_hours(settings: dict[str, Any]) -> bool:
        from core.services.auto_reporting import is_quiet_hours
        return is_quiet_hours(settings)

    # -------------------------------------------------------- preferences

    def _get_preferences(
        self, event_type: str, user_id: Optional[int]
    ) -> list[dict[str, Any]]:
        """Load every matching preference row with user override semantics.

        Returns a list of dicts (one per destination) where user-specific
        rows shadow workspace-default rows on the same destination. Rows
        whose destination is unique to the workspace default survive.
        """
        result = self.db.execute(
            text(
                "SELECT user_id, destination, enabled, channel_connection_id "
                "FROM notification_preferences "
                "WHERE workspace_id = :ws_id "
                "  AND event_type = :event_type "
                "  AND (user_id = :user_id OR user_id IS NULL)"
            ),
            {
                "ws_id": self.workspace_id,
                "event_type": event_type,
                "user_id": user_id,
            },
        )
        rows = result.fetchall() if hasattr(result, "fetchall") else list(result)

        # Bucket by destination. If a user-specific row exists, drop the
        # workspace-default row for that destination.
        user_rows: dict[str, dict[str, Any]] = {}
        default_rows: dict[str, dict[str, Any]] = {}

        for row in rows:
            row_dict = self._row_to_dict(row)
            dest = row_dict.get("destination") or "in_app"
            if row_dict.get("user_id") is not None:
                user_rows[dest] = row_dict
            else:
                default_rows[dest] = row_dict

        merged: dict[str, dict[str, Any]] = {**default_rows, **user_rows}
        return list(merged.values())

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        """Normalise a SQLAlchemy Row (or tuple/dict) into a plain dict."""
        if isinstance(row, dict):
            return dict(row)
        if hasattr(row, "_mapping"):
            return dict(row._mapping)
        if hasattr(row, "_asdict"):
            return row._asdict()
        # Plain tuple fallback following the SELECT column order above.
        keys = ("user_id", "destination", "enabled", "channel_connection_id")
        return dict(zip(keys, row))

    # ----------------------------------------------------------- in-app

    def _insert_in_app(
        self,
        user_id: Optional[int],
        event_type: str,
        title: str,
        message: Optional[str],
        link_type: Optional[str],
        link_id: Optional[str],
        agent_id: Optional[int],
        agent_name: Optional[str],
        status: str,
    ) -> None:
        """Insert an in-app notification row. Does NOT commit."""
        self.db.execute(
            text(
                "INSERT INTO notifications "
                "(workspace_id, user_id, event_type, title, message, "
                " link_type, link_id, agent_id, agent_name, status) "
                "VALUES (:ws_id, :user_id, :event_type, :title, :message, "
                " :link_type, :link_id, :agent_id, :agent_name, :status)"
            ),
            {
                "ws_id": self.workspace_id,
                "user_id": user_id,
                "event_type": event_type,
                "title": title,
                "message": message,
                "link_type": link_type,
                "link_id": str(link_id) if link_id is not None else None,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "status": status,
            },
        )

    # ------------------------------------------------- external formatting

    @staticmethod
    def _format_external_message(
        title: str,
        status: str,
        agent_name: Optional[str],
        message: Optional[str],
    ) -> str:
        """Render a concise multi-line body for Telegram / Slack / webhook.

        Layout::

            {icon} {title}
            Agent: {agent_name}
            {truncated message}
            🔗 (deep link TODO)
        """
        icon = _STATUS_ICON.get((status or "ok").lower(), "ℹ️")
        lines: list[str] = [f"{icon} {title}"]
        if agent_name:
            lines.append(f"Agent: {agent_name}")
        if message:
            truncated = message.strip()
            if len(truncated) > _MESSAGE_TRUNCATE:
                truncated = truncated[: _MESSAGE_TRUNCATE - 1].rstrip() + "…"
            lines.append(truncated)
        # TODO(prd-128): append deep link once link_type → URL resolver exists.
        lines.append("🔗 (deep link pending)")
        return "\n".join(lines)

    # ------------------------------------------------- specific channel

    async def _send_to_channel_connection(
        self, connection_id: str, formatted: str
    ) -> bool:
        """Look up a specific channel_connection row and deliver via its platform."""
        row = self.db.execute(
            text(
                "SELECT platform FROM channel_connections "
                "WHERE id = :id AND workspace_id = :ws_id"
            ),
            {"id": connection_id, "ws_id": self.workspace_id},
        ).fetchone()

        if not row:
            logger.warning(
                "[Dispatcher] channel_connection %s not found for ws=%s",
                connection_id,
                self.workspace_id,
            )
            return False

        platform = row[0] if not hasattr(row, "_mapping") else row._mapping["platform"]
        if platform not in _EXTERNAL_DESTINATIONS:
            logger.warning(
                "[Dispatcher] channel_connection %s has unsupported platform '%s'",
                connection_id,
                platform,
            )
            return False

        return await send_workspace_notification(
            self.workspace_id, formatted, channel=platform
        )
