"""PRD-128 US-003: Unified NotificationDispatcher.

Single entry point every caller (heartbeat, task, mission, playbook, report,
trigger, agent error) uses to raise a platform event. The dispatcher reads
``notification_preferences`` for the current workspace / user, fans the
event out to every enabled destination, and returns a summary of what was
dispatched.

Design notes:

* **Never commits, never aborts the caller's work.** In-app rows are inserted
  via ``db.execute(...)`` on the caller's session, and the caller owns the
  transaction — notification writes roll back if the main work fails. Each
  statement the dispatcher runs there after the preferences (the in-app
  insert, a channel lookup) runs in a SAVEPOINT on the caller's connection
  (``_savepoint``): one that fails rolls back only itself and is logged, and
  the caller's transaction and its commit go on (26 Oct 2026: a too-long
  title aborted a Mission's transaction and lost the step's result). Every
  in-app row is cut to fit its columns first
  (``core.services.notification_columns``).
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
* **Push** (PRD-266 US-010d, K7) is never a preference destination: a
  ``push`` row is each person's switch, read by the fan-out. After the
  destinations, the three push types go to ``services.push.fanout.hand_over``
  with the in-app row's id; quiet hours hold them unless urgent or security.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import text

from core.services.notification_columns import fit_columns
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
        # PRD-251 Wave 3 (US-301): how a social post's publish ended.
        "social_post_published",
        "social_post_failed",
        "social_post_missed",
        # PRD-225: an agent raised a free-text question that parks its subject
        # until a human answers. Respects quiet hours unless the ask blocks a
        # large downstream cascade (the caller passes severity='urgent').
        "question_pending",
        "mission_plan_ready",
        "mission_step_complete",
        "mission_complete",
        # PRD-204 S4: the silent failure holes -- a failed mission and a
        # budget-paused mission previously reached no subscriber at all.
        "mission_failed",
        "mission_budget_paused",
        "playbook_step_complete",
        "playbook_complete",
        "playbook_failed",
        # PRD-204 S4: scheduler skipped a cron fire on an open breaker with
        # only a log line -- now user-visible, once per breaker-open period.
        "playbook_benched",
        # F132: a scheduled run that did not start (missed its grace, or the
        # workspace was at its run limit) -- it used to leave no trace at all.
        "playbook_schedule_skipped",
        # F197: the AI provider's account has credit again (the bell said it ran out).
        "credit_restored",
        # PRD-265: the workspace's Automatos credits are running low / have run out.
        "credits_low",
        "credits_exhausted",
        # PRD-265 W2: a subscription payment failed (Stripe retries), or a paid plan ended.
        "billing_payment_failed",
        "billing_subscription_ended",
        # PRD-265 W2: a platform admin offered the workspace the Business plan (by invite).
        "billing_business_offered",
        # PRD-204 S4: watcher-plane events (verdicts S5/S6, corrective
        # actions S7/S8, escalations).
        "watch_verdict",
        "watch_action",
        "watch_escalation",
        "trigger_fired",
        "report_submitted",
        # PRD-266 US-016a: a member reported an Auto message or a Deliverable; the
        # owner's notice opens Command Center -> Governance (link_type content_report).
        "content_reported",
        # PRD-266 US-012a: a ticket entered Review (core/models/review_entries.py, the
        # sweeper); in-app by default, a push type (Ready for review), linked to the ticket.
        "review_pending",
        # PRD-266 US-012b: Auto posted in a person's chat unprompted (a scheduled task, the
        # daily digest; services/notification_events/auto_message.py); in-app by default, a
        # push type (Auto messaged you), linked to the chat.
        "auto_message",
        # PRD-266 US-014a (K12): the per-person daily-digest switch, a 'push' row only
        # (services/push/preferences.py); the digest itself is posted as Auto's message and
        # pushed as auto_message, never raised as an event of its own.
        "daily_digest",
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

# PRD-266 K7: the push destination, never delivered as a row (the fan-out reads it).
_PUSH = "push"
# Severities quiet hours never hold.
_THROUGH_QUIET_HOURS: frozenset[str] = frozenset({"urgent", "security"})

# The bell's row (``_insert_in_app``); bound parameters only, its values cut to fit first.
_INSERT_IN_APP = text(
    "INSERT INTO notifications "
    "(workspace_id, user_id, event_type, title, message, "
    " link_type, link_id, agent_id, agent_name, status, created_at) "
    "VALUES (:ws_id, :user_id, :event_type, :title, :message, "
    " :link_type, :link_id, :agent_id, :agent_name, :status, clock_timestamp()) "
    "RETURNING id"
)


@dataclass(frozen=True)
class _Notice:
    """One dispatch's event, as the caller raised it."""

    event_type: str
    title: str
    message: Optional[str]
    link_type: Optional[str]
    link_id: Optional[str]
    agent_id: Optional[int]
    agent_name: Optional[str]
    status: str
    user_id: Optional[int]
    severity: Optional[str]


def _row(destination: str, user_id: Optional[int]) -> dict[str, Any]:
    """A preference row the dispatcher makes itself (a route, the default, quiet hours)."""
    return {
        "destination": destination,
        "enabled": True,
        "channel_connection_id": None,
        "user_id": user_id,
    }


def _live_destination(pref: dict[str, Any]) -> Optional[str]:
    """The row's destination, or None when the row is off or silent."""
    if not pref.get("enabled", True):
        return None
    destination = pref.get("destination") or "in_app"
    return None if destination == "silent" else destination


def _returned_id(result: Any) -> Optional[str]:
    """The id an ``INSERT … RETURNING id`` gave back, as a string; None when it gave none."""
    scalar = getattr(result, "scalar", None)
    value = scalar() if callable(scalar) else None
    return str(value) if value is not None else None


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

        PRD-266 US-010d: after the destinations, an event in
        ``services.push.payload.PUSH_EVENT_TYPES`` is handed to the push
        fan-out (outside quiet hours, or when urgent or security), carrying the
        in-app row's id.

        Returns a dict with a ``dispatched_to`` list naming every
        destination that was actually fired (``in_app``, ``telegram``,
        ``slack``, ``webhook``, ``channel:<uuid>``, ``push``). Silent and failed
        destinations are not included.
        """
        notice = _Notice(
            event_type=event_type,
            title=title,
            message=message,
            link_type=link_type,
            link_id=link_id,
            agent_id=agent_id,
            agent_name=agent_name,
            status=status,
            user_id=user_id,
            severity=severity,
        )
        held, notice = self._hold_for_credit(notice)
        if held is not None:
            return held
        prefs, quiet = self._plan(notice)
        dispatched, notification_id = await self._deliver(notice, prefs)
        if self._push(notice, notification_id, quiet):
            dispatched = [*dispatched, _PUSH]
        return {"dispatched_to": dispatched}

    # ------------------------------------------------------- dispatch steps

    def _hold_for_credit(self, notice: _Notice) -> tuple[Optional[dict[str, Any]], _Notice]:
        """F197: running out of model credit reaches the bell once per outage, in plain words.

        Returns ``(held, notice)``: ``held`` is the dispatch's result when the bell already
        says credit ran out (night 6: 17 raw 402 notices in 14 minutes), else None; the first
        notice of an outage is reworded.
        """
        from core.llm.credit import (
            CREDIT_EVENTS, OUT_OF_CREDIT, OUTAGE_NOTICE, OUTAGE_TITLE, first_notice_of_outage, is_out_of_credit,
        )

        if notice.event_type not in CREDIT_EVENTS or not is_out_of_credit(notice.message):
            return None, notice
        if not first_notice_of_outage(self.workspace_id):
            logger.info("[F197] %s held back: the bell already says credit ran out", notice.event_type)
            return {"dispatched_to": [], "held_back": OUT_OF_CREDIT}, notice
        return None, replace(notice, title=OUTAGE_TITLE, message=OUTAGE_NOTICE)

    def _plan(self, notice: _Notice) -> tuple[list[dict[str, Any]], bool]:
        """The rows to deliver, and whether quiet hours hold this event.

        Wave 2 order: auto_reporting loads first (a failed load is ``{}``, read as
        disabled, so a transient error never engages routing with empty state), then the
        preferences (the dispatch's first ``db.execute``). A route overrides them; no rows
        at all → one in_app row; quiet hours funnel non-urgent traffic to in_app. K7: a
        ``push`` row is never a delivery destination (dropped before the default, so it
        never costs the bell row); a route to ``push`` delivers the bell row.
        """
        ar_settings = self._load_auto_reporting()
        ar_enabled = bool(ar_settings) and bool(ar_settings.get("enabled", False))
        override: Optional[str] = None
        if ar_enabled:
            override = self._auto_reporting_destination(ar_settings, notice.event_type, notice.severity)
        prefs = [
            pref for pref in self._get_preferences(notice.event_type, notice.user_id)
            if (pref.get("destination") or "in_app") != _PUSH
        ]
        quiet = (
            ar_enabled
            and notice.severity not in _THROUGH_QUIET_HOURS
            and self._is_quiet_hours(ar_settings)
        )
        if quiet:
            return [_row("in_app", notice.user_id)], True
        if override:
            return [_row("in_app" if override == _PUSH else override, notice.user_id)], False
        return prefs or [_row("in_app", notice.user_id)], False

    async def _deliver(
        self, notice: _Notice, prefs: list[dict[str, Any]]
    ) -> tuple[list[str], Optional[str]]:
        """Fire every enabled, non-silent row; a row's failure is logged, never raised.

        A failed in-app row was written in a savepoint (``_insert_in_app``), so it is taken
        back alone: the caller's transaction stays usable. Returns the destinations fired and
        the in-app row's id (the first written, or None when none was).
        """
        dispatched: list[str] = []
        notification_id: Optional[str] = None
        for pref in prefs:
            destination = _live_destination(pref)
            if destination is None:
                continue
            try:
                fired, row_id = await self._fire(notice, pref, destination)
            except Exception:  # the dispatcher never raises
                logger.exception(
                    "[Dispatcher] Failed to deliver event_type=%s destination=%s ws=%s",
                    notice.event_type,
                    destination,
                    self.workspace_id,
                )
                continue
            if fired:
                dispatched.append(fired)
            notification_id = notification_id or row_id
        return dispatched, notification_id

    async def _fire(
        self, notice: _Notice, pref: dict[str, Any], destination: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Deliver one row: ``(what fired or None, the in-app row's id or None)``."""
        if destination == "in_app":
            row_id = self._insert_in_app(
                user_id=pref.get("user_id", notice.user_id),
                event_type=notice.event_type,
                title=notice.title,
                message=notice.message,
                link_type=notice.link_type,
                link_id=notice.link_id,
                agent_id=notice.agent_id,
                agent_name=notice.agent_name,
                status=notice.status,
            )
            return "in_app", row_id
        if destination in _EXTERNAL_DESTINATIONS:
            sent = await send_workspace_notification(
                self.workspace_id, self._external_body(notice), channel=destination
            )
            return (destination if sent else None), None
        if destination == "channel":
            return await self._fire_channel(notice, pref), None
        logger.debug(
            "[Dispatcher] Unknown destination '%s' (ws=%s event=%s) — skipping",
            destination,
            self.workspace_id,
            notice.event_type,
        )
        return None, None

    async def _fire_channel(self, notice: _Notice, pref: dict[str, Any]) -> Optional[str]:
        """Deliver a 'channel' row through its channel_connection; ``channel:<id>`` when sent."""
        connection_id = pref.get("channel_connection_id")
        if not connection_id:
            logger.warning(
                "[Dispatcher] 'channel' destination missing "
                "channel_connection_id (ws=%s event=%s)",
                self.workspace_id,
                notice.event_type,
            )
            return None
        if await self._send_to_channel_connection(str(connection_id), self._external_body(notice)):
            return f"channel:{connection_id}"
        return None

    def _external_body(self, notice: _Notice) -> str:
        """The Telegram / Slack / webhook / channel text for this notice."""
        return self._format_external_message(
            title=notice.title,
            status=notice.status,
            agent_name=notice.agent_name,
            message=notice.message,
        )

    def _push(self, notice: _Notice, notification_id: Optional[str], quiet: bool) -> bool:
        """PRD-266 US-010d: hand a push event to the fan-out (K6); never raises.

        Only ``PUSH_EVENT_TYPES`` reach ``hand_over`` (FR-7), and none while quiet hours
        hold the event (``quiet`` already lets urgent and security through). True when
        the fan-out took it. ``notification_id`` is None when no in-app row was written (none
        was asked for, or its insert failed and was rolled back to its savepoint): the push
        still goes, carrying its link but no id.
        """
        try:
            from services.push.payload import PUSH_EVENT_TYPES

            if quiet or notice.event_type not in PUSH_EVENT_TYPES:
                return False
            from services.push.fanout import PushEvent, hand_over

            return bool(hand_over(self.db, PushEvent(
                workspace_id=self.workspace_id,
                event_type=notice.event_type,
                notification_id=notification_id,
                link_type=notice.link_type,
                link_id=str(notice.link_id) if notice.link_id is not None else None,
                user_id=notice.user_id,
                agent_name=notice.agent_name,
                severity=notice.severity,
            )))
        except Exception:  # the dispatcher never raises
            logger.exception(
                "[Dispatcher] Push hand-over failed for event_type=%s ws=%s",
                notice.event_type,
                self.workspace_id,
            )
            return False

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

    def _savepoint(self) -> Any:
        """A SAVEPOINT on the caller's connection: a statement that fails in it is rolled back
        alone, and the caller's transaction stays usable.

        Taken on the connection, not with ``Session.begin_nested()``: that flushes the caller's
        pending ORM changes first (``SessionLocal`` runs with autoflush off), so a notice would
        change when the caller's own work is written, and a failure in that flush would surface
        here, logged as a failed notice.
        """
        return self.db.connection().begin_nested()

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
    ) -> Optional[str]:
        """Insert an in-app notification row and return its id (None when none came back).

        Does NOT commit. PRD-266 US-010d: the id rides in the push payload.

        Every bounded column's value is cut to fit it (``fit_columns``: a long title ends in
        '…'), and the insert runs in a SAVEPOINT (``_savepoint``): when it fails, only the row
        is rolled back and the error is raised to ``_deliver``, which logs it; the caller's
        transaction and its commit go on.

        F209: its time is when it is written (clock_timestamp()). The column's
        default, now(), is the transaction's START: a run's task_complete,
        written minutes into a long transaction, sorted before things that
        happened earlier and read "N min ago" from when the run began."""
        params = fit_columns({
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
        })
        with self._savepoint():
            return _returned_id(self.db.execute(_INSERT_IN_APP, params))

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
        """Look up a specific channel_connection row and deliver via its platform.

        The lookup runs in a savepoint (``_savepoint``: a failed read never aborts the caller's
        transaction), and before the send: no savepoint is held across an await.
        """
        with self._savepoint():
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
