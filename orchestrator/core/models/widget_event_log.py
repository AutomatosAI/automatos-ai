"""
WidgetEventLog ORM model (PRD-008-A Phase 4).

Append-only event log for the storefront widget — proactive popups,
callback requests + dispatch outcomes, cart-idle fires, settings
changes. Backs the dashboard's per-Site telemetry rollups and is the
source of truth that downstream sinks (PostHog, Amplitude) consume.

Pattern follows PRD-139 ``ToolExecutionLog``: single table, JSONB
event payload, fire-and-forget writer in
``modules/widgets/telemetry.py``.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base


# Event-type allow-list. Keeps the dashboard rollup queries tight and
# catches typos in callers. Add new event types here AS the features
# that emit them ship — silent drift breaks the dashboard.
WIDGET_EVENT_TYPES: frozenset[str] = frozenset({
    # PRD-007 — proactive engagement
    "proactive_fired",
    "proactive_dismissed",
    # PRD-008-A Feature B — callback handoff
    "callback_requested",
    "callback_delivered",
    "callback_failed",
    # PRD-008-A Feature C1 — cart-idle proactive
    "cart_idle_fired",
    "cart_idle_dismissed",
    # PRD-008-A audit
    "settings_changed",
})


class WidgetEventLog(Base):
    __tablename__ = "widget_event_log"
    __table_args__ = (
        Index("idx_widget_event_log_site_created", "site_id", "created_at"),
        Index("idx_widget_event_log_type_created", "event_type", "created_at"),
        {"extend_existing": True},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Cascades from sites — when a Site is hard-deleted (rare; usually
    # status='disconnected') its event history goes with it.
    site_id = Column(PGUUID(as_uuid=True), nullable=False)

    # Browser session — opaque string; no validation. Used to dedupe
    # callback requests and group dashboard analytics by session.
    session_id = Column(String(64), nullable=True)

    event_type = Column(String(64), nullable=False)

    # Free-form per-event payload. Keep small (<2KB typical). Schema
    # depends on event_type:
    #   callback_requested  → {phone_hash, name, product_context, ...}
    #   callback_delivered  → {destination_type, latency_ms, attempt}
    #   callback_failed     → {destination_type, error, attempt}
    #   cart_idle_fired     → {idle_seconds, cart_item_count}
    #   settings_changed    → {path, old, new, user_id}
    event_data = Column(JSONB, nullable=False, server_default="{}")

    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<WidgetEventLog id={self.id} site={self.site_id} "
            f"type={self.event_type!r}>"
        )
