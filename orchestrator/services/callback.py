"""
Callback service (PRD-008-A Feature B).

Pure-functional helpers backing the ``POST /api/widgets/callback``
endpoint. Validation, GDPR-friendly phone hashing, idempotency
deduplication, rate limiting, and SLA-phrase synthesis live here so
the endpoint stays a thin HTTP wrapper.

Notes
-----
- Phone numbers are NEVER persisted in Automatos. Only a salted hash
  is written to ``widget_event_log`` to support 5-minute idempotency.
  The plaintext phone is forwarded to the merchant's destinations only.
- Working-hours computation reads ``site.settings.callback`` — never
  hardcodes anything per-merchant.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Optional
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# E.164: leading '+' followed by 8-15 digits. Strict to keep dispatcher
# downstream sane — "01234" without country code is rejected.
PHONE_E164_REGEX = re.compile(r"^\+[1-9]\d{7,14}$")

# Idempotency window — back-to-back duplicate submissions in this window
# are deduplicated to a single dispatch.
IDEMPOTENCY_WINDOW = timedelta(minutes=5)

# Rate limits — small + safe defaults. Per-Site is the only one
# currently configurable; per-IP requires Redis (deferred).
DEFAULT_PER_SESSION_COOLDOWN = timedelta(seconds=60)
DEFAULT_PER_SITE_HOURLY_CAP = 100

# Days of week as JSON keys in site.settings.callback.working_hours.
DOW_KEYS: tuple[str, ...] = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def normalise_phone(raw: str) -> str:
    """Strip whitespace, keep '+' and digits. Returns the normalised form
    suitable for E.164 validation."""
    if raw is None:
        return ""
    cleaned = "".join(ch for ch in raw if ch.isdigit() or ch == "+")
    return cleaned


def is_valid_phone(phone: str) -> bool:
    return bool(PHONE_E164_REGEX.match(phone))


# ---------------------------------------------------------------------------
# GDPR phone hashing
# ---------------------------------------------------------------------------

def compute_phone_hash(phone: str, site_id: UUID) -> str:
    """SHA-256 of phone + site_id as salt. Deterministic per Site so we
    can use it for idempotency lookups; not reversible (and rainbow-tables
    over the small phone space need a per-Site salt to be meaningful).

    Truncated to 32 hex chars (128 bits) — collision space is enormous
    for a 5-minute dedup window.
    """
    norm = normalise_phone(phone)
    digest = hashlib.sha256(f"{norm}|{site_id}".encode("utf-8")).hexdigest()
    return digest[:32]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def find_recent_duplicate(
    db: Session,
    *,
    site_id: UUID,
    session_id: str,
    phone_hash: str,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """Look up a ``callback_requested`` event in the last 5 minutes
    matching the same session_id + phone_hash. Returns its request_id
    if found, else None.
    """
    from core.models.widget_event_log import WidgetEventLog

    now = now or datetime.now(timezone.utc)
    cutoff = now - IDEMPOTENCY_WINDOW

    candidates = (
        db.query(WidgetEventLog)
        .filter(
            WidgetEventLog.site_id == site_id,
            WidgetEventLog.event_type == "callback_requested",
            WidgetEventLog.session_id == session_id,
            WidgetEventLog.created_at >= cutoff,
        )
        .all()
    )
    for row in candidates:
        data = row.event_data or {}
        if data.get("phone_hash") == phone_hash:
            return data.get("request_id")
    return None


# ---------------------------------------------------------------------------
# Rate limiting (per-session + per-Site, IP deferred until Redis)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    reason: Optional[str] = None
    retry_after_seconds: Optional[int] = None


def check_rate_limits(
    db: Session,
    *,
    site_id: UUID,
    session_id: str,
    per_site_hourly_cap: int = DEFAULT_PER_SITE_HOURLY_CAP,
    now: Optional[datetime] = None,
) -> RateLimitDecision:
    """Per-session cooldown (60s) + per-Site hourly cap.

    Per-IP rate limit is deferred until Redis-backed atomic counters
    land (need atomicity to prevent races; SQL-only is too racy).
    """
    from core.models.widget_event_log import WidgetEventLog

    now = now or datetime.now(timezone.utc)

    # Per-session cooldown
    session_cutoff = now - DEFAULT_PER_SESSION_COOLDOWN
    recent_session_count = (
        db.query(WidgetEventLog)
        .filter(
            WidgetEventLog.site_id == site_id,
            WidgetEventLog.event_type == "callback_requested",
            WidgetEventLog.session_id == session_id,
            WidgetEventLog.created_at >= session_cutoff,
        )
        .count()
    )
    if recent_session_count > 0:
        return RateLimitDecision(
            allowed=False,
            reason="per_session_cooldown",
            retry_after_seconds=int(DEFAULT_PER_SESSION_COOLDOWN.total_seconds()),
        )

    # Per-Site hourly cap
    site_cutoff = now - timedelta(hours=1)
    site_hourly_count = (
        db.query(WidgetEventLog)
        .filter(
            WidgetEventLog.site_id == site_id,
            WidgetEventLog.event_type == "callback_requested",
            WidgetEventLog.created_at >= site_cutoff,
        )
        .count()
    )
    if site_hourly_count >= per_site_hourly_cap:
        return RateLimitDecision(
            allowed=False,
            reason="per_site_hourly_cap",
            retry_after_seconds=3600,
        )

    return RateLimitDecision(allowed=True)


# ---------------------------------------------------------------------------
# SLA phrase synthesis
# ---------------------------------------------------------------------------

def _parse_hhmm(s: str) -> time:
    """Parse "HH:MM" → time. Caller guards against bad input."""
    h, m = s.split(":")
    return time(int(h), int(m))


def _is_within_working_hours(
    now_local: datetime,
    working_hours: dict,
) -> bool:
    """``working_hours`` is the per-day dict from site.settings.callback.
    Each weekday key is either 'closed' or {'start': 'HH:MM', 'end': 'HH:MM'}.
    """
    dow_key = DOW_KEYS[now_local.weekday()]
    spec = working_hours.get(dow_key)
    if spec is None or spec == "closed":
        return False
    try:
        start = _parse_hhmm(spec["start"])
        end = _parse_hhmm(spec["end"])
    except (KeyError, TypeError, ValueError):
        return False
    return start <= now_local.time() <= end


def compute_eta_phrase(
    callback_settings: dict,
    *,
    product_context: Optional[str] = None,
    now: Optional[datetime] = None,
) -> str:
    """Generate the user-facing reassurance line based on the merchant's
    callback config + the current time.

    Honours:
    - working_hours_only: if True and outside hours, uses outside-hours phrasing
    - team_capacity: 'limited' softens "we'll" → "we'll aim to"
    - sla_hours: numeric cap surfaced in the phrase

    Never raises — falls back to a safe generic phrase on any config issue.
    """
    try:
        sla_hours = int(callback_settings.get("sla_hours", 4))
        capacity = callback_settings.get("team_capacity", "limited")
        working_hours_only = bool(callback_settings.get("working_hours_only", True))
        working_hours = callback_settings.get("working_hours", {})
        tz_name = working_hours.get("tz", "UTC")

        try:
            tz = ZoneInfo(tz_name)
        except Exception:  # noqa: BLE001
            tz = timezone.utc

        now_utc = now or datetime.now(timezone.utc)
        now_local = now_utc.astimezone(tz)

        is_in_hours = (
            _is_within_working_hours(now_local, working_hours) if working_hours else True
        )

        verb = "aim to call" if capacity == "limited" else "call"
        product_clause = f" about the {product_context}" if product_context else ""

        if working_hours_only and not is_in_hours:
            return (
                f"We're closed right now — we'll {verb} you "
                f"first thing the next working day{product_clause}."
            )

        return (
            f"We'll {verb} you within {sla_hours} working hours{product_clause}."
        )
    except Exception:  # noqa: BLE001 — phrasing must never fail the request
        return "Thanks — we'll be in touch."


# ---------------------------------------------------------------------------
# Request ID
# ---------------------------------------------------------------------------

def new_request_id() -> str:
    """Short opaque request_id surfaced to the widget for idempotent retry."""
    return f"cb_{secrets.token_urlsafe(12)}"
