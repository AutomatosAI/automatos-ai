"""PRD-222 W1·S9 — the $5 onboarding trial ledger.

The ledger has NO table of its own. A workspace's trial lives inside
``workspaces.onboarding.trial`` JSONB (``{granted_usd, spent_usd, state}``) and
the platform-wide daily spend lives as a single ``system_settings`` row keyed by
day. Spend is DERIVED from the platform's existing usage telemetry — never a
second bookkeeping system (PRD-222 D4).

This slice (US-004) is the GRANT side:

* read today's global trial spend (``get_daily_trial_spend``),
* decide whether a new Clerk user gets the one-time credit
  (``decide_trial_grant`` — a pure function),
* stamp it at provisioning (``grant_trial_at_provisioning``), one per Clerk user,
  paused by the global daily cap or the ``TRIAL_ENABLED`` kill switch.

US-005 adds the SPEND side into this same file (``resolve_trial_routing`` at the
LLM key-resolution choke point, per-request accumulation, and the
``active → warned → exhausted`` transitions), reusing ``daily_spend_key`` below
so grant and spend read/write ONE counter.

Trial states: ``active`` → ``warned`` (>=80%) → ``exhausted`` (>=100%); a
validated key save flips any of these to ``converted`` (US-006). A ``None`` /
absent trial means never-granted or paused — the client snapshot renders it as
``trial: null`` (see ``services.onboarding_state.public_snapshot``).
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Any, Optional

from sqlalchemy import text

from config import config
from core.models.system_settings import SettingCategory

logger = logging.getLogger(__name__)

# Trial states carried in workspaces.onboarding.trial.state.
TRIAL_ACTIVE = "active"
TRIAL_WARNED = "warned"
TRIAL_EXHAUSTED = "exhausted"
TRIAL_CONVERTED = "converted"

# The platform-wide daily trial-spend counter lives in the existing
# system_settings plane under the cost-audit category (no new table/plane).
DAILY_SPEND_CATEGORY: str = SettingCategory.LLM_COST_AUDIT.value  # "llm_cost_audit"


# --------------------------------------------------------------------------- #
# Daily global counter — the read side (US-005 owns the write/accumulate side).
# --------------------------------------------------------------------------- #


def daily_spend_key(day: Optional[date] = None) -> str:
    """``system_settings`` key for a day's aggregate trial spend.

    Format ``trial_spend_YYYY-MM-DD`` (UTC). US-004's grant reads it; US-005's
    per-request accumulation writes it — one key, one counter.
    """
    d = day or datetime.now(timezone.utc).date()
    return f"trial_spend_{d.isoformat()}"


def _read_setting(db: Any, category: str, key: str) -> Optional[str]:
    """Read a raw ``system_settings`` value, preferring a provided session.

    ``db`` given → query on that session (pure-testable, single transaction).
    ``db is None`` → fall through to the shared ``get_system_setting`` reader,
    which opens its own short-lived session.
    """
    if db is not None:
        row = db.execute(
            text(
                "SELECT value FROM system_settings "
                "WHERE category = :c AND key = :k LIMIT 1"
            ),
            {"c": category, "k": key},
        ).fetchone()
        return row[0] if row else None
    from core.llm.manager import get_system_setting

    return get_system_setting(category, key, None)


def get_daily_trial_spend(db: Any = None, *, day: Optional[date] = None) -> float:
    """Today's platform-wide trial spend in USD (``0.0`` when unset/unreadable)."""
    raw = _read_setting(db, DAILY_SPEND_CATEGORY, daily_spend_key(day))
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        logger.warning("Unparseable trial daily-spend value %r — treating as 0.0", raw)
        return 0.0


# --------------------------------------------------------------------------- #
# Grant decision (pure) + one-per-user check + write.
# --------------------------------------------------------------------------- #


def decide_trial_grant(
    *,
    enabled: bool,
    already_held: bool,
    daily_spend: float,
    daily_cap: float,
    credit_usd: float,
) -> tuple[Optional[dict[str, Any]], str]:
    """Pure grant decision → ``(trial | None, reason)``.

    ``reason`` is a short machine string for the log line:
    ``granted`` | ``disabled`` | ``already_held`` | ``daily_cap_reached``.
    A ``None`` trial means "do not write anything" — the workspace's onboarding
    stays trial-less and the snapshot renders ``trial: null``.
    """
    if not enabled:
        return None, "disabled"
    if already_held:
        return None, "already_held"
    if daily_spend >= daily_cap:
        return None, "daily_cap_reached"
    trial = {
        "granted_usd": round(float(credit_usd), 2),
        "spent_usd": 0,
        "state": TRIAL_ACTIVE,
    }
    return trial, "granted"


def user_has_held_trial(db: Any, owner_id: Any) -> bool:
    """True if ANY workspace owned by this user already carries a trial.

    One trial per Clerk user (PRD-222 FR-16), keyed off the actual presence of a
    ``trial`` object — a grant that was *paused* (cap/kill-switch) writes nothing,
    so that user stays eligible on a later attempt. ``db is None`` / no owner →
    ``False`` (can't check, don't block; provisioning always passes a session).
    """
    if db is None or owner_id is None:
        return False
    row = db.execute(
        text(
            "SELECT 1 FROM workspaces "
            "WHERE owner_id = :uid AND onboarding -> 'trial' IS NOT NULL "
            "LIMIT 1"
        ),
        {"uid": owner_id},
    ).fetchone()
    return row is not None


def _write_trial(db: Any, workspace_id: Any, trial: dict[str, Any]) -> None:
    """Stamp ``onboarding.trial`` via a full server-side JSONB reassignment.

    Raw SQL (matching ``_provision_new_user_workspace``'s style) using
    ``jsonb_set``, which replaces the whole value server-side — no ORM in-place
    mutation, so the PRD-220 JSONB silent-loss bug class cannot occur here.
    Does NOT commit; the caller owns the transaction.
    """
    if db is None:
        return
    db.execute(
        text(
            "UPDATE workspaces "
            "SET onboarding = jsonb_set(COALESCE(onboarding, '{}'::jsonb), "
            "'{trial}', CAST(:trial AS jsonb), true) "
            "WHERE id = :ws_id"
        ),
        {"trial": json.dumps(trial), "ws_id": str(workspace_id)},
    )


def grant_trial_at_provisioning(
    db: Any, workspace_id: Any, *, owner_id: Any
) -> Optional[dict[str, Any]]:
    """Grant the one-time trial credit for a freshly-provisioned workspace.

    Returns the trial dict when granted (caller commits), or ``None`` when
    skipped (kill switch off / user already held one / global daily cap reached),
    always logging the reason. Never raises for a decline — a pause is a normal,
    visible outcome, not an error.
    """
    already = user_has_held_trial(db, owner_id)
    daily = get_daily_trial_spend(db)
    trial, reason = decide_trial_grant(
        enabled=config.TRIAL_ENABLED,
        already_held=already,
        daily_spend=daily,
        daily_cap=config.TRIAL_GLOBAL_DAILY_USD,
        credit_usd=config.TRIAL_CREDIT_USD,
    )
    if trial is None:
        logger.info(
            "Trial grant skipped for workspace %s (owner=%s): %s "
            "(enabled=%s, daily_spend=%.2f/%.2f)",
            workspace_id, owner_id, reason,
            config.TRIAL_ENABLED, daily, config.TRIAL_GLOBAL_DAILY_USD,
        )
        return None
    _write_trial(db, workspace_id, trial)
    logger.info(
        "Trial granted for workspace %s (owner=%s): $%.2f active",
        workspace_id, owner_id, trial["granted_usd"],
    )
    return trial
