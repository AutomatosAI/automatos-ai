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
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Optional

from sqlalchemy import text

from config import config
from core.models.system_settings import SettingCategory
from services.onboarding_state import get_onboarding

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


# =========================================================================== #
# US-005 — the SPEND side: enforcement at the LLM key-resolution choke point,
# per-request accrual, and the "no background burn" predicate.
# =========================================================================== #

# Stable, frontend-facing error code for an exhausted trial (US-014 renders it
# deterministically). Kept as a constant so both the raiser and any consumer
# reference one string.
TRIAL_EXHAUSTED_CODE = "trial_exhausted"

# Fraction of the grant at which state flips active -> warned (Auto warns the
# user); at 1.0 it flips to exhausted and requests are blocked.
WARN_THRESHOLD = 0.80

# resolve_trial_routing action verbs.
ACTION_BYOK = "byok"                 # workspace has its own key — zero trial involvement
ACTION_PLATFORM_TRIAL = "platform_trial"  # route to platform key, pin model, accrue
ACTION_BLOCKED = "blocked"           # exhausted — raise the typed error
ACTION_PASSTHROUGH = "passthrough"   # no active trial — existing resolution unchanged

# The three non-converted trial states that constitute "on the trial" — the set
# that must NOT accrue background/scheduled burn (converted + never-granted run).
_ON_TRIAL_STATES = (TRIAL_ACTIVE, TRIAL_WARNED, TRIAL_EXHAUSTED)


class TrialExhaustedError(Exception):
    """Raised at the LLM key-resolution choke point when a trial workspace has
    spent its credit. Carries a STABLE ``error_code`` the frontend can render
    deterministically (US-014); the workspace's ``trial.state`` is also flipped
    to ``exhausted`` on the request that trips it, so ``GET /api/workspaces/current``
    (US-002 snapshot) reflects it independently of this in-flight error.
    """

    error_code = TRIAL_EXHAUSTED_CODE

    def __init__(self, message: str = "Your trial credit is used up — add a provider key to keep Auto running."):
        super().__init__(message)
        self.message = message


@dataclass
class TrialRouting:
    """The routing verdict at the key-resolution seam (see ``resolve_trial_routing``)."""

    action: str
    model: Optional[str] = None
    error_code: Optional[str] = None
    reason: str = ""


def _trial_of(workspace: Any) -> Optional[dict[str, Any]]:
    if workspace is None:
        return None
    return (get_onboarding(workspace) or {}).get("trial")


def is_trial_active_workspace(workspace: Any) -> bool:
    """True when the workspace holds a NON-converted trial (active/warned/exhausted).

    This is the "no background burn" set (PRD-222 FR-17): heartbeats and scheduled
    execution must skip these until the trial converts. Converted (paid) and
    never-granted workspaces return ``False`` and run normally.
    """
    trial = _trial_of(workspace)
    return bool(trial) and trial.get("state") in _ON_TRIAL_STATES


def resolve_trial_routing(
    workspace: Any, requested_model: str, *, is_byok: bool
) -> TrialRouting:
    """The trial gate at the LLM key-resolution choke point (#610 seam).

    PURE — reads ``workspace.onboarding.trial`` + config only, no DB writes:

    1. ``is_byok`` (the workspace resolved its OWN key) → ``byok``: zero trial
       involvement. This is the provable bypass (AC2).
    2. no active/warned/exhausted trial (never-granted or ``converted``) →
       ``passthrough``: existing resolution is left exactly as today.
    3. trial ``exhausted`` → ``blocked`` with ``error_code='trial_exhausted'``.
    4. trial ``active``/``warned`` → ``platform_trial``: the request pins to the
       requested model passes through unchanged — the trial is a spend cap,
       not a model gate (2026-08-29).
       never dead-ends on model choice).
    """
    if is_byok:
        return TrialRouting(ACTION_BYOK, model=requested_model, reason="byok")

    trial = _trial_of(workspace)
    state = (trial or {}).get("state")
    if not trial or state not in _ON_TRIAL_STATES:
        return TrialRouting(ACTION_PASSTHROUGH, model=requested_model, reason="no_active_trial")
    if state == TRIAL_EXHAUSTED:
        return TrialRouting(
            ACTION_BLOCKED, error_code=TRIAL_EXHAUSTED_CODE, reason="trial_exhausted"
        )

    # 2026-08-29 (Gerard): the trial is a SPEND CAP, not a model gate — "$5 is
    # $5 no matter what model they choose." Model pinning + TRIAL_MODEL_ALLOWLIST
    # deleted; an expensive model simply exhausts the credit sooner. Metering,
    # the 80% warn, the hard stop, and the global daily cap are the controls.
    return TrialRouting(ACTION_PLATFORM_TRIAL, model=requested_model, reason="trial_active")


def compute_trial_state(spent_usd: float, granted_usd: float, *, warn_ratio: float = WARN_THRESHOLD) -> str:
    """Pure spend→state map: >=100% exhausted, >=warn_ratio warned, else active.

    ``granted<=0`` is treated as exhausted (nothing to spend).
    """
    if granted_usd <= 0:
        return TRIAL_EXHAUSTED
    ratio = spent_usd / granted_usd
    if ratio >= 1.0:
        return TRIAL_EXHAUSTED
    if ratio >= warn_ratio:
        return TRIAL_WARNED
    return TRIAL_ACTIVE


def _price_request(db: Any, model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost for one request, via the platform's SINGLE cost source.

    Reuses ``modules.policy.pricing`` (the registry-backed calc that already
    replaced four hardcoded price maps) — never a second pricing table. Falls
    back to the module's documented flat last-resort when a model is unpriceable
    so a trial ALWAYS accrues (and therefore always eventually exhausts).
    """
    from modules.policy.pricing import estimate_cost_usd, price_total_tokens_usd

    in_tok = int(input_tokens or 0)
    out_tok = int(output_tokens or 0)
    cost = estimate_cost_usd(db, model_id, in_tok, out_tok)
    if cost is None:
        cost = price_total_tokens_usd(db, model_id, in_tok + out_tok)
    return round(float(cost or 0.0), 6)


def _increment_daily_spend(db: Any, cost_usd: float, *, day: Optional[date] = None) -> float:
    """Add ``cost_usd`` to today's platform-wide trial counter (the WRITE side).

    Read-then-update-or-insert on the ``system_settings`` row US-004 reads — no
    reliance on a (category,key) unique constraint (none is declared). The tiny
    concurrent-insert race is acceptable for a daily aggregate; the per-workspace
    cap is the hard per-user backstop.
    """
    key = daily_spend_key(day)
    new_total = round(get_daily_trial_spend(db, day=day) + cost_usd, 6)
    row = db.execute(
        text("SELECT id FROM system_settings WHERE category = :c AND key = :k LIMIT 1"),
        {"c": DAILY_SPEND_CATEGORY, "k": key},
    ).fetchone()
    if row:
        db.execute(
            text("UPDATE system_settings SET value = :v, updated_at = now() WHERE id = :id"),
            {"v": str(new_total), "id": row[0]},
        )
    else:
        db.execute(
            text(
                "INSERT INTO system_settings (category, key, value, value_type, description) "
                "VALUES (:c, :k, :v, 'number', :d)"
            ),
            {"c": DAILY_SPEND_CATEGORY, "k": key, "v": str(new_total), "d": "PRD-222 trial daily spend"},
        )
    return new_total


def record_trial_spend(
    workspace_id: Any,
    *,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    db: Any = None,
) -> Optional[str]:
    """Accrue one trial request's cost into the workspace trial + the daily counter.

    Called from the usage-tracking seam (``LLMManager._track_usage``) ONLY for
    platform-trial requests. Prices via :func:`_price_request` (the single cost
    source), rebuilds ``onboarding.trial`` with a NEW dict (never mutates — jsonb_set
    server-side), flips ``active → warned(>=80%) → exhausted(>=100%)``, and
    increments the daily counter. Opens its own session when ``db is None``
    (mirroring ``UsageTracker.track``) so it never touches the caller's
    transaction. NEVER raises. Returns the new state, or ``None`` when nothing
    accrued (non-trial / zero-cost / workspace gone).
    """
    own = db is None
    try:
        if own:
            from core.database.database import SessionLocal

            db = SessionLocal()
        try:
            cost = _price_request(db, model_id, input_tokens, output_tokens)
            if cost <= 0:
                return None

            from core.models.workspaces import Workspace

            ws = db.query(Workspace).get(workspace_id)
            trial = _trial_of(ws)
            # Only accrue on a live (active/warned) trial. exhausted is already
            # blocked upstream; converted/none are not on the trial.
            if not trial or trial.get("state") not in (TRIAL_ACTIVE, TRIAL_WARNED):
                return None

            granted = float(trial.get("granted_usd") or 0.0)
            spent = round(float(trial.get("spent_usd") or 0.0) + cost, 6)
            new_state = compute_trial_state(spent, granted)
            new_trial = {**trial, "spent_usd": spent, "state": new_state}

            _write_trial(db, workspace_id, new_trial)
            _increment_daily_spend(db, cost)
            db.commit()

            if new_state != trial.get("state"):
                logger.info(
                    "Trial %s: %s -> %s ($%.4f / $%.2f)",
                    workspace_id, trial.get("state"), new_state, spent, granted,
                )
            return new_state
        finally:
            if own:
                db.close()
    except Exception:
        logger.warning(
            "Trial spend accrual failed for workspace %s — non-fatal", workspace_id,
            exc_info=True,
        )
        if own and db is not None:
            try:
                db.rollback()
            except Exception:
                pass
        return None


# =========================================================================== #
# US-006 — the CONVERT side: a validated BYOK key save flips the trial.
# =========================================================================== #


def mark_trial_converted(db: Any, workspace_id: Any) -> bool:
    """Flip an on-trial workspace to ``converted`` after a validated key save.

    Called from the BYOK key-save seam (``api.user_api_keys.add_api_key``, US-006)
    once a LIVE provider test passes. Only an ON-trial workspace
    (active/warned/exhausted) converts; a ``converted`` or never-granted workspace
    is a no-op — so re-saving a key never rewrites history and a paying customer
    is never dragged back onto the ledger. Rebuilds ``onboarding.trial`` with a
    NEW dict via ``_write_trial`` (jsonb_set, server-side — PRD-220-safe) and does
    NOT commit; the caller (the key-save handler) owns the transaction. Returns
    ``True`` when a conversion was written, ``False`` otherwise.
    """
    if db is None or workspace_id is None:
        return False
    from core.models.workspaces import Workspace

    ws = db.query(Workspace).get(workspace_id)
    trial = _trial_of(ws)
    if not trial or trial.get("state") not in _ON_TRIAL_STATES:
        return False
    new_trial = {**trial, "state": TRIAL_CONVERTED}
    _write_trial(db, workspace_id, new_trial)
    logger.info(
        "Trial converted for workspace %s (%s -> converted, validated key saved)",
        workspace_id, trial.get("state"),
    )
    return True
