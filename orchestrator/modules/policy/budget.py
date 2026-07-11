"""Policy plane — pre-call budget admission (PRD-174 F086 + F059).

Today no ``BudgetExceeded`` exists and no loop checks spend *before* issuing a
call — the LLM manager only cost-logs *after*. This adds the admission gate the
``on_pre_tool`` seam calls: it reads the workspace's cost/token budget from
``Workspace.plan_limits`` and compares it to spend-to-date (summed from
``llm_usage``, which the manager already writes model-aware dollars into). Over
budget → :class:`BudgetExceeded`, surfaced to the model as errors-as-data.

Budget lives on ``plan_limits`` (JSONB, already present — holds concurrency
caps) under a ``budget`` key, so no new column:

    workspace.plan_limits = {
        "max_agents": 10, ...,                # existing concurrency caps
        "budget": {                            # NEW (this PRD) — all optional
            "max_cost_usd": 25.0,             # spend ceiling for the window
            "max_total_tokens": 5_000_000,    # token ceiling for the window
            "window": "day" | "month" | "all" # rolling window (default "day")
        }
    }

No budget key ⇒ no ceiling ⇒ the gate is inert (allow) — EXCEPT autonomy-enabled
workspaces, which get a default monthly ceiling (PRD-192 S3, locked:
``config.AUTONOMY_DEFAULT_BUDGET_USD`` = 50/month; explicit budgets always
win; no migration). Pricing for the pending call is model-aware via
:mod:`modules.policy.pricing`. SQLAlchemy is imported lazily so this module
loads in the stdlib-only unit-test env.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_VALID_WINDOWS = frozenset({"day", "month", "all"})
_DEFAULT_WINDOW = "day"


def _enforcement_active() -> bool:
    """Guarded read of the enforce-stage flag — never raises (PRD-192 S1)."""
    try:
        from modules.policy.flag import enforcement_active

        return enforcement_active()
    except Exception:
        return False


class BudgetExceeded(Exception):
    """Raised (or returned as a deny) when a call would breach the workspace budget.

    Carries the numbers so the caller can build errors-as-data the model reads.
    """

    def __init__(
        self,
        *,
        reason: str,
        limit: float,
        spent: float,
        projected: float,
        dimension: str,
    ) -> None:
        self.reason = reason
        self.limit = limit
        self.spent = spent
        self.projected = projected
        self.dimension = dimension  # "cost_usd" | "total_tokens"
        super().__init__(reason)


@dataclass(frozen=True)
class BudgetDecision:
    """Outcome of a budget admission check."""

    allowed: bool
    reason: str
    dimension: Optional[str] = None  # which ceiling bound (cost/tokens), if any
    limit: Optional[float] = None
    spent: Optional[float] = None
    projected: Optional[float] = None

    def audit_snapshot(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "dimension": self.dimension,
            "limit": self.limit,
            "spent": round(self.spent, 6) if self.spent is not None else None,
            "projected": round(self.projected, 6) if self.projected is not None else None,
        }


def load_budget(db: Any, workspace_id: Any) -> Dict[str, Any]:
    """Return the workspace's ``plan_limits.budget`` merged onto defaults.

    Empty/missing ⇒ ``{}`` (no ceiling) — except autonomy-enabled workspaces,
    which get the PRD-192 S3 code-default monthly ceiling (explicit budgets
    always win). Fail posture on an unreadable row: ``{}`` in off/shadow (the
    gate is a *cost* control and must not wedge execution), re-raise under the
    enforce stages so the gate's single except owns the posture (PRD-192 S1).
    """
    if db is None or workspace_id is None:
        return {}
    try:
        from core.models.workspaces import Workspace

        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if ws is None:
            return {}
        budget = (ws.plan_limits or {}).get("budget") or {}
        if not isinstance(budget, dict):
            return {}
        if not budget:
            # PRD-192 S3 (locked #2a): no explicit budget — autonomy-enabled
            # workspaces get the code-default monthly ceiling; everyone else
            # stays ceiling-less exactly as before.
            return _default_autonomy_budget(db, workspace_id)
        window = budget.get("window")
        if window not in _VALID_WINDOWS:
            window = _DEFAULT_WINDOW
        out: Dict[str, Any] = {"window": window}
        if isinstance(budget.get("max_cost_usd"), (int, float)):
            out["max_cost_usd"] = float(budget["max_cost_usd"])
        if isinstance(budget.get("max_total_tokens"), (int, float)):
            out["max_total_tokens"] = int(budget["max_total_tokens"])
        return out
    except Exception:
        logger.warning(
            "[policy.budget] budget read failed for workspace=%s", workspace_id,
            exc_info=True,
        )
        # PRD-192 S1: under an enforce stage a budget-read fault must not
        # silently decide "no ceiling ⇒ allow" — re-raise so the gate's single
        # except owns the fail posture. off/shadow keep the historical swallow.
        if _enforcement_active():
            raise
        return {}


def _default_autonomy_budget(db: Any, workspace_id: Any) -> Dict[str, Any]:
    """The code-default ceiling for autonomy-enabled workspaces (PRD-192 S3).

    Locked decision #2a: a workspace dialled to full autonomy that has NO
    explicit ``plan_limits.budget`` gets ``max_cost_usd =
    config.AUTONOMY_DEFAULT_BUDGET_USD`` per ``month`` — autonomous spend is
    never unbounded by omission. Explicit budgets always win (the caller only
    reaches here when none is set); supervised workspaces stay ceiling-less as
    today. Fail-safe: an unreadable autonomy dial or config ⇒ no default
    (``{}``), never a surprise ceiling.
    """
    try:
        from core.services.auto_autonomy import is_full_autonomy

        if not is_full_autonomy(db, workspace_id):
            return {}
        from config import config

        ceiling = float(config.AUTONOMY_DEFAULT_BUDGET_USD)
        if ceiling <= 0:
            return {}
        return {"window": "month", "max_cost_usd": ceiling, "default_applied": True}
    except Exception:
        logger.warning(
            "[policy.budget] autonomy default-ceiling read failed for "
            "workspace=%s — no default applied", workspace_id, exc_info=True,
        )
        return {}


def set_budget(
    db: Any,
    workspace_id: Any,
    *,
    max_cost_usd: Optional[float] = None,
    max_total_tokens: Optional[int] = None,
    window: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist the workspace budget to ``plan_limits.budget`` (PRD-196 S4).

    The writer beside :func:`load_budget`, mirroring ``set_policy_document``'s
    ``flag_modified`` JSONB pattern (no new column — budget lives under the
    existing ``plan_limits`` JSONB). Validates ONLY the documented keys; an
    invalid value raises ``ValueError`` (the API surfaces it as 422). Only
    provided fields change. Caller owns the transaction (flushes here).
    """
    from core.models.workspaces import Workspace
    from sqlalchemy.orm.attributes import flag_modified

    if max_cost_usd is not None:
        if isinstance(max_cost_usd, bool) or not isinstance(max_cost_usd, (int, float)) or max_cost_usd < 0:
            raise ValueError("max_cost_usd must be a number >= 0")
    if max_total_tokens is not None:
        if isinstance(max_total_tokens, bool) or not isinstance(max_total_tokens, int) or max_total_tokens < 0:
            raise ValueError("max_total_tokens must be an integer >= 0")
    if window is not None and window not in _VALID_WINDOWS:
        raise ValueError(f"window must be one of {sorted(_VALID_WINDOWS)}")

    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if ws is None:
        raise ValueError(f"workspace {workspace_id} not found")

    plan_limits = dict(ws.plan_limits or {})
    budget = dict(plan_limits.get("budget") or {})
    if max_cost_usd is not None:
        budget["max_cost_usd"] = float(max_cost_usd)
    if max_total_tokens is not None:
        budget["max_total_tokens"] = int(max_total_tokens)
    if window is not None:
        budget["window"] = window
    plan_limits["budget"] = budget
    ws.plan_limits = plan_limits
    flag_modified(ws, "plan_limits")
    db.flush()
    return load_budget(db, workspace_id)


def _window_start(window: str, now: Optional[datetime] = None) -> Optional[datetime]:
    now = now or datetime.now(timezone.utc)
    if window == "day":
        return now - timedelta(days=1)
    if window == "month":
        return now - timedelta(days=30)
    return None  # "all" — no lower bound


def spend_to_date(db: Any, workspace_id: Any, window: str) -> Dict[str, float]:
    """Sum this workspace's spend + tokens over the window from ``llm_usage``.

    Returns ``{"cost_usd": float, "total_tokens": float}``. Fail-safe: a read
    error returns zeros — an unreadable ledger must not block work, only the
    ceilings do, and they only bind when a spend number actually clears them.
    """
    zero = {"cost_usd": 0.0, "total_tokens": 0.0}
    if db is None or workspace_id is None:
        return dict(zero)
    try:
        from sqlalchemy import func as _func
        from core.models.core import LLMUsage

        start = _window_start(window)
        q = db.query(
            _func.coalesce(_func.sum(LLMUsage.total_cost), 0.0),
            _func.coalesce(_func.sum(LLMUsage.total_tokens), 0.0),
        ).filter(LLMUsage.workspace_id == workspace_id)
        if start is not None:
            q = q.filter(LLMUsage.created_at >= start)
        cost, tokens = q.one()
        return {"cost_usd": float(cost or 0.0), "total_tokens": float(tokens or 0.0)}
    except Exception:
        logger.warning(
            "[policy.budget] spend-to-date read failed for workspace=%s", workspace_id,
            exc_info=True,
        )
        # PRD-192 S1: with a ceiling configured, zeros-on-fault silently allow.
        # Enforce stages re-raise so the gate's except owns the posture.
        if _enforcement_active():
            raise
        return dict(zero)


def check_budget(
    db: Any,
    workspace_id: Any,
    *,
    projected_cost_usd: float = 0.0,
    projected_tokens: int = 0,
) -> BudgetDecision:
    """Admission check: would this call breach the workspace budget?

    ``projected_*`` is the pending call's estimate (model-aware, from
    :mod:`modules.policy.pricing`). We compare *spend-to-date + projected*
    against each configured ceiling. First ceiling crossed denies; if no ceiling
    is configured the call is allowed (the gate is inert).
    """
    budget = load_budget(db, workspace_id)
    if not budget or ("max_cost_usd" not in budget and "max_total_tokens" not in budget):
        return BudgetDecision(True, "no budget ceiling configured")

    window = budget.get("window", _DEFAULT_WINDOW)
    spent = spend_to_date(db, workspace_id, window)

    max_cost = budget.get("max_cost_usd")
    if max_cost is not None:
        projected_total = spent["cost_usd"] + max(0.0, float(projected_cost_usd))
        if projected_total > max_cost:
            return BudgetDecision(
                False,
                (
                    f"cost ${projected_total:.4f} (spent ${spent['cost_usd']:.4f} + "
                    f"call ${max(0.0, float(projected_cost_usd)):.4f}) over "
                    f"{window} ceiling ${max_cost:.2f}"
                ),
                dimension="cost_usd",
                limit=max_cost,
                spent=spent["cost_usd"],
                projected=projected_total,
            )

    max_tokens = budget.get("max_total_tokens")
    if max_tokens is not None:
        projected_total_tok = spent["total_tokens"] + max(0, int(projected_tokens))
        if projected_total_tok > max_tokens:
            return BudgetDecision(
                False,
                (
                    f"tokens {projected_total_tok:.0f} over {window} ceiling "
                    f"{max_tokens}"
                ),
                dimension="total_tokens",
                limit=float(max_tokens),
                spent=spent["total_tokens"],
                projected=projected_total_tok,
            )

    return BudgetDecision(
        True, f"within {window} budget", dimension="cost_usd",
        limit=max_cost, spent=spent["cost_usd"],
    )
