"""PRD-181 S2 — the dollar-ceiling helper, generalised beyond OrchestrationRun.

PRD-163 S5 put a dollar ceiling on missions, but the band logic lived inside
``MissionDispatcher`` bound to ``OrchestrationRun`` (``_budget_ceiling_usd`` /
``_get_budget_status`` / ``_pre_dispatch_budget_check``). F060 needs the same
ceiling on board tasks and playbook runs, so the *arithmetic* is lifted here as
pure functions any surface can call against its own ``(ceiling, used)`` numbers.
The mission dispatcher keeps its OrchestrationRun accessors and delegates the
banding here, so there is one definition of the bands, not two.

Convention (unchanged from the mission gate): ``ceiling_usd <= 0`` means
**unlimited** (HEALTHY always). Bands: <50% HEALTHY, 50–79% WARNING, 80–100%
CRITICAL, >100% EXCEEDED.
"""
from __future__ import annotations

from enum import Enum


class BudgetBand(str, Enum):
    """Budget health as a fraction of the dollar ceiling consumed."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"


def budget_status(*, ceiling_usd: float, used_usd: float) -> BudgetBand:
    """Return the band for a (ceiling, used) pair. ``ceiling <= 0`` ⇒ unlimited."""
    ceiling = float(ceiling_usd or 0.0)
    if ceiling <= 0:
        return BudgetBand.HEALTHY
    pct = (float(used_usd or 0.0) / ceiling) * 100.0
    if pct > 100:
        return BudgetBand.EXCEEDED
    if pct >= 80:
        return BudgetBand.CRITICAL
    if pct >= 50:
        return BudgetBand.WARNING
    return BudgetBand.HEALTHY


def would_exceed(*, ceiling_usd: float, used_usd: float, next_step_usd: float = 0.0) -> bool:
    """True when ``used + next_step`` breaches the ceiling. ``ceiling <= 0`` ⇒ never."""
    ceiling = float(ceiling_usd or 0.0)
    if ceiling <= 0:
        return False
    return (float(used_usd or 0.0) + max(0.0, float(next_step_usd or 0.0))) > ceiling


def playbook_can_afford(*, ceiling_usd: float, used_usd: float, next_step_usd: float) -> bool:
    """Admission check for the next playbook step — allow iff it stays in budget.

    Mirrors the mission ``_pre_dispatch_budget_check`` 'block' rule: a step that
    would push cumulative spend over the ceiling is refused. No ceiling ⇒ allow.
    """
    return not would_exceed(
        ceiling_usd=ceiling_usd, used_usd=used_usd, next_step_usd=next_step_usd
    )
