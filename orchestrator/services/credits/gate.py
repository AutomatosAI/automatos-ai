"""The gate in front of the platform's AI keys (PRD-265 US-004, FR-1/FR-2).

``LLMManager`` calls :func:`before_platform_call` before every model call, so
every construction site — the agent factory's managers and the 31 system call
sites alike — passes the same check. What it lets through without a look:

- calls on the customer's own key (``is_byok``) and calls with no workspace;
- the local edition;
- platform overhead (``classification.is_overhead``): absorbed under D-3.

Everything else is customer work on the platform's keys, metered afterwards
(``meter.py``) at its real cost × ``CREDITS_MARKUP``. The gate's part is to stop
work that can't be paid for:

- a plan on an explicit allowance (enterprise) is never stopped;
- otherwise the workspace needs spendable credits, or the call stops with
  ``CreditsExhaustedError``;
- a product's workspace also stays under its month's cap (``caps.py``), the smaller of the two being the
  budget, and stops with ``WorkspaceCapReachedError`` once it is reached;
- inside an ask, the balance read at its first call is the ask's budget: once its
  running cost (in credits) reaches it, the next call stops — a long mission ends
  when the credits do, instead of overdrawing them by more than one call.

:func:`after_platform_call` adds the call's cost to the ask in scope.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from services.credits import asks
from services.credits.classification import is_overhead
from services.credits.routing import (
    FUNDED_BY_CREDITS,
    CreditsExhaustedError,
    WorkspaceCapReachedError,
    credits_for,
    paused_error,
    workspace_funding,
)

logger = logging.getLogger(__name__)


def _exempt(workspace_id: Any, is_byok: bool, purpose: Optional[str]) -> bool:
    from config import config

    return bool(is_byok or not workspace_id or config.AUTH_EDITION == "local" or is_overhead(purpose))


def _within_budget(state: asks.AskState) -> None:
    if state.funded_by != FUNDED_BY_CREDITS:
        return
    if credits_for(state.cost_usd) >= state.budget_credits:
        logger.info("[credits] ask %s reached its workspace's %s (%d credits)",
                    state.ask_id, "cap" if state.capped else "balance", state.budget_credits)
        if state.capped:
            raise WorkspaceCapReachedError()
        raise CreditsExhaustedError(own_keys_plan=state.own_keys_plan)


def _check_balance(workspace_id: Any, state: Optional[asks.AskState]) -> None:
    from core.database.database import SessionLocal
    from services.credits.caps import cap_left

    db = SessionLocal()
    try:
        funding = workspace_funding(db, workspace_id, background=bool(state and state.background))
        left = cap_left(db, workspace_id) if funding.funded_by == FUNDED_BY_CREDITS else None
    finally:
        db.close()
    if funding.funded_by != FUNDED_BY_CREDITS:
        asks.set_budget(state, funded_by=funding.funded_by)
        return
    if funding.spendable <= 0:
        raise paused_error(funding)
    if left is not None and left <= 0:  # a product's workspace at its month's cap (services/credits/caps)
        raise WorkspaceCapReachedError()
    capped = left is not None and left < funding.spendable
    asks.set_budget(state, funded_by=funding.funded_by, budget_credits=left if capped else funding.spendable,
                    own_keys_plan=funding.own_keys_plan, capped=capped)


def before_platform_call(*, workspace_id: Any, is_byok: bool, purpose: Optional[str]) -> None:
    """Let a model call through, or raise ``CreditsExhaustedError``."""
    if _exempt(workspace_id, is_byok, purpose):
        return
    state = asks.current_ask()
    if state is not None and state.checked:
        _within_budget(state)
        return
    _check_balance(workspace_id, state)


def after_platform_call(*, workspace_id: Any, is_byok: bool, purpose: Optional[str], cost_usd: float) -> None:
    """Add a finished call's cost to the ask in scope (for its budget)."""
    if _exempt(workspace_id, is_byok, purpose):
        return
    asks.note_cost(cost_usd)
