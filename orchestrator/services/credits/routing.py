"""Who pays for a model call: the customer's own key, credits, or nobody (PRD-265 US-004).

The rules (PRD-265 §3–§4, FR-1–FR-3):

- **Own key** — a call on the workspace's own key is never debited.
- **Local edition** — never metered: the keys are the operator's own.
- **Allowance** — a plan with ``metered: False`` (enterprise, contract-billed) runs
  on the platform's keys without a credit check; its calls are still recorded.
- **Credits** — everything else on the platform's keys needs credits. Trial
  credits pay on any plan. Monthly and top-up credits pay only on a plan whose
  ``platform_ai`` is on: Own keys (the entry plan) has no platform lane once its
  trial is spent, for any provider, including the OpenRouter fallback.
- **Background work** (heartbeats, scheduled tasks, playbook schedules) is never
  paid from the trial: an idle trial workspace burns nothing (PRD-222 FR-17).

A call's price is metered (Gerard, 20 Oct 2026, like OpenRouter): its real cost
in cents × ``CREDITS_MARKUP``, rounded up (:func:`credits_for`). At zero,
platform-funded work pauses with :class:`CreditsExhaustedError`, whose
``error_code`` the client renders as the paused state; own-key calls keep working.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

from services.credits.ledger import BUCKET_TRIAL, SPEND_ORDER, Balance, read_balance

logger = logging.getLogger(__name__)

CREDITS_EXHAUSTED_CODE = "credits_exhausted"
CREDIT_STOP_CODES = frozenset({CREDITS_EXHAUSTED_CODE})

FUNDED_BY_OWN_KEY = "own_key"
FUNDED_BY_CREDITS = "credits"
FUNDED_BY_ALLOWANCE = "allowance"
FUNDED_BY_LOCAL = "local"

_OWN_KEYS_MESSAGE = (
    "Your trial credits are used up. On the Own keys plan the AI runs on your own keys: "
    "add your Anthropic, OpenAI or OpenRouter key in Settings to keep going."
)
_PAUSED_MESSAGE = (
    "Your credits are used up, so work on Automatos's AI keys is paused. "
    "Top up, or add your own AI key in Settings to keep going."
)


class CreditsExhaustedError(Exception):
    """Platform-funded work paused: no credits can pay for it.

    ``error_code`` is stable; the chat client renders the paused banner on it.
    ``own_keys_plan`` says which way out to offer first (add a key vs top up).
    """

    error_code = CREDITS_EXHAUSTED_CODE

    def __init__(self, message: Optional[str] = None, *, own_keys_plan: bool = False):
        self.own_keys_plan = own_keys_plan
        self.message = message or (_OWN_KEYS_MESSAGE if own_keys_plan else _PAUSED_MESSAGE)
        super().__init__(self.message)


def is_credit_stop(exc: BaseException) -> bool:
    """True for a credit stop (work paused for want of credits): it ends the work
    with its own code and is never retried, compacted or folded into a reply."""
    return getattr(exc, "error_code", None) in CREDIT_STOP_CODES


def credits_for(cost_usd: float) -> int:
    """The credits a platform cost of ``cost_usd`` comes to: cents × markup, rounded
    up so a call is never under-charged (1 credit = 1 US cent at Automatos prices)."""
    from config import config

    if not cost_usd or cost_usd <= 0:
        return 0
    return int(math.ceil(round(float(cost_usd) * 100 * float(config.CREDITS_MARKUP), 6)))


@dataclass(frozen=True)
class Funding:
    """How a workspace's platform-key calls are paid for right now."""

    funded_by: str
    spend_buckets: tuple = ()
    spendable: int = 0
    own_keys_plan: bool = False


def _plan_facts(plan: Optional[str]) -> tuple[bool, bool]:
    from services.plan_tiers import is_metered, platform_ai_allowed

    return is_metered(plan), platform_ai_allowed(plan)


def spend_buckets_for(plan: Optional[str], *, background: bool = False) -> tuple:
    """The buckets that may pay for ``plan``'s platform calls, in spend order."""
    _, platform_ai = _plan_facts(plan)
    buckets = SPEND_ORDER if platform_ai else (BUCKET_TRIAL,)
    if background:
        buckets = tuple(b for b in buckets if b != BUCKET_TRIAL)
    return buckets


def funding_for(plan: Optional[str], balance: Balance, *, edition: str, background: bool = False) -> Funding:
    """Pure: how platform-key calls for a workspace on ``plan`` are paid for."""
    if edition == "local":
        return Funding(FUNDED_BY_LOCAL)
    metered, platform_ai = _plan_facts(plan)
    if not metered:
        return Funding(FUNDED_BY_ALLOWANCE)
    buckets = spend_buckets_for(plan, background=background)
    return Funding(FUNDED_BY_CREDITS, buckets, balance.spendable(buckets), own_keys_plan=not platform_ai)


def workspace_funding(db: Any, workspace_id: Any, *, background: bool = False) -> Funding:
    """:func:`funding_for` for a stored workspace (its plan and its balance)."""
    from config import config
    from core.models.workspaces import Workspace

    if config.AUTH_EDITION == "local":
        return Funding(FUNDED_BY_LOCAL)
    workspace = db.query(Workspace).get(workspace_id)
    plan = getattr(workspace, "plan", None)
    return funding_for(plan, read_balance(db, workspace_id), edition=config.AUTH_EDITION, background=background)


def can_pay(funding: Funding, credits: int) -> bool:
    """True when ``funding`` covers a charge of ``credits`` (always, off credits)."""
    return funding.funded_by != FUNDED_BY_CREDITS or funding.spendable >= credits


def paused_error(funding: Funding) -> CreditsExhaustedError:
    return CreditsExhaustedError(own_keys_plan=funding.own_keys_plan)


def has_own_keys(workspace: Any) -> bool:
    """True when the workspace has switched on any key of its own (BYOK)."""
    settings = getattr(workspace, "settings", None) or {}
    overrides = settings.get("byok_overrides") or {}
    return isinstance(overrides, dict) and any(bool(v) for v in overrides.values())


def background_allowed(db: Any, workspace_id: Any) -> bool:
    """Whether scheduled work (heartbeats, scheduled tasks, playbook schedules) may
    start: when credits other than the trial remain, or when the workspace has
    keys of its own for its agents to run on. The call-time gate still stops any
    platform call the run makes without credits."""
    from core.models.workspaces import Workspace

    if can_pay(workspace_funding(db, workspace_id, background=True), 1):
        return True
    return has_own_keys(db.query(Workspace).get(workspace_id))
