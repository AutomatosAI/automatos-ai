"""PRD-251 D13, P251W1-RVW-5: a post's paid media is booked before the provider can charge for it.

Footage, stills and a voice from the workspace's Composio toolkits cost money
once the toolkit takes the job, and a job runs for minutes. So each job's spend
is booked before its toolkit is called (:func:`commit`): one ``pending`` row in
``llm_usage`` on the media lane against the post, at what the job is committed
to (a footage shot's fal.ai estimate, or its ceiling on a credit-billed toolkit;
a voice script's priced amount), committed before the recipe makes the call. A
process that stops while the job runs (a redeploy's SIGKILL, an OOM, a crash)
leaves that row in place, and everything that reads the media lane counts it:
the post's cap and the workspace's monthly media cap (``media_caps.py``), the
workspace budget gate (``modules/policy/budget.spend_to_date``) and the daily
spend guard. The boot reaper leaves it there.

When the job ends, the recipe settles the booking to what the toolkit spent
(:func:`settle`): the same row, updated in place, so each job has exactly one
amount in ``llm_usage`` at every moment, never a second row beside the first. A
job the toolkit failed, or never took, has its booking reversed: the row is
deleted. Only a ``pending`` row is settled, so a booking is settled once.

Each function opens a session of its own and commits before it returns: call it
off the event loop (``asyncio.to_thread``), so the row is written at once.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

from core.llm.usage_context import LANE_MEDIA
from core.llm.usage_tracker import STATUS_SUCCESS
from core.models.core import LLMUsage
from modules.socials.media_caps import post_execution_id

logger = logging.getLogger(__name__)

# A booking whose job has not ended; once settled it is a success, like any usage row.
PENDING_STATUS = "pending"
SETTLED_STATUS = STATUS_SUCCESS
# As UsageTracker.track_media books the media lane: the workspace's own toolkit
# (BYOK), priced by the recipe, never at a token price.
MEDIA_TIER = "direct"
COST_DECIMALS = 8
UNKNOWN = "unknown"


class BookingFailed(Exception):
    """A job's spend could not be booked, so the job must not be submitted."""


@dataclass(frozen=True)
class Settlement:
    """What one job's booking settles to when the job ends."""

    booking: int  # the job's llm_usage row (:func:`commit`)
    units: float = 0.0
    usd: float = 0.0
    error_message: Optional[str] = None
    # The toolkit said the job failed, or never took it: the booking is deleted.
    reverse: bool = False


def _amount(value: Any) -> Optional[float]:
    """``value`` as an amount to book: a finite number of at least 0, else ``None``
    (a booking never lowers the media lane, and garbage is never booked as $0)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    amount = float(value)
    return amount if math.isfinite(amount) and amount >= 0 else None


def _booked(units: Any, usd: Any) -> Optional[Tuple[int, float]]:
    """(units rounded up, as the media lane counts them, so a quota never
    under-counts; dollars) to book, or ``None`` when either is not an amount."""
    count, cost = _amount(units), _amount(usd)
    if count is None or cost is None:
        return None
    return math.ceil(count), round(cost, COST_DECIMALS)


def commit(
    session_factory: Callable[[], Any],
    workspace_id: Any,
    post_id: Any,
    *,
    provider: str,
    model_id: str,
    units: float,
    usd: float,
) -> int:
    """Book ``usd`` against the post before its job is submitted: one ``pending``
    row on the media lane, committed before this returns. Its id, which the job
    settles. :class:`BookingFailed` when the price is not an amount of dollars or
    the row cannot be written: then nothing may be submitted."""
    toolkit = str(provider or "").strip().lower() or UNKNOWN
    amounts = _booked(units, usd)
    if amounts is None:
        logger.error("[SocialsLedger] %s's price for post %s is not an amount to book: %r, %r units", toolkit,
                     post_id, usd, units)
        raise BookingFailed(f"{toolkit}'s price ({usd!r}) is not an amount of dollars to book")
    count, cost = amounts
    db = session_factory()
    try:
        row = LLMUsage(
            workspace_id=workspace_id, model_id=model_id or UNKNOWN, provider=toolkit, tier=MEDIA_TIER,
            execution_id=post_execution_id(post_id), request_type=LANE_MEDIA,
            input_tokens=count, output_tokens=0, total_tokens=count, cache_read_tokens=0, cache_write_tokens=0,
            input_cost=cost, output_cost=0.0, total_cost=cost, is_byok=True, status=PENDING_STATUS,
        )
        db.add(row)
        db.flush()
        booking = int(row.id)
        db.commit()
        return booking
    except Exception as exc:  # noqa: BLE001 — a spend that cannot be booked is never submitted
        db.rollback()
        logger.exception("[SocialsLedger] booking %s's %s for post %s failed", toolkit, model_id, post_id)
        raise BookingFailed(f"{toolkit}'s spend could not be booked") from exc
    finally:
        db.close()


def _apply(db: Any, item: Settlement, latency_ms: Optional[int]) -> None:
    """One settlement, in the caller's transaction; only a ``pending`` row is touched."""
    pending = db.query(LLMUsage).filter(LLMUsage.id == item.booking, LLMUsage.status == PENDING_STATUS)
    if item.reverse:
        pending.delete(synchronize_session=False)
        return
    amounts = _booked(item.units, item.usd)
    if amounts is None:
        logger.error("[SocialsLedger] booking %s cannot settle to %r (%r units): it keeps its committed amount",
                     item.booking, item.usd, item.units)
        return
    count, cost = amounts
    pending.update(
        {
            LLMUsage.input_tokens: count,
            LLMUsage.total_tokens: count,
            LLMUsage.input_cost: cost,
            LLMUsage.total_cost: cost,
            LLMUsage.latency_ms: latency_ms,
            LLMUsage.status: SETTLED_STATUS,
            LLMUsage.error_message: item.error_message,
        },
        synchronize_session=False,
    )


def settle(
    session_factory: Callable[[], Any], settlements: Sequence[Settlement], latency_ms: Optional[int] = None
) -> bool:
    """Settle each booking to what its job cost, in one transaction: its row
    updated in place, or deleted when the job is reversed. ``False`` when that
    transaction failed: each booking then keeps its committed amount, which is
    what the caps and the budget gate go on counting."""
    if not settlements:
        return True
    db = session_factory()
    try:
        for item in settlements:
            _apply(db, item, latency_ms)
        db.commit()
        return True
    except Exception:  # noqa: BLE001 — the jobs have ended: each keeps its committed amount, loudly
        db.rollback()
        logger.exception(
            "[SocialsLedger] settling bookings %s failed: they keep their committed amounts",
            [item.booking for item in settlements],
        )
        return False
    finally:
        db.close()
