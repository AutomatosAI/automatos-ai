"""P256-FIX-SOC-5 (F436, socials day 14 Oct): why a plan's slot is empty, said at read time.

Nothing told the owner, the calendar or Auto why a slot had no post: a slot answered only
``made`` or ``planned``, a plan in the middle of its batch looked the same as a failed one, and a
stall recorded nothing. Each slot now carries a phase, derived from the plan, the clock and the
same predicates the make tick uses, so it cannot go stale when the owner adds a topic, and it adds
no write to ``plan.make`` (which owner saves race). :func:`slot_phase`, in this order:

* ``made``: a post holds the slot; ``why`` is what it waits for (SOC-2's ``needs_input``) or why its
  render failed, when either is the case;
* ``skipped``: its batch recorded it skipped, and why (no connected channel posts its format, ...);
  or its time came while nothing made it;
* ``waiting``: the moment that makes it has not come; ``batch`` says which batch and when;
* ``stalled``: it is due and the tick cannot make it: Socials is off, the plan is paused or has
  ended, the day has its ``max_per_day`` posts (``plans.within_daily_cap``), or no unused topic
  suits its format (``topics.fitting``, the rule ``next_topic`` reads);
* ``late``: due for more than :data:`LATE_TICKS` ticks, and no post of its batch was made in that
  time;
* ``building``: due, and the tick is making it.

:func:`current_batch` is the latest batch whose moment has come: building, stalled, late, ready
(announced, nothing left to make) or complete (nothing left, not announced yet). Pure: the facts
(``SlotFacts``) are read once per request by ``plan_store.slot_facts``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from config import config
from core.social_render_state import render_block
from modules.socials import batches, plans, topics

MADE, SKIPPED, WAITING, STALLED, LATE, BUILDING = "made", "skipped", "waiting", "stalled", "late", "building"
READY, COMPLETE = "ready", "complete"
LATE_TICKS = 3
PAUSED_WHY = "the plan is paused"
ENDED_WHY = "the plan has ended"
DAILY_CAP_WHY = "the day already has its {cap} posts (the plan's most a day)"
NO_TOPIC_WHY = "no unused topic in the bank suits a {format}"
LATE_WHY = "due since {since}; nothing made for {ticks} ticks"
_MADE_REASONS = ("waiting", "failed")  # render_block states whose ``why`` a made slot shows


@dataclass(frozen=True)
class SlotFacts:
    """What a slot's phase reads beyond the plan and the clock, read once per request."""

    socials_off: Optional[str]
    taken: frozenset
    unused: Tuple[Any, ...]
    # The newest post made for each batch (``None``: a daily plan's posts), as UTC.
    last_made: Mapping[Optional[str], datetime]


def batch_of(plan: Any, slot: plans.Slot) -> Dict[str, Any]:
    """The batch that makes the slot, and when: a weekly or monthly plan's window, a daily plan's
    make time (no key)."""
    window = batches.window_of(plan, slot.local_date)
    if window is None:
        return {"key": None, "label": None, "made_at": plans.make_at(plan, slot).isoformat()}
    return {"key": window.key, "label": batches.label(plan, window), "made_at": window.moment.isoformat()}


def due_at(plan: Any, slot: plans.Slot, now: datetime) -> datetime:
    """When the slot is (or was) due: its batch's moment (a slot moved before the latest batch made
    is due with it, as ``batches.due_slots`` makes it), or a daily plan's make time."""
    window = batches.window_of(plan, slot.local_date)
    if window is None:
        return plans.make_at(plan, slot)
    through = batches.made_through(plan, now)
    if slot.moved and through is not None and slot.at < plans.local_to_utc(through.end, "00:00", plans.zone_of(plan)):
        return min(window.moment, through.moment)
    return window.moment


def _made_why(post: Any) -> Optional[str]:
    block = render_block(getattr(post, "status", None), getattr(post, "review_log", None))
    return block["why"] if block is not None and block["state"] in _MADE_REASONS else None


def _skipped_why(plan: Any, slot: plans.Slot) -> Optional[str]:
    key = batches.batch_key(plan, slot.local_date)
    record = batches.records(plan).get(key) if key is not None else None
    skipped = (record or {}).get(batches.SKIPPED) if isinstance(record, Mapping) else None
    return skipped.get(slot.key) if isinstance(skipped, Mapping) else None


def stalled_why(plan: Any, slot: plans.Slot, facts: SlotFacts) -> Optional[str]:
    """Why the tick cannot make the due ``slot`` now (the module note), or ``None``."""
    if facts.socials_off:
        return facts.socials_off
    status = getattr(plan, "status", None)
    if status in (plans.PAUSED, plans.ENDED):
        return PAUSED_WHY if status == plans.PAUSED else ENDED_WHY
    if not plans.within_daily_cap(plan, [slot], facts.taken):
        return DAILY_CAP_WHY.format(cap=plans.make_settings(plan)["max_per_day"])
    if not topics.fitting(facts.unused, slot.format, slot.local_date):
        return NO_TOPIC_WHY.format(format=slot.format)
    return None


def late_why(plan: Any, slot: plans.Slot, due: datetime, now: datetime, facts: SlotFacts) -> Optional[str]:
    """Why a due slot reads late: due for more than ``LATE_TICKS`` ticks and nothing of its batch
    made in that time; ``None`` otherwise."""
    quiet = timedelta(seconds=LATE_TICKS * int(config.SOCIALS_PLAN_TICK_SECONDS))
    last = facts.last_made.get(batches.batch_key(plan, slot.local_date))
    if now - due < quiet or (last is not None and now - last < quiet):
        return None
    return LATE_WHY.format(since=f"{due.astimezone(plans.zone_of(plan)):%H:%M}", ticks=LATE_TICKS)


def slot_phase(plan: Any, slot: plans.Slot, now: datetime, post: Any, facts: SlotFacts) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """(phase, why, batch) for one slot of the plan (the module note)."""
    batch = batch_of(plan, slot)
    if post is not None:
        return MADE, _made_why(post), batch
    skipped = _skipped_why(plan, slot)
    if skipped:
        return SKIPPED, skipped, batch
    if slot.at <= now:
        return SKIPPED, batches.PASSED_UNMADE, batch
    due = due_at(plan, slot, now)
    if due > now:
        return WAITING, None, batch
    stalled = stalled_why(plan, slot, facts)
    if stalled:
        return STALLED, stalled, batch
    late = late_why(plan, slot, due, now, facts)
    return (LATE, late, batch) if late else (BUILDING, None, batch)


def _state_of(plan: Any, pending: Sequence[plans.Slot], now: datetime, facts: SlotFacts) -> Tuple[str, Optional[str]]:
    """The batch's state while it has slots left: the first stall, else the first late slot, else building."""
    phases = [slot_phase(plan, slot, now, None, facts) for slot in pending]
    for wanted in (STALLED, LATE):
        found = next((why for phase, why, _batch in phases if phase == wanted), None)
        if found:
            return wanted, found
    return BUILDING, None


def current_batch(plan: Any, now: datetime, facts: SlotFacts) -> Optional[Dict[str, Any]]:
    """The latest batch whose moment has come, and where it stands (the module note); ``None`` for
    a daily plan. What is left is read first, so a batch announced early (before this rule) never
    reads ready while it is still being made."""
    through = batches.made_through(plan, now)
    if through is None:
        return None
    pending = batches.pending(plan, through.key, now, facts.taken)
    made = sum(1 for key in facts.taken if batches.key_of_slot(plan, key) == through.key)
    if pending:
        state, why = _state_of(plan, pending, now, facts)
    else:
        state, why = (READY if batches.announced(plan, through.key) else COMPLETE), None
    return {
        "key": through.key, "label": batches.label(plan, through), "since": through.moment.isoformat(),
        "state": state, "made": made, "left": len(pending), "why": why,
    }


__all__ = [
    "BUILDING", "COMPLETE", "LATE", "MADE", "READY", "SKIPPED", "STALLED", "WAITING", "SlotFacts", "batch_of",
    "current_batch", "due_at", "slot_phase",
]
