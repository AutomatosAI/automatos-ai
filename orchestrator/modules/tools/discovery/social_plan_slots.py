"""P256-FIX-SOC-5 (F436, socials day 14 Oct): Auto reads why a plan's slot is empty.

Asked why Thursday was empty, Auto had only ``platform_get_social_plan``: the plan's fixed keys,
the whole content bank and the history, no slot at all; and a platform tool's answer reaches the
chat as indented JSON cut to about 2,000 tokens, so slot rows added there would be cut first. No
tool listed the plans either. ``platform_get_social_plan_slots`` answers one short line a slot, in
the plan's timezone, built from the slot rows the Plan page reads (``plan_store.plan_slots``, the
phases of ``modules/socials/slot_phases.py``), and one line for each plan's current batch:

    Thu 22 Oct 09:00 | Countdown | r1 | waiting: made by the batch of Wed 21 Oct 13:44
    Wed 21 Oct 09:00 | Countdown | r1 | made (draft): it waits for you: Fill in Big number before rendering.
    Tue 20 Oct 09:00 | Countdown | r2 | skipped: no connected channel posts a video for tiktok
    Countdown | the week of 19 Oct (made Sun 18 Oct 17:00) | building: 3 made, 4 left

With no plan named it reads every active or paused plan. At most :data:`MAX_LINES` lines, then
how many more and how to narrow it, which keeps the answer near 1,200 tokens. The separator is
ASCII: the chat's formatter writes JSON with non-ASCII escaped (a middle dot becomes a 6-character
escape of about five tokens, three a line).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from modules.socials import plan_store, plans, slot_phases

DEFAULT_DAYS, MAX_DAYS, MAX_LINES = 7, 14, 40
READ_STATUSES = (plans.ACTIVE, plans.PAUSED)
MORE = "…and {count} more: name a plan or fewer days"
DAYS_REFUSED = f"days must be a whole number from 1 to {MAX_DAYS}."
FROM_REFUSED = "from must be a day, YYYY-MM-DD."
NO_PLANS = "This workspace has no active or paused Socials plan."
SEP = " | "


class SlotsRefused(ValueError):
    """A call the tool refuses before reading anything; its message is the answer."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def days_asked(value: Any) -> int:
    """``days``: a whole number from 1 to ``MAX_DAYS``, ``DEFAULT_DAYS`` when left out."""
    if value is None:
        return DEFAULT_DAYS
    number = int(value) if isinstance(value, str) and value.strip().isdigit() else value
    if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= MAX_DAYS:
        raise SlotsRefused(DAYS_REFUSED)
    return number


def day_asked(value: Any) -> Optional[date]:
    """``from``: a day, or ``None`` for each plan's today."""
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise SlotsRefused(FROM_REFUSED) from None


def _local(moment: Any, plan: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(moment)).astimezone(plans.zone_of(plan))
    except (TypeError, ValueError):
        return None


def _when(moment: Optional[datetime]) -> str:
    """"Wed 21 Oct 13:44"."""
    return f"{moment:%a} {moment.day} {moment:%b %H:%M}" if moment is not None else "its make time"


def _made(row: Dict[str, Any]) -> str:
    post = row.get("post") or {}
    status, why = post.get("status"), row.get("why")
    if not why:
        return f"made ({status})"
    reason = f"its render failed: {why}" if status == "failed" else f"it waits for you: {why}"
    return f"made ({status}): {reason}"


def _phase(plan: Any, row: Dict[str, Any]) -> str:
    phase, why = row.get("phase"), row.get("why")
    if phase == slot_phases.MADE:
        return _made(row)
    if phase == slot_phases.WAITING:
        batch = row.get("batch") or {}
        made_at = _when(_local(batch.get("made_at"), plan))
        return f"waiting: made by the batch of {made_at}" if batch.get("key") else f"waiting: made {made_at}"
    return f"{phase}: {why}" if why else str(phase)


def slot_line(plan: Any, row: Dict[str, Any]) -> str:
    """One slot: "Thu 22 Oct 09:00 | <plan> | <row> | <phase>"."""
    day = date.fromisoformat(row["local_date"])
    when = f"{day:%a} {day.day} {day:%b} {row['local_time']}"
    return SEP.join((when, plan.name, row["row_id"], _phase(plan, row)))


def batch_line(plan: Any, batch: Optional[Dict[str, Any]]) -> Optional[str]:
    """The plan's current batch: "<plan> | the week of 19 Oct (made Sun 18 Oct 17:00) | building: 3 made, 4 left"."""
    if not batch:
        return None
    made = f"{batch['label']} (made {_when(_local(batch['since'], plan))})"
    state = f"{batch['state']}: {batch['made']} made, {batch['left']} left"
    return SEP.join((plan.name, made, f"{state}; {batch['why']}" if batch.get("why") else state))


def _plan_lines(db: Any, plan: Any, first: Optional[date], days: int, now: datetime) -> List[str]:
    zone = plans.zone_of(plan)
    start_day = first or now.astimezone(zone).date()
    start = plans.local_to_utc(start_day, "00:00", zone)
    end = plans.local_to_utc(start_day + timedelta(days=days), "00:00", zone)
    facts = plan_store.slot_facts(db, plan)
    rows = plan_store.plan_slots(db, plan, start, end, now, facts)
    batch = batch_line(plan, slot_phases.current_batch(plan, now, facts))
    return ([batch] if batch else []) + [slot_line(plan, row) for row in rows]


def plans_read(db: Any, workspace_id: Any) -> List[Any]:
    """The workspace's active and paused plans, newest first."""
    return [plan for plan in plan_store.list_plans(db, workspace_id) if plan.status in READ_STATUSES]


def slot_lines(db: Any, workspace_id: Any, chosen: Optional[Sequence[Any]], first: Any, days: Any) -> Dict[str, Any]:
    """The answer: each plan's current batch and its slots from ``first`` for ``days`` days, one line
    each, at most ``MAX_LINES``, then how many more. ``chosen`` is the plan asked for, else every
    active or paused plan."""
    count, day = days_asked(days), day_asked(first)
    read = list(chosen) if chosen else plans_read(db, workspace_id)
    if not read:
        return {"success": True, "lines": [NO_PLANS], "count": 0}
    now = _now()
    lines = [line for plan in read for line in _plan_lines(db, plan, day, count, now)]
    shown = lines[:MAX_LINES] + ([MORE.format(count=len(lines) - MAX_LINES)] if len(lines) > MAX_LINES else [])
    return {"success": True, "lines": shown, "count": len(lines)}


__all__ = ["DEFAULT_DAYS", "MAX_DAYS", "MAX_LINES", "SlotsRefused", "batch_line", "slot_line", "slot_lines"]
