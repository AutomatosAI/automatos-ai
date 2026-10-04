"""PRD-251C (C7, C9; US-C403): the weekly note, how the plan's week went and what to change.

Once per plan and week (``services/socials_weekly_notes.py``): with the weekly batch for a
weekly plan (once its moment has come), on Mondays at the plan's make time for a daily or
monthly plan. It says:

* the week's posts (those that went out in the seven days before it), each with its numbers;
* the best and the worst, by engagement (O7 is open: ``result_reads.engagement``);
* what the best share: their format, their time of day, the best one's topic;
* the plan's health (US-C407) and Auto's proposals (US-C404), each acted on in the plan;

and links to the plan's Posted view. Pure: the sender reads, this writes the words.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple

from modules.socials import batches, plans

NOTE_EVENT = ("social_plan_weekly_note", "Your week on socials: ", "ok")
NOTE_KEY = "weekly_note"
POSTED_PATH = "/deliverables?tab=socials&view=posted&plan={plan_id}"
WEEK_DAYS = 7
MONDAY = 0
MAX_LISTED = 10
MORNING_END, AFTERNOON_END = 12, 17


@dataclass(frozen=True)
class WeekPost:
    """One post of the week: what the note says of it."""

    title: str
    format: str
    clock: Optional[str]  # when it went out, HH:MM in the plan's zone
    topic: Optional[str]
    line: str  # its numbers as one line (posted-model's words)
    engagement: Optional[int]  # None until read


def note_key(plan: Any, now: datetime) -> Optional[str]:
    """This week's key when the note is due now, else ``None`` (the module docstring)."""
    if batches.rhythm_of(plan) == plans.WEEKLY:
        window = batches.made_through(plan, now)
        return window.key if window is not None and window.moment <= now else None
    local = now.astimezone(plans.zone_of(plan))
    if local.weekday() != MONDAY or local.strftime("%H:%M") < plans.make_settings(plan)["time"]:
        return None
    iso = local.date().isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def day_part(clock: Optional[str]) -> Optional[str]:
    if not clock:
        return None
    hour = int(clock.split(":")[0])
    return "mornings" if hour < MORNING_END else "afternoons" if hour < AFTERNOON_END else "evenings"


def shared_by_best(posts: Sequence[WeekPost]) -> Optional[str]:
    """What the better half of the read posts share: "images, in the mornings, on 'The stand'"."""
    read = sorted((post for post in posts if post.engagement is not None), key=lambda post: -(post.engagement or 0))
    if not read:
        return None
    top = read[: max(1, len(read) // 2)]
    fmt = Counter(post.format for post in top).most_common(1)[0][0]
    part = Counter(part for part in (day_part(post.clock) for post in top) if part).most_common(1)
    topic = top[0].topic or top[0].title
    when = f", in the {part[0][0]}" if part else ""
    return f"{fmt} posts{when}; the best was on “{topic}”"


def _best_worst(posts: Sequence[WeekPost]) -> List[str]:
    read = [post for post in posts if post.engagement is not None]
    if not read:
        return ["No numbers read yet: posts are read a day after they go out."]
    best, worst = max(read, key=lambda p: p.engagement), min(read, key=lambda p: p.engagement)
    lines = [f"Best: “{best.title}”, {best.engagement} engagements."]
    if worst is not best:
        lines.append(f"Worst: “{worst.title}”, {worst.engagement} engagements.")
    return lines


def compose(plan_name: str, posts: Sequence[WeekPost], health: Sequence[str], proposals: Sequence[str], url: str) -> Tuple[str, str]:
    """The note's title and message."""
    title = f"{plan_name}: {len(posts)} post{'' if len(posts) == 1 else 's'} went out this week"
    lines = [f"• {post.title}: {post.line}" for post in posts[:MAX_LISTED]]
    lines += _best_worst(posts) if posts else []
    shared = shared_by_best(posts)
    lines += [f"What the best share: {shared}."] if shared else []
    lines += [f"Plan health: {'; '.join(health)}."] if health else []
    lines += [f"Auto proposes: {'; '.join(proposals)}. Open the plan to apply one."] if proposals else []
    lines.append(f"Everything that went out: {url}")
    return title, "\n".join(lines)
