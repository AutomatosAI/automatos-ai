"""PRD-251C (C7, US-C404): Auto's proposals for a plan, from its posts' results.

From the plan's posts that went out in the last ``LOOKBACK_DAYS`` and were read
(``results.post_numbers``), each proposal is a plan change with its reason, applied by one
click through the plan's ``PUT`` (its ``changes`` are the whole field, as the Plan page
saves it) and never by itself:

* **time**: a time of the plan's posts did at least ``LIFT`` times better than a row's own
  time, over ``MIN_POSTS`` posts each: move the row to that time;
* **format**: a format did at least ``LIFT`` times better than another, over ``MIN_POSTS``
  posts each: make the weaker format's row the stronger format (its template is left to Auto);
* **angle**: the best post's topic, as a note research reads ("more like it"), unless the
  notes already say so.

Engagement measures "better" (``result_reads.engagement``; O7 is open). Few posts, or no lift,
propose nothing. Pure over what the caller read.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from modules.socials import plans

LOOKBACK_DAYS = 28
MIN_POSTS = 3
LIFT = 1.5
ANGLE_NOTE = 'More like "{title}": it did best.'
ANGLE_NOTE_START, ANGLE_NOTE_END = 'More like "', '": it did best.'


@dataclass(frozen=True)
class ReadPost:
    """A post of the plan that went out and was read: what the proposals weigh."""

    title: str
    topic: Optional[str]
    format: str
    row_id: Optional[str]
    time: Optional[str]  # its slot's HH:MM
    engagement: int


@dataclass(frozen=True)
class Proposal:
    id: str
    kind: str
    title: str
    why: str
    changes: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "title": self.title, "why": self.why, "changes": dict(self.changes)}


def _means(posts: Sequence[ReadPost], key) -> Dict[Any, Tuple[float, int]]:
    groups: Dict[Any, List[int]] = {}
    for post in posts:
        if key(post) is not None:
            groups.setdefault(key(post), []).append(post.engagement)
    return {name: (mean(values), len(values)) for name, values in groups.items() if len(values) >= MIN_POSTS}


def _with_row(rows: Sequence[Mapping[str, Any]], row_id: str, change: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [{**row, **change} if row.get("id") == row_id else dict(row) for row in rows]


def _time_proposal(rows: Sequence[Mapping[str, Any]], posts: Sequence[ReadPost]) -> Optional[Proposal]:
    by_time, by_row = _means(posts, lambda p: p.time), _means(posts, lambda p: p.row_id)
    if not by_time or not by_row:
        return None
    best_time, (best_mean, _) = max(by_time.items(), key=lambda item: item[1][0])
    candidates = [row for row in rows if row.get("id") in by_row and row.get("time") != best_time
                  and best_mean >= LIFT * max(by_row[row["id"]][0], 1)]
    if not candidates:
        return None
    row = min(candidates, key=lambda r: by_row[r["id"]][0])
    return Proposal(
        f"time:{row['id']}:{best_time}", "time", f"Post the {row['format']} row at {best_time}",
        f"Posts at {best_time} got {best_mean:.0f} engagements on average; this row's at {row['time']} got {by_row[row['id']][0]:.0f}.",
        {"cadence": _with_row(rows, row["id"], {"time": best_time})},
    )


def _format_proposal(rows: Sequence[Mapping[str, Any]], posts: Sequence[ReadPost]) -> Optional[Proposal]:
    by_format = _means(posts, lambda p: p.format)
    if len(by_format) < 2:
        return None
    (best, (best_mean, _)), (worst, (worst_mean, _)) = (max(by_format.items(), key=lambda i: i[1][0]),
                                                        min(by_format.items(), key=lambda i: i[1][0]))
    row = next((r for r in rows if r.get("format") == worst and not r.get("kind")), None)
    if row is None or best_mean < LIFT * max(worst_mean, 1):
        return None
    change = {"format": best, "template_id": None, "length_seconds": None, "visual": None}
    return Proposal(
        f"format:{row['id']}:{best}", "format", f"Make the {worst} row on {', '.join(row.get('days') or [])} a {best}",
        f"{best.capitalize()} posts got {best_mean:.0f} engagements on average, {worst} posts {worst_mean:.0f}.",
        {"cadence": _with_row(rows, row["id"], change)},
    )


def _angle_proposal(sources: Mapping[str, Any], posts: Sequence[ReadPost]) -> Optional[Proposal]:
    best = max(posts, key=lambda p: p.engagement, default=None)
    if best is None or best.engagement <= 0:
        return None
    note = ANGLE_NOTE.format(title=best.topic or best.title)
    notes = str(sources.get("notes") or "")
    if note in notes:
        return None
    # The newest angle replaces an earlier one, so the notes never grow past their limit by proposals.
    kept = [line for line in notes.splitlines() if not (line.startswith(ANGLE_NOTE_START) and line.endswith(ANGLE_NOTE_END))]
    changed = "\n".join([*kept, note]).strip()
    if len(changed) > plans.NOTES_MAX_CHARS:
        return None
    return Proposal(
        f"angle:{best.topic or best.title}", "angle", f'More topics like "{best.topic or best.title}"',
        f"It got {best.engagement} engagements, the most of the plan's recent posts. Research reads the note.",
        {"sources": {**dict(sources), "notes": changed}},
    )


def proposals(plan: Any, posts: Sequence[ReadPost]) -> List[Proposal]:
    """The plan's proposals (the module docstring), from its recent read posts."""
    rows = plans.cadence_rows(plan)
    found = [_time_proposal(rows, posts), _format_proposal(rows, posts), _angle_proposal(plans.validate_sources(plan.sources), posts)]
    return [item for item in found if item is not None]


def _aware(moment: Optional[datetime]) -> datetime:
    """A stored time as UTC (SQLite gives it back without a zone); the epoch when unknown."""
    if moment is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def read_posts(db: Any, plan: Any, since: datetime) -> List[ReadPost]:
    """The plan's posts that went out since ``since`` and were read."""
    from core.models.socials import SocialPost
    from modules.socials import history, posted, results

    posts = (
        db.query(SocialPost)
        .filter(SocialPost.campaign_id == plan.id, SocialPost.workspace_id == plan.workspace_id, SocialPost.status.in_(posted.POSTED_STATUSES))
        .all()
    )
    went = [post for post in posts if _aware(history.post_date(post)) >= since]
    numbers = results.post_numbers(db, plan.workspace_id, [post.id for post in went])
    topics = history.topics_by_post(db, plan.workspace_id, [post.id for post in went])
    out = []
    for post in went:
        if post.id not in numbers:
            continue
        parsed = plans.parse_slot_key(post.slot_key or "")
        out.append(ReadPost(post.title, topics[post.id].title if post.id in topics else history.first_line(post.brief), post.format,
                            parsed[0] if parsed else None, parsed[2] if parsed else None, numbers[post.id].engagement))
    return out
