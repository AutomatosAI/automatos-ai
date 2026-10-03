"""PRD-251B (3 Oct 2026 pass): "Plan with Auto", a plan drafted from what the person says.

The person writes what their socials should do ("three posts this week about our autumn
offer on Instagram, use our website, one should be a customer review"). One call to the
workspace's model, given the channels the workspace has connected and what each can post,
the formats its templates make, the brand kit's voice and style and today's date, answers a
plan as the Plan page holds it: a name, the goal, the audience, its dates (this week unless
they said otherwise), a cadence (channels, format, days and time per row) and the
suggestions it heard, as content bank topics.

Nothing is saved here. The Plan page opens with the draft; the person checks it and saves,
the suggestions join the content bank, and research fills the rest from the sources they
chose (``sources``, which the request sets, never the model; ``notes`` carries their words).

The model's JSON is never trusted (``checked_draft``): a channel that is not connected, a
format, day or time the plan cannot carry, dates in the past or out of order are dropped or
replaced with the form's default, each with a warning that says so. The Plan page's own
checks run again when it saves.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from modules.socials import campaigns, plans, topics
from modules.socials.compose import parse_answer

logger = logging.getLogger(__name__)

SERVICE_NAME = "socials"
REQUEST_TYPE = "socials_plan_draft"
ATTEMPTS = 2  # the first answer, and one retry when its JSON is unusable
REQUEST_MAX_CHARS = 2000
DRAFT_FORMATS = ("image", "carousel", "video", "text")
DEFAULT_FORMAT = "image"
DEFAULT_DAYS = ("mon", "wed", "fri")
DEFAULT_TIME = "09:00"
DEFAULT_WEEK_DAYS = 7
MAX_DRAFT_TOPICS = 10
NAME_FALLBACK = "Plan from {day}"
NOTES_LEAD = "What the person asked for: "
RETRY_NOTE = "Your answer was not the JSON object asked for. Answer again with ONLY that JSON object, no prose."

_ANSWER_SHAPE = {
    "name": "a short name for the plan",
    "goal": "what the posts are for, in one or two sentences",
    "audience": "who they are for, or empty",
    "starts_on": "YYYY-MM-DD",
    "ends_on": "YYYY-MM-DD",
    "cadence": [{"channels": ["<toolkit>"], "format": "image | carousel | video | text",
                 "days": ["mon", "wed"], "time": "HH:MM"}],
    "topics": [{"title": "one post idea", "angle": "how to tell it, or empty", "formats": ["image"]}],
}


class DraftFailed(Exception):
    """The model gave no usable plan (502)."""


class DraftTimedOut(Exception):
    """The model did not answer in time (504)."""


@dataclass(frozen=True)
class DraftContext:
    """What the model is given: the person's words and the workspace as it is now."""

    request: str
    today: date
    timezone: str
    channels: Sequence[Mapping[str, Any]]  # {toolkit, label, kinds}: connected, with the kinds they can post now
    templates: Sequence[Mapping[str, str]] = ()  # {name, format}
    voice: Mapping[str, Any] = field(default_factory=dict)
    style: str = ""


@dataclass(frozen=True)
class DraftSources:
    """The research sources the person chose for the plan (the request's, never the model's)."""

    knowledge: bool = True
    website: bool = True
    deliverables: bool = True


# ── the prompt ──────────────────────────────────────────────────────────────
def build_messages(ctx: DraftContext) -> List[Dict[str, str]]:
    """The one conversation the model is asked: what to answer, then the workspace's material."""
    system = "\n\n".join([
        "You plan a small business's social media posts. Answer with ONE JSON object only, shaped:",
        json.dumps(_ANSWER_SHAPE),
        "Plan what the person asked for, and nothing they did not. Use only the channels listed, each "
        "for a format it can post. Unless they named other dates, the plan runs from today for a week. "
        "Each post idea they mention, and each one that would serve their goal, is a topic: a short title "
        "and how to tell it. Never invent prices, figures, offers or claims they did not give.",
    ])
    material = {
        "request": ctx.request,
        "today": ctx.today.isoformat(),
        "weekday": plans.WEEKDAYS[ctx.today.weekday()],
        "timezone": ctx.timezone,
        "channels": [dict(c) for c in ctx.channels],
        "templates": [dict(t) for t in ctx.templates],
        "brand_voice": dict(ctx.voice),
    }
    if ctx.style:
        material["brand_style"] = ctx.style
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(material, default=str)}]


# ── the checks ─────────────────────────────────────────────────────────────
def _text(value: Any, limit: int) -> str:
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def _day(value: Any) -> Optional[date]:
    try:
        return date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _dates(raw: Mapping[str, Any], today: date, warnings: List[str]) -> Tuple[date, date]:
    """From today (or the day asked, never in the past) for a week unless another end was asked."""
    starts = _day(raw.get("starts_on")) or today
    if starts < today:
        warnings.append("The plan cannot start in the past: it starts today.")
        starts = today
    ends = _day(raw.get("ends_on")) or starts + timedelta(days=DEFAULT_WEEK_DAYS - 1)
    if ends < starts:
        warnings.append("The plan's last day was before its first: it runs for a week.")
        ends = starts + timedelta(days=DEFAULT_WEEK_DAYS - 1)
    last = starts + timedelta(days=plans.MAX_PLAN_DAYS - 1)
    if ends > last:
        warnings.append(f"A plan runs at most {plans.MAX_PLAN_DAYS} days: it ends on {last.isoformat()}.")
        ends = last
    return starts, ends


def _row_channels(raw: Any, connected: Sequence[str], where: str, warnings: List[str]) -> List[str]:
    asked = [str(c).strip().lower() for c in raw] if isinstance(raw, list) else []
    kept = [c for c in dict.fromkeys(asked) if c in connected][: plans.MAX_ROW_CHANNELS]
    dropped = [c for c in asked if c not in connected]
    if dropped:
        warnings.append(f"{where}: {', '.join(dropped)} is not connected here, so it was left out.")
    if not kept:
        warnings.append(f"{where}: pick the channel it posts to.")
    return kept


def _row(raw: Any, index: int, connected: Sequence[str], warnings: List[str]) -> Dict[str, Any]:
    """One cadence row as the Plan page holds it; Auto picks each post's template on its day."""
    row = raw if isinstance(raw, Mapping) else {}
    where = f"Row {index + 1}"
    asked_days = row.get("days") if isinstance(row.get("days"), list) else []
    fmt = row.get("format") if row.get("format") in DRAFT_FORMATS else None
    days = [d for d in plans.WEEKDAYS if d in asked_days]
    time = row.get("time") if isinstance(row.get("time"), str) and plans.CLOCK.match(row["time"]) else None
    unusable = [name for name, value in (("format", fmt), ("days", days), ("time", time)) if not value]
    if unusable:
        warnings.append(f"{where}: its {', '.join(unusable)} could not be used, so the form's default is.")
    channels = _row_channels(row.get("channels"), connected, where, warnings)
    fmt, days, time = fmt or DEFAULT_FORMAT, days or list(DEFAULT_DAYS), time or DEFAULT_TIME
    return {"channels": channels, "format": fmt, "days": days, "time": time, "template_id": None, "length_seconds": None}


def _cadence(raw: Any, connected: Sequence[str], warnings: List[str]) -> List[Dict[str, Any]]:
    rows = [_row(r, i, connected, warnings) for i, r in enumerate(raw[: plans.MAX_CADENCE_ROWS])] if isinstance(raw, list) else []
    if rows:
        return rows
    warnings.append("No rhythm was given: one post on Monday, Wednesday and Friday at 09:00 to start from.")
    return [{"channels": list(connected[:1]), "format": DEFAULT_FORMAT, "days": list(DEFAULT_DAYS), "time": DEFAULT_TIME,
             "template_id": None, "length_seconds": None}]


def _topics(raw: Any) -> List[Dict[str, Any]]:
    """The suggestions as content bank topics: a title, how to tell it, the formats it suits."""
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in raw if isinstance(raw, list) else []:
        topic = item if isinstance(item, Mapping) else {}
        title = _text(topic.get("title"), topics.TITLE_MAX_CHARS)
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        formats = [f for f in topic.get("formats") or [] if f in DRAFT_FORMATS] if isinstance(topic.get("formats"), list) else []
        out.append({"title": title, "angle": _text(topic.get("angle"), topics.ANGLE_MAX_CHARS) or None, "formats": formats})
        if len(out) == MAX_DRAFT_TOPICS:
            break
    return out


def checked_draft(raw: Mapping[str, Any], ctx: DraftContext, sources: DraftSources) -> Dict[str, Any]:
    """The plan and its suggestions as the Plan page opens them, every field checked against ``ctx``."""
    warnings: List[str] = []
    connected = [str(c["toolkit"]) for c in ctx.channels]
    if not connected:
        warnings.append("No social channel is connected yet: connect one in Composio, then pick it in the cadence.")
    starts, ends = _dates(raw, ctx.today, warnings)
    plan = {
        "name": _text(raw.get("name"), campaigns.CAMPAIGN_NAME_MAX_CHARS) or NAME_FALLBACK.format(day=ctx.today.strftime("%d %b")),
        "goal": _text(raw.get("goal"), plans.GOAL_MAX_CHARS) or _text(ctx.request, plans.GOAL_MAX_CHARS),
        "audience": _text(raw.get("audience"), plans.AUDIENCE_MAX_CHARS),
        "starts_on": starts.isoformat(),
        "ends_on": ends.isoformat(),
        "timezone": ctx.timezone,
        "cadence": _cadence(raw.get("cadence"), connected, warnings),
        "sources": {
            "knowledge": sources.knowledge, "website": sources.website, "deliverables": sources.deliverables,
            "github": False, "notes": (NOTES_LEAD + ctx.request)[: plans.NOTES_MAX_CHARS], "never_say": [],
        },
    }
    return {"plan": plan, "topics": _topics(raw.get("topics")), "warnings": warnings}


# ── the model ───────────────────────────────────────────────────────────────
def llm_factory(workspace_id: Any) -> Callable[[], Any]:
    """The workspace's model, through the platform's LLM manager: usage tracked as ``socials_plan_draft``."""

    def build() -> Any:
        from core.llm import create_llm_manager

        return create_llm_manager(service_name=SERVICE_NAME, workspace_id=workspace_id, request_type=REQUEST_TYPE)

    return build


async def draft(ctx: DraftContext, sources: DraftSources, factory: Callable[[], Any], timeout: float) -> Dict[str, Any]:
    """The checked draft for ``ctx``: one call, and one retry when its JSON is unusable."""
    llm = factory()
    messages = build_messages(ctx)
    for attempt in range(ATTEMPTS):
        try:
            response = await asyncio.wait_for(llm.generate_response(messages), timeout=timeout)
        except asyncio.TimeoutError:
            raise DraftTimedOut(f"Auto did not answer within {timeout:g} seconds. Try again.") from None
        raw = parse_answer(getattr(response, "content", None))
        if raw is not None:
            return checked_draft(raw, ctx, sources)
        logger.warning("[Socials] plan draft answer %d was not JSON", attempt + 1)
        messages = [*messages, {"role": "user", "content": RETRY_NOTE}]
    raise DraftFailed("Auto's answer could not be read as a plan. Try again, or fill the plan in yourself.")
