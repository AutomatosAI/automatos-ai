"""PRD-251B Wave 2 (B6, B7, B11; US-B202): plans and their slots.

A plan is a campaign of kind ``plan`` (B6): dates, a timezone, a cadence, what to
research, how and when its posts are made, and what a passed slot does. Series
approval works on it as on any campaign.

Nothing is generated ahead (B7). The cadence expands into slots on the fly
(:func:`expand_slots`, pure): a slot is one cadence row on one local day at its time,
in the plan's timezone. Its key ``<row id>|<YYYY-MM-DD>|<HH:MM>`` is unique within the
plan, and the post made for it carries the key (``social_posts.slot_key``).
``slot_overrides`` moves a slot (``{"to": ISO datetime}``) or skips it
(``{"skip": true}``) and leaves the cadence alone.

Validation returns the cleaned columns or raises :class:`InvalidPlan` (a 422).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.models.socials import SOCIAL_LATE_POLICIES, SOCIAL_POST_FORMATS
from modules.socials.service import InvalidPost, SocialsError
from modules.socials.targets import TOOLKIT_NAME

PLAN = "plan"
ACTIVE, PAUSED, ENDED = "active", "paused", "ended"
SKIP, NEXT_SLOT = "skip", "next_slot"
VIDEO, TEXT = "video", "text"
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
CLOCK = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
ROW_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

DEFAULT_MAKE_TIME = "07:00"
DEFAULT_VIDEO_DAYS_EARLY = 1
MAX_VIDEO_DAYS_EARLY = 3
# A post is made at least this long before its slot, or a day earlier.
MIN_MAKE_LEAD = timedelta(hours=2)
DEFAULT_RESEARCH = {"enabled": True, "day": "mon", "time": "06:00"}
DEFAULT_SOURCES = {"knowledge": True, "deliverables": True, "website": True, "github": False, "notes": "", "never_say": []}
SOURCE_SWITCHES = ("knowledge", "deliverables", "website", "github")
VISUAL_MIX_KEYS = ("templates", "library", "ai_images", "ai_footage")

MAX_CADENCE_ROWS = 20
MAX_ROW_CHANNELS = 10
MAX_PLAN_DAYS = 366
MAX_WINDOW_DAYS = 62
MAX_MOVE_DAYS = 14
MAX_PER_DAY = 20
NEXT_SLOT_HORIZON_DAYS = 60
MAX_NEVER_SAY = 50
NEVER_SAY_MAX_CHARS = 80
NOTES_MAX_CHARS = 2000
GOAL_MAX_CHARS = 1000
AUDIENCE_MAX_CHARS = 500


class InvalidPlan(InvalidPost):
    """A plan field the plan cannot carry (422)."""


class PlanNotFound(SocialsError):
    """No plan of this workspace has the id (404)."""


# TemplateInfo: (template format, declared lengths) of a workspace social template, by id.
TemplateInfo = Tuple[str, List[int]]


# ── checks ─────────────────────────────────────────────────────────────────


def _clock(value: Any, where: str) -> str:
    if not isinstance(value, str) or not CLOCK.match(value):
        raise InvalidPlan(f"{where} must be a time of day, HH:MM")
    return value


def _days(value: Any, where: str) -> List[str]:
    if not isinstance(value, (list, tuple)) or not value or any(day not in WEEKDAYS for day in value):
        raise InvalidPlan(f"{where} must list days among {', '.join(WEEKDAYS)}")
    return [day for day in WEEKDAYS if day in value]


def _channels(value: Any, where: str) -> List[str]:
    names = sorted({str(v).strip().lower() for v in value}) if isinstance(value, (list, tuple)) else []
    if not names or len(names) > MAX_ROW_CHANNELS or not all(TOOLKIT_NAME.match(name) for name in names):
        raise InvalidPlan(f"{where} must name 1 to {MAX_ROW_CHANNELS} channels by their Composio toolkit, e.g. linkedin")
    return names


def _template_lengths(row: Mapping[str, Any], where: str, templates: Mapping[str, TemplateInfo]) -> Optional[str]:
    """The row's template, checked against its format; ``None`` lets Auto pick at make time."""
    template_id = row.get("template_id")
    if template_id in (None, ""):
        return None
    info = templates.get(str(template_id))
    wanted = "social_video" if row.get("format") == VIDEO else "social_image"
    if info is None or row.get("format") == TEXT or info[0] != wanted:
        raise InvalidPlan(f"{where}.template_id is not one of this workspace's {row.get('format')} templates")
    return str(template_id)


def _row_length(row: Mapping[str, Any], where: str, template_id: Optional[str], templates: Mapping[str, TemplateInfo]) -> Optional[int]:
    """A video row's length: one its template declares, or, with no template, one some video template does."""
    length = row.get("length_seconds")
    if row.get("format") != VIDEO or length is None:
        return None
    offered = templates[template_id][1] if template_id else sorted({d for fmt, ds in templates.values() if fmt == "social_video" for d in ds})
    if isinstance(length, bool) or not isinstance(length, int) or length not in offered:
        raise InvalidPlan(f"{where}.length_seconds must be one of the lengths offered: {offered}")
    return length


def _cadence_row(row: Any, index: int, templates: Mapping[str, TemplateInfo]) -> Dict[str, Any]:
    where = f"cadence[{index}]"
    if not isinstance(row, Mapping):
        raise InvalidPlan(f"{where} must be an object with channels, format, days and time")
    row_id = str(row.get("id") or f"r{index + 1}")
    if not ROW_ID.match(row_id):
        raise InvalidPlan(f"{where}.id must be short letters, digits, - or _")
    if row.get("format") not in SOCIAL_POST_FORMATS:
        raise InvalidPlan(f"{where}.format must be one of {', '.join(SOCIAL_POST_FORMATS)}")
    template_id = _template_lengths(row, where, templates)
    return {
        "id": row_id,
        "channels": _channels(row.get("channels"), f"{where}.channels"),
        "format": row["format"],
        "length_seconds": _row_length(row, where, template_id, templates),
        "template_id": template_id,
        "days": _days(row.get("days"), f"{where}.days"),
        "time": _clock(row.get("time"), f"{where}.time"),
    }


def validate_cadence(rows: Any, templates: Mapping[str, TemplateInfo]) -> List[Dict[str, Any]]:
    """The cadence rows, checked: known formats, channels, templates and their lengths, days and times."""
    if not isinstance(rows, (list, tuple)) or not 0 < len(rows) <= MAX_CADENCE_ROWS:
        raise InvalidPlan(f"cadence must list 1 to {MAX_CADENCE_ROWS} rows")
    clean = [_cadence_row(row, i, templates) for i, row in enumerate(rows)]
    ids = [row["id"] for row in clean]
    if len(set(ids)) != len(ids):
        raise InvalidPlan("cadence row ids must differ")
    return clean


def _text(value: Any, where: str, limit: int) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > limit:
        raise InvalidPlan(f"{where} must be text of at most {limit} characters")
    return value.strip() or None


def validate_timezone(value: Any) -> str:
    try:
        ZoneInfo(str(value))
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise InvalidPlan("timezone must be an IANA name, e.g. Europe/London") from exc
    return str(value)


def validate_dates(starts_on: Any, ends_on: Any) -> Tuple[date, date]:
    if not isinstance(starts_on, date) or not isinstance(ends_on, date):
        raise InvalidPlan("starts_on and ends_on must be dates")
    if ends_on < starts_on or (ends_on - starts_on).days >= MAX_PLAN_DAYS:
        raise InvalidPlan(f"a plan ends on or after its start, within {MAX_PLAN_DAYS} days")
    return starts_on, ends_on


def validate_sources(value: Any) -> Dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    clean: Dict[str, Any] = {key: bool(raw.get(key, DEFAULT_SOURCES[key])) for key in SOURCE_SWITCHES}
    clean["notes"] = _text(raw.get("notes"), "sources.notes", NOTES_MAX_CHARS) or ""
    phrases = raw.get("never_say") or []
    if not isinstance(phrases, (list, tuple)) or len(phrases) > MAX_NEVER_SAY:
        raise InvalidPlan(f"sources.never_say must list at most {MAX_NEVER_SAY} phrases")
    said = [_text(p, "sources.never_say", NEVER_SAY_MAX_CHARS) for p in phrases]
    clean["never_say"] = sorted({p for p in said if p}, key=str.lower)
    return clean


def _visual_mix(value: Any) -> Dict[str, int]:
    mix = dict(value) if isinstance(value, Mapping) and value else {"templates": 100}
    if any(key not in VISUAL_MIX_KEYS for key in mix) or not all(
        isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 100 for v in mix.values()
    ) or sum(mix.values()) != 100:
        raise InvalidPlan(f"make.visual_mix shares 100 among {', '.join(VISUAL_MIX_KEYS)}")
    return mix


def validate_make(value: Any) -> Dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    early = raw.get("video_days_early", DEFAULT_VIDEO_DAYS_EARLY)
    if isinstance(early, bool) or not isinstance(early, int) or not 0 <= early <= MAX_VIDEO_DAYS_EARLY:
        raise InvalidPlan(f"make.video_days_early must be 0 to {MAX_VIDEO_DAYS_EARLY}")
    per_day = raw.get("max_per_day")
    if per_day is not None and (isinstance(per_day, bool) or not isinstance(per_day, int) or not 0 < per_day <= MAX_PER_DAY):
        raise InvalidPlan(f"make.max_per_day must be 1 to {MAX_PER_DAY}")
    return {
        "time": _clock(raw.get("time", DEFAULT_MAKE_TIME), "make.time"),
        "video_days_early": early,
        "max_per_day": per_day,
        "visual_mix": _visual_mix(raw.get("visual_mix")),
    }


def validate_research(value: Any) -> Dict[str, Any]:
    raw = {**DEFAULT_RESEARCH, **(value if isinstance(value, Mapping) else {})}
    if raw["day"] not in WEEKDAYS:
        raise InvalidPlan(f"research.day must be one of {', '.join(WEEKDAYS)}")
    return {"enabled": bool(raw["enabled"]), "day": raw["day"], "time": _clock(raw["time"], "research.time")}


def validate_late_policy(value: Any) -> str:
    if value not in SOCIAL_LATE_POLICIES:
        raise InvalidPlan(f"late_policy must be one of {', '.join(SOCIAL_LATE_POLICIES)}")
    return value


_FIELD_CHECKS: Dict[str, Callable[[Any], Any]] = {
    "goal": lambda v: _text(v, "goal", GOAL_MAX_CHARS),
    "audience": lambda v: _text(v, "audience", AUDIENCE_MAX_CHARS),
    "timezone": validate_timezone,
    "sources": validate_sources,
    "make": validate_make,
    "research": validate_research,
    "late_policy": validate_late_policy,
}


def validate_fields(fields: Mapping[str, Any], templates: Mapping[str, TemplateInfo]) -> Dict[str, Any]:
    """The plan columns among ``fields``, each checked; dates are checked as a pair by the caller."""
    clean = {key: check(fields[key]) for key, check in _FIELD_CHECKS.items() if key in fields}
    if "cadence" in fields:
        clean["cadence"] = validate_cadence(fields["cadence"], templates)
    return clean


# ── slots ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Slot:
    key: str
    row_id: str
    channels: Tuple[str, ...]
    format: str
    length_seconds: Optional[int]
    template_id: Optional[str]
    local_date: date
    local_time: str
    at: datetime  # UTC
    moved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "row_id": self.row_id, "channels": list(self.channels), "format": self.format,
            "length_seconds": self.length_seconds, "template_id": self.template_id,
            "local_date": self.local_date.isoformat(), "local_time": self.local_time,
            "at": self.at.isoformat(), "moved": self.moved,
        }


def slot_key(row_id: str, day: date, clock: str) -> str:
    return f"{row_id}|{day.isoformat()}|{clock}"


def parse_slot_key(key: str) -> Optional[Tuple[str, date, str]]:
    parts = key.split("|") if isinstance(key, str) else []
    if len(parts) != 3 or not CLOCK.match(parts[2]):
        return None
    try:
        return parts[0], date.fromisoformat(parts[1]), parts[2]
    except ValueError:
        return None


def zone_of(plan: Any) -> ZoneInfo:
    try:
        return ZoneInfo(getattr(plan, "timezone", None) or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def local_to_utc(day: date, clock: str, zone: ZoneInfo) -> datetime:
    """A local day and time in ``zone`` as UTC. A time the clocks skip lands an hour on;
    a time they repeat is its first occurrence (fold 0)."""
    hours, minutes = (int(part) for part in clock.split(":"))
    return datetime.combine(day, time(hours, minutes), tzinfo=zone).astimezone(timezone.utc)


def _parse_utc(value: Any) -> Optional[datetime]:
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment.astimezone(timezone.utc) if moment.tzinfo else None


def _slot(plan: Any, row: Mapping[str, Any], day: date, zone: ZoneInfo) -> Optional[Slot]:
    """The row's slot on ``day``, moved as ``slot_overrides`` says; ``None`` when skipped."""
    key = slot_key(row["id"], day, row["time"])
    override = (getattr(plan, "slot_overrides", None) or {}).get(key) or {}
    if override.get("skip"):
        return None
    moved_to = _parse_utc(override.get("to")) if override.get("to") else None
    return Slot(
        key=key, row_id=row["id"], channels=tuple(row.get("channels") or ()), format=row["format"],
        length_seconds=row.get("length_seconds"), template_id=row.get("template_id"),
        local_date=day, local_time=row["time"], at=moved_to or local_to_utc(day, row["time"], zone), moved=moved_to is not None,
    )


def _days_between(first: date, last: date) -> Iterable[date]:
    for offset in range((last - first).days + 1):
        yield first + timedelta(days=offset)


def _rows(plan: Any) -> List[Mapping[str, Any]]:
    return [row for row in (getattr(plan, "cadence", None) or []) if isinstance(row, Mapping) and row.get("id")]


def expand_slots(plan: Any, start: datetime, end: datetime, rows: Optional[Sequence[str]] = None) -> List[Slot]:
    """The plan's slots whose time falls in [start, end), in time order: each cadence row on
    each of its days between the plan's dates, moved or skipped as ``slot_overrides`` says. Pure."""
    starts_on, ends_on = getattr(plan, "starts_on", None), getattr(plan, "ends_on", None)
    if starts_on is None or ends_on is None or end <= start:
        return []
    zone = zone_of(plan)
    pad = timedelta(days=MAX_MOVE_DAYS + 1)
    first = max(starts_on, (start - pad).astimezone(zone).date())
    last = min(ends_on, (end + pad).astimezone(zone).date())
    wanted = [row for row in _rows(plan) if rows is None or row["id"] in rows]
    found = []
    for day in _days_between(first, last):
        weekday = WEEKDAYS[day.weekday()]
        found += [_slot(plan, row, day, zone) for row in wanted if weekday in (row.get("days") or ())]
    return sorted((s for s in found if s is not None and start <= s.at < end), key=lambda s: (s.at, s.key))


def slot_for_key(plan: Any, key: str) -> Optional[Slot]:
    """The plan's slot with ``key`` (moved as its override says), or ``None``: no such
    slot in the cadence and dates, or it is skipped."""
    parsed = parse_slot_key(key)
    row = next((r for r in _rows(plan) if parsed and r["id"] == parsed[0]), None)
    if parsed is None or row is None:
        return None
    _, day, clock = parsed
    starts_on, ends_on = getattr(plan, "starts_on", None), getattr(plan, "ends_on", None)
    if clock != row["time"] or WEEKDAYS[day.weekday()] not in (row.get("days") or ()):
        return None
    if starts_on is None or ends_on is None or not starts_on <= day <= ends_on:
        return None
    return _slot(plan, row, day, zone_of(plan))


def make_settings(plan: Any) -> Dict[str, Any]:
    try:
        return validate_make(getattr(plan, "make", None))
    except InvalidPlan:
        return validate_make(None)


def make_at(plan: Any, slot: Slot) -> datetime:
    """When the slot's post is made (B7): the plan's make time on the slot's day, a video
    ``video_days_early`` days before; a day earlier when that leaves less than
    MIN_MAKE_LEAD before the slot."""
    settings = make_settings(plan)
    early = settings["video_days_early"] if slot.format == VIDEO else 0
    moment = local_to_utc(slot.local_date - timedelta(days=early), settings["time"], zone_of(plan))
    return moment if moment <= slot.at - MIN_MAKE_LEAD else moment - timedelta(days=1)


def due_slots(plan: Any, now: datetime, made: Set[str]) -> List[Slot]:
    """The slots to make now: their make time has come, their time has not, and no post
    holds their key. Earliest first."""
    horizon = now + timedelta(days=MAX_VIDEO_DAYS_EARLY + 2)
    return [slot for slot in expand_slots(plan, now, horizon) if slot.key not in made and make_at(plan, slot) <= now]


def next_free_slot(plan: Any, row_id: str, after: datetime, taken: Set[str]) -> Optional[Slot]:
    """The late policy's ``next_slot`` (B11): the first slot of the same cadence row after
    ``after`` that no post holds, within NEXT_SLOT_HORIZON_DAYS; ``None`` when there is none."""
    horizon = after + timedelta(days=NEXT_SLOT_HORIZON_DAYS)
    return next((s for s in expand_slots(plan, after, horizon, rows=[row_id]) if s.key not in taken), None)


def window(start: datetime, end: datetime) -> Tuple[datetime, datetime]:
    """A slots query's window, checked: UTC-aware, in order, at most MAX_WINDOW_DAYS long."""
    if start.tzinfo is None or end.tzinfo is None:
        raise InvalidPlan("start and end must carry a timezone, e.g. 2026-10-01T00:00:00Z")
    if not start < end or end - start > timedelta(days=MAX_WINDOW_DAYS):
        raise InvalidPlan(f"end must come after start, within {MAX_WINDOW_DAYS} days")
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)
