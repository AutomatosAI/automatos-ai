"""PRD-251B Wave 2 (B8; US-B203, US-B204, US-B205): a plan's content bank.

A topic is an idea a post can be made from: a title, an angle, the facts it rests on
and the formats it suits. Research (the seeded playbook, through
``platform_add_social_topics``) or a person adds it. The server refuses:

* a fact without a source: every fact names where it comes from, its ``kind`` one of
  ``knowledge|deliverable|web|github|note``, with a reference and a label (D7's rule
  for claims, carried into the bank);
* a title the plan's bank already holds (case and spacing aside);
* anything on the plan's "never say" list (``sources.never_say``), in the title, the
  angle or a fact;
* from research only (PRD-251C US-C104): a title too close to a topic in any of the
  workspace's banks, or to a post of its history within the plan's repeat window
  (``modules/socials/repeats.py``). A person's topic is added, with a warning.

The make tick takes the next unused topic that suits a slot's format, a topic pinned
to the slot's day first, and records the post that used it. Nothing here commits.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from sqlalchemy import func, or_

from core.models.socials import SOCIAL_POST_FORMATS, SOCIAL_TOPIC_ORIGINS, SocialCampaign, SocialPost, SocialTopic
from modules.socials import repeats
from modules.socials.service import InvalidPost, SocialsError

FACT_SOURCE_KINDS = ("knowledge", "deliverable", "web", "github", "note")
RESEARCH, PERSON = "research", "person"
TITLE_MAX_CHARS = 200
ANGLE_MAX_CHARS = 1000
FACT_MAX_CHARS = 500
REF_MAX_CHARS = 1000
LABEL_MAX_CHARS = 200
MAX_FACTS = 12
MAX_ADDED_AT_ONCE = 30
MAX_BANK = 500
# PRD-251C (C7, US-C405): the next topic is the one most like the plan's best performers: its
# posts read in the last BEST_LOOKBACK_DAYS, the BEST_COUNT with the most engagement.
BEST_LOOKBACK_DAYS = 90
BEST_COUNT = 3


class InvalidTopic(InvalidPost):
    """A topic the bank refuses (422), with the reason."""


class TopicNotFound(SocialsError):
    """No topic of this plan has the id (404)."""


def _text(value: Any, where: str, limit: int, *, required: bool) -> Optional[str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise InvalidTopic(f"{where} is required")
        return None
    if not isinstance(value, str) or len(value) > limit:
        raise InvalidTopic(f"{where} must be text of at most {limit} characters")
    return " ".join(value.split())


def validate_fact(fact: Any, where: str) -> Dict[str, Any]:
    """``{"text", "source": {"kind", "ref", "label"}}``: a fact without a source is refused."""
    if not isinstance(fact, Mapping):
        raise InvalidTopic(f"{where} must be an object with text and source")
    text = _text(fact.get("text"), f"{where}.text", FACT_MAX_CHARS, required=True)
    source = fact.get("source")
    if not isinstance(source, Mapping) or source.get("kind") not in FACT_SOURCE_KINDS:
        raise InvalidTopic(f"{where} has no source: every fact names one, its kind one of {', '.join(FACT_SOURCE_KINDS)}")
    return {
        "text": text,
        "source": {
            "kind": source["kind"],
            "ref": _text(source.get("ref"), f"{where}.source.ref", REF_MAX_CHARS, required=True),
            "label": _text(source.get("label"), f"{where}.source.label", LABEL_MAX_CHARS, required=True),
        },
    }


def never_said(texts: Sequence[Optional[str]], never_say: Sequence[str]) -> Optional[str]:
    """The first "never say" phrase any of ``texts`` contains (case aside), or ``None``."""
    haystack = " \n ".join(t.lower() for t in texts if t)
    return next((phrase for phrase in never_say if phrase and phrase.lower() in haystack), None)


def plan_never_say(plan: SocialCampaign) -> List[str]:
    return [p for p in ((plan.sources or {}).get("never_say") or []) if isinstance(p, str)]


def validate_topic(fields: Mapping[str, Any], never_say: Sequence[str]) -> Dict[str, Any]:
    """A topic's columns from ``fields``, every rule above checked but the duplicate."""
    facts = fields.get("facts") or []
    if not isinstance(facts, (list, tuple)) or len(facts) > MAX_FACTS:
        raise InvalidTopic(f"facts must list at most {MAX_FACTS} facts")
    formats = fields.get("formats") or []
    if not isinstance(formats, (list, tuple)) or any(f not in SOCIAL_POST_FORMATS for f in formats):
        raise InvalidTopic(f"formats must be among {', '.join(SOCIAL_POST_FORMATS)}")
    clean = {
        "title": _text(fields.get("title"), "title", TITLE_MAX_CHARS, required=True),
        "angle": _text(fields.get("angle"), "angle", ANGLE_MAX_CHARS, required=False),
        "facts": [validate_fact(fact, f"facts[{i}]") for i, fact in enumerate(facts)],
        "formats": [f for f in SOCIAL_POST_FORMATS if f in formats],
    }
    said = never_said([clean["title"], clean["angle"], *(f["text"] for f in clean["facts"])], never_say)
    if said:
        raise InvalidTopic(f'"{said}" is on the plan\'s never-say list')
    return clean


def _same_title(title: str) -> str:
    return " ".join(title.lower().split())


def has_title(db: Any, plan: SocialCampaign, title: str, *, other_than: Optional[UUID] = None) -> bool:
    query = db.query(SocialTopic.id).filter(SocialTopic.campaign_id == plan.id, func.lower(SocialTopic.title) == _same_title(title))
    if other_than is not None:
        query = query.filter(SocialTopic.id != other_than)
    return query.first() is not None


def bank_size(db: Any, plan: SocialCampaign) -> int:
    return db.query(func.count(SocialTopic.id)).filter(SocialTopic.campaign_id == plan.id).scalar() or 0


def add_topic(db: Any, plan: SocialCampaign, fields: Mapping[str, Any], *, created_by: str, origin: str = PERSON) -> SocialTopic:
    """A new topic in the plan's bank, checked; not committed."""
    if origin not in SOCIAL_TOPIC_ORIGINS:
        raise InvalidTopic(f"origin must be one of {', '.join(SOCIAL_TOPIC_ORIGINS)}")
    clean = validate_topic(fields, plan_never_say(plan))
    if has_title(db, plan, clean["title"]):
        raise InvalidTopic(f'the bank already has "{clean["title"]}"')
    if bank_size(db, plan) >= MAX_BANK:
        raise InvalidTopic(f"a plan's bank holds at most {MAX_BANK} topics")
    # created_at from the clock, not now(): topics one transaction adds keep their order (F209).
    topic = SocialTopic(
        workspace_id=plan.workspace_id, campaign_id=plan.id, created_by=created_by, origin=origin,
        created_at=datetime.now(timezone.utc),
        pinned_on=_pinned_on(fields.get("pinned_on"), plan), **clean,
    )
    db.add(topic)
    db.flush()
    return topic


def _pinned_on(value: Any, plan: SocialCampaign) -> Optional[date]:
    """The day a topic is pinned to (PRD-251C C9: research's dated topics, a countdown's days):
    a date, or one as text, within the plan's dates."""
    if value in (None, ""):
        return None
    try:
        day = value if isinstance(value, date) else date.fromisoformat(str(value))
    except ValueError:
        raise InvalidTopic("pinned_on must be a day, such as 2026-11-05") from None
    starts_on, ends_on = getattr(plan, "starts_on", None), getattr(plan, "ends_on", None)
    if starts_on is not None and ends_on is not None and not starts_on <= day <= ends_on:
        raise InvalidTopic("pinned_on must fall within the plan's dates")
    return day


def _refuse_a_repeat(title: Any, earlier: Sequence[repeats.Earlier]) -> None:
    found = repeats.closest(title, earlier) if earlier else None
    if found is not None:
        raise InvalidTopic(repeats.refusal(found))


def add_topics(
    db: Any, plan: SocialCampaign, topics: Sequence[Any], *, created_by: str, origin: str = RESEARCH
) -> Tuple[List[SocialTopic], List[Dict[str, str]]]:
    """Each of ``topics`` added or refused with its reason (the research tool's write). Research's
    are held against everything the workspace has, the ones this call adds included (US-C104)."""
    if not isinstance(topics, (list, tuple)) or not 0 < len(topics) <= MAX_ADDED_AT_ONCE:
        raise InvalidTopic(f"topics must list 1 to {MAX_ADDED_AT_ONCE} topics")
    earlier = repeats.earlier_for(db, plan) if origin == RESEARCH else ()
    added, refused = [], []
    for i, item in enumerate(topics):
        fields = item if isinstance(item, Mapping) else {}
        try:  # every refusal comes before the topic's write, so nothing is left to undo
            _refuse_a_repeat(fields.get("title"), earlier)
            topic = add_topic(db, plan, fields, created_by=created_by, origin=origin)
        except InvalidTopic as exc:
            refused.append({"index": str(i), "title": str(fields.get("title") or ""), "reason": str(exc)})
            continue
        added.append(topic)
        if origin == RESEARCH:
            earlier = (*earlier, repeats.of_topic(topic.title, topic.created_at, plan.name))
    return added, refused


def update_topic(db: Any, plan: SocialCampaign, topic: SocialTopic, fields: Mapping[str, Any]) -> SocialTopic:
    merged = {"title": topic.title, "angle": topic.angle, "facts": topic.facts, "formats": topic.formats, **fields}
    clean = validate_topic(merged, plan_never_say(plan))
    if has_title(db, plan, clean["title"], other_than=topic.id):
        raise InvalidTopic(f'the bank already has "{clean["title"]}"')
    for key, value in clean.items():
        setattr(topic, key, value)
    return topic


def pin(topic: SocialTopic, day: Optional[date]) -> SocialTopic:
    """Pin the topic to ``day`` (the slots of that day take it first), or unpin it."""
    if topic.used_at is not None:
        raise InvalidTopic("a used topic cannot be pinned")
    topic.pinned_on = day
    return topic


def get_topic(db: Any, plan: SocialCampaign, topic_id: UUID) -> SocialTopic:
    topic = db.query(SocialTopic).filter(SocialTopic.id == topic_id, SocialTopic.campaign_id == plan.id).first()
    if topic is None:
        raise TopicNotFound(str(topic_id))
    return topic


def list_topics(db: Any, plan: SocialCampaign) -> List[SocialTopic]:
    """The bank: unused topics first (pinned ones by their day), then the used, newest use first."""
    rows = db.query(SocialTopic).filter(SocialTopic.campaign_id == plan.id).order_by(SocialTopic.created_at).all()
    unused = sorted((t for t in rows if t.used_at is None), key=lambda t: (t.pinned_on is None, t.pinned_on or date.max))
    used = sorted((t for t in rows if t.used_at is not None), key=lambda t: t.used_at, reverse=True)
    return unused + used


def best_performers(db: Any, plan: SocialCampaign, now: Optional[datetime] = None) -> List[frozenset]:
    """The content words of the plan's best performers (US-C405): its posts read in the last
    ``BEST_LOOKBACK_DAYS``, the ``BEST_COUNT`` with the most engagement (none without any)."""
    from modules.socials import results

    since = (now or datetime.now(timezone.utc)) - timedelta(days=BEST_LOOKBACK_DAYS)
    posts = db.query(SocialPost).filter(SocialPost.campaign_id == plan.id, SocialPost.created_at >= since).all()
    numbers = results.post_numbers(db, plan.workspace_id, [post.id for post in posts])
    read = sorted((p for p in posts if p.id in numbers and numbers[p.id].engagement > 0), key=lambda p: -numbers[p.id].engagement)
    used = {topic.used_post_id: topic for topic in db.query(SocialTopic).filter(SocialTopic.used_post_id.in_([p.id for p in read]))}
    return [repeats.words(f"{post.title} {used[post.id].title if post.id in used else ''}") for post in read[:BEST_COUNT]]


def _likeness(topic: SocialTopic, best: Sequence[frozenset]) -> Tuple[float, ...]:
    """How like each best performer the topic is, the best first: the most like the top one wins."""
    mine = repeats.words(f"{topic.title} {topic.angle or ''}")
    return tuple(repeats.overlap(mine, theirs) for theirs in best)


def next_topic(db: Any, plan: SocialCampaign, post_format: str, day: date) -> Optional[SocialTopic]:
    """The topic a slot of ``post_format`` on ``day`` is made from: an unused one pinned to
    that day first; then the unused one not pinned to another day most like the plan's best
    performers (US-C405), the oldest of equals (the oldest, without results). Either must suit
    the format (no formats suits any)."""
    candidates = (
        db.query(SocialTopic)
        .filter(SocialTopic.campaign_id == plan.id, SocialTopic.used_at.is_(None))
        .filter(or_(SocialTopic.pinned_on.is_(None), SocialTopic.pinned_on == day))
        .order_by(SocialTopic.pinned_on.is_(None), SocialTopic.created_at)
        .all()
    )
    fitting = [t for t in candidates if not t.formats or post_format in t.formats]
    if not fitting or fitting[0].pinned_on == day:
        return fitting[0] if fitting else None
    best = best_performers(db, plan)
    return max(enumerate(fitting), key=lambda item: (_likeness(item[1], best), -item[0]))[1] if best else fitting[0]


def mark_used(topic: SocialTopic, post: SocialPost, now: datetime) -> None:
    topic.used_post_id = post.id
    topic.used_at = now
    topic.pinned_on = None


def sources_of(topic: SocialTopic) -> List[Dict[str, Any]]:
    """The topic's facts as the composer's material: each fact with its source."""
    return [dict(fact) for fact in (topic.facts or [])]
