"""PRD-251C Wave 1 (C5; US-C104): no topic repeats by accident.

Research may not add a topic too close to:

* any topic in any of the workspace's content banks (every plan, used or not); or
* a post in the workspace's history (``modules/socials/history.py``) within the plan's repeat
  window, ``research.repeat_after_days`` (default 60).

The refusal names the earlier topic or post and its date. A person's topic is never refused
for this: they may repeat on purpose, and are warned instead ("Posted 5 Oct 2026 as 'What is
a Mission?'"). The content bank says the same on each topic close to a post (``post_notes``,
US-C106), with the post to open.

"Too close" is deterministic and cheap, with no model or embedding call per topic (the 15 Sep
embedding storm). Two titles are too close when they are the same once case, punctuation and
contractions are folded ("What's a mission" and "What is a Mission?"), or when their content
words (stopwords out, a plural's "s" off) overlap, shared words over all words, at or above
``SOCIALS_REPEAT_OVERLAP``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple

from config import config
from core.models.socials import SocialCampaign, SocialTopic
from modules.socials import history, plans

_CONTRACTIONS = (
    ("can't", "cannot"), ("won't", "will not"), ("n't", " not"), ("'re", " are"), ("'ll", " will"),
    ("'ve", " have"), ("'m", " am"), ("'d", " would"), ("'s", " is"),
)
_APOSTROPHES = re.compile("[\u2018\u2019`\u00b4]")  # curly quotes and accents read as an apostrophe
_NOT_A_WORD = re.compile(r"[^a-z0-9]+")
STOPWORDS = frozenset(
    "a about am an and are as at be been being but by can cannot could did do does for from had has have how i "
    "in into is it its me my no not of on or our over should that the these this those to us vs was we were "
    "what when where which who why will with would you your".split()
)
SINGULAR_MIN_CHARS = 4  # "missions" loses its s; "is", "bus" and "class" keep theirs
WENT_OUT = frozenset({"publishing", "published", "partially_published"})


def fold(title: Any) -> str:
    """``title`` with case, punctuation and contractions folded: "What's a Mission?" is
    "what is a mission"."""
    text = _APOSTROPHES.sub("'", str(title or "").lower())
    for short, full in _CONTRACTIONS:
        text = text.replace(short, full)
    return " ".join(_NOT_A_WORD.sub(" ", text).split())


def _singular(word: str) -> str:
    return word[:-1] if len(word) >= SINGULAR_MIN_CHARS and word.endswith("s") and not word.endswith("ss") else word


def words(title: Any) -> FrozenSet[str]:
    """The title's content words: folded, stopwords out, a plural's "s" off."""
    return frozenset(_singular(word) for word in fold(title).split() if word not in STOPWORDS)


def overlap(first: FrozenSet[str], second: FrozenSet[str]) -> float:
    """Shared words over all words (0 when either has none)."""
    return len(first & second) / len(first | second) if first and second else 0.0


def threshold() -> float:
    return float(config.SOCIALS_REPEAT_OVERLAP)


@dataclass(frozen=True)
class Earlier:
    """A topic in one of the workspace's banks, or a post in its history, a new title is held against."""

    title: str
    texts: Tuple[str, ...]  # what is compared: a topic's title; a post's title and its topic
    day: Optional[str]  # "5 Oct 2026"
    plan_name: Optional[str] = None  # a bank topic's plan
    went_out: bool = False  # a post that went out (else it is approved, scheduled or waiting)
    is_post: bool = False
    post_id: Optional[str] = None
    folded: Tuple[str, ...] = ()  # the texts folded, and their content words: worked out once
    word_sets: Tuple[FrozenSet[str], ...] = ()

    def score(self, folded: str, title_words: FrozenSet[str], at_least: float) -> float:
        """How close a title (folded, and its content words) is: 2 for the same folded title,
        else the best overlap at or over ``at_least``; 0 when it is not close."""
        if folded and folded in self.folded:
            return 2.0
        best = max((overlap(title_words, other) for other in self.word_sets), default=0.0)
        return best if best >= at_least else 0.0


def _earlier(title: str, texts: Tuple[str, ...], **fields: Any) -> Earlier:
    return Earlier(title=title, texts=texts, folded=tuple(fold(text) for text in texts),
                   word_sets=tuple(words(text) for text in texts), **fields)


def _day(moment: Any) -> Optional[str]:
    if isinstance(moment, str):
        try:
            moment = datetime.fromisoformat(moment)
        except ValueError:
            return None
    return f"{moment.day} {moment:%b %Y}" if isinstance(moment, datetime) else None


def of_topic(title: str, created_at: Any, plan_name: Optional[str]) -> Earlier:
    return _earlier(title, (title,), day=_day(created_at), plan_name=plan_name)


def of_post(item: Dict[str, Any]) -> Earlier:
    """A history item (``history.history``) as something a new title is held against."""
    texts = tuple(text for text in (item.get("title"), item.get("topic")) if text)
    return _earlier(str(item.get("title") or ""), texts, day=_day(item.get("date")),
                    went_out=item.get("state") in WENT_OUT, is_post=True, post_id=item.get("id"))


def recent_posts(db: Any, plan: SocialCampaign, now: Optional[datetime] = None) -> Tuple[Earlier, ...]:
    """The workspace's history within the plan's repeat window."""
    days = plans.validate_research(plan.research)["repeat_after_days"]
    posts = history.history(db, plan.workspace_id, days=days, limit=history.MAX_LIMIT, now=now or datetime.now(timezone.utc))
    return tuple(of_post(item) for item in posts)


def earlier_for(db: Any, plan: SocialCampaign, now: Optional[datetime] = None) -> Tuple[Earlier, ...]:
    """Every topic in the workspace's banks, and its history within the plan's repeat window."""
    rows = (
        db.query(SocialTopic.title, SocialTopic.created_at, SocialCampaign.name)
        .join(SocialCampaign, SocialCampaign.id == SocialTopic.campaign_id)
        .filter(SocialTopic.workspace_id == plan.workspace_id)
        .all()
    )
    return (*(of_topic(title, made, name) for title, made, name in rows), *recent_posts(db, plan, now))


def closest(title: Any, earlier: Iterable[Earlier], at_least: Optional[float] = None) -> Optional[Earlier]:
    """The earlier topic or post ``title`` is closest to, when it is too close to any; else None."""
    folded = fold(title) if isinstance(title, str) else ""
    if not folded:
        return None
    floor, title_words = threshold() if at_least is None else at_least, words(title)
    scored = [(item.score(folded, title_words, floor), item) for item in earlier]
    best = max(scored, key=lambda pair: pair[0], default=(0.0, None))
    return best[1] if best[0] > 0 else None


def _named(earlier: Earlier) -> str:
    when = f" ({earlier.day})" if earlier.day else ""
    if not earlier.is_post:
        return f"\"{earlier.title}\" in the bank of the plan \"{earlier.plan_name}\"{when}"
    verb = "posted" if earlier.went_out else "planned"
    return f"\"{earlier.title}\", {verb}{' ' + earlier.day if earlier.day else ''}"


def refusal(earlier: Earlier) -> str:
    """Why research may not add the topic: it names the earlier topic or post and its date."""
    return f"too close to {_named(earlier)}: research what neither the history nor the bank covers"


def warning(earlier: Earlier) -> str:
    """What a person adding a close topic is told; they may repeat on purpose."""
    if not earlier.is_post:
        when = f" (added {earlier.day})" if earlier.day else ""
        return f"The plan \"{earlier.plan_name}\" has \"{earlier.title}\" in its bank{when}."
    verb = "Posted" if earlier.went_out else "Planned"
    return f"{verb}{' ' + earlier.day if earlier.day else ''} as \"{earlier.title}\"."


def post_notes(db: Any, plan: SocialCampaign, topics: Iterable[SocialTopic]) -> Dict[Any, Dict[str, Any]]:
    """US-C106: for each of the plan's topics close to a post in the history (its own post
    aside), the post to open and what the bank says ("Posted 5 Oct 2026 as ..."), by topic id."""
    posts = recent_posts(db, plan)
    notes: Dict[Any, Dict[str, Any]] = {}
    for topic in topics:
        own = str(topic.used_post_id) if topic.used_post_id else None
        found = closest(topic.title, (post for post in posts if post.post_id != own)) if posts else None
        if found is not None:
            notes[topic.id] = {"post_id": found.post_id, "note": warning(found)}
    return notes


def warning_for(db: Any, plan: SocialCampaign, title: Any) -> Optional[str]:
    """A person's new topic: the warning when it is close to what the workspace has, else None."""
    found = closest(title, earlier_for(db, plan)) if isinstance(title, str) else None
    return warning(found) if found is not None else None
