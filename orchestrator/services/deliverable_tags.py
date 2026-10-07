"""A Deliverable's tags (Gerard, 7 Oct): short labels the owner finds their work by.

Deliverables had no tags. They live in the Deliverable's ``extra`` JSON as
``extra["tags"]`` (no migration): a list of short strings, trimmed, lowercase, each
once, at most :data:`MAX_TAGS` of at most :data:`MAX_TAG_CHARS` characters.

* :func:`validated_tags` reads tags where they enter the platform (the generate
  route's body, generate_document's arguments) and refuses what breaks the rules,
  saying which rule.
* :func:`tags_of` reads tags already in the platform (a stored ``extra``, a board
  card's own tags) leniently: what cannot be a tag is left out, the first ten kept.
* :func:`card_tags_on_register` wraps ``DeliverableService.register``: a Deliverable a
  board card produced (``source_type='task'``, the card's id; a Claude Code session's
  files, a document generated while working the card, a file an agent wrote on a board
  run) carries the card's tags too: the owner's and the nights' (``sim-night-…``), never
  the tags the platform writes on its own cards (:func:`is_system_tag`).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional

from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

TAGS_KEY = "tags"
MAX_TAGS = 10
MAX_TAG_CHARS = 40
# A tag string an agent sends as one piece ("invoice, q3") separates on commas.
TAG_SEPARATOR = ","
# The Deliverable source a board card's work is registered under, with the card's id.
CARD_SOURCE_TYPE = "task"

# The tags the platform writes on the cards it files itself, which say how a card came to
# be, not what its work is about, so they stay off its Deliverables (7 Oct):
# cli_ticket_lane ``session``; orchestration_board_bridge ``mission``, ``orchestration``,
# ``mission:<goal>``; coordinator_service ``mission``; board_task_bridge ``recipe`` and the
# Playbook run's trigger; escalation_service and the notification tool ``escalation``,
# ``auto-escalation``, ``blocked:``/``watch:``/``stalled:``/``urgency:``; harness_service
# ``harness``, ``org-review``, ``risk-``, ``rx:``; watch_actions ``watch-spawned``,
# ``blueprint:``.
SYSTEM_TAGS = frozenset({
    "session", "mission", "orchestration", "recipe", "escalation", "auto-escalation", "harness",
    "harness-ledger-unreadable", "org-review", "watch-spawned",
    # A Playbook card's trigger (board_task_bridge: ``['recipe', triggered_by]``).
    "manual", "anonymous", "cron_scheduler", "webhook", "workspace_webhook", "composio_trigger", "credit_back",
    "platform_action", "rerun", "watch_rerun", "socials_plan_research",
})
SYSTEM_TAG_PREFIXES = (
    "mission:", "blocked:", "watch:", "stalled:", "urgency:", "risk-", "rx:", "blueprint:", "user:",
)
# A Playbook run a person started carries their email as its trigger (api/playbook_run_start.py).
EMAIL_MARK = "@"

NOT_A_LIST = "tags must be a list of short words or phrases, for example [\"invoice\", \"q3\"]."
NOT_TEXT = "each tag must be text; {value!r} is not."
TOO_LONG = "a tag has at most {limit} characters; {tag!r} has {length}."
TOO_MANY = "a Deliverable has at most {limit} tags; {count} were given."


class TagsRefused(ValueError):
    """Tags that break the rules; the message says which rule, in plain words."""


def clean_tag(tag: str) -> str:
    """``tag`` trimmed, its inner runs of space made one, lowercase. Pure."""
    return " ".join(tag.split()).lower()


def _unique(tags: List[str]) -> List[str]:
    return list(dict.fromkeys(tag for tag in tags if tag))


def validated_tags(raw: Any) -> List[str]:
    """The tags ``raw`` gives, cleaned and each once, or :class:`TagsRefused`. Pure.

    ``None`` is no tags; a string is read as comma-separated tags."""
    if raw is None:
        return []
    values = raw.split(TAG_SEPARATOR) if isinstance(raw, str) else raw
    if not isinstance(values, (list, tuple)):
        raise TagsRefused(NOT_A_LIST)
    bad = next((value for value in values if not isinstance(value, str)), None)
    if bad is not None:
        raise TagsRefused(NOT_TEXT.format(value=bad))
    tags = _unique([clean_tag(value) for value in values])
    long = next((tag for tag in tags if len(tag) > MAX_TAG_CHARS), None)
    if long is not None:
        raise TagsRefused(TOO_LONG.format(limit=MAX_TAG_CHARS, tag=long, length=len(long)))
    if len(tags) > MAX_TAGS:
        raise TagsRefused(TOO_MANY.format(limit=MAX_TAGS, count=len(tags)))
    return tags


def tags_of(raw: Any) -> List[str]:
    """The tags in ``raw`` (a stored list, a card's tags), leniently: what is not text or
    is too long is left out, each tag once, the first :data:`MAX_TAGS` kept. Pure."""
    values = raw if isinstance(raw, (list, tuple)) else []
    cleaned = [clean_tag(value) for value in values if isinstance(value, str)]
    return _unique([tag for tag in cleaned if len(tag) <= MAX_TAG_CHARS])[:MAX_TAGS]


def is_system_tag(tag: str) -> bool:
    """True for a (cleaned) tag the platform wrote on a card it filed itself. Pure."""
    return tag in SYSTEM_TAGS or tag.startswith(SYSTEM_TAG_PREFIXES) or EMAIL_MARK in tag


def owner_card_tags(raw: Any) -> List[str]:
    """A card's tags without the platform's own (:func:`is_system_tag`), leniently. Pure."""
    values = raw if isinstance(raw, (list, tuple)) else []
    return tags_of([value for value in values if isinstance(value, str) and not is_system_tag(clean_tag(value))])


def tags_in(extra: Any) -> List[str]:
    """A Deliverable's tags, from its ``extra``. Pure."""
    return tags_of(extra.get(TAGS_KEY)) if isinstance(extra, Mapping) else []


def card_of(source_type: Any, source_id: Any) -> Optional[int]:
    """The board card a Deliverable came from, or ``None`` when it came from no card. Pure."""
    if source_type != CARD_SOURCE_TYPE or isinstance(source_id, bool):
        return None
    try:
        return int(str(source_id))
    except (TypeError, ValueError):
        return None


def card_tags(db: Any, workspace_id: Any, card_id: int) -> List[str]:
    """The owner's tags on the workspace's board card ``card_id``, the platform's own left
    out; none when it has none or the read fails (logged: the Deliverable is registered
    without them)."""
    from core.models.core import BoardTask

    try:
        raw = (db.query(BoardTask.tags)
               .filter(BoardTask.id == card_id, BoardTask.workspace_id == str(workspace_id))
               .scalar())
    except SQLAlchemyError:
        logger.exception("[Deliverables] card %s's tags could not be read: registered without them", card_id)
        db.rollback()
        return []
    return owner_card_tags(raw)


def with_card_tags(db: Any, workspace_id: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """``register``'s keyword arguments with the card's tags after the Deliverable's own;
    the arguments unchanged when it came from no card or no tag applies."""
    card = card_of(kwargs.get("source_type"), kwargs.get("source_id"))
    if card is None:
        return kwargs
    extra = dict(kwargs.get("extra") or {})
    tags = tags_of([*tags_in(extra), *card_tags(db, workspace_id, card)])
    if not tags:
        return kwargs
    return {**kwargs, "extra": {**extra, TAGS_KEY: tags}}


def card_tags_on_register(register: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Wrap ``DeliverableService.register`` (called with keywords): a card's Deliverable
    carries the card's tags."""
    @functools.wraps(register)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return register(self, *args, **with_card_tags(self.db, self.workspace_id, kwargs))
    return wrapped


__all__ = [
    "MAX_TAGS", "MAX_TAG_CHARS", "TAGS_KEY", "TagsRefused", "card_of", "card_tags", "card_tags_on_register",
    "clean_tag", "is_system_tag", "owner_card_tags", "tags_in", "tags_of", "validated_tags", "with_card_tags",
]
