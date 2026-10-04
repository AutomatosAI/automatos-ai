"""PRD-251C (C7, US-C402): a published post's numbers, read through its channel's own read
action in Composio (Composio only, D15: no client of our own).

:data:`READS` holds each channel's read, from its adapter's ``results`` (the channel data,
``channel_adapters.py``): its action, the params (``$remote_id`` is the target's id on the
platform, as its publish recorded it) and where each number sits in the answer. The actions
(docs.composio.dev/toolkits, checked 2026-10-04):

* X: ``TWITTER_POST_LOOKUP_BY_POST_ID`` with ``tweet_fields`` ``public_metrics``: views
  (impressions), likes, reposts, replies and quotes.
* Instagram: ``INSTAGRAM_GET_IG_MEDIA_INSIGHTS``, the media's insights by metric name: views,
  reach, likes, comments, shares and saves (a story: views, reach, replies and shares).
* LinkedIn: a member's post gives its reactions only (``LINKEDIN_LIST_REACTIONS``); the share
  statistics action is an organisation's, not a post's.
* YouTube: ``YOUTUBE_GET_VIDEO_DETAILS_BATCH`` with the ``statistics`` part: views, likes and
  comments.
* TikTok has none: its Composio toolkit has no statistics for a video, and its publish
  returns a publish id, not the video's. Its numbers stay empty.

A read runs only where the capability registry says the workspace can run it now
(``capabilities.runnable_actions``: connected, the action synced, not deny-listed). A
number the platform does not give is left out, never a zero. Pure: no database, no Composio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from modules.socials.channel_adapters import CHANNEL_ADAPTERS

REMOTE_ID = "$remote_id"
READINGS: Tuple[int, ...] = (1, 7)  # days after a target went out
# Every number a read may give, in the order the Posted view shows them.
NUMBER_KEYS = ("views", "reach", "likes", "comments", "shares", "saves", "replies", "reposts", "quotes", "reactions")
# The ones that are someone acting on the post: their sum is its engagement (O7, open: this wave's measure).
ENGAGEMENT_KEYS = ("likes", "comments", "shares", "saves", "replies", "reposts", "quotes", "reactions")
_READ_KEYS = frozenset({"action", "params", "numbers", "insights", "kind_params"})


@dataclass(frozen=True)
class ResultRead:
    """One channel's read: the action, its params, and each number's place in the answer."""

    toolkit: str
    action: str
    params: Mapping[str, Any]
    numbers: Mapping[str, str]  # number → "a.b|c.d": the first path holding it; or a metric name (insights)
    insights: bool = False  # the answer is Graph API insights: [{"name", "values": [{"value"}]}]
    kind_params: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: MappingProxyType({}))


def parse_read(toolkit: str, raw: Any) -> ResultRead:
    """A channel's ``results`` entry, checked; ``ValueError`` naming what is wrong."""
    if not isinstance(raw, Mapping) or set(raw) - _READ_KEYS or not isinstance(raw.get("action"), str):
        raise ValueError(f"{toolkit}.results is an object of {sorted(_READ_KEYS)} with an action")
    numbers = raw.get("numbers")
    if not isinstance(numbers, Mapping) or not numbers or set(numbers) - set(NUMBER_KEYS):
        raise ValueError(f"{toolkit}.results.numbers maps numbers among {', '.join(NUMBER_KEYS)} to where they are")
    return ResultRead(
        toolkit=toolkit, action=raw["action"].upper(), params=MappingProxyType(dict(raw.get("params") or {})),
        numbers=MappingProxyType({key: str(where) for key, where in numbers.items()}), insights=raw.get("insights") is True,
        kind_params=MappingProxyType({kind: MappingProxyType(dict(extra)) for kind, extra in (raw.get("kind_params") or {}).items()}),
    )


READS: Mapping[str, ResultRead] = MappingProxyType({
    toolkit: parse_read(toolkit, adapter["results"]) for toolkit, adapter in CHANNEL_ADAPTERS.items() if adapter.get("results")
})


def params_for(read: ResultRead, remote_id: str, post_kind: str) -> Dict[str, Any]:
    """The read's params for one target: ``$remote_id`` filled in, a kind's own params over them."""
    def fill(value: Any) -> Any:
        if isinstance(value, list):
            return [fill(item) for item in value]
        return remote_id if value == REMOTE_ID else value

    return {name: fill(value) for name, value in {**read.params, **read.kind_params.get(post_kind, {})}.items()}


def _number(value: Any) -> Optional[int]:
    """A count as the platform gave it (YouTube's are text); ``None`` when it is not one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value >= 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _at(answer: Any, path: str) -> Any:
    node = answer
    for part in path.split("."):
        if isinstance(node, Mapping):
            node = node.get(part)
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def _insight(answer: Any, name: str) -> Any:
    """A Graph API insight's value by metric name: ``values[0].value`` or ``total_value.value``."""
    rows = answer.get("data") if isinstance(answer, Mapping) else answer
    for row in rows if isinstance(rows, list) else ():
        if isinstance(row, Mapping) and row.get("name") == name:
            return _at(row, "values.0.value") if _at(row, "values.0.value") is not None else _at(row, "total_value.value")
    return None


def numbers_from(read: ResultRead, answer: Any) -> Dict[str, int]:
    """The numbers the answer gives, by key; one it does not give is left out."""
    found: Dict[str, int] = {}
    for key, where in read.numbers.items():
        value = _insight(answer, where) if read.insights else next(
            (v for v in (_at(answer, path) for path in where.split("|")) if v is not None), None
        )
        number = _number(value)
        if number is not None:
            found[key] = number
    return found


def engagement(numbers: Mapping[str, Any]) -> int:
    """Everyone who acted on the post, as far as its numbers say."""
    return sum(int(numbers.get(key) or 0) for key in ENGAGEMENT_KEYS)
