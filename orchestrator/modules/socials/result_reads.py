"""PRD-251C (C7, US-C402): a published post's numbers, read through its channel's own read
action in Composio (Composio only, D15: no client of our own).

:data:`READS` holds each channel's read: its action, the params (``$remote_id`` is the
target's id on the platform, as its publish recorded it) and where each number sits in the
answer. The actions (docs.composio.dev/toolkits, checked 2026-10-04):

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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

REMOTE_ID = "$remote_id"
READINGS: Tuple[int, ...] = (1, 7)  # days after a target went out
# Every number a read may give, in the order the Posted view shows them.
NUMBER_KEYS = ("views", "reach", "likes", "comments", "shares", "saves", "replies", "reposts", "quotes", "reactions")
# The ones that are someone acting on the post: their sum is its engagement (O7, open: this wave's measure).
ENGAGEMENT_KEYS = ("likes", "comments", "shares", "saves", "replies", "reposts", "quotes", "reactions")
_TWEET = ("data.public_metrics.{key}", "public_metrics.{key}")
_VIDEO = ("items.0.statistics.{key}", "data.items.0.statistics.{key}")


def _paths(templates: Sequence[str], key: str) -> str:
    return "|".join(template.format(key=key) for template in templates)


@dataclass(frozen=True)
class ResultRead:
    """One channel's read: the action, its params, and each number's place in the answer."""

    toolkit: str
    action: str
    params: Mapping[str, Any]
    numbers: Mapping[str, str]  # number → "a.b|c.d": the first path holding it; or a metric name (insights)
    insights: bool = False  # the answer is Graph API insights: [{"name", "values": [{"value"}]}]
    kind_params: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: MappingProxyType({}))


READS: Mapping[str, ResultRead] = MappingProxyType({
    "twitter": ResultRead(
        "twitter", "TWITTER_POST_LOOKUP_BY_POST_ID", {"id": REMOTE_ID, "tweet_fields": ["public_metrics"]},
        {"views": _paths(_TWEET, "impression_count"), "likes": _paths(_TWEET, "like_count"),
         "reposts": _paths(_TWEET, "retweet_count"), "replies": _paths(_TWEET, "reply_count"), "quotes": _paths(_TWEET, "quote_count")},
    ),
    "instagram": ResultRead(
        "instagram", "INSTAGRAM_GET_IG_MEDIA_INSIGHTS",
        {"ig_media_id": REMOTE_ID, "metric": ["views", "reach", "likes", "comments", "shares", "saved"]},
        {"views": "views", "reach": "reach", "likes": "likes", "comments": "comments", "shares": "shares", "saves": "saved",
         "replies": "replies"},
        insights=True,
        kind_params={"story": {"metric": ["views", "reach", "replies", "shares"]}},
    ),
    "linkedin": ResultRead(
        "linkedin", "LINKEDIN_LIST_REACTIONS", {"entity": REMOTE_ID, "count": 1},
        {"reactions": "paging.total|data.paging.total|response_dict.paging.total"},
    ),
    "youtube": ResultRead(
        "youtube", "YOUTUBE_GET_VIDEO_DETAILS_BATCH", {"id": [REMOTE_ID], "parts": ["statistics"]},
        {"views": _paths(_VIDEO, "viewCount"), "likes": _paths(_VIDEO, "likeCount"), "comments": _paths(_VIDEO, "commentCount")},
    ),
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
