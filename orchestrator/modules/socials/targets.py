"""PRD-251 US-204 (S3.3-prep): a post's channels, its targets, are approved content.

A post's targets are where it publishes: one per channel (a Composio toolkit) and
post kind, each with its options, the values of the kind's ``$option.<name>``
sources in ``channel_adapters.py`` (LinkedIn's author, TikTok's privacy level and
AI label, YouTube's category, privacy and tags). Before Wave 2 the content hash
(D6) did not cover them, so a channel added after approval would have published
unapproved. Now it does ("Waves 2+3 build"):

* ``target_set`` is what an approval covers of them: each target's toolkit, kind
  and options, sorted. ``service.compute_content_hash`` adds it to the hash when
  the post has any, so a post with no target hashes as it did before.
* ``validate_targets`` checks the shape of a replacement set, and
  ``replace_targets`` writes it onto the post: one row per channel and kind, keyed
  ``sp:{post}:{toolkit}:{kind}``. ``service.update_post`` runs both, so a change
  to an approved or scheduled post's channels voids its approval like any edit.
* A row's ``action_plan`` holds its options and ``steps``, the kind's action
  sequence as the channel registry resolved it (the api adds them:
  ``api/socials_targets.py``). The steps are how the platform publishes, not what
  was approved: outside the hash.

Pure: no database, no registry and no FastAPI. A problem raises ``ValueError``
naming it, which the lifecycle answers as ``InvalidPost`` (422).
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from uuid import uuid4

from core.models.socials import SOCIAL_TARGET_POST_KINDS, SocialPost, SocialPostTarget

TARGETS = "targets"  # the post's field: its relationship, and the edit's key
TARGET_KEYS = ("toolkit", "post_kind", "options")  # what an approval covers
STEPS = "steps"  # the registry's resolved action sequence, set by the api, never a client
PENDING = "pending"  # a new target's status: Wave 3 publishes it
KEY_PREFIX = "sp"
TARGETS_MAX = 50  # channel kinds per post

# A Composio toolkit's name. At most TOOLKIT_MAX_CHARS, so a target's key
# (sp:{post}:{toolkit}:{kind}) fits social_post_targets.idempotency_key's 128.
TOOLKIT_MAX_CHARS = 64
TOOLKIT_NAME = re.compile(rf"^[a-z][a-z0-9_]{{0,{TOOLKIT_MAX_CHARS - 1}}}$")

# An option is text, a number, true or false, or a list of text (YouTube's tags).
OPTION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")  # as the adapter data's $option.<name>
OPTION_TEXT_MAX_CHARS = 500
OPTION_LIST_MAX_ITEMS = 30


# ---- what an approval covers -----------------------------------------------


def _order(target: Mapping[str, Any]) -> Tuple[str, str]:
    return str(target["toolkit"]), str(target["post_kind"])


def _approved_part(target: Any) -> Dict[str, Any]:
    """A row, or a mapping of the same keys (``to_dict``'s or ``validate_targets``')."""
    if isinstance(target, Mapping):
        toolkit, kind, options = target.get("toolkit"), target.get("post_kind"), target.get("options")
    else:
        toolkit, kind, options = target.toolkit, target.post_kind, target.options
    return {"toolkit": toolkit, "post_kind": kind, "options": dict(options or {})}


def approved_set(targets: Any) -> List[Dict[str, Any]]:
    """What an approval covers of ``targets``: each one's toolkit, post kind and
    options, sorted by toolkit, then kind."""
    return sorted((_approved_part(target) for target in targets or []), key=_order)


def target_set(post: Any) -> List[Dict[str, Any]]:
    """The post's channels as an approval covers them (``approved_set``)."""
    return approved_set(getattr(post, TARGETS, None))


# ---- a replacement set, checked --------------------------------------------


def _option_value(value: Any, where: str) -> Any:
    if isinstance(value, (bool, int)) or (isinstance(value, float) and math.isfinite(value)):
        return value
    if isinstance(value, str) and len(value) <= OPTION_TEXT_MAX_CHARS:
        return value
    texts = isinstance(value, list) and all(isinstance(v, str) and len(v) <= OPTION_TEXT_MAX_CHARS for v in value)
    if texts and len(value) <= OPTION_LIST_MAX_ITEMS:
        return list(value)
    raise ValueError(
        f"{where} must be text of at most {OPTION_TEXT_MAX_CHARS} characters, a number, true or false, "
        f"or a list of at most {OPTION_LIST_MAX_ITEMS} texts"
    )


def _options(value: Any, where: str) -> Dict[str, Any]:
    """``{name: value}``; ``None`` leaves an option unset. Which names a kind takes
    is the registry's to say."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object of option names and values")
    clean: Dict[str, Any] = {}
    for name, option in value.items():
        if not isinstance(name, str) or not OPTION_NAME.match(name):
            raise ValueError(f"{where}: {name!r} is not an option name (lower-case letters, digits and _)")
        if option is not None:
            clean[name] = _option_value(option, f"{where}.{name}")
    return clean


def _target(item: Any, where: str) -> Dict[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError(f"{where} must be an object with toolkit, post_kind and options")
    unknown = [key for key in item if key not in TARGET_KEYS + (STEPS,)]
    if unknown:
        raise ValueError(f"{where} keys must be {list(TARGET_KEYS)}, got {unknown!r}")
    toolkit = item.get("toolkit")
    toolkit = toolkit.strip().lower() if isinstance(toolkit, str) else ""
    if not TOOLKIT_NAME.match(toolkit):
        raise ValueError(f"{where}.toolkit must name a channel's Composio toolkit, such as linkedin")
    if item.get("post_kind") not in SOCIAL_TARGET_POST_KINDS:
        raise ValueError(f"{where}.post_kind must be one of {list(SOCIAL_TARGET_POST_KINDS)}")
    steps = item.get(STEPS) or []
    if not isinstance(steps, (list, tuple)) or not all(isinstance(step, Mapping) for step in steps):
        raise ValueError(f"{where}.{STEPS} must be the channel registry's list of steps")
    return {
        "toolkit": toolkit,
        "post_kind": item["post_kind"],
        "options": _options(item.get("options"), f"{where}.options"),
        STEPS: [dict(step) for step in steps],
    }


def validate_targets(value: Any) -> List[Dict[str, Any]]:
    """A replacement set: a list of ``{"toolkit", "post_kind", "options"}`` (plus the
    api's ``steps``), each channel and kind once, returned sorted by them. The shape
    only: whether the workspace can post a kind now is the registry's to say."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("targets must be a list of {toolkit, post_kind, options}")
    if len(value) > TARGETS_MAX:
        raise ValueError(f"a post publishes to at most {TARGETS_MAX} channel kinds")
    clean = sorted((_target(item, f"targets[{i}]") for i, item in enumerate(value)), key=_order)
    for before, after in zip(clean, clean[1:]):
        if _order(before) == _order(after):
            raise ValueError(f"targets lists {before['toolkit']} {before['post_kind']} twice")
    return clean


# ---- the rows ----------------------------------------------------------------


def target_key(post_id: Any, toolkit: str, post_kind: str) -> str:
    """A target's idempotency key: one per post, channel and kind, kept across edits."""
    return f"{KEY_PREFIX}:{post_id}:{toolkit}:{post_kind}"


def replace_targets(post: SocialPost, targets: Sequence[Mapping[str, Any]]) -> None:
    """``post``'s targets become ``targets`` (``validate_targets``'): a channel and
    kind the post already has keeps its row, so its key and receipt stay; a new one
    is a pending row keyed ``sp:{post}:{toolkit}:{kind}``; one left out is deleted
    (the relationship's delete-orphan). A new list, and a new ``action_plan`` on
    every row: JSON is reassigned, never edited in place."""
    if post.id is None:
        raise ValueError("a post needs its id before its channels are set")
    kept = {(row.toolkit, row.post_kind): row for row in post.targets or []}
    rows = []
    for target in targets:
        toolkit, kind = target["toolkit"], target["post_kind"]
        row = kept.get((toolkit, kind)) or SocialPostTarget(
            id=uuid4(),
            toolkit=toolkit,
            post_kind=kind,
            idempotency_key=target_key(post.id, toolkit, kind),
            status=PENDING,
            attempts=0,
        )
        row.action_plan = {"options": dict(target["options"]), STEPS: [dict(step) for step in target[STEPS]]}
        rows.append(row)
    post.targets = rows
