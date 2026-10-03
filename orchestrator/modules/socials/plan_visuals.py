"""PRD-251B Wave 3 (US-B305 with US-B207): a plan's visual mix, post by post.

A plan shares 100 among ``templates``, ``library``, ``ai_images`` and ``ai_footage``
(``plans.VISUAL_MIX_KEYS``). Each slot draws its visual from the mix by its own key, so a
slot always gets the same one, and over many slots the shares hold:

* **templates**: the template's own visuals, as before;
* **library**: one of the workspace's image or video Deliverables that fits the topic
  becomes the post's media, as the editor's Library does (the plan maker picks it);
* **ai_images** / **ai_footage**: the template's image or video slots an AI tool may fill
  ask for it (``post.footage``), each with the composer's prompt for it, else the topic's.
  The render makes them through the workspace's default toolkit, priced, capped and booked
  (D13), exactly as for a post a person asked for; the brand kit's style follows each prompt.

A visual the slot cannot have (no such slots in its template, nothing in the library that
fits) leaves the template's own visuals.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from core.social_templates import IMAGE_SLOT, VIDEO_SLOT, slot_generatable
from modules.socials.plans import VISUAL_MIX_KEYS

TEMPLATES, LIBRARY, AI_IMAGES, AI_FOOTAGE = VISUAL_MIX_KEYS
SLOT_KIND = {AI_IMAGES: IMAGE_SLOT, AI_FOOTAGE: VIDEO_SLOT}
PERCENT = 100
TOPIC_PROMPT_MAX_CHARS = 600
MIN_WORD_CHARS = 4
_WORD = re.compile(r"[a-z0-9]+")


def visual_for(mix: Mapping[str, Any], slot_key: str) -> str:
    """The slot's visual: a stable draw from the mix by the slot's key."""
    point = int(hashlib.sha256(slot_key.encode("utf-8")).hexdigest(), 16) % PERCENT
    total = 0
    for visual in VISUAL_MIX_KEYS:
        total += int(mix.get(visual) or 0)
        if point < total:
            return visual
    return TEMPLATES


def ai_slots(blocks: Optional[Mapping[str, Any]], visual: str) -> List[Dict[str, str]]:
    """The template's slots an AI tool fills for ``visual``: ``{slot, kind, label}`` each."""
    kind = SLOT_KIND.get(visual)
    slots = (blocks or {}).get("slots") if kind else None
    if not isinstance(slots, Mapping):
        return []
    return [
        {"slot": name, "kind": kind, "label": str(spec.get("label") or name)}
        for name, spec in slots.items()
        if isinstance(spec, Mapping) and spec.get("kind") == kind and slot_generatable(spec)
    ]


def topic_prompt(title: str, angle: Optional[str]) -> str:
    """The prompt a slot takes when the composer wrote none for it: the topic itself."""
    text = " ".join(part.strip() for part in (title, angle or "") if part and part.strip())
    return text[:TOPIC_PROMPT_MAX_CHARS]


def footage_asks(slots: Sequence[Mapping[str, str]], prompts: Mapping[str, str], fallback: str) -> Dict[str, Dict[str, str]]:
    """``post.footage`` for the slots: each with the composer's prompt for it, else ``fallback``."""
    return {slot["slot"]: {"prompt": (prompts.get(slot["slot"]) or fallback).strip()} for slot in slots}


def _words(text: Any) -> set:
    return {word for word in _WORD.findall(str(text or "").lower()) if len(word) >= MIN_WORD_CHARS}


def best_fit(items: Iterable[Mapping[str, Any]], title: str, angle: Optional[str]) -> Optional[Mapping[str, Any]]:
    """The item whose title and summary share most words with the topic (the first of equals:
    the newest, as listed); ``None`` when none shares a word."""
    wanted = _words(title) | _words(angle)
    best, best_score = None, 0
    for item in items:
        score = len(wanted & (_words(item.get("title")) | _words(item.get("summary"))))
        if score > best_score:
            best, best_score = item, score
    return best
