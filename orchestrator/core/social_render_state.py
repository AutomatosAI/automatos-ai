"""P256-FIX-SOC-5 (F437, socials day 14 Oct): where a post's render stands, read from its history.

A post showed "Rendering" with no start time and no deadline, and Auto could not confirm a render
it started: the start was only a ``render`` entry in ``review_log``, and ``SocialPost.to_dict`` had
no render field. :func:`render_block` derives one from the history, as ``to_dict``'s ``render``
(placed right after ``status``, so a cut tool answer still carries it):

* ``rendering``: the post is rendering; ``started_at`` is the last ``render`` entry's time and
  ``gives_up_at`` when that render gives up (``service.start_render``);
* ``done`` / ``failed``: the newest render outcome; ``finished_at`` is its time and ``why`` a
  failure's reason (a draft the platform refused has no ``started_at``);
* ``waiting``: a draft whose last entry asks for the owner's input (``service.waits_for_input``),
  ``why`` what it asks;
* ``None``: no render yet.

The action names are literal here (``core`` imports nothing under ``modules``); a test pins
them equal to ``modules.socials.service``'s ``ACTION_*``. The name ``finished_at`` is the one
render time the block adds: a target's own ``published_at`` (P256-FIX-SOC-4) is unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

RENDER, RENDER_DONE, RENDER_FAILED, NEEDS_INPUT = "render", "render_done", "render_failed", "needs_input"
RENDERING_STATUS, DRAFT_STATUS = "rendering", "draft"
RENDERING, DONE, FAILED, WAITING = "rendering", "done", "failed", "waiting"
_OUTCOMES = {RENDER_DONE: DONE, RENDER_FAILED: FAILED}


def _entries(review_log: Any) -> Sequence[Mapping[str, Any]]:
    return [entry for entry in (review_log or []) if isinstance(entry, Mapping)]


def _last(entries: Sequence[Mapping[str, Any]], *actions: str) -> int:
    """The index of the newest entry of one of ``actions``; -1 when there is none."""
    return next((index for index in range(len(entries) - 1, -1, -1) if entries[index].get("action") in actions), -1)


def _block(state: str, started: Optional[Mapping[str, Any]], *, finished_at: Any = None, why: Any = None) -> Dict[str, Any]:
    return {
        "state": state,
        "started_at": started.get("at") if started else None,
        "gives_up_at": started.get("gives_up_at") if started else None,
        "finished_at": finished_at,
        "why": why,
    }


def render_block(status: Any, review_log: Any) -> Optional[Dict[str, Any]]:
    """Where the post's render stands (the module note); ``None`` before any render."""
    entries = _entries(review_log)
    start = _last(entries, RENDER)
    started = entries[start] if start >= 0 else None
    if status == RENDERING_STATUS:
        return _block(RENDERING, started)
    if status == DRAFT_STATUS and entries and entries[-1].get("action") == NEEDS_INPUT:
        return _block(WAITING, None, why=entries[-1].get("comment"))
    end = _last(entries, RENDER_DONE, RENDER_FAILED)
    if end < 0 or end < start:
        return None  # no render ended, or the newest one has not ended (it was rolled back)
    outcome = entries[end]
    if _last(entries[:end], RENDER_DONE, RENDER_FAILED) > start:
        started = None  # an earlier render ended after that start: this outcome never started (a refusal)
    why = outcome.get("comment") if outcome.get("action") == RENDER_FAILED else None
    return _block(_OUTCOMES[outcome["action"]], started, finished_at=outcome.get("at"), why=why)


__all__ = ["DONE", "FAILED", "NEEDS_INPUT", "RENDER", "RENDERING", "RENDER_DONE", "RENDER_FAILED", "WAITING", "render_block"]
