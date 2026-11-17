"""PRD-251D (US-D108): whether a post's file was made from what the post holds now.

D6 says a post is approved against the exact content that goes out. A post's file (its rendered
image or video, in ``media``) is made from its template and fields, and an edit keeps the file. So
a post that waits for approval, edited and saved, would be approved with a file that shows its old
words. Nothing recorded what a file was made from, so nothing could tell.

A render now records it: :func:`render_inputs_hash`, a sha256 over what a render reads, goes on the
render's ``render_done`` history entry as ``made_from`` (``service.finish_render``). The post is
still ``rendering`` then and cannot be edited, so what the hash reads is what the render read: the
fields Auto wrote before the render started included, and the footage the render made. Then
:func:`file_current` compares the last record with the post as it is now. ``SocialPost.to_dict``
answers it as ``file_current``, and ``service.approve`` refuses a post whose file is not current
with :data:`FILE_OLDER_THAN_WORDS`.

What the hash covers, from the render path (``api/socials.py`` ``render_post``,
``modules/socials/render.py`` ``bundle_for`` and ``footage_plan_for``):

* the template, the format and the chosen length (the cut of the template's timeline);
* each field's value, as ``post_fields.resolved_fields`` reads it. Whether a field is marked a
  claim is not on the file, and a save may mark one again, so the mark is left out;
* each slot's footage or photo as a render reads it: the prompt it asks with, and the file it holds
  once it is done (a shot a render made, an AI option picked, the person's own picture). The AI
  options offered for a slot and what a shot cost change nothing on the file;
* for a video, the voice and the music: the render speaks the script with the voice and mixes the
  music, so a new voice or track leaves the file older than the post. An image render reads neither.

Left out, because they do not change what the file shows: the caption, the title, the brief (Auto
writes empty fields from it before a render starts, and the record covers those fields), the
sources, the planned slot and the channels. The channels choose which sizes a still is rendered at
and whether a 9:16 file keeps a story's safe zone, but a channel change is not a change to the
file's content: the publisher sends each channel the file closest to its shape. The AI-generated
label is checked on its own (``ai_media.label_mismatch``). The brand kit and the template's own
file are the workspace's, not the post's.

Always current: a text post (nothing to render) and a post with no template (the person's own file,
or one left to "Let Auto pick" that a render has not made yet). A post with no record is current
too: a post rendered before this change has none, and treating it as current means no post that
exists today is suddenly refused; a post that was never rendered has none either, and the editor's
Submit renders a post that has no file (``frontend/.../studio/editor-submit.ts``).

``core`` imports nothing under ``modules``: the action and status names are literal here, and
``tests/test_prd251d_w2_file_current.py`` pins them equal to ``modules.socials``'s.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional

from core.social_render_state import RENDER_DONE

# The render_done entry's record of what the file was made from (render_inputs_hash).
MADE_FROM = "made_from"
TEXT_FORMAT = "text"
VIDEO_FORMAT = "video"
FOOTAGE_DONE = "done"  # service.FOOTAGE_DONE: a slot that holds its file
# The refusal an approval gives, and the sentence the Queue and the editor show in place of Approve.
FILE_OLDER_THAN_WORDS = (
    "This file was made before the post last changed, so it does not show the post as it is now. "
    "Render it again, then approve it."
)


def _field_values(variables: Any) -> Dict[str, Any]:
    """Each field's value, as the render reads it; a claim mark is not on the file. A field with
    no value reads as one that is not there (the render takes its default either way), so a save
    that only adds or drops such a field leaves the file current."""
    if not isinstance(variables, Mapping):
        return {}
    return {
        str(name): spec.get("value") for name, spec in variables.items()
        if isinstance(spec, Mapping) and spec.get("value") is not None
    }


def _slot_files(footage: Any) -> Dict[str, Any]:
    """Each slot as a render reads it: the prompt it asks with, and its file once it is done."""
    if not isinstance(footage, Mapping):
        return {}
    return {
        str(slot): {
            "prompt": record.get("prompt"),
            "file": record.get("deliverable_id") if record.get("status") == FOOTAGE_DONE else None,
        }
        for slot, record in footage.items()
        if isinstance(record, Mapping)
    }


def render_inputs(post: Any) -> Dict[str, Any]:
    """What a render of ``post`` reads (the module note), as a plain dict."""
    post_format = getattr(post, "format", None)
    template_id = getattr(post, "template_id", None)
    inputs = {
        "template_id": str(template_id) if template_id is not None else None,
        "format": post_format,
        "length_seconds": getattr(post, "length_seconds", None),
        "fields": _field_values(getattr(post, "variables", None)),
        "slots": _slot_files(getattr(post, "footage", None)),
    }
    if post_format != VIDEO_FORMAT:
        return inputs
    return {**inputs, "voice": getattr(post, "voice", None) or None, "music": getattr(post, "music", None) or None}


def render_inputs_hash(post: Any) -> str:
    """sha256 over canonical JSON of :func:`render_inputs` (sorted keys, no spaces), lowercase hex."""
    canonical = json.dumps(render_inputs(post), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def last_made_from(review_log: Any) -> Optional[str]:
    """The ``made_from`` of the post's newest ``render_done`` entry; None when it has none, or the
    entry is from before this record existed."""
    entries = review_log if isinstance(review_log, (list, tuple)) else ()
    for entry in reversed(entries):
        if isinstance(entry, Mapping) and entry.get("action") == RENDER_DONE:
            made_from = entry.get(MADE_FROM)
            return made_from if isinstance(made_from, str) else None
    return None


def file_current(post: Any) -> bool:
    """Whether the post's file was made from what it holds now (the module note)."""
    if getattr(post, "format", None) == TEXT_FORMAT or getattr(post, "template_id", None) is None:
        return True
    made_from = last_made_from(getattr(post, "review_log", None))
    # No record: rendered before US-D108 recorded one, or never rendered. Current, so no post that
    # exists today is refused; a post with no file renders when it is submitted from the editor.
    return made_from is None or made_from == render_inputs_hash(post)


__all__ = [
    "FILE_OLDER_THAN_WORDS", "FOOTAGE_DONE", "MADE_FROM", "TEXT_FORMAT", "VIDEO_FORMAT", "file_current",
    "last_made_from", "render_inputs", "render_inputs_hash",
]
