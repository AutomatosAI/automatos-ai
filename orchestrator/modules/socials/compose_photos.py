"""F378 (night 11, 7 Oct): the composer knows which templates show a photo, and asks for it.

B19: for a brief saying "I haven't got the photo yet", compose picked "Just the photo",
said nothing (``visual_prompts: {}``), and the render was a giant brand-name wordmark.
The picker saw only each template's name and fields, never its photo spots. Now:

* each template the composer is given lists its photo spots (``photo_slots``: the image
  slots, each with its label and whether the post renders only once it is filled,
  ``"required": true`` in the template, ``core.social_templates.slot_required``);
* a brief that says there is no photo (:func:`brief_says_no_photo`) never gets a
  template whose photo is required, unless the owner chose that template;
* a proposal whose template shows a photo says so: a required photo is a warning and an
  owner's question ("A photo for 'Photo'"), an optional one a warning that the brand's
  colours show there until one is added. A slot an AI tool is asked to fill
  (``ComposeContext.visual_slots``, the plan maker's) needs neither.

The render refuses a post whose required photo is still empty (``render.bundle_for``),
with :data:`NEEDS_PHOTO`, so a saved post never prints the stand-in a preview shows.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from core.social_templates import IMAGE_SLOT, slot_required

NEEDS_PHOTO = "This template needs a photo: add one to '{label}' (upload one, pick one from the Library, or let AI make one)"
SHOWS_PHOTO = "This template shows a photo in '{label}': add one, or your brand's colours show there"
PHOTO_QUESTION = "A photo for '{label}'"
_NO_PHOTO = re.compile(
    r"\b(?:no|without(?:\s+(?:a|any|the|my))?|(?:have\s*n't|haven't|don't\s+have|do\s+not\s+have|not\s+got|have\s+no)"
    r"(?:\s+got)?(?:\s+(?:a|any|the|my|our))?)\s+(?:photos?|pictures?|images?|pics?|shots?)\b"
    r"|\b(?:photos?|pictures?)\s+(?:later|to\s+come|(?:is|are)\s*(?:n't|\s+not)\s+ready|not\s+ready)\b",
    re.IGNORECASE,
)


def photo_slots(blocks: Any) -> List[Dict[str, Any]]:
    """The template's photo spots (its image slots), each ``{slot, label, required}``."""
    slots = blocks.get("slots") if isinstance(blocks, Mapping) and isinstance(blocks.get("slots"), Mapping) else {}
    return [
        {"slot": name, "label": str(spec.get("label") or name), "required": slot_required(spec)}
        for name, spec in slots.items()
        if isinstance(spec, Mapping) and spec.get("kind") == IMAGE_SLOT
    ]


def brief_says_no_photo(brief: str) -> bool:
    """Whether the brief says there is no photo ("I haven't got the photo yet", "no picture")."""
    return bool(_NO_PHOTO.search((brief or "").replace("’", "'")))


def needs_a_photo(template: Mapping[str, Any]) -> bool:
    """Whether a template entry has a photo spot the post renders only once it is filled."""
    return any(slot.get("required") for slot in template.get("photo_slots") or ())


def without_required_photos(templates: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """The templates a brief without a photo may use: none whose photo is required."""
    return [template for template in templates if not needs_a_photo(template)]


def photo_notes(proposal: Mapping[str, Any], ctx: Any) -> Tuple[List[str], List[str]]:
    """The warnings and the owner's questions for the photo spots of the proposal's template."""
    filled_by_ai = {str(slot.get("slot")) for slot in getattr(ctx, "visual_slots", ()) or ()}
    warnings: List[str] = []
    questions: List[str] = []
    for slot in (proposal.get("template") or {}).get("photo_slots") or ():
        if slot.get("slot") in filled_by_ai:
            continue
        if slot.get("required"):
            warnings.append(NEEDS_PHOTO.format(label=slot["label"]))
            questions.append(PHOTO_QUESTION.format(label=slot["label"]))
        else:
            warnings.append(SHOWS_PHOTO.format(label=slot["label"]))
    return warnings, questions
