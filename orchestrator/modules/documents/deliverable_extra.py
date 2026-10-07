"""What a generated document's Deliverable records in ``extra``, and who it is for.

F354 (5 Oct): the owner adds an invoice or a letter to knowledge so that later "every
file for customer X" finds it. The name of the client, customer or recipient is known
for sure only in the generation data (the presets read ``data.client_name`` and
``data.recipient_name``; the data-details block reads ``customer_name``), and that data
is gone once the file is rendered. :func:`the_parties_are_remembered` wraps
``DocumentGenerationService.generate`` and keeps those names on the result
(``GeneratedDocument.parties``); :func:`deliverable_extra` writes them to the
Deliverable's ``extra.parties`` beside what it already recorded (the render quality of
P2-09 S4, the template of PRD-242 S4, the music of PRD-251 S1.6, the tags of 7 Oct),
which moved here out of ``register_as_deliverable`` (generation_service.py is over 800 lines).
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from services.deliverable_tags import TAGS_KEY, tags_of

Async = Callable[..., Awaitable[Any]]

# The data keys that name who a document is for, and how the knowledge copy labels them.
PARTY_LABELS: Dict[str, str] = {
    "client_name": "Client",
    "customer_name": "Customer",
    "recipient_name": "Recipient",
}
PARTY_CHARS = 200


def parties_named_in(data: Any) -> Dict[str, str]:
    """The client, customer and recipient names ``data`` gives, trimmed; blanks left out."""
    if not isinstance(data, dict):
        return {}
    named = {key: data.get(key) for key in PARTY_LABELS}
    return {key: value.strip()[:PARTY_CHARS] for key, value in named.items()
            if isinstance(value, str) and value.strip()}


def the_parties_are_remembered(generate: Async) -> Async:
    """Wrap ``DocumentGenerationService.generate`` (called with keywords): its result
    carries the names its data gave (a copy of the result, the original untouched)."""
    @functools.wraps(generate)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await generate(self, *args, **kwargs)
        parties = parties_named_in(kwargs.get("data"))
        if not parties or not dataclasses.is_dataclass(result):
            return result
        return dataclasses.replace(result, parties=parties)
    return wrapped


def deliverable_extra(result: Any, template_id: Optional[Any] = None, tags: Sequence[str] = ()) -> Dict[str, Any]:
    """The ``extra`` a generated document's Deliverable is registered with; ``tags`` the
    Deliverable's own tags (7 Oct), already validated where they came in."""
    extra: Dict[str, Any] = {
        "render": {
            "unresolved_count": len(result.unresolved),
            "unknown_count": len(result.unknown),
            "template_lane": result.template_lane,
        }
    }
    if template_id:
        extra["template_id"] = str(template_id)
    if getattr(result, "template_name", None):
        extra["template_name"] = result.template_name
    # PRD-251 S1.6: the music a social video mixed; a post that attaches
    # this Deliverable carries its credit line (modules/socials/credits.py).
    if getattr(result, "music", None):
        extra["music"] = dict(result.music)
    if getattr(result, "parties", None):
        extra["parties"] = dict(result.parties)
    own_tags = tags_of(list(tags))
    if own_tags:
        extra[TAGS_KEY] = own_tags
    return extra


def _named_parties(extra: Any) -> List[Tuple[str, str]]:
    """``[("Client", "Northwind Traders"), …]`` from a Deliverable's ``extra``."""
    parties = extra.get("parties") if isinstance(extra, dict) else None
    if not isinstance(parties, dict):
        return []
    return [(PARTY_LABELS[key], value.strip()) for key, value in parties.items()
            if key in PARTY_LABELS and isinstance(value, str) and value.strip()]


def party_lines(extra: Any) -> List[str]:
    """``["Client: Northwind Traders", …]``: who the document is for, as the knowledge
    copy's description says it."""
    return [f"{label}: {name}" for label, name in _named_parties(extra)]


def party_tags(extra: Any) -> List[str]:
    """``["client:Northwind Traders", …]``: the same names as document tags."""
    return [f"{label.lower()}:{name}" for label, name in _named_parties(extra)]


__all__ = ["PARTY_LABELS", "deliverable_extra", "parties_named_in", "party_lines", "party_tags",
           "the_parties_are_remembered"]
