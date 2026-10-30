"""PRD-270 K10 (§5.7) — what a public Marketplace read may say about an item: one allowlist per type.

A visitor sees an item's display fields and nothing else. The member routes' answers can't be reused:
``api/marketplace.py`` ``list_items`` answers an agent's configuration and model config, a Playbook's
steps, inputs, outputs and configs, and a Playbook creator's email. So each reader maps its row field by
field through ``display_item``, which refuses any key outside the type's allowlist, and a field can't
leave by accident:

* every type: ``key`` (the id the signed-in tab deep-links by: an agent's or a Playbook's catalogue id,
  a package's slug), ``name``, ``description``, ``category``, ``icon``, ``tags``, ``publisher`` (a
  display name, never an email: ``publishers.py``) and ``requiredApps`` (lower-case app slugs);
* a package also its ``members``, each only its ``type`` and ``name``.

No workspace, user, owner, install or clone id is on any list. Text is answered as the row holds it,
trimmed; a list keeps only its non-empty strings, so a malformed JSON column answers nothing rather
than its raw value.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional

MARKETPLACE_OWNER = "marketplace"
DEFAULT_PUBLISHER = "Automatos"

DISPLAY_FIELDS: FrozenSet[str] = frozenset(
    {"key", "name", "description", "category", "icon", "tags", "publisher", "requiredApps"}
)
AGENT_FIELDS = DISPLAY_FIELDS
PLAYBOOK_FIELDS = DISPLAY_FIELDS
PACKAGE_FIELDS: FrozenSet[str] = DISPLAY_FIELDS | {"members"}
PACKAGE_MEMBER_FIELDS: FrozenSet[str] = frozenset({"type", "name"})

# The answer around the items, and each type's item and nested allowlists, as the route and its tests read them.
ANSWER_FIELDS: FrozenSet[str] = frozenset({"type", "items", "limit", "offset", "more"})
ITEM_FIELDS: Mapping[str, FrozenSet[str]] = MappingProxyType(
    {"agents": AGENT_FIELDS, "playbooks": PLAYBOOK_FIELDS, "packages": PACKAGE_FIELDS}
)
NESTED_FIELDS: Mapping[str, FrozenSet[str]] = MappingProxyType({"members": PACKAGE_MEMBER_FIELDS})


class FieldNotAllowed(ValueError):
    """A reader asked to answer a field its type's allowlist doesn't name: a programming error, never sent."""


def allowed_item(allowed: FrozenSet[str], **fields: Any) -> Dict[str, Any]:
    """``fields`` as one public item, or ``FieldNotAllowed`` when any key is outside ``allowed``."""
    outside = sorted(set(fields) - allowed)
    if outside:
        raise FieldNotAllowed(f"not on the public allowlist: {', '.join(outside)}")
    return dict(fields)


def text_or_none(value: Any) -> Optional[str]:
    """``value`` trimmed when it is a non-empty string, else ``None``."""
    if not isinstance(value, str):
        return None
    return value.strip() or None


def strings(value: Any) -> List[str]:
    """The non-empty strings of a JSON list, trimmed; anything that isn't a list answers none."""
    if not isinstance(value, (list, tuple)):
        return []
    return [text for text in (text_or_none(entry) for entry in value) if text is not None]


def app_slugs(values: Iterable[Any]) -> List[str]:
    """The distinct app slugs among ``values``, lower-case, in first-seen order."""
    return list(dict.fromkeys(text.lower() for text in strings(list(values))))


def display_item(
    allowed: FrozenSet[str],
    *,
    key: Any,
    name: Any,
    description: Any = None,
    category: Any = None,
    icon: Any = None,
    tags: Any = (),
    publisher: str = DEFAULT_PUBLISHER,
    required_apps: Iterable[Any] = (),
    **extra: Any,
) -> Dict[str, Any]:
    """One item's display fields, each mapped from the row's value; ``extra`` is the type's own plain facts.

    Raises ``FieldNotAllowed`` when ``extra`` names a key outside ``allowed``.
    """
    return allowed_item(
        allowed,
        key=str(key), name=text_or_none(name) or "", description=text_or_none(description),
        category=text_or_none(category), icon=text_or_none(icon), tags=strings(tags), publisher=publisher,
        requiredApps=app_slugs(required_apps), **extra,
    )


__all__ = [
    "AGENT_FIELDS", "ANSWER_FIELDS", "DEFAULT_PUBLISHER", "DISPLAY_FIELDS", "FieldNotAllowed", "ITEM_FIELDS",
    "MARKETPLACE_OWNER", "NESTED_FIELDS", "PACKAGE_FIELDS", "PACKAGE_MEMBER_FIELDS", "PLAYBOOK_FIELDS",
    "allowed_item", "app_slugs", "display_item", "strings", "text_or_none",
]
