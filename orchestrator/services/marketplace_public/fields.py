"""PRD-270 K10 (§5.7) — what a public Marketplace read may say about an item: one allowlist per type.

A visitor sees an item's display fields and nothing else. The member routes' answers can't be reused:
``api/marketplace.py`` ``list_items`` answers an agent's configuration and model config, a Playbook's
steps, inputs, outputs and configs, and a Playbook creator's email. So each reader maps its row field by
field through ``display_item``, which refuses any key outside the type's allowlist, and a field can't
leave by accident:

* every type: ``key`` (the id the signed-in tab deep-links by: an agent's or a Playbook's catalogue id,
  a package's slug, an application's app slug, a plugin's id, a skill's name, an LLM's route key),
  ``name``, ``description``, ``category``, ``icon``, ``tags``, ``publisher`` (a display name, never an
  email: ``publishers.py``) and ``requiredApps`` (lower-case app slugs);
* a package also its ``members``, each only its ``type`` and ``name``;
* an application its ``actionCount`` and ``triggerCount``; a plugin its ``version`` and its
  ``skillsCount``, ``commandsCount``, ``agentsCount`` and ``hooksCount``; a skill its ``version``;
* an LLM route its ``provider`` (the serving provider's slug, the catalogue's ``provider`` filter),
  ``vendor``, ``contextWindow``, ``priceTier`` and ``capabilities``, each only its ``name`` and ``rating``;
* a Featured or Popular item (``featured.py``: an agent or a Playbook on the page's hero, rail and Popular
  strip) also its ``type`` (``agents`` or ``playbooks``), ``version`` and ``installCount``, the catalogue
  row's own aggregate count; the counts read (``counts.py``) answers only ``items`` and ``installs``.

No workspace, user, owner, install, clone or connection id is on any list, nor a workspace's connected,
enabled, installed or key state. Text is answered as the row holds it, trimmed; a list keeps only its
non-empty strings, so a malformed JSON column answers nothing rather than its raw value; a count is a
whole number, else 0; an icon a browser reads as a URL (a scheme, ``word:`` included, or scheme-relative:
``//``, a ``\\`` reading as ``/``, once tabs, newlines and leading control characters are dropped as a
browser drops them) is answered only when it is ``https``, any other icon (an emoji) as its text.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional
from urllib.parse import urlsplit

MARKETPLACE_OWNER = "marketplace"
DEFAULT_PUBLISHER = "Automatos"

DISPLAY_FIELDS: FrozenSet[str] = frozenset(
    {"key", "name", "description", "category", "icon", "tags", "publisher", "requiredApps"}
)
AGENT_FIELDS = DISPLAY_FIELDS
PLAYBOOK_FIELDS = DISPLAY_FIELDS
PACKAGE_FIELDS: FrozenSet[str] = DISPLAY_FIELDS | {"members"}
PACKAGE_MEMBER_FIELDS: FrozenSet[str] = frozenset({"type", "name"})
APPLICATION_FIELDS: FrozenSet[str] = DISPLAY_FIELDS | {"actionCount", "triggerCount"}
PLUGIN_FIELDS: FrozenSet[str] = DISPLAY_FIELDS | {
    "version", "skillsCount", "commandsCount", "agentsCount", "hooksCount",
}
SKILL_FIELDS: FrozenSet[str] = DISPLAY_FIELDS | {"version"}
LLM_FIELDS: FrozenSet[str] = DISPLAY_FIELDS | {"provider", "vendor", "contextWindow", "priceTier", "capabilities"}
LLM_CAPABILITY_FIELDS: FrozenSet[str] = frozenset({"name", "rating"})
FEATURED_FIELDS: FrozenSet[str] = DISPLAY_FIELDS | {"type", "version", "installCount"}
FEATURED_ANSWER_FIELDS: FrozenSet[str] = frozenset({"featured", "popular"})
COUNTS_FIELDS: FrozenSet[str] = frozenset({"items", "installs"})
HTTPS = "https"
SCHEME_RELATIVE = "//"  # a browser loads it on the page's own scheme, from any host
# What a browser's URL parser drops before it reads a URL: tab, LF and CR anywhere, C0 controls and spaces in front.
URL_DROPPED = str.maketrans("", "", "\t\n\r")
URL_LEADING = "".join(chr(code) for code in range(0x21))

# The answer around the items, and each type's item and nested allowlists, as the route and its tests read them.
ANSWER_FIELDS: FrozenSet[str] = frozenset({"type", "items", "limit", "offset", "more"})
ITEM_FIELDS: Mapping[str, FrozenSet[str]] = MappingProxyType({
    "agents": AGENT_FIELDS, "playbooks": PLAYBOOK_FIELDS, "packages": PACKAGE_FIELDS,
    "applications": APPLICATION_FIELDS, "plugins": PLUGIN_FIELDS, "skills": SKILL_FIELDS, "llms": LLM_FIELDS,
})
NESTED_FIELDS: Mapping[str, FrozenSet[str]] = MappingProxyType(
    {"members": PACKAGE_MEMBER_FIELDS, "capabilities": LLM_CAPABILITY_FIELDS}
)


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


def count_of(value: Any) -> int:
    """``value`` when it is a whole number above 0, else 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def https_url_or_none(value: Any) -> Optional[str]:
    """``value`` trimmed when it is an ``https`` URL with a host and no user name or password, else ``None``."""
    text = text_or_none(value)
    if text is None or any(char.isspace() for char in text):
        return None
    try:
        parts = urlsplit(text)
    except ValueError:  # an unbalanced IPv6 bracket: not a URL a visitor's page should load
        return None
    if parts.scheme != HTTPS or not parts.hostname or "@" in parts.netloc:
        return None
    return text


def _reads_as_url(text: str) -> bool:
    """Whether a browser reads ``text`` as a URL: it has a scheme, or starts ``//`` (a ``\\`` reads as ``/``)."""
    probe = text.translate(URL_DROPPED).lstrip(URL_LEADING).replace("\\", "/")
    if probe.startswith(SCHEME_RELATIVE):
        return True
    try:
        return bool(urlsplit(probe).scheme)
    except ValueError:  # an unbalanced IPv6 bracket after ``//``: a URL, which https_url_or_none refuses
        return True


def icon_or_none(value: Any) -> Optional[str]:
    """An item's icon: a URL only when ``https_url_or_none`` keeps it; any other text (an emoji) trimmed."""
    text = text_or_none(value)
    if text is None or not _reads_as_url(text):
        return text
    return https_url_or_none(text)


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
        category=text_or_none(category), icon=icon_or_none(icon), tags=strings(tags), publisher=publisher,
        requiredApps=app_slugs(required_apps), **extra,
    )


__all__ = [
    "AGENT_FIELDS", "ANSWER_FIELDS", "APPLICATION_FIELDS", "COUNTS_FIELDS", "DEFAULT_PUBLISHER", "DISPLAY_FIELDS",
    "FEATURED_ANSWER_FIELDS", "FEATURED_FIELDS", "FieldNotAllowed", "ITEM_FIELDS", "LLM_CAPABILITY_FIELDS",
    "LLM_FIELDS", "MARKETPLACE_OWNER", "NESTED_FIELDS", "PACKAGE_FIELDS",
    "PACKAGE_MEMBER_FIELDS", "PLAYBOOK_FIELDS", "PLUGIN_FIELDS", "SKILL_FIELDS",
    "allowed_item", "app_slugs", "count_of", "display_item", "https_url_or_none", "icon_or_none", "strings",
    "text_or_none",
]
