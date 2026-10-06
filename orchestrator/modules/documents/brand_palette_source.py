"""How a save's colour roles become the stored ones (F366, night 10c): a GET body sent back changes nothing.

GET answers every effective role (``palette``) and whether each is ``set`` or
``derived`` (``palette_source``). Sent back as it is, every derived role used to
come back ``set``, pinned at today's colour, and stopped following the kit's
colours; ``palette_source`` in the body was ignored. Now, per role:

* a colour sent empty (``null`` or ``""``) goes back to derived, as before;
* ``palette_source`` ``"derived"`` returns the role to derived (or keeps it there),
  unless the body sends it a colour other than the one it has now: a changed
  colour is the owner's choice, and is set;
* ``palette_source`` ``"set"`` pins the role: at the colour sent, else at the one
  it has now;
* a role derived now, sent at the colour it derives to after the save, stays
  derived. One sent at another colour is set: so a body that changes the primary
  and sends the accent at today's colour keeps that accent;
* a role set now keeps the colour sent.

``palette_source: "derived"`` (the string, not a map) is every role. So a role is
reset with ``{"palette": {"accent": null}}`` or ``{"palette_source": {"accent":
"derived"}}``, and the whole palette with ``{"palette_source": "derived"}``.
A role this does not know is left for the palette's own validation to refuse.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from core.brand_palette import (
    KIT_COLOUR_FIELDS,
    PALETTE_ROLES,
    ROLE_DERIVED,
    ROLE_SET,
    derive_palette,
    effective_palette,
    parse_hex,
)
from modules.documents.brand_system import PALETTE_FIELD, PALETTE_SOURCE_FIELD

# A role the body does not send: nothing changes for it.
_UNSENT = object()
# A role derived now that the body sends a colour, with no palette_source for it:
# derived if that colour is what it derives to after the save, else set.
_DERIVED_IF_SAME = object()


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _same_colour(value: Any, other: Optional[str]) -> bool:
    colour = parse_hex(value)
    return colour is not None and colour == parse_hex(other)


def asked_sources(sources: Any) -> Dict[str, str]:
    """``palette_source`` as each role's asked source: ``"derived"`` alone is every role; anything else is none."""
    if sources == ROLE_DERIVED:
        return {role: ROLE_DERIVED for role in PALETTE_ROLES}
    return dict(sources) if isinstance(sources, Mapping) else {}


def _role_outcome(value: Any, asked: Optional[str], current: Optional[str], current_source: Optional[str]) -> Any:
    """One role's outcome: a colour (set), ``None`` (derived), ``_UNSENT`` (no change) or ``_DERIVED_IF_SAME``."""
    sent = value is not _UNSENT
    if sent and _blank(value):
        return None
    if asked == ROLE_DERIVED:
        return value if sent and not _same_colour(value, current) else None
    if asked == ROLE_SET:
        return value if sent else current
    if sent and current_source == ROLE_DERIVED:
        return _DERIVED_IF_SAME
    return value if sent else _UNSENT


def _derived_after(base: Mapping[str, Any], colours: Mapping[str, Any], outcomes: Mapping[str, Any]) -> Dict[str, str]:
    """The roles as they derive once the save's kit colours and settled roles apply (the undecided ones unset)."""
    stored = base.get(PALETTE_FIELD) if isinstance(base.get(PALETTE_FIELD), Mapping) else {}
    settled = {role: outcome for role, outcome in outcomes.items() if outcome is not _UNSENT}
    palette = {
        role: settled.get(role, stored.get(role))
        for role in PALETTE_ROLES
        if isinstance(settled.get(role, stored.get(role)), str)
    }
    return derive_palette({**base, **colours, PALETTE_FIELD: palette})


def resolved_palette(
    sent: Mapping[str, Any], sources: Any, base: Mapping[str, Any], colours: Mapping[str, Any],
) -> Dict[str, Any]:
    """The palette record a save merges into ``base`` (the stored kit): each role's colour, or ``None`` for derived.

    ``colours`` are the kit colours the same save changes: a role derives from them.
    """
    current, current_sources = effective_palette(base)
    asked = asked_sources(sources)
    outcomes = {
        role: _role_outcome(sent.get(role, _UNSENT), asked.get(role), current.get(role), current_sources.get(role))
        for role in PALETTE_ROLES
    }
    undecided = [role for role, outcome in outcomes.items() if outcome is _DERIVED_IF_SAME]
    if undecided:
        settled = {role: (_UNSENT if role in undecided else outcome) for role, outcome in outcomes.items()}
        after = _derived_after(base, colours, settled)
        outcomes = {
            **outcomes,
            **{role: None if _same_colour(sent[role], after.get(role)) else sent[role] for role in undecided},
        }
    resolved = {role: value for role, value in sent.items() if role not in PALETTE_ROLES}
    return {**resolved, **{role: outcome for role, outcome in outcomes.items() if outcome is not _UNSENT}}


def with_palette_resolved(patch: Mapping[str, Any], base: Mapping[str, Any]) -> Dict[str, Any]:
    """``patch`` without ``palette_source``, its ``palette`` (if any) read by the rules above against ``base``."""
    rest = {key: value for key, value in patch.items() if key != PALETTE_SOURCE_FIELD}
    sources = patch.get(PALETTE_SOURCE_FIELD)
    palette = rest.get(PALETTE_FIELD)
    if palette is not None and not isinstance(palette, Mapping):
        return rest  # not a record: the kit's validation refuses it
    if palette is None and sources is None:
        return rest
    colours = {field: rest[field] for field in KIT_COLOUR_FIELDS if rest.get(field) is not None}
    return {**rest, PALETTE_FIELD: resolved_palette(palette or {}, sources, base, colours)}


__all__ = ["asked_sources", "resolved_palette", "with_palette_resolved"]
