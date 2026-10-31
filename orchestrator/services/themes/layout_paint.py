"""PRD-271 §4.4 — a theme's layout slots resolved for one look, as the phone gets them.

Every slot of the contract is resolved, in the contract's order:

* a slot the block leaves out takes the contract's ``default`` (a required one has none, so it can't be
  resolved; the manifest check refuses that block anyway);
* a literal is the phone's as written (a number as a whole ``int``);
* ``{"look": "--name"}`` is the variable the web paints for the look through ``boot/prepaint.json``
  (``mobile_tokens.paint_vars``, the walk ``paint_look`` makes, any form kept), and ``{"look": "knob"}``
  the painted look's knob (``mobile_tokens.painted_look``), each held to its slot's kind
  (``layout_slots``).

The look is painted as the web paints it: the look when it fits the table, else the table's defaults
whole. When a look paints a value that doesn't fit its slot, the defaults are resolved instead, whole,
as the web does with a look that doesn't fit; when the defaults can't fill a slot either, the layout
can't be drawn (``Resolved.unresolved`` names the slots) and the phone draws the theme's Studio skin.

:func:`layout_tone` is the contract's tone rule over the resolved slots (D-7): ``dark`` under a
``ground`` rule when the ground slot is darker than the text slot (look-kit ``isDarkGround``,
``mobile_tokens.ground_tone``), else ``light``; a fixed rule's own tone otherwise.

The answer (``mobile.py``), the reader (``mobile_layout.py``) and the review (``layout_contrast.py``) share
this one resolution, so the phone is served exactly what was checked and measured.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

from services.themes.layout_contracts import TONE_GROUND, LayoutContract, contract_of
from services.themes.layout_slots import knob_slot_value, look_ref, slot_value, takes_variable
from services.themes.mobile_tokens import GROUND_TOKEN, TEXT_TOKEN, ground_tone, paint_vars, painted_look

LAYOUT_FIELD = "layout"
SLOTS_FIELD = "slots"
LAYOUT_FIELDS: Tuple[str, ...] = (LAYOUT_FIELD, SLOTS_FIELD)


class Resolved(NamedTuple):
    """A look's slots: every slot's value, and the slots it couldn't fill (empty when the phone can draw it)."""

    slots: Dict[str, Any]
    unresolved: Tuple[str, ...]


def _object(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def block_contract(block: Mapping[str, Any]) -> Optional[LayoutContract]:
    """The contract a ``mobile`` block names, or None (no ``layout``, or one the host doesn't know)."""
    return contract_of(block.get(LAYOUT_FIELD))


def look_slots(block: Mapping[str, Any]) -> Dict[str, str]:
    """``{slot: the variable or knob it names}`` for each slot the block takes from the look."""
    refs = {name: look_ref(value) for name, value in _object(block.get(SLOTS_FIELD)).items()}
    return {name: ref for name, ref in refs.items() if ref is not None}


class _Painted(NamedTuple):
    """What a look paints: the variables its slots name, and its knobs."""

    variables: Mapping[str, str]
    knobs: Mapping[str, Any]


def _painted(contract: LayoutContract, refs: Mapping[str, str], table: Any, look: Any) -> _Painted:
    """The look painted through ``table`` (None: the defaults): the variables the colour and tint slots name."""
    names = [ref for name, ref in refs.items() if name in contract.slots and takes_variable(contract.slots[name])]
    return _Painted(paint_vars(table, look, names) if names else {}, painted_look(table, look) or {})


def _slot(spec: Mapping[str, Any], name: str, given: Mapping[str, Any], painted: _Painted) -> Any:
    """One slot's value, or None when it can't be filled (module docstring)."""
    if name not in given:
        return spec.get("default")
    ref = look_ref(given[name])
    if ref is None:
        return slot_value(spec, given[name])
    if takes_variable(spec):
        return slot_value(spec, painted.variables.get(ref))
    return knob_slot_value(spec, painted.knobs.get(ref))


def _resolve_once(contract: LayoutContract, given: Mapping[str, Any], painted: _Painted) -> Resolved:
    values = {name: _slot(spec, name, given, painted) for name, spec in contract.slots.items()}
    unresolved = tuple(name for name, value in values.items() if value is None)
    return Resolved({name: value for name, value in values.items() if value is not None}, unresolved)


def resolve_slots(contract: LayoutContract, block: Mapping[str, Any], table: Any, look: Any) -> Resolved:
    """Every slot of ``contract`` for ``look`` (None: the defaults); the defaults whole when the look can't fill one."""
    given, refs = _object(block.get(SLOTS_FIELD)), look_slots(block)
    resolved = _resolve_once(contract, given, _painted(contract, refs, table, look))
    if not resolved.unresolved or look is None or not refs:
        return resolved
    return _resolve_once(contract, given, _painted(contract, refs, table, None))


def layout_tone(contract: LayoutContract, slots: Mapping[str, Any]) -> str:
    """The contract's tone rule over resolved slots (module docstring)."""
    rule = contract.tone["rule"]
    if rule != TONE_GROUND:
        return rule
    return ground_tone({GROUND_TOKEN: slots.get(contract.tone["ground"]), TEXT_TOKEN: slots.get(contract.tone["text"])})


def layout_answer(contract: LayoutContract, resolved: Resolved) -> Dict[str, Any]:
    """The answer's ``layout`` for slots the phone can draw: the contract, its tone and every slot."""
    return {"contract": contract.layout, "tone": layout_tone(contract, resolved.slots), "slots": dict(resolved.slots)}


__all__ = [
    "LAYOUT_FIELD", "LAYOUT_FIELDS", "Resolved", "SLOTS_FIELD", "block_contract", "layout_answer", "layout_tone",
    "look_slots", "resolve_slots",
]
