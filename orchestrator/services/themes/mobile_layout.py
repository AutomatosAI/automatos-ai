"""PRD-271 C02 (D-1, D-10, host 1.2.0) — a ``mobile`` block's ``layout`` and ``slots``, checked.

A theme names one of Automatos Go's phone layouts and fills the slots its contract lists
(``layout_contracts.py``; the table is ``data/phone-layouts.json``)::

    "mobile": {"fromLooks": true, "layout": "candy@1",
               "slots": {"page": "#FFF4F9", "accent": {"look": "--bubble-accent"}, "roundness": {"look": "roundness"}}}

:func:`layout_errors` is the manifest's half, called by ``mobile_manifest.mobile_errors``. Each refusal is
``BAD_MOBILE`` at ``mobile.layout`` or ``mobile.slots.<name>``:

* ``layout`` and ``slots`` come together;
* ``layout`` names a contract the host knows (``"<id>@<major>"``: an unknown id or major is refused);
* ``slots`` is an object: no slot the contract lacks, every slot it requires, each value a literal of its
  slot's kind or ``{"look": …}`` naming a variable (colour and tint) or a knob (number and choice) as the
  browser's first paint keeps one (``layout_slots.py``).

``engines.automatos`` admitting a host below ``contract.LAYOUT_MIN_HOST`` is ``mobile_manifest``'s refusal,
and an older host refuses ``layout`` as ``UNKNOWN_FIELD``.

:func:`layout_package_errors` is the reader's half (``package.py``), on a block the manifest check passed:
a slot taken from the look needs ``boot/prepaint.json``; each ``{"look": …}`` names a variable the table's
knobs or pairs write, or a knob of its looks, as the first paint keeps them (``mobile_tokens.boot_table``
and ``first_paint.spec_knobs``, the ports of boot-sanitize.ts and lookSpecOf); and the table's defaults
fill every such slot (``layout_paint.resolve_slots``). Each refusal is ``MOBILE_CANNOT_PAINT`` naming the
slot. The TypeScript twin of the manifest's half is ``frontend/lib/theme-host/mobile-layout.ts``; the
shared corpus (``tests/fixtures/themes/cases.json``) holds both to the same verdicts.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Mapping, Optional

from services.themes.contract import ThemeError, echo, parse_json, theme_error
from services.themes.css_value import PREPAINT_PATH
from services.themes.first_paint import spec_knobs
from services.themes.layout_contracts import KIND_COLOUR, KIND_NUMBER, KIND_TINT, LayoutContract, layout_names
from services.themes.layout_paint import LAYOUT_FIELD, SLOTS_FIELD, block_contract, look_slots, resolve_slots
from services.themes.layout_slots import is_look_source, look_ref, slot_value, takes_variable
from services.themes.mobile_tokens import BootTable, boot_table

MOBILE_CODE = "BAD_MOBILE"
CANNOT_PAINT = "MOBILE_CANNOT_PAINT"
LOOK_VARIABLE = '{"look": "--variable"}'
LOOK_KNOB = '{"look": "knob"}'


def _bad(path: str, message: str) -> List[ThemeError]:
    return [theme_error(MOBILE_CODE, path, message)]


def _kind_sentence(where: str, spec: Mapping[str, Any]) -> str:
    """What a slot of this kind takes."""
    kind = spec["kind"]
    if kind == KIND_COLOUR:
        return f'"{where}" must be a colour ("#rrggbb" or "H S% L%") or {LOOK_VARIABLE}.'
    if kind == KIND_TINT:
        return f'"{where}" must be a colour, "rgba(r,g,b,a)" or "transparent", or {LOOK_VARIABLE}.'
    if kind == KIND_NUMBER:
        return f'"{where}" must be a whole number from {spec["min"]} to {spec["max"]}, or {LOOK_KNOB}.'
    choices = ", ".join(f'"{choice}"' for choice in spec["choices"])
    return f'"{where}" must be one of {choices}, or {LOOK_KNOB}.'


def _source_sentence(where: str, spec: Mapping[str, Any]) -> str:
    """What ``{"look": …}`` names for a slot of this kind."""
    if takes_variable(spec):
        return (
            f'"{where}" takes {LOOK_VARIABLE}: a variable {PREPAINT_PATH} writes, "--" then 1 to 64 lower-case '
            'letters, digits or dashes.'
        )
    return f'"{where}" takes {LOOK_KNOB}: a look knob\'s name, a letter then up to 31 letters, digits or _.'


def _value_errors(name: str, spec: Mapping[str, Any], value: Any) -> List[ThemeError]:
    where = f"mobile.slots.{name}"
    if isinstance(value, dict):
        return [] if is_look_source(spec, look_ref(value)) else _bad(where, _source_sentence(where, spec))
    return [] if slot_value(spec, value) is not None else _bad(where, _kind_sentence(where, spec))


def _slots_errors(contract: LayoutContract, slots: Any) -> List[ThemeError]:
    """No slot the contract lacks, every one it requires, each value of its kind or from the look."""
    if not isinstance(slots, dict):
        return _bad("mobile.slots", f'"mobile.slots" must be an object of {contract.layout}\'s slots and their values.')
    errors = [
        theme_error(MOBILE_CODE, f"mobile.slots.{echo(name)}", f'"{echo(name)}" is not a slot of {contract.layout}.')
        for name in slots if name not in contract.slots
    ]
    for name, spec in contract.slots.items():
        if name in slots:
            errors += _value_errors(name, spec, slots[name])
        elif "default" not in spec:
            errors += _bad(f"mobile.slots.{name}", f'"mobile.slots" needs "{name}": {contract.layout} has no default for it.')
    return errors


def _unknown_layout() -> List[ThemeError]:
    known = ", ".join(f'"{name}"' for name in layout_names())
    return _bad(f"mobile.{LAYOUT_FIELD}", f'"mobile.layout" must name a phone layout and its contract major: one of {known}.')


def layout_errors(block: Mapping[str, Any]) -> List[ThemeError]:
    """Every reason the block's ``layout`` and ``slots`` are refused (module docstring); empty without them."""
    has_layout, has_slots = LAYOUT_FIELD in block, SLOTS_FIELD in block
    if not has_layout and not has_slots:
        return []
    if not has_slots:
        return _bad(f"mobile.{SLOTS_FIELD}", '"mobile.layout" needs "mobile.slots": the values of the layout\'s slots.')
    if not has_layout:
        return _bad(f"mobile.{LAYOUT_FIELD}", (
            '"mobile.slots" needs "mobile.layout": the layout and contract major its slots fill, such as "candy@1".'
        ))
    contract = block_contract(block)
    return _unknown_layout() if contract is None else _slots_errors(contract, block[SLOTS_FIELD])


# ---------------------------------------------------------------------------
# The reader's half: the slots taken from the look, against the package's prepaint table
# ---------------------------------------------------------------------------


def _cannot(name: str, message: str) -> ThemeError:
    return theme_error(CANNOT_PAINT, f"mobile.slots.{name}", message)


def _table_variables(table: BootTable) -> FrozenSet[str]:
    """Every variable the kept table's knobs and pairs write; a numeric rule's is painted from a number instead."""
    entries = [entry for options in table.knobs.values() for entry in options.values()]
    entries += [cell for pair in table.pairs for row in pair.table.values() for cell in row.values()]
    return frozenset(name for entry in entries for name in entry) - {rule.var for rule in table.numeric}


def _unknown_sources(contract: LayoutContract, refs: Mapping[str, str], table: BootTable) -> Dict[str, str]:
    """``{slot: why}`` for each look slot naming a variable the table never writes, or a knob its looks lack."""
    variables, knobs = _table_variables(table), frozenset(spec_knobs(table))
    unknown: Dict[str, str] = {}
    for name, ref in refs.items():
        taken = f'"mobile.slots.{name}" takes "{ref}" from the look'
        if takes_variable(contract.slots[name]) and ref not in variables:
            unknown = {**unknown, name: f"{taken}, which {PREPAINT_PATH} never writes."}
        elif not takes_variable(contract.slots[name]) and ref not in knobs:
            unknown = {**unknown, name: f"{taken}, which is no knob of {PREPAINT_PATH}."}
    return unknown


def _defaults_errors(
    contract: LayoutContract, block: Mapping[str, Any], table: Any, reported: FrozenSet[str],
) -> List[ThemeError]:
    """A slot the table's defaults don't fill (one already refused for its source isn't named twice)."""
    unfilled = [name for name in resolve_slots(contract, block, table, None).unresolved if name not in reported]
    return [
        _cannot(name, (
            f'{PREPAINT_PATH}\'s default look gives "mobile.slots.{name}" nothing a {contract.slots[name]["kind"]} slot '
            f'takes, so a phone can\'t draw {contract.layout}. Make the defaults paint one, or give the slot a value.'
        ))
        for name in unfilled
    ]


def layout_package_errors(block: Mapping[str, Any], prepaint: Optional[bytes]) -> List[ThemeError]:
    """``MOBILE_CANNOT_PAINT`` for each slot taken from the look that the package's table can't fill (module docstring).

    ``prepaint`` is the package's ``boot/prepaint.json`` bytes, None when it has none. A table the reader
    refuses on its own (not JSON, or dropped whole by the first paint) adds nothing here.
    """
    contract, refs = block_contract(block), look_slots(block)
    if contract is None or not refs:
        return []
    if prepaint is None:
        names = ", ".join(f'"{name}"' for name in refs)
        return [theme_error(CANNOT_PAINT, PREPAINT_PATH, (
            f'"mobile.slots" takes {names} from the look through {PREPAINT_PATH}, which isn\'t in the package. '
            'Add the table, or give those slots values.'
        ))]
    document, error = parse_json(prepaint, PREPAINT_PATH)
    table = boot_table(document) if error is None else None
    if table is None:
        return []
    unknown = _unknown_sources(contract, refs, table)
    refused = [_cannot(name, why) for name, why in unknown.items()]
    return refused + _defaults_errors(contract, block, document, frozenset(unknown))
