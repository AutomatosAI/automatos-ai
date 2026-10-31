"""PRD-271 C01 (D-2) — the phone layout contracts, as the host holds them.

Automatos Go draws a theme on one of its built-in layouts (native code: App Store guideline 2.5.2
allows no downloaded code). Each layout publishes a **contract**: the slots it paints from. A theme's
``mobile`` block names one as ``"<layout>@<major>"`` and fills its slots (``mobile_layout.py`` checks
the block, ``layout_paint.py`` resolves it for the phone, ``layout_contrast.py`` measures it).

The contracts are data, ``data/phone-layouts.json``, read at runtime like ``core-pages.json`` and
``signing-keys.json``: the host's contract, not seed content. The kit and Automatos Go copy the file
byte for byte; ``frontend/lib/theme-host/phone-layouts.json`` is the browser's copy, the file less each layout's
``home`` line (no frontend runtime file names a first-party theme's id, PRD-261 A-20), held equal by
``validator-parity.test.ts`` and ``tests/test_prd271_layout_contracts.py``. Its shape::

    {"schema": "automatos-phone-layouts/v1",
     "layouts": {"<id>": {"name", "home": "<the first-party theme that owns it>",
       "versions": {"<major>": {"revision", "tone": {"rule": "ground", "ground", "text"} | {"rule": "dark"},
         "slots": {"<slot>": {"kind": "colour" | "tint" | "number" | "choice", "group",
                              "min", "max" (number), "choices" (choice), "default" (optional slots only)}},
         "fixed": {"<name>": "<colour the layout always paints>"},
         "pairs": [["<text or mark>", "<ground>", "text" | "ui"]],
         "fonts": {"sans" | "serif" | "mono": "<what that role draws>"},
         "mascot": "bundled" | "drawn", "example": {"<slot>": <the home theme's default look>}}}}}}

A slot with a ``default`` is optional, one without is required. Within a major the only change is a
new optional slot (``revision`` + 1); anything else is the next major, kept beside the old one.

Standard library only, like ``contract.py``: nothing here imports the rest of the host.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

LAYOUTS_FILE = Path(__file__).resolve().parent / "data" / "phone-layouts.json"
LAYOUTS_SCHEMA = "automatos-phone-layouts/v1"
# "<layout id>@<contract major>": the id as the table keys it, the major without a leading zero.
LAYOUT_PATTERN = re.compile(r"([a-z][a-z0-9-]{0,31})@([1-9][0-9]{0,3})")
SLOT_KINDS: Tuple[str, ...] = ("colour", "tint", "number", "choice")
KIND_COLOUR, KIND_TINT, KIND_NUMBER, KIND_CHOICE = SLOT_KINDS
# A contract's tone: "ground" compares its ground slot with its text slot; "light" and "dark" are fixed.
TONE_GROUND = "ground"
TONE_RULES: Tuple[str, ...] = (TONE_GROUND, "light", "dark")
MASCOT_KINDS: Tuple[str, ...] = ("bundled", "drawn")
PAIR_KINDS: Tuple[str, ...] = ("text", "ui")


class LayoutContract(NamedTuple):
    """One major of one layout, read-only: every mapping is a view of the loaded table."""

    layout: str
    layout_id: str
    major: int
    name: str
    home: str
    revision: int
    tone: Mapping[str, str]
    slots: Mapping[str, Mapping[str, Any]]
    fixed: Mapping[str, str]
    pairs: Tuple[Tuple[str, str, str], ...]
    fonts: Mapping[str, str]
    mascot: str
    example: Mapping[str, Any]


def _frozen(value: Any) -> Any:
    """``value`` with every object a read-only view and every list a tuple, so no caller can change the table."""
    if isinstance(value, dict):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


@lru_cache(maxsize=1)
def layouts_table() -> Mapping[str, Any]:
    """``data/phone-layouts.json``, read once and frozen."""
    return _frozen(json.loads(LAYOUTS_FILE.read_text(encoding="utf-8")))


def _contract(layout_id: str, major: str, layout: Mapping[str, Any], version: Mapping[str, Any]) -> LayoutContract:
    return LayoutContract(
        layout=f"{layout_id}@{major}", layout_id=layout_id, major=int(major), name=layout["name"], home=layout["home"],
        revision=version["revision"], tone=version["tone"], slots=version["slots"], fixed=version["fixed"],
        pairs=tuple(tuple(pair) for pair in version["pairs"]), fonts=version["fonts"], mascot=version["mascot"],
        example=version["example"],
    )


def contract_of(layout: Any) -> Optional[LayoutContract]:
    """The contract ``"<id>@<major>"`` names, or None for anything else (an unknown id or major, a wrong shape)."""
    match = LAYOUT_PATTERN.fullmatch(layout) if isinstance(layout, str) else None
    if match is None:
        return None
    layout_id, major = match.groups()
    entry = layouts_table()["layouts"].get(layout_id)
    version = entry["versions"].get(major) if entry is not None else None
    return None if version is None else _contract(layout_id, major, entry, version)


def layout_names() -> Tuple[str, ...]:
    """Every contract the host knows, as a theme names it (``"play@1"``), in the table's order."""
    layouts = layouts_table()["layouts"]
    return tuple(f"{layout_id}@{major}" for layout_id, entry in layouts.items() for major in entry["versions"])


def home_layouts() -> Dict[str, str]:
    """``{theme id: layout id}``: the first-party theme that owns each layout (D-4; Go's and the kit's old id list)."""
    return {entry["home"]: layout_id for layout_id, entry in layouts_table()["layouts"].items()}


def required_slots(contract: LayoutContract) -> Tuple[str, ...]:
    """The contract's slots with no default, in its order: a theme must fill each."""
    return tuple(name for name, spec in contract.slots.items() if "default" not in spec)


__all__ = [
    "KIND_CHOICE", "KIND_COLOUR", "KIND_NUMBER", "KIND_TINT", "LAYOUTS_FILE", "LAYOUT_PATTERN", "LayoutContract",
    "SLOT_KINDS", "TONE_GROUND", "TONE_RULES", "contract_of", "home_layouts", "layout_names", "layouts_table",
    "required_slots",
]
