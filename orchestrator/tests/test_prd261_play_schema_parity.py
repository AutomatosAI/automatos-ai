"""PRD-261 A-09a — builtin Play's committed look schema matches Play's look code in frontend/.

The server checks builtin Play's looks against ``services/themes/data/play-look-schema.json``
(the running image has no ``frontend/`` folder) until A-16c ships ``looks/schema.json`` in the
Play package. This test holds that JSON to the files it mirrors:

* ``frontend/lib/looks/types.ts`` — the look's schema id, its interface, its knob fields and
  their key types, and the glow's range (``GLOW_MIN``/``GLOW_MAX``, whole numbers);
* ``frontend/lib/looks/catalogue.ts`` — every catalogue section's keys, in order, are the
  choices of the knob whose key type names it (``GroundKey = keyof typeof GROUNDS``);
* ``frontend/lib/looks/presets.ts`` — every knob's default is ``DEFAULT_LOOK``'s value, the
  server's ``BUILTIN_PLAY_DEFAULT_LOOK`` is its id, and every preset fits the schema.

The files are read as text, so the test needs nothing but the checkout. A change to Play's
catalogue fails it until the JSON is edited to match.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

ORCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ORCH_ROOT.parent
TYPES_FILE = REPO_ROOT / "frontend/lib/looks/types.ts"
CATALOGUE_FILE = REPO_ROOT / "frontend/lib/looks/catalogue.ts"
PRESETS_FILE = REPO_ROOT / "frontend/lib/looks/presets.ts"
SCHEMA_FILE = ORCH_ROOT / "services" / "themes" / "data" / "play-look-schema.json"
LOOKS_SERVICE = ORCH_ROOT / "services" / "themes" / "looks.py"

LOOK_HEADER_FIELDS = {"schema", "id", "name", "sub", "interface"}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _const(source: str, name: str) -> str:
    match = re.search(rf"export const {name} = '?([^'\n]+?)'?$", source, re.MULTILINE)
    assert match, f"{name} not found"
    return match.group(1)


def _look_fields(types_source: str) -> List[Tuple[str, str]]:
    """``[(field, type)]`` of ``interface Look`` in types.ts, in order."""
    block = re.search(r"export interface Look \{(.*?)\n\}", types_source, re.DOTALL)
    assert block, "interface Look not found in frontend/lib/looks/types.ts"
    return re.findall(r"readonly (\w+): ([\w ]+?)\s*$", block.group(1), re.MULTILINE)


def _sections(catalogue_source: str) -> Dict[str, str]:
    """``{key type: catalogue section}``, e.g. ``{"GroundKey": "GROUNDS"}``."""
    return dict(re.findall(r"export type (\w+Key) = keyof typeof (\w+)", catalogue_source))


def _section_keys(catalogue_source: str, section: str) -> List[str]:
    block = re.search(rf"export const {section} = \{{(.*?)\n\}} as const", catalogue_source, re.DOTALL)
    assert block, f"{section} not found in frontend/lib/looks/catalogue.ts"
    return re.findall(r"^  ([a-z][A-Za-z0-9]*):", block.group(1), re.MULTILINE)


def _presets(presets_source: str) -> Dict[str, Dict[str, object]]:
    """``{CONST: {field: value}}`` for every ``preset({...})`` in presets.ts."""
    found = {}
    for name, body in re.findall(r"export const (\w+): Look = preset\(\{(.*?)\}\)", presets_source):
        values: Dict[str, object] = dict(re.findall(r"(\w+): '([^']*)'", body))
        values.update({key: int(number) for key, number in re.findall(r"(\w+): (\d+)", body)})
        found[name] = values
    assert found, "no presets found in frontend/lib/looks/presets.ts"
    return found


def _schema() -> dict:
    return json.loads(_text(SCHEMA_FILE))


def test_the_header_is_types_ts_look_schema_and_interface():
    types_source = _text(TYPES_FILE)
    look = _schema()["look"]
    assert look == {"schema": _const(types_source, "LOOK_SCHEMA"), "interface": _const(types_source, "LOOK_INTERFACE")}
    assert _schema()["schema"] == "automatos-look-schema/v1"


def test_the_knobs_are_the_look_fields_in_order():
    fields = [name for name, _type in _look_fields(_text(TYPES_FILE)) if name not in LOOK_HEADER_FIELDS]
    assert fields == ["ground", "accent", "font", "shell", "light", "glow"]
    assert list(_schema()["knobs"]) == fields


def test_every_choice_knob_lists_its_catalogue_sections_keys():
    catalogue_source = _text(CATALOGUE_FILE)
    sections = _sections(catalogue_source)
    knobs = _schema()["knobs"]
    choice_fields = [(name, kind) for name, kind in _look_fields(_text(TYPES_FILE)) if kind.endswith("Key")]
    assert len(choice_fields) == 5
    for name, key_type in choice_fields:
        assert knobs[name]["kind"] == "choice", name
        assert knobs[name]["choices"] == _section_keys(catalogue_source, sections[key_type]), name


def test_the_glow_is_types_ts_whole_number_range():
    types_source = _text(TYPES_FILE)
    glow = _schema()["knobs"]["glow"]
    assert dict(_look_fields(types_source))["glow"] == "number"
    assert (glow["kind"], glow["min"], glow["max"], glow["integer"]) == (
        "number", int(_const(types_source, "GLOW_MIN")), int(_const(types_source, "GLOW_MAX")), True,
    )


def test_every_default_is_the_default_looks_value():
    presets_source = _text(PRESETS_FILE)
    default_name = _const(presets_source, "DEFAULT_LOOK: Look")
    default = _presets(presets_source)[default_name]
    for name, knob in _schema()["knobs"].items():
        assert knob["default"] == default[name], name
    service = re.search(r'^BUILTIN_PLAY_DEFAULT_LOOK = "([^"]+)"$', _text(LOOKS_SERVICE), re.MULTILINE)
    assert service and service.group(1) == default["id"]


def test_every_preset_fits_the_schema():
    knobs = _schema()["knobs"]
    for name, preset in _presets(_text(PRESETS_FILE)).items():
        for knob_name, knob in knobs.items():
            value = preset[knob_name]
            if knob["kind"] == "choice":
                assert value in knob["choices"], (name, knob_name, value)
            else:
                assert knob["min"] <= value <= knob["max"], (name, knob_name, value)
