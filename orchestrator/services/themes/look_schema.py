"""PRD-261 — a theme's looks, checked against the theme's own ``looks/schema.json``.

PRD §4.2 "Looks": there is no fixed field list in core. Each package declares its knobs, and
every look (a preset in the package, or one a member saves, A-09a) is checked against them.

``looks/schema.json`` (schema ``automatos-look-schema/v1``)::

    {"schema": "automatos-look-schema/v1",
     "look": {"schema": "<the look documents' schema id>", "interface": "<optional>"},
     "knobs": {"ground": {"kind": "choice", "label": "Background",
                          "choices": ["graphite", "paper"], "default": "graphite"},
               "glow": {"kind": "number", "label": "Glow", "min": 0, "max": 40,
                        "integer": true, "default": 24}}}

A look is ``{schema, interface (when the schema names one), id, name, sub, <every knob>}``:
Play's look (the theme kit's ``themes/play/src/lib/looks/types.ts``) is exactly this shape with
its six knobs. ``default`` is what a knob falls back to when a newer version drops or changes it
(A-09a).

The TypeScript twin (``frontend/lib/theme-host/look-schema.ts``, A-06b) is held to the same
corpus (``tests/fixtures/themes/cases.json``, ``lookSchemas`` and ``looks``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from services.themes.contract import (
    LOOK_SCHEMA_SCHEMA, ThemeError, echo, is_finite_number, is_whole_number, parse_json, plain_text_problem,
    theme_error,
)

SCHEMA_PATH = "looks/schema.json"
PRESETS_DIR = "looks/presets/"
PRESET_SUFFIX = ".look.json"

LOOK_ERROR_CODES: Tuple[str, ...] = (
    "BAD_LOOK_SCHEMA", "BAD_KNOB", "LOOK_NOT_AN_OBJECT", "LOOK_UNKNOWN_FIELD", "LOOK_WRONG_SCHEMA", "LOOK_BAD_ID",
    "LOOK_BAD_TEXT", "LOOK_BAD_VALUE",
)
SCHEMA_FIELDS: Tuple[str, ...] = ("schema", "look", "knobs")
LOOK_HEADER_FIELDS: Tuple[str, ...] = ("schema", "interface")
LOOK_TEXT_FIELDS: Tuple[str, ...] = ("name", "sub")
LOOK_RESERVED_NAMES = frozenset({"schema", "interface", "id", "name", "sub"})
CHOICE_FIELDS: Tuple[str, ...] = ("kind", "label", "choices", "default")
NUMBER_FIELDS: Tuple[str, ...] = ("kind", "label", "min", "max", "integer", "default")
KNOB_KINDS: Tuple[str, ...] = ("choice", "number")

SCHEMA_ID_PATTERN = re.compile(r"[a-z][a-z0-9.-]{0,47}/v[0-9]{1,3}")
INTERFACE_PATTERN = re.compile(r"[a-z][a-z0-9-]{0,31}")
KNOB_NAME_PATTERN = re.compile(r"[a-z][A-Za-z0-9]{0,31}")
CHOICE_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
# frontend lib/look-kit/validate.ts LOOK_ID_PATTERN: safe as a storage key, a React key and a file name.
LOOK_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")

MAX_KNOBS = 32
MAX_CHOICES = 64
KNOB_NUMBER_LIMIT = 1_000_000
LABEL_MAX = 40
LOOK_TEXT_MAX = 40


class LookSchema(NamedTuple):
    """A checked looks/schema.json: the look documents' header and the knobs by name."""

    look_schema: str
    interface: Optional[str]
    knobs: Mapping[str, Mapping[str, Any]]


def _error(code: str, path: str, message: str) -> List[ThemeError]:
    return [theme_error(code, path, message)]


def _unknown(record: Mapping[str, Any], known: Tuple[str, ...], code: str, where: str) -> List[ThemeError]:
    """``code`` for every key of ``record`` outside ``known``; ``where`` is the record's dotted path."""
    return [
        theme_error(code, f"{where}.{echo(key)}" if where else echo(key), f'"{echo(key)}" isn\'t part of {where or "a look"}. Remove it.')
        for key in record if key not in known
    ]


def _label_errors(knob: Mapping[str, Any], where: str) -> List[ThemeError]:
    if "label" not in knob:
        return []
    problem = plain_text_problem(knob["label"], f"{where}.label", LABEL_MAX)
    return _error("BAD_KNOB", f"{where}.label", problem) if problem else []


def _choice_errors(knob: Mapping[str, Any], where: str) -> List[ThemeError]:
    choices = knob.get("choices")
    valid = isinstance(choices, list) and 0 < len(choices) <= MAX_CHOICES and all(
        isinstance(choice, str) and CHOICE_PATTERN.fullmatch(choice) for choice in choices
    )
    if not valid or len(set(choices)) != len(choices):
        return _error("BAD_KNOB", f"{where}.choices", (
            f'"{where}.choices" must list 1 to {MAX_CHOICES} different keys: lower-case letters, digits and dashes.'
        ))
    if knob.get("default") not in choices:
        return _error("BAD_KNOB", f"{where}.default", f'"{where}.default" must be one of its choices.')
    return []


def _number_errors(knob: Mapping[str, Any], where: str) -> List[ThemeError]:
    low, high, integer = knob.get("min"), knob.get("max"), knob.get("integer", False)
    in_limit = all(is_finite_number(value) and abs(value) <= KNOB_NUMBER_LIMIT for value in (low, high))
    if not in_limit or not low < high:
        return _error("BAD_KNOB", where, f'"{where}" needs "min" below "max", both within ±{KNOB_NUMBER_LIMIT}.')
    if not isinstance(integer, bool):
        return _error("BAD_KNOB", f"{where}.integer", f'"{where}.integer" must be true or false.')
    if integer and not (is_whole_number(low) and is_whole_number(high)):
        return _error("BAD_KNOB", where, f'"{where}" is whole numbers only, so "min" and "max" must be too.')
    if _number_problem(knob, knob.get("default")) is not None:
        return _error("BAD_KNOB", f"{where}.default", f'"{where}.default" must be a number the knob accepts.')
    return []


def _knob_errors(name: str, knob: Any) -> List[ThemeError]:
    where = f"knobs.{echo(name)}"
    if KNOB_NAME_PATTERN.fullmatch(name) is None or name in LOOK_RESERVED_NAMES:
        return _error("BAD_KNOB", where, f'"{where}" needs a name of 1 to 32 letters and digits, not a look field.')
    if not isinstance(knob, dict) or knob.get("kind") not in KNOB_KINDS:
        return _error("BAD_KNOB", where, f'"{where}" must be an object whose "kind" is "choice" or "number".')
    known = CHOICE_FIELDS if knob["kind"] == "choice" else NUMBER_FIELDS
    unknown = _unknown(knob, known, "BAD_KNOB", where)
    shape = _choice_errors(knob, where) if knob["kind"] == "choice" else _number_errors(knob, where)
    return unknown + _label_errors(knob, where) + shape


def _header_errors(document: Mapping[str, Any]) -> List[ThemeError]:
    errors = _unknown(document, SCHEMA_FIELDS, "BAD_LOOK_SCHEMA", SCHEMA_PATH)
    if document.get("schema") != LOOK_SCHEMA_SCHEMA:
        errors += _error("BAD_LOOK_SCHEMA", "schema", f'"schema" must be "{LOOK_SCHEMA_SCHEMA}".')
    look = document.get("look")
    if not isinstance(look, dict):
        return errors + _error("BAD_LOOK_SCHEMA", "look", '"look" must be an object naming the looks\' "schema".')
    errors += _unknown(look, LOOK_HEADER_FIELDS, "BAD_LOOK_SCHEMA", "look")
    if not isinstance(look.get("schema"), str) or SCHEMA_ID_PATTERN.fullmatch(look["schema"]) is None:
        errors += _error("BAD_LOOK_SCHEMA", "look.schema", '"look.schema" must be an id such as "automatos-interface-look/v1".')
    if "interface" in look and (not isinstance(look["interface"], str) or not INTERFACE_PATTERN.fullmatch(look["interface"])):
        errors += _error("BAD_LOOK_SCHEMA", "look.interface", '"look.interface" must be 1 to 32 lower-case letters, digits and dashes.')
    return errors


def validate_look_schema(document: Any) -> Tuple[Optional[LookSchema], List[ThemeError]]:
    """A checked looks/schema.json, or the errors saying why it isn't one."""
    if not isinstance(document, dict):
        return None, _error("BAD_LOOK_SCHEMA", SCHEMA_PATH, f"{SCHEMA_PATH} must be a JSON object.")
    errors = _header_errors(document)
    knobs = document.get("knobs")
    if not isinstance(knobs, dict) or not 0 < len(knobs) <= MAX_KNOBS:
        errors += _error("BAD_LOOK_SCHEMA", "knobs", f'"knobs" must be an object of 1 to {MAX_KNOBS} knobs.')
    else:
        errors += [error for name, knob in knobs.items() for error in _knob_errors(name, knob)]
    if errors:
        return None, errors
    look = document["look"]
    return LookSchema(look["schema"], look.get("interface"), dict(knobs)), []


def _number_problem(knob: Mapping[str, Any], value: Any) -> Optional[str]:
    if not is_finite_number(value) or not knob["min"] <= value <= knob["max"]:
        return f'a number from {knob["min"]} to {knob["max"]}'
    if knob.get("integer", False) and not is_whole_number(value):
        return f'a whole number from {knob["min"]} to {knob["max"]}'
    return None


def _knob_value_problem(knob: Mapping[str, Any], value: Any) -> Optional[str]:
    if knob["kind"] == "choice":
        return None if isinstance(value, str) and value in knob["choices"] else f'one of {", ".join(knob["choices"])}'
    return _number_problem(knob, value)


def _look_header_errors(schema: LookSchema, look: Mapping[str, Any]) -> List[ThemeError]:
    errors: List[ThemeError] = []
    if look.get("schema") != schema.look_schema:
        errors += _error("LOOK_WRONG_SCHEMA", "schema", f'This look is for another theme: "schema" must be "{schema.look_schema}".')
    if schema.interface is not None and look.get("interface") != schema.interface:
        errors += _error("LOOK_WRONG_SCHEMA", "interface", f'"interface" must be "{schema.interface}".')
    look_id = look.get("id")
    if not isinstance(look_id, str) or LOOK_ID_PATTERN.fullmatch(look_id) is None:
        errors += _error("LOOK_BAD_ID", "id", '"id" must be 1 to 64 letters, digits, dashes or underscores, starting with a letter or digit.')
    for field in LOOK_TEXT_FIELDS:
        problem = plain_text_problem(look.get(field), field, LOOK_TEXT_MAX)
        if problem and not (field == "sub" and look.get(field) == ""):
            errors += _error("LOOK_BAD_TEXT", field, problem)
    return errors


def validate_look(schema: LookSchema, look: Any) -> List[ThemeError]:
    """Every reason ``look`` doesn't fit ``schema``; empty when it does. ``sub`` may be empty."""
    if not isinstance(look, dict):
        return _error("LOOK_NOT_AN_OBJECT", "", "A look must be a JSON object.")
    known = ("schema", "id", "name", "sub", *(("interface",) if schema.interface is not None else ()), *schema.knobs)
    errors = _unknown(look, tuple(known), "LOOK_UNKNOWN_FIELD", "") + _look_header_errors(schema, look)
    for name, knob in schema.knobs.items():
        problem = _knob_value_problem(knob, look.get(name))
        if problem is not None:
            errors += _error("LOOK_BAD_VALUE", name, f'"{name}" must be {problem}.')
    return errors


def preset_paths(paths: Any) -> List[str]:
    """The preset files among a package's paths, in order."""
    return sorted(path for path in paths if path.startswith(PRESETS_DIR) and path.endswith(PRESET_SUFFIX))


def _preset_errors(schema: LookSchema, path: str, raw: bytes) -> List[ThemeError]:
    document, error = parse_json(raw, path)
    if error is not None:
        return [error]
    errors = [{**item, "path": f"{path}#{item['path']}", "message": f"{path}: {item['message']}"} for item in validate_look(schema, document)]
    stem = path[len(PRESETS_DIR):-len(PRESET_SUFFIX)]
    if not errors and document.get("id") != stem:
        errors += _error("PRESET_ID_MISMATCH", path, f'{path}: its "id" must be "{echo(stem)}", the file\'s name.')
    return errors


def look_file_errors(files: Mapping[str, bytes]) -> Tuple[Optional[LookSchema], List[ThemeError]]:
    """looks/schema.json checked, then every preset against it (PA-14: at least one preset)."""
    if SCHEMA_PATH not in files:
        return None, _error("MISSING_FILE", SCHEMA_PATH, f"A theme must ship {SCHEMA_PATH}: every theme has a look editor.")
    document, error = parse_json(files[SCHEMA_PATH], SCHEMA_PATH)
    if error is not None:
        return None, [error]
    schema, errors = validate_look_schema(document)
    if schema is None:
        return None, [{**item, "path": f"{SCHEMA_PATH}#{item['path']}"} for item in errors]
    presets = preset_paths(files)
    if not presets:
        return schema, _error("NO_PRESET", PRESETS_DIR, f"A theme must ship at least one preset in {PRESETS_DIR}.")
    return schema, [item for path in presets for item in _preset_errors(schema, path, files[path])]


def preset_ids(files: Mapping[str, bytes]) -> Dict[str, str]:
    """``{id: path}`` of the presets, by file name (a preset's id equals its file name)."""
    return {path[len(PRESETS_DIR):-len(PRESET_SUFFIX)]: path for path in preset_paths(files)}
