"""PRD-270 W2 (K8) — the strict check every item bundle passes, on both sides.

``validate_bundle`` walks the bundle against its type's spec (schema.py) and refuses the first
problem it finds with a BundleError: a key outside the spec at any depth, a wrong type, a value
over its cap or off its pattern (a scan time that is no real date and time), a reference of the
wrong kind (a package member's, of a kind its type does not take), a step naming no agent of the
bundle, an agent step without an agent or a prompt, a document step naming an agent. The hosted
export never lists a bundle this refuses, and a local install writes no row before its bundle
passes. The whole document's size is checked by canonical.capped_bytes, against the cap its
caller gives.

Messages name the place by the format's own field names and list positions, never by a key or a
value of the bundle. Standard library only: the hosted export and the local import share it.
"""
from __future__ import annotations

import math
from datetime import datetime
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping

from services.marketplace_bundles.errors import BundleError
from services.marketplace_bundles.refs import check_ref
from services.marketplace_bundles.schema import (
    BUNDLE_SCHEMA, DOCUMENT_STEP_TYPE, SCALAR_NUMBER_MAX, SCALAR_TEXT_MAX, TYPE_SPECS, Spec, is_plain_text, number, text,
)

__all__ = ["BundleError", "validate_bundle"]

ROOT = "bundle"
# A scalar that is text or a number: any text, or any number either side of zero, up to its cap.
SCALAR_TEXT = text(SCALAR_TEXT_MAX)
SCALAR_NUMBER = number(SCALAR_NUMBER_MAX, min_value=-SCALAR_NUMBER_MAX)


def validate_bundle(data: Any) -> Dict[str, Any]:
    """``data`` when it is a schema 1 bundle the format allows; else raise BundleError."""
    spec = _type_spec(data)
    _walk(data, spec, ROOT)
    if data["type"] == "playbook":
        _check_steps(data["playbook"])
    return data


def _type_spec(data: Any) -> Spec:
    """The spec of ``data``'s type, once it is an object that says schema 1 and a known type."""
    if not isinstance(data, dict):
        raise BundleError("type", "A bundle is a JSON object.")
    schema = data.get("bundle")
    if type(schema) is not int or schema != BUNDLE_SCHEMA:
        raise BundleError("schema", "The document is not a schema 1 item bundle.")
    kind = data.get("type")
    spec = TYPE_SPECS.get(kind) if isinstance(kind, str) else None
    if spec is None:
        raise BundleError("schema", "The bundle's type is not one this install reads.")
    return spec


def _walk(value: Any, spec: Spec, where: str) -> None:
    """Check ``value`` against ``spec``; ``where`` names its place for a message."""
    if value is None and spec.get("nullable"):
        return
    _CHECKS[spec["kind"]](value, spec, where)


def _expect(ok: bool, where: str) -> None:
    if not ok:
        raise BundleError("type", f"{where} has the wrong type.")


def _check_object(value: Any, spec: Spec, where: str) -> None:
    _expect(isinstance(value, dict), where)
    fields = spec["fields"]
    if any(key not in fields for key in value):
        raise BundleError("unknown-key", f"{where} has a key the bundle format does not allow.")
    for key, field_spec in fields.items():
        if key in value:
            _walk(value[key], field_spec, f"{where}.{key}")
        elif key in spec["required"]:
            raise BundleError("schema", f"{where}.{key} is missing.")


def _check_tagged(value: Any, spec: Spec, where: str) -> None:
    """An object checked against the variant its ``key`` names (a package member, by its type)."""
    _expect(isinstance(value, dict), where)
    key = spec["key"]
    if key not in value:
        raise BundleError("schema", f"{where}.{key} is missing.")
    tag = value[key]
    _expect(isinstance(tag, str), f"{where}.{key}")
    variant = spec["variants"].get(tag)
    if variant is None:
        raise BundleError("pattern", f"{where}.{key} is not one of the values the bundle format allows.")
    _check_object(value, variant, where)


def _check_map(value: Any, spec: Spec, where: str) -> None:
    _expect(isinstance(value, dict), where)
    if len(value) > spec["max"]:
        raise BundleError("too-many", f"{where} has more than {spec['max']} entries.")
    for key, item in value.items():
        if not isinstance(key, str) or spec["key"].fullmatch(key) is None:
            raise BundleError("pattern", f"{where} has a name the bundle format does not allow.")
        _walk(item, spec["values"], f"{where}.*")


def _check_list(value: Any, spec: Spec, where: str) -> None:
    _expect(isinstance(value, list), where)
    if len(value) > spec["max"]:
        raise BundleError("too-many", f"{where} has more than {spec['max']} entries.")
    if len(value) < spec["min"]:
        raise BundleError("schema", f"{where} needs at least {spec['min']} entries.")
    for index, item in enumerate(value):
        _walk(item, spec["items"], f"{where}[{index}]")


def _check_text(value: Any, spec: Spec, where: str) -> None:
    _expect(isinstance(value, str), where)
    if len(value) > spec["max"]:
        raise BundleError("too-long", f"{where} is longer than {spec['max']} characters.")
    if len(value) < spec["min"]:
        raise BundleError("schema", f"{where} is empty.")
    pattern = spec["pattern"]
    if not is_plain_text(value) or (pattern is not None and pattern.fullmatch(value) is None):
        raise BundleError("pattern", f"{where} is not in the form the bundle format allows.")


def _check_instant(value: Any, spec: Spec, where: str) -> None:
    _check_text(value, spec["text"], where)
    try:
        datetime.fromisoformat(value)
    except ValueError:
        raise BundleError("pattern", f"{where} is not a real date and time.") from None


def _is_number(value: Any, integer: bool) -> bool:
    """A JSON number (never true or false); a float only when ``integer`` is off, and finite."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return not integer and isinstance(value, float) and math.isfinite(value)


def _check_number(value: Any, spec: Spec, where: str) -> None:
    _expect(_is_number(value, spec["integer"]), where)
    if value > spec["max"]:
        raise BundleError("too-long", f"{where} is over {spec['max']}.")
    if value < spec["min"]:
        raise BundleError("schema", f"{where} is under {spec['min']}.")


def _check_flag(value: Any, spec: Spec, where: str) -> None:
    _expect(isinstance(value, bool), where)


def _check_choice(value: Any, spec: Spec, where: str) -> None:
    _expect(isinstance(value, str), where)
    if value not in spec["values"]:
        raise BundleError("pattern", f"{where} is not one of the values the bundle format allows.")


def _check_ref(value: Any, spec: Spec, where: str) -> None:
    check_ref(value, spec["field"], where)


def _check_local_id(value: Any, spec: Spec, where: str) -> None:
    _expect(value is None or isinstance(value, str), where)


def _check_scalar(value: Any, spec: Spec, where: str) -> None:
    if value is None or isinstance(value, bool):
        return
    if _is_number(value, False):
        _check_number(value, SCALAR_NUMBER, where)
        return
    _check_text(value, SCALAR_TEXT, where)


def _check_scalars(value: Any, spec: Spec, where: str) -> None:
    if not (spec["lists"] and isinstance(value, list)):
        _check_scalar(value, spec, where)
        return
    if len(value) > spec["max_items"]:
        raise BundleError("too-many", f"{where} has more than {spec['max_items']} entries.")
    for index, item in enumerate(value):
        _check_scalar(item, spec, f"{where}[{index}]")


def _check_steps(playbook: Mapping[str, Any]) -> None:
    """A document step names no agent; any other step names an agent of the Playbook's ``agents``
    and carries its prompt_template (core.py PLAYBOOK_STEP_FIELDS), so the import writes a step
    that can run."""
    local_ids = playbook["agents"]
    for index, step in enumerate(playbook["steps"]):
        where = f"{ROOT}.playbook.steps[{index}]"
        agent = step["agent"]
        if step.get("type") == DOCUMENT_STEP_TYPE:
            if agent is not None:
                raise BundleError("ref", f"{where}.agent is set on a document step, which runs no agent.")
            continue
        if agent is None or agent not in local_ids:
            raise BundleError("ref", f"{where}.agent names no agent of the bundle.")
        if "prompt_template" not in step:
            raise BundleError("schema", f"{where}.prompt_template is missing.")


_CHECKS: Mapping[str, Callable[[Any, Spec, str], None]] = MappingProxyType({
    "object": _check_object,
    "tagged": _check_tagged,
    "map": _check_map,
    "list": _check_list,
    "text": _check_text,
    "instant": _check_instant,
    "number": _check_number,
    "flag": _check_flag,
    "choice": _check_choice,
    "ref": _check_ref,
    "local_id": _check_local_id,
    "scalars": _check_scalars,
})
