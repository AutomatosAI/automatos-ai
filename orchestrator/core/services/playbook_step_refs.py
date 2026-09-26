"""What a fixed playbook step reads of earlier steps: ``{{ step_N.key }}`` (PRD-251 US-117).

A fixed step (``generate_document``) has no agent to read the run's context,
so it names what it needs. Step N offers:

* ``output``: its whole answer, as text;
* each key it saved with ``scratchpad_write`` (the channel meant for this);
* when its answer is a JSON object (bare, or in one ```json fence), each of
  that object's keys.

A saved key wins over an answer key of the same name; ``output`` is always the
whole answer unless the step saved a key called ``output``. A reference may go
on into a JSON value: ``{{ step_1.variables.headline }}``.

A reference that is a whole string takes the value itself, so a social
template's variables can come from one saved key: ``"data": "{{ step_1.variables }}"``.
Text stays text (``"3.10"`` is never the number 3.1), except text holding a
JSON object or array, which is that object or array (numbers inside it stay
numbers). A reference inside longer text is replaced by the value's text.

A reference nothing answers raises :class:`UnresolvedStepReference` naming
every one, so a fixed step never renders a literal ``{{ step_1.headline }}``.

Pure (stdlib only): the step loop records each finished step's values with
:func:`step_values` and resolves a fixed step's config with
:func:`resolve_step_references`.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

STEP_REFERENCE = re.compile(r"\{\{\s*step_(\d+)\.(\w+(?:\.\w+)*)\s*\}\}")
OUTPUT_KEY = "output"
_FENCED_JSON = re.compile(r"^```(?:json)?[ \t]*\n(.*)\n[ \t]*```$", re.DOTALL | re.IGNORECASE)

StepValues = Mapping[int, Mapping[str, Any]]


class UnresolvedStepReference(ValueError):
    """References no earlier step answers; ``references`` names each once."""

    def __init__(self, references: List[str]):
        self.references = list(references)
        super().__init__(
            f"This step reads {', '.join(self.references)}, which no earlier step produced. A step offers "
            f"{{{{ step_N.{OUTPUT_KEY} }}}}, each key it saved with scratchpad_write, and the keys of an "
            "answer that is a JSON object."
        )


def structured(text: str) -> Any:
    """``text`` as the JSON object or array it holds (bare or in one ```json fence), else ``text``."""
    candidate = text.strip()
    fenced = _FENCED_JSON.match(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    if candidate[:1] not in ("{", "["):
        return text
    try:
        value = json.loads(candidate)
    except ValueError:
        return text
    return value if isinstance(value, (dict, list)) else text


def step_values(output: Optional[str], saved: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """What a later step can reference of a finished one: its answer's keys, ``output``, its saved keys."""
    text = output if isinstance(output, str) else ""
    answer = structured(text)
    values: Dict[str, Any] = dict(answer) if isinstance(answer, dict) else {}
    values[OUTPUT_KEY] = text
    values.update(saved or {})
    return values


def resolve_step_references(value: Any, steps: StepValues) -> Any:
    """``value`` (a string, or a JSON structure of them) with every reference resolved; a new object.

    Raises :class:`UnresolvedStepReference` when any reference is unanswered.
    """
    unresolved: List[str] = []
    resolved = _resolve(value, steps, unresolved)
    if unresolved:
        raise UnresolvedStepReference(list(dict.fromkeys(unresolved)))
    return resolved


def _resolve(value: Any, steps: StepValues, unresolved: List[str]) -> Any:
    if isinstance(value, str):
        return _resolve_text(value, steps, unresolved)
    if isinstance(value, Mapping):
        return {key: _resolve(item, steps, unresolved) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, steps, unresolved) for item in value]
    return value


def _resolve_text(text: str, steps: StepValues, unresolved: List[str]) -> Any:
    whole = STEP_REFERENCE.fullmatch(text.strip())
    if whole:
        found, value = _lookup(steps, whole)
        if not found:
            unresolved.append(_name(whole))
            return text
        return copy.deepcopy(structured(value) if isinstance(value, str) else value)

    def replace(match: "re.Match[str]") -> str:
        found, value = _lookup(steps, match)
        if not found:
            unresolved.append(_name(match))
            return match.group(0)
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    return STEP_REFERENCE.sub(replace, text)


def _lookup(steps: StepValues, match: "re.Match[str]") -> Tuple[bool, Any]:
    values = steps.get(int(match.group(1)))
    head, *path = match.group(2).split(".")
    if values is None or head not in values:
        return False, None
    value = values[head]
    for key in path:
        value = structured(value) if isinstance(value, str) else value
        if not isinstance(value, Mapping) or key not in value:
            return False, None
        value = value[key]
    return True, value


def _name(match: "re.Match[str]") -> str:
    return f"{{{{ step_{match.group(1)}.{match.group(2)} }}}}"
