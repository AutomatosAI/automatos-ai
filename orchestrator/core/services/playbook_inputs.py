"""F113 (run 4): a playbook run's input is key-value pairs, whatever a caller sends.

Auto ran "Friday cafe payment chase" with input_data as one string (the unpaid
invoices as CSV); the run died on ``'str' object has no attribute 'items'``
and the owner's ticket carried the Python error as its result. A playbook that
takes one text reads it as ``{input}`` (``substitute_playbook_input``), so a
plain string becomes ``{"input": <the text>}``; a string holding a JSON object
is that object; anything else is refused with a reason that names the
parameter. The tool handler and the executor both read input through here.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

INPUT_KEY = "input"


def playbook_inputs(value: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """``(inputs, problem)`` — the key-value pairs to run with, or a reason the
    value cannot be one. Nothing given is ``{}``."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return {}, None
    if isinstance(value, dict):
        return dict(value), None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed, None
        return {INPUT_KEY: value}, None
    return {}, (
        f"input_data must be key-value pairs (an object), or one text that the playbook reads "
        f'as {{{INPUT_KEY}}} — got a {type(value).__name__}. Pass e.g. {{"{INPUT_KEY}": "…"}}.'
    )


# ---------------------------------------------------------------------------
# F182 (night 6): what a run needs before step 1
# ---------------------------------------------------------------------------
# Run 207 of "New Cafe Onboarding" started with no inputs, and its first step
# wrote to another café's contact from memory. A playbook's ``inputs`` column
# declares what each run needs, the way a function signature does:
# {name: {"required": true, "description": "…", "default": …}}. When it declares
# nothing, the placeholders a step cannot run without are the contract: F055's
# {input.<name>} and a bare {input}. A {{name}} blank is not (F130 leaves one the
# run does not supply as written: the prompt may be asking for a template), nor
# a single-brace {name}, which the engine never fills.

_INPUT_FIELD_RE = re.compile(r"\{input\.([A-Za-z0-9_]+)\}")
_BARE_INPUT = "{" + INPUT_KEY + "}"
# One "name: value" (or "name = value") line of the owner's answer. The value is
# the rest of the line, stripped in code: a lazy value before a trailing \s*$
# backtracked quadratically on a long run of spaces (CodeQL py/polynomial-redos).
_ANSWER_LINE_RE = re.compile(r"^\s*(?:[-*\u2022]\s*)?([A-Za-z_][A-Za-z0-9_ -]{0,60}?)\s*[:=](.*)$")
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _absent(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or value in ([], {})


def declared_inputs(inputs: Any) -> Dict[str, Dict[str, Any]]:
    """A playbook's declared inputs, read the way they are saved: the playbook
    editor saves the column as JSON text, and a name may carry only its
    description (a string) or only whether it is required (a boolean)."""
    if isinstance(inputs, str):
        try:
            inputs = json.loads(inputs)
        except ValueError:
            return {}
    if not isinstance(inputs, dict):
        return {}
    declared: Dict[str, Dict[str, Any]] = {}
    for name, spec in inputs.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if isinstance(spec, dict):
            declared[name.strip()] = dict(spec)
        elif isinstance(spec, bool):
            declared[name.strip()] = {"required": spec}
        else:
            declared[name.strip()] = {"description": str(spec)} if spec else {}
    return declared


def derived_inputs(steps: Any) -> Dict[str, Dict[str, Any]]:
    """The inputs the steps cannot run without, each required."""
    derived: Dict[str, Dict[str, Any]] = {}
    for step in steps or []:
        prompt = str(step.get("prompt_template") or "") if isinstance(step, dict) else ""
        for name in _INPUT_FIELD_RE.findall(prompt):
            derived.setdefault(name, {"required": True, "derived": True})
        if _BARE_INPUT in prompt:
            derived.setdefault(INPUT_KEY, {"required": True, "derived": True,
                                           "description": "the text this playbook works on"})
    return derived


def input_contract(inputs: Any, steps: Any) -> Dict[str, Dict[str, Any]]:
    """What a run of this playbook needs: its declared inputs, else what its steps read."""
    return declared_inputs(inputs) or derived_inputs(steps)


def contract_of(playbook: Any) -> Dict[str, Dict[str, Any]]:
    """``input_contract`` of a playbook row."""
    return input_contract(getattr(playbook, "inputs", None), getattr(playbook, "steps", None))


def with_defaults(contract: Dict[str, Dict[str, Any]], input_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The run's inputs, with a declared default for each one not given. An
    input without a default stays missing: it is never filled with ''."""
    filled = dict(input_data or {})
    for name, spec in contract.items():
        if _absent(filled.get(name)) and not _absent(spec.get("default")):
            filled[name] = spec["default"]
    return filled


def missing_inputs(contract: Dict[str, Dict[str, Any]], input_data: Optional[Dict[str, Any]]) -> List[str]:
    """The required inputs this run was not given. A bare {input} reads the whole
    input, so any value given satisfies it."""
    given = input_data or {}
    missing = []
    for name, spec in contract.items():
        if not spec.get("required"):
            continue
        if name == INPUT_KEY and spec.get("derived") and any(not _absent(v) for v in given.values()):
            continue
        if _absent(given.get(name)):
            missing.append(name)
    return missing


def _needed_lines(missing: List[str], contract: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = []
    for name in missing:
        description = str((contract.get(name) or {}).get("description") or "").strip()
        lines.append(f"- {name}: {description}" if description else f"- {name}")
    return lines


def inputs_question(playbook_name: str, missing: List[str], contract: Dict[str, Dict[str, Any]]) -> str:
    """The owner's question when a run starts without inputs it needs, with the
    answer's format (``inputs_from_answer`` reads it)."""
    lines = [f"'{playbook_name}' needs {'this' if len(missing) == 1 else 'these'} before it can run:",
             *_needed_lines(missing, contract), "",
             "Answer with lines like:", *(f"{name}: …" for name in missing)]
    if len(missing) == 1:
        lines.append("(or just the value)")
    return "\n".join(lines)


def inputs_needed_error(playbook_name: str, missing: List[str], contract: Dict[str, Dict[str, Any]]) -> str:
    """The tool's refusal when a run would start without inputs it needs."""
    example = json.dumps({"input_data": {name: "…" for name in missing}}, ensure_ascii=False)
    return "\n".join([
        f"'{playbook_name}' needs {', '.join(missing)} before it can run, and this call did not give "
        f"{'it' if len(missing) == 1 else 'them'}. Nothing was started.",
        *_needed_lines(missing, contract),
        f"Ask the owner if you do not know them, then pass them like this: {example}",
    ])


def _normal(name: str) -> str:
    return re.sub(r"[\s-]+", "_", name.strip().lower())


def inputs_from_answer(answer: str, missing: List[str], contract: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """The owner's answer to ``inputs_question`` as input values: each
    "name: value" line for a name the playbook takes, and, when one input was
    missing and no line names it, the whole answer."""
    text = (answer or "").strip()
    if not text:
        return {}
    names = {_normal(name): name for name in [*contract, *missing]}
    values: Dict[str, str] = {}
    for line in text.splitlines():
        match = _ANSWER_LINE_RE.match(line)
        name = names.get(_normal(match.group(1))) if match else None
        value = match.group(2).strip() if name else ""
        if value:
            values[name] = value
    if not values and len(missing) == 1:
        values[missing[0]] = text
    return values


def inputs_problem(value: Any) -> Optional[str]:
    """Why ``value`` cannot be a playbook's declared inputs, or None."""
    if not isinstance(value, dict):
        return ('inputs must be an object of names, e.g. {"cafe_name": {"required": true, '
                '"description": "The café\'s name"}}.')
    bad = [name for name in value if not isinstance(name, str) or not _NAME_RE.fullmatch(name)]
    if bad:
        return f"inputs names must be letters, digits and _ (a step reads each as {{{{name}}}}): {bad}"
    return None
