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
from typing import Any, Dict, Optional, Tuple

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


def with_input_defaults(inputs: Dict[str, Any], declared: Any) -> Dict[str, Any]:
    """``inputs`` plus the declared default of each input the run was not given (a new dict).

    The run route fills them before a run starts; a scheduled run, a tool's run
    and a retry start with only what they were given (PRD-251 US-120: the weekly
    Socials Playbook is scheduled). An input declared without a default stays
    missing, and F055 names it.
    """
    filled = dict(inputs)
    if not isinstance(declared, dict):
        return filled
    for name, spec in declared.items():
        if name not in filled and isinstance(spec, dict) and "default" in spec:
            filled[name] = spec["default"]
    return filled
