"""PRD-256 US-006: a direct call is held to its schema's required fields, as a dispatched one is.

``platform_execute`` refuses a call that leaves out a required field before anything runs
(F027-C: the refusal spells out the exact call). A platform action called by its own name
skipped that check and reached its handler with the field missing. Auto's writes are now
first-class tools (PRD-256 US-006), so most of its writes are direct calls: a direct call to a
first-class (promoted) action is refused the same way, in the form the caller used, and nothing
runs. Any other action keeps its handler's own check.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def missing_required(action_def: Any, params: Dict[str, Any]) -> List[str]:
    """The required fields of ``action_def``'s schema that ``params`` leaves out."""
    required = (getattr(action_def, "parameters", None) or {}).get("required") or []
    return [name for name in required if name not in params]


def missing_on_a_direct_call(action_name: str, action_def: Any, params: Dict[str, Any]) -> Optional[str]:
    """The refusal of a direct call to a first-class (promoted) action that leaves out a required
    field, or None: the fields missing, the exact call with each required field and its type,
    and that the call is the caller's to make again."""
    from modules.tools.execution.unified_executor import REFUSED_CALL_IS_YOURS, _placeholder

    missing = missing_required(action_def, params) if getattr(action_def, "promoted", False) else []
    if not missing:
        return None
    schema = action_def.parameters or {}
    props = schema.get("properties") or {}
    example = ", ".join(f'"{key}": {_placeholder(props.get(key) or {})}' for key in schema.get("required") or [])
    lines = [
        f"Missing required params for '{action_name}': {missing}, so nothing was done.",
        f"Call it exactly like this: {action_name}({{{example}}})",
        REFUSED_CALL_IS_YOURS,
    ]
    lines += [f"  {name}: {props[name].get('description', props[name].get('type', '?'))}"
              for name in missing if name in props]
    return "\n".join(lines)


__all__ = ["missing_on_a_direct_call", "missing_required"]
