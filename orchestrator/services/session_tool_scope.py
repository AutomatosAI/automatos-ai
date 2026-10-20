"""#942 / F430: how a grouped session tool's spec forwards a call, and its schema's shape.

The work tools (``session_work_tools``) and the Socials tools (``session_socials_tools``)
are built with these, and ``session_work_tools`` lists the Socials tools after its own,
so the helpers live apart from both: neither spec module imports the other's helpers,
and there is no import cycle. Plain functions; the refusal type is read at CALL time
(``session_tools`` imports the spec modules to build its list).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Tuple

SCOPED = Callable[[Dict[str, Any], Any], Dict[str, Any]]


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def forward(fields: Mapping[str, str], required: Tuple[str, ...] = (), needs: str = "",
            one_of: Tuple[str, ...] = ()) -> SCOPED:
    """A scope that forwards the session's ``fields`` under the action's names
    (``{session_name: action_name}``) and drops everything else. A missing
    ``required`` field, or none of ``one_of``, is refused with ``needs``, the
    sentence the model reads."""

    def scope(params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
        out = {fields[key]: value for key, value in params.items() if key in fields and not _blank(value)}
        missing = any(fields[key] not in out for key in required)
        if missing or (one_of and not any(fields[key] in out for key in one_of)):
            from services.session_tools import SessionToolRefused

            raise SessionToolRefused(needs)
        return out

    return scope


def object_schema(properties: Dict[str, Any], required: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """A tool's input schema: an object with ``properties``, ``required`` as a list."""
    return {"type": "object", "properties": properties, "required": list(required)}
