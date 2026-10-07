"""The ``tags`` argument of generate_document, declared in each schema an agent or Auto reads (7 Oct).

generate_document is declared twice: its ToolSpec (modules/tools/registry/
app_and_document_tools.py, the board, missions and scheduled runs) and its inline chat
schema, built inside ``AgentPlatformTools.get_available_tools``, one long function in a
file over 800 lines. The ToolSpec takes the parameter from :func:`tags_parameter`; the
chat schema is wrapped here (as services/past_work_schemas.py adds ``scope``), so both
say the same thing. The rules themselves are services/deliverable_tags.py's.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List

from services.deliverable_tags import MAX_TAG_CHARS, MAX_TAGS

GENERATE_DOCUMENT = "generate_document"
TAGS = "tags"
TAGS_DESCRIPTION = (
    f"Optional tags for the document's Deliverable, so the owner can find it by them: up to {MAX_TAGS} "
    f"short words or phrases of at most {MAX_TAG_CHARS} characters, for example [\"invoice\", \"harbourline\"]. "
    "Saved lowercase. Tags on the card you are working are added too."
)
# The limits are said in words, not as maxItems / maxLength: not every provider reads those.
TAGS_PROPERTY: Dict[str, Any] = {"type": "array", "items": {"type": "string"}, "description": TAGS_DESCRIPTION}


def with_tags(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """A JSON-schema ``parameters`` object with the ``tags`` property added. Pure."""
    return {**parameters, "properties": {**(parameters.get("properties") or {}), TAGS: dict(TAGS_PROPERTY)}}


def chat_document_takes_tags(build: Callable[..., List[Dict[str, Any]]]) -> Callable[..., List[Dict[str, Any]]]:
    """Wrap ``AgentPlatformTools.get_available_tools``: generate_document takes ``tags``."""
    @functools.wraps(build)
    def wrapped(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        return [{**tool, "parameters": with_tags(tool.get("parameters") or {})}
                if isinstance(tool, dict) and tool.get("name") == GENERATE_DOCUMENT else tool
                for tool in build(*args, **kwargs)]
    return wrapped


def tags_parameter() -> Any:
    """generate_document's ``tags`` as a ToolSpec parameter (the registry's lane)."""
    from modules.tools.registry.tool_registry import ToolParameter

    return ToolParameter(name=TAGS, type="array", description=TAGS_DESCRIPTION, required=False,
                         items={"type": "string"})


__all__ = ["TAGS_DESCRIPTION", "TAGS_PROPERTY", "chat_document_takes_tags", "tags_parameter", "with_tags"]
