"""PRD-256 FX-013 (night 12, M2): ``platform_store_memory``'s ``source_type`` has a default.

Night 12: six ``store_memory`` calls were refused on ``source_type``, until Auto asked the owner
to pick "claude_reports, current_status, inference or platform_verified" (A536). Provenance is
the platform's to record, never a question for the owner. So ``source_type`` is optional: a
memory stored on a turn a person drives is ``platform_verified`` (the owner said it), any other
is ``claude_reports`` (the assistant's own claim). A value outside the four is stored as that
default, and the answer says so: the call is never refused for it.

Who drives the turn is the server's ``_driving_user_id`` (platform executor, strip-then-inject),
never a model argument.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, Tuple

from sqlalchemy.orm import Session

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

SOURCE_TYPE = "source_type"
SOURCE_TYPES = ("platform_verified", "claude_reports", "current_status", "inference")
SAID_BY_THE_OWNER = "platform_verified"
THE_ASSISTANTS_CLAIM = "claude_reports"
DRIVER = "_driving_user_id"
NOTE = "note"
NOT_A_SOURCE_TYPE = "source_type '{said}' is not one of {types}: stored as {default}."


def default_source_type(params: Dict[str, Any]) -> str:
    """platform_verified on a turn a person drives, else claude_reports."""
    return SAID_BY_THE_OWNER if params.get(DRIVER) is not None else THE_ASSISTANTS_CLAIM


def source_type_read(params: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """``params`` with a source type the handler takes, and what the answer says of a value
    it replaced ('' when none). A new dict, never the caller's."""
    said = params.get(SOURCE_TYPE)
    said = said.strip() if isinstance(said, str) else said
    if said in SOURCE_TYPES:
        return {**params, SOURCE_TYPE: said}, ""
    default = default_source_type(params)
    note = "" if said in (None, "") else NOT_A_SOURCE_TYPE.format(said=said, types=", ".join(SOURCE_TYPES),
                                                                  default=default)
    return {**params, SOURCE_TYPE: default}, note


def defaults_the_source_type(handler: Handler) -> Handler:
    """Wrap ``store_memory``: the source type left out (or not one of the four) is the default."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        read, note = source_type_read(params or {})
        out = await handler(db, workspace_id, read)
        if not note or not isinstance(out, dict) or out.get("success") is not True:
            return out
        return {**out, NOTE: note}
    return wrapped


__all__ = ["SOURCE_TYPES", "default_source_type", "defaults_the_source_type", "source_type_read"]
