"""An API agent reads a generated document's brand note (night 9b, prep for night 10).

A generate_document result can carry :data:`DOCUMENT_NOTE_KEY`, the plain sentence
naming the words the brand kit bans that its document uses
(``modules.tools.execution.document_brand_check``). A session reads the result itself;
an API agent reads ``ToolResultFormatter.format_for_llm``'s summary, which lists fixed
fields only, so :func:`the_documents_brand_check_is_read` adds the note to it. Night 9b's
PDF 49c0c2b1 said "exquisite" and nothing told the agent.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional

DOCUMENT_NOTE_KEY = "brand_check"
GENERATE_DOCUMENT = "generate_document"


def document_note(result: Any) -> Optional[str]:
    """The brand note on a generate_document result's first row, if it has one."""
    rows = result.get("results") if isinstance(result, dict) else None
    first = rows[0] if isinstance(rows, list) and rows else None
    note = first.get(DOCUMENT_NOTE_KEY) if isinstance(first, dict) else None
    return note if isinstance(note, str) and note else None


def the_documents_brand_check_is_read(fmt: Callable[..., str]) -> Callable[..., str]:
    """Wrap ``ToolResultFormatter.format_for_llm``: the summary ends with the note."""
    @functools.wraps(fmt)
    def wrapped(result: Dict[str, Any], tool_name: str, *args: Any, **kwargs: Any) -> str:
        text = fmt(result, tool_name, *args, **kwargs)
        note = document_note(result) if tool_name == GENERATE_DOCUMENT else None
        return f"{text}\n{note}" if note else text
    return wrapped


__all__ = ["DOCUMENT_NOTE_KEY", "document_note", "the_documents_brand_check_is_read"]
