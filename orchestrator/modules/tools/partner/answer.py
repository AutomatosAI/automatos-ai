"""Partner tools: the result a call returns, and how the model reads it (PRD-275).

Every result of a partner tool carries ``PARTNER_KEY``: the pack, the tool and the operation
it came from. ``ToolResultFormatter.format_for_llm`` reads a result's ``results`` and keeps
the first four, which would drop a partner's records, so a result with the key is read whole
on its seam (``reads_a_partner_answer_whole``): the data is already cut to the pack's caps.

A refusal carries the same key, an ``error`` the model can act on and an ``error_code``.
Neither holds the key, a header or the request's query.
"""
from __future__ import annotations

import functools
import json
from typing import Any, Callable, Dict, Optional

from modules.tools.partner.pack import Operation
from modules.tools.partner.request import PartnerRefusal
from modules.tools.partner.shape import Shaped

PARTNER_KEY = "partner_tool"
DEFAULT_LLM_CHARS = 20_000
CUT_NOTE = "\n... (cut short: ask for less)"


def _source(pack_name: str, tool_name: str, operation: Optional[Operation]) -> Dict[str, Any]:
    return {"pack": pack_name, "tool": tool_name, "operation": operation.name if operation else None}


def succeeded(pack_name: str, tool_name: str, operation: Operation, status_code: int, kept: Shaped) -> Dict[str, Any]:
    """The result of a call the partner answered."""
    counts = {} if kept.items is None else {"items": kept.items, "total_items": kept.total_items}
    return {
        "success": True,
        PARTNER_KEY: _source(pack_name, tool_name, operation),
        "data": kept.data,
        "metadata": {"source": f"GET {operation.path}", "status_code": status_code,
                     "truncated": kept.truncated, **counts},
    }


def refused(pack_name: str, tool_name: str, refusal: PartnerRefusal,
            operation: Optional[Operation] = None) -> Dict[str, Any]:
    """The result of a call that was not made, or whose answer is not passed on."""
    return {
        "success": False,
        PARTNER_KEY: _source(pack_name, tool_name, operation),
        "error": refusal.message,
        "error_code": refusal.code,
        "status_code": refusal.status_code,
        "tool": tool_name,
    }


def answer_text(result: Dict[str, Any], tool_name: str) -> str:
    """A partner tool's result as the model reads it: the data and where it came from."""
    if not result.get("success"):
        return f"Tool {tool_name} failed: {result.get('error', 'Unknown error')}"
    read = {"data": result.get("data"), "metadata": result.get("metadata")}
    return f"Tool: {tool_name}\nStatus: success\n\n{json.dumps(read, default=str, ensure_ascii=False)}"


def reads_a_partner_answer_whole(format_for_llm: Callable[..., str]) -> Callable[..., str]:
    """Wrap ``ToolResultFormatter.format_for_llm``: a partner tool's result is given as it is."""

    @functools.wraps(format_for_llm)
    def wrapped(result: Dict[str, Any], tool_name: str, max_chars: int = DEFAULT_LLM_CHARS) -> str:
        if not isinstance(result, dict) or PARTNER_KEY not in result:
            return format_for_llm(result, tool_name, max_chars)
        text = answer_text(result, tool_name)
        # The data is already within its pack's caps; only a caller with a smaller budget cuts it, and says so.
        return text if len(text) <= max_chars else text[:max(0, max_chars - len(CUT_NOTE))] + CUT_NOTE

    return wrapped


__all__ = ["PARTNER_KEY", "answer_text", "reads_a_partner_answer_whole", "refused", "succeeded"]
