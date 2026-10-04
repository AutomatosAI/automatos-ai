"""The status a model call's ``llm_usage`` row is booked with.

F295 (night 8): an empty completion, no text and no tool call, was booked
``success``. Night 8 had 27 of them on claude-sonnet-4 alone (2 or 3 output
tokens each, $0.01 to $0.05 a call), and eight board runs failed on them
("Empty response from LLM"), yet the night's report counted 9 failed calls of
17,529: it counts rows whose status is not success. A call that answered nothing
failed, so its row says ``error``, as the analytics' error counts read it. Its
tokens and cost stay on the row.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

STATUS_SUCCESS = "success"
STATUS_ERROR = "error"


def empty_completion(response: Any) -> bool:
    """The model returned neither text nor a tool call."""
    if getattr(response, "tool_calls", None):
        return False
    return not (getattr(response, "content", None) or "").strip()


def call_status(response: Any) -> str:
    """``error`` for an empty completion, else ``success``."""
    if not empty_completion(response):
        return STATUS_SUCCESS
    logger.warning("[usage] empty completion (model=%s): booked as a failed call", getattr(response, "model", None))
    return STATUS_ERROR
