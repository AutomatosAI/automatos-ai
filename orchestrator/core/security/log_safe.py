"""A value a request (or a model's tool call) supplies, made safe to log.

F155: line breaks and other control characters become spaces, so a value
cannot forge a log line, and a long value is cut to ``limit`` characters.
"""
from __future__ import annotations

from typing import Any


def log_safe(value: Any, limit: int = 80) -> str:
    """``value`` as one printable log line of at most ``limit`` characters."""
    if value is None:
        return "<none>"
    text = "".join(ch if ch.isprintable() else " " for ch in str(value))
    return text if len(text) <= limit else text[:limit] + f"…(+{len(text) - limit})"
