"""What a database tool hands back to the agent that asked (F299, night 9).

F299 — the board agents' answer is ``json.dumps(raw)`` (agent_factory's tool callback),
which raises on the ``Decimal`` and ``date`` values a shop's money, kilos and dates come
back as: "Object of type Decimal is not JSON serializable" on #1858 (twice), #1856, #1869,
#1879, #1880, #1881 — 72 Decimal and 18 date failures in the backend log on 4 Oct. Auto's
chat read the same rows because its formatter writes them with
``json.dumps(..., default=str)`` (``ToolResultFormatter.format_for_llm``). The answer now
goes out in that same form, so every lane gets plain JSON.
"""
from __future__ import annotations

import json
from typing import Any


def json_safe(value: Any) -> Any:
    """``value`` as plain JSON: Decimal, date, datetime and UUID become text, written
    exactly as Auto's chat writes them (``json.dumps(..., default=str)``)."""
    return json.loads(json.dumps(value, default=str))
