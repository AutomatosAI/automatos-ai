"""The two client-sent ids a turn's timing may carry, checked before they are kept.

A phone may send a turn reference (``turnRef`` on ``POST /api/chat`` and on the speech
requests of that turn), and a request id can come from the caller's ``X-Request-ID``. Both
end up on a log line and on a reply's trace, so both are checked for shape and length here:
a value that does not fit is dropped (``None``), never an error.

Standard library only, so a request model can import this and nothing else.
"""
from __future__ import annotations

import re
from typing import Any, Optional

REF_PATTERN = re.compile(r"^[A-Za-z0-9_-]{6,40}$")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _fitting(value: Any, pattern: "re.Pattern[str]") -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value if pattern.fullmatch(value) else None


def clean_ref(value: Any) -> Optional[str]:
    """The turn reference when it is a string that fits ``REF_PATTERN``, else None. Never raises."""
    return _fitting(value, REF_PATTERN)


def clean_request_id(value: Any) -> Optional[str]:
    """The request id when it is a string that fits ``REQUEST_ID_PATTERN``, else None. Never raises."""
    return _fitting(value, REQUEST_ID_PATTERN)
