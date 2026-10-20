"""What a Qdrant call raises (PRD-262 row #20), shaped as qdrant-client raised it.

``UnexpectedResponse``: Qdrant answered with a status that is not a success (a 408
when it timed out a write, a 400 for a wrong vector size, a 404 for a missing
collection). ``ResponseHandlingException``: no usable answer at all, the transport's
error (a connect error, the client's own timeout) or an answer that is not Qdrant's
JSON, kept as ``source``. core/qdrant_writes.py reads both to tell a timeout from
anything else.
"""
from __future__ import annotations

from typing import Optional

import httpx

CONTENT_SHOWN = 200  # bytes of a refused answer's body in the message


class UnexpectedResponse(Exception):
    """Qdrant answered ``status_code``, which is not a success."""

    def __init__(self, status_code: Optional[int], reason_phrase: str, content: bytes, headers: httpx.Headers) -> None:
        self.status_code = status_code
        self.reason_phrase = reason_phrase
        self.content = content
        self.headers = headers
        shown = content[:CONTENT_SHOWN].decode("utf-8", "replace")
        super().__init__(f"Unexpected Response: {status_code} ({reason_phrase}) Raw response content: {shown}")

    @classmethod
    def for_response(cls, response: httpx.Response) -> "UnexpectedResponse":
        return cls(response.status_code, response.reason_phrase, response.content, response.headers)


class ResponseHandlingException(Exception):
    """No usable answer from Qdrant: ``source`` is the transport's error or the parse error."""

    def __init__(self, source: BaseException) -> None:
        self.source = source
        super().__init__(f"{type(source).__name__}: {source}")
