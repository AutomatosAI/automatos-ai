"""PRD-282 US-301: one JSON record per line, with a fixed list of fields (``LOG_FORMAT=json``).

A record carries exactly the fields in ``JSON_LOG_FIELDS`` (PRD-282 §5.7) and nothing
else: a record's ``extra`` and any other attribute never reach it. The first six are on
every line. The request's fields are there when the request has one: the id, the
workspace id, the method and path, and the trace and span ids when the trace is being
recorded. An exception, or a stack a caller asked for, is one field, ``exception``: the
record stays one line, since ``json`` escapes the newlines in it.

The request's fields come from a callable the formatter is given (the API's collapses a
file hash in the path first), so this module needs nothing but the standard library and
holds no context of its own. The workspace worker's image holds only its own folder, so it
keeps a byte-identical copy (``services/workspace-worker/json_log_format.py``).

The filters that keep things out of the logs run before a formatter, on the handler or
the logger (the API's request-log guard and file-hash filter), and they change the
record's message: what they keep out of a text line is kept out of a JSON one.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Callable, Mapping

JSON_LOG_FIELDS = (
    "time", "level", "logger", "message", "service", "version",
    "request_id", "workspace_id", "method", "path", "trace_id", "span_id",
    "exception",
)
# The request's own fields, in JSON_LOG_FIELDS: written only when they hold a value.
REQUEST_LOG_FIELDS = JSON_LOG_FIELDS[6:12]
EXCEPTION_FIELD = JSON_LOG_FIELDS[12]

RequestFields = Callable[[], Mapping[str, str]]


def _utc_time(created: float) -> str:
    """``created`` as ISO 8601 in UTC, to the millisecond: ``2026-11-22T09:30:00.123Z``."""
    stamp = datetime.fromtimestamp(created, tz=timezone.utc).isoformat(timespec="milliseconds")
    return stamp.replace("+00:00", "Z")


class JsonLineFormatter(logging.Formatter):
    """A record as one line of JSON holding ``JSON_LOG_FIELDS`` and no other key."""

    def __init__(self, service: str, version: str, request_fields: RequestFields) -> None:
        super().__init__()
        self._service = service
        self._version = version
        self._request_fields = request_fields

    def format(self, record: logging.LogRecord) -> str:
        line = {
            "time": _utc_time(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self._service,
            "version": self._version,
        }
        context = self._request_fields()
        line.update((name, str(context[name])) for name in REQUEST_LOG_FIELDS if context.get(name))
        trace = self._trace(record)
        if trace:
            line[EXCEPTION_FIELD] = trace
        return json.dumps(line, default=str)

    def _trace(self, record: logging.LogRecord) -> str:
        """The exception's traceback and the stack a caller asked for, as one text; the record is not changed."""
        parts = []
        if record.exc_info and record.exc_info[0] is not None:
            parts.append(self.formatException(record.exc_info))
        elif record.exc_text:
            parts.append(record.exc_text)
        if record.stack_info:
            parts.append(self.formatStack(record.stack_info))
        return "\n".join(parts)


__all__ = ["EXCEPTION_FIELD", "JSON_LOG_FIELDS", "JsonLineFormatter", "REQUEST_LOG_FIELDS", "RequestFields"]
