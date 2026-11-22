"""PRD-282 US-301: the web server's own lines in the install's format.

uvicorn configures its loggers before it imports the app: ``uvicorn`` (``uvicorn.error``
writes through it) and ``uvicorn.access`` each have a handler of their own and do not
propagate, so the app's console handler never sees their lines, and log-relay never ships
them. ``setup_logging`` calls ``format_server_lines`` after uvicorn has done that, in each
worker process. Each of uvicorn's three loggers takes ``LOG_LEVEL`` (``level_server_lines``),
so ``WARNING`` drops the access lines and the start-up lines too, and the formatter on
uvicorn's own handlers changes (``format_server_lines``):

* ``json``: they write the same record as the app's console (``JsonLineFormatter``), so
  every line a worker writes has one shape.
* ``text``: their lines are as uvicorn writes them, and an access line ends with
  `` [req=<id>]``, the request's id, when it has one.

The filters stay where they are: ``path_digests`` on the access logger, before any handler.

With more than one worker, uvicorn's supervising process writes a few lines of its own (its
start, a worker that exits, its stop). It never imports the app: with ``json`` the start script
gives it ``server_log_config.json`` (``server_log_config``) for those lines.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

SERVER_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")
ACCESS_LOGGER = "uvicorn.access"


class RequestIdSuffix(logging.Formatter):
    """Another formatter's line, then `` [req=<id>]`` when the request has an id."""

    def __init__(self, inner: logging.Formatter, request_id: Callable[[], str]) -> None:
        super().__init__()
        self._inner = inner
        self._request_id = request_id

    def format(self, record: logging.LogRecord) -> str:
        line = self._inner.format(record)
        request_id = self._request_id()
        return f"{line} [req={request_id}]" if request_id else line


def level_server_lines(level: int) -> None:
    """Put each of uvicorn's loggers at ``level``, the install's ``LOG_LEVEL``."""
    for name in SERVER_LOGGERS:
        logging.getLogger(name).setLevel(level)


def format_server_lines(json_formatter: Optional[logging.Formatter], request_id: Callable[[], str]) -> None:
    """Give uvicorn's handlers ``json_formatter``; with none (text), add the request id to its access line, once."""
    if json_formatter is not None:
        for name in SERVER_LOGGERS:
            for handler in logging.getLogger(name).handlers:
                handler.setFormatter(json_formatter)
        return
    for handler in logging.getLogger(ACCESS_LOGGER).handlers:
        if not isinstance(handler.formatter, RequestIdSuffix):
            handler.setFormatter(RequestIdSuffix(handler.formatter or logging.Formatter(), request_id))


__all__ = ["ACCESS_LOGGER", "SERVER_LOGGERS", "RequestIdSuffix", "format_server_lines", "level_server_lines"]
