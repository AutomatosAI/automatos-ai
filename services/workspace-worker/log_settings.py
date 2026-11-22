"""PRD-282 US-301: what ``LOG_FORMAT`` and ``LOG_LEVEL`` may be, checked when logging starts.

Each container's own config module reads both as the text in the environment and never
refuses one (``orchestrator/config.py``, like the settings PRD-277 and PRD-278 added; the
workspace worker's ``worker_config.py``): the check is here, where the value is used. Each
``setup_logging`` calls it before the first handler is made, so an install with a value it
cannot use does not start, and the refusal names the setting and what it may be, never the
value.

* ``LOG_FORMAT``: ``text`` (the default) or ``json``, one JSON record per line.
* ``LOG_LEVEL``: a level name (``DEBUG``, ``INFO``, ``WARNING`` or ``WARN``, ``ERROR``,
  ``CRITICAL`` or ``FATAL``); ``INFO`` is the default. ``NOTSET`` and numbers are refused.

Both are read without regard to case or surrounding spaces, and an empty value is the
default, so an install that sets neither, or sets one empty, logs exactly as before.

It imports nothing but ``logging``. The workspace worker's image holds only its own folder,
so it keeps a byte-identical copy (``services/workspace-worker/log_settings.py``).
"""
from __future__ import annotations

import logging
from typing import Optional

LOG_FORMAT_TEXT = "text"
LOG_FORMAT_JSON = "json"
LOG_FORMATS = (LOG_FORMAT_TEXT, LOG_FORMAT_JSON)
DEFAULT_LOG_FORMAT = LOG_FORMAT_TEXT

# The names Python's logging gives its levels, two of them aliases; NOTSET is left out,
# since a root logger at NOTSET writes every line of every library.
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.CRITICAL,
}
DEFAULT_LOG_LEVEL = "INFO"

REFUSED = "{name} cannot be used: it must be one of {allowed}, or unset for {default}."


class LogSettingRefused(RuntimeError):
    """Start-up refused: ``LOG_FORMAT`` or ``LOG_LEVEL`` holds a value logging cannot use."""


def _plain(raw: Optional[str]) -> str:
    return (raw or "").strip()


def log_format(raw: Optional[str]) -> str:
    """``text`` or ``json`` for the ``LOG_FORMAT`` text ``raw``; ``text`` when it is unset or empty."""
    value = _plain(raw).lower() or DEFAULT_LOG_FORMAT
    if value not in LOG_FORMATS:
        raise LogSettingRefused(REFUSED.format(name="LOG_FORMAT", allowed=", ".join(LOG_FORMATS),
                                               default=DEFAULT_LOG_FORMAT))
    return value


def log_level(raw: Optional[str]) -> int:
    """The ``logging`` level for the ``LOG_LEVEL`` text ``raw``; ``INFO`` when it is unset or empty."""
    value = _plain(raw).upper() or DEFAULT_LOG_LEVEL
    if value not in LOG_LEVELS:
        raise LogSettingRefused(REFUSED.format(name="LOG_LEVEL", allowed=", ".join(LOG_LEVELS),
                                               default=DEFAULT_LOG_LEVEL))
    return LOG_LEVELS[value]


__all__ = [
    "DEFAULT_LOG_FORMAT", "DEFAULT_LOG_LEVEL", "LOG_FORMATS", "LOG_FORMAT_JSON", "LOG_FORMAT_TEXT",
    "LOG_LEVELS", "LogSettingRefused", "log_format", "log_level",
]
