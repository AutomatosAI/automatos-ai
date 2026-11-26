"""PRD-282 US-301: what ``LOG_FORMAT`` and ``LOG_LEVEL`` may be, read when logging starts.

Each container's own config module reads both as the text in the environment
(``orchestrator/config.py``; the workspace worker's ``worker_config.py``), and each
``setup_logging`` reads them here, before the first handler is made.

* ``LOG_FORMAT``: ``text`` (the default) or ``json``, one JSON record per line.
* ``LOG_LEVEL``: a level name (``DEBUG``, ``INFO``, ``WARNING`` or ``WARN``, ``ERROR``,
  ``CRITICAL`` or ``FATAL``); ``INFO`` is the default.

Both are read without regard to case or surrounding spaces, and an empty value is the
default, so an install that sets neither, or sets one empty, logs exactly as before.

A value logging cannot use (a misspelling, ``NOTSET``, a number) never stops a service:
the default is used, and ``unusable`` gives the one warning ``setup_logging`` writes for
it, which names the setting and what it may be, never the value.

It imports nothing but ``logging``. The workspace worker's image holds only its own folder,
so it keeps a byte-identical copy (``services/workspace-worker/log_settings.py``).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

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

UNUSABLE = "{name} is not one of {allowed}: using {default}."


def _format_of(raw: Optional[str]) -> Optional[str]:
    """``text`` or ``json`` for ``raw``; the default for empty; None for any other value."""
    value = (raw or "").strip().lower() or DEFAULT_LOG_FORMAT
    return value if value in LOG_FORMATS else None


def _level_name_of(raw: Optional[str]) -> Optional[str]:
    """The level name for ``raw``; the default for empty; None for any other value."""
    value = (raw or "").strip().upper() or DEFAULT_LOG_LEVEL
    return value if value in LOG_LEVELS else None


def log_format(raw: Optional[str]) -> str:
    """``text`` or ``json`` for the ``LOG_FORMAT`` text ``raw``; ``text`` for anything else."""
    return _format_of(raw) or DEFAULT_LOG_FORMAT


def log_level(raw: Optional[str]) -> int:
    """The ``logging`` level for the ``LOG_LEVEL`` text ``raw``; ``INFO`` for anything else."""
    return LOG_LEVELS[_level_name_of(raw) or DEFAULT_LOG_LEVEL]


def unusable(format_raw: Optional[str], level_raw: Optional[str]) -> Tuple[str, ...]:
    """The warning for each of the two settings that holds a value logging cannot use."""
    warnings = []
    if _format_of(format_raw) is None:
        warnings.append(UNUSABLE.format(name="LOG_FORMAT", allowed=", ".join(LOG_FORMATS),
                                        default=DEFAULT_LOG_FORMAT))
    if _level_name_of(level_raw) is None:
        warnings.append(UNUSABLE.format(name="LOG_LEVEL", allowed=", ".join(LOG_LEVELS),
                                        default=DEFAULT_LOG_LEVEL))
    return tuple(warnings)


__all__ = [
    "DEFAULT_LOG_FORMAT", "DEFAULT_LOG_LEVEL", "LOG_FORMATS", "LOG_FORMAT_JSON", "LOG_FORMAT_TEXT",
    "LOG_LEVELS", "UNUSABLE", "log_format", "log_level", "unusable",
]
