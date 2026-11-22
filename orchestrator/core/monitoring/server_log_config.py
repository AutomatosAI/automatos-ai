"""PRD-282 US-301: the JSON line for uvicorn's supervising process (``LOG_FORMAT=json``).

With more than one server process, uvicorn's parent writes a few lines of its own (its
start, a process that exits, its stop) and never imports the app. With ``json`` the start
script (``docker-entrypoint.sh``) points uvicorn at ``server_log_config.json`` beside this
module (``UVICORN_LOG_CONFIG``), whose formatter and filter are made here. This module
imports neither ``config`` nor ``main``, so the parent stays as light as it is: the service
name is fixed, the version and the level are read from the environment as config.py reads
them, and a line carries no request fields. Each server process then replaces the formatter
with the app's own (``server_log_lines.format_server_lines``).
"""
from __future__ import annotations

import logging
import os

from core.monitoring import log_settings
from core.monitoring.json_log_format import JsonLineFormatter

SERVICE = "automatos-backend"
UNVERSIONED_BUILD = "dev"   # config.UNVERSIONED_BUILD


def _setting(name: str) -> str:
    return os.environ.get(name, "")  # noqa: TID251 — the parent process never imports config


def json_formatter() -> JsonLineFormatter:
    """The parent's JSON line: the API's fields, its service and version, no request fields."""
    return JsonLineFormatter(SERVICE, _setting("AUTOMATOS_VERSION").strip() or UNVERSIONED_BUILD, dict)


class LevelFilter(logging.Filter):
    """A handler filter passing records at ``LOG_LEVEL`` and above (INFO for an unusable value)."""

    def __init__(self) -> None:
        super().__init__()
        self.level = log_settings.log_level(_setting("LOG_LEVEL"))

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.level


__all__ = ["LevelFilter", "SERVICE", "UNVERSIONED_BUILD", "json_formatter"]
