"""httpx never writes a request URL to the logs.

A request URL can carry a secret in its path: python-telegram-bot polls
https://api.telegram.org/bot<token>/getUpdates. httpx logs every request line
at INFO, so after #783 enabled the Telegram driver, prod's console and the
log-relay (Loki) carried the bot token on every poll. setup_logging holds httpx
and httpcore at WARNING: request lines are dropped, and warnings still show.
"""
from __future__ import annotations

import logging

from core.monitoring.automatos_logging import setup_logging

LINE = 'HTTP Request: POST https://api.telegram.org/bot8300000000:AAtestTOKENvalue/getUpdates "HTTP/1.1 200 OK"'


def test_a_request_line_with_a_token_in_its_url_is_not_logged(caplog, monkeypatch):
    for name in ("httpx", "httpcore"):  # a fresh process: nothing set these yet
        monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)
    setup_logging(service="test", enable_relay=False)
    with caplog.at_level(logging.INFO):
        logging.getLogger("httpx").info(LINE)
        logging.getLogger("httpcore").info(LINE)
    assert "AAtestTOKENvalue" not in caplog.text


def test_an_httpx_warning_still_shows(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("httpx"), "level", logging.NOTSET)
    setup_logging(service="test", enable_relay=False)
    with caplog.at_level(logging.INFO):
        logging.getLogger("httpx").warning("connection pool is full")
    assert "connection pool is full" in caplog.text
