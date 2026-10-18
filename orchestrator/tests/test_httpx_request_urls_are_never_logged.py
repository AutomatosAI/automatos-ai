"""httpx never writes a request URL to the logs.

A request URL can carry a secret in its path: python-telegram-bot polls
https://api.telegram.org/bot<token>/getUpdates. httpx logs every request line
at INFO, so after #783 enabled the Telegram driver, prod's console and the
log-relay (Loki) carried the bot token on every poll. setup_logging holds httpx
and httpcore at WARNING: request lines are dropped, and warnings still show.

#1051: openai 3.x sends through httpx's fork, httpx2 (with httpcore2), which logs
the same request line under its own name, so every model call's URL reached the
logs. Both services' setup_logging now hold the forks at WARNING too.
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest

from core.monitoring.automatos_logging import setup_logging

LINE = 'HTTP Request: POST https://api.telegram.org/bot8300000000:AAtestTOKENvalue/getUpdates "HTTP/1.1 200 OK"'
GUARDED = ("httpx", "httpcore", "httpx2", "httpcore2")
WORKER_LOGGING = Path(__file__).resolve().parents[2] / "services" / "workspace-worker" / "automatos_logging.py"


def _worker_setup_logging():
    spec = importlib.util.spec_from_file_location("worker_automatos_logging", WORKER_LOGGING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.setup_logging


@pytest.fixture(autouse=True)
def _fresh_loggers(monkeypatch):
    for name in GUARDED:  # a fresh process: nothing set these yet
        monkeypatch.setattr(logging.getLogger(name), "level", logging.NOTSET)


@pytest.mark.parametrize("service", ["backend", "workspace-worker"])
@pytest.mark.parametrize("name", GUARDED)
def test_a_request_line_with_a_token_in_its_url_is_not_logged(caplog, service, name):
    setup = setup_logging if service == "backend" else _worker_setup_logging()
    setup(service="test", level=logging.INFO, enable_relay=False)
    with caplog.at_level(logging.INFO):
        logging.getLogger(name).info(LINE)
    assert "AAtestTOKENvalue" not in caplog.text


def test_an_httpx_warning_still_shows(caplog):
    setup_logging(service="test", enable_relay=False)
    with caplog.at_level(logging.INFO):
        logging.getLogger("httpx").warning("connection pool is full")
        logging.getLogger("httpx2").warning("connection pool is full (httpx2)")
    assert "connection pool is full" in caplog.text and "(httpx2)" in caplog.text


def test_a_model_call_through_the_openai_sdk_logs_no_request_line(caplog):
    import httpx2
    import openai

    def answer(request):
        return httpx2.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]})

    setup_logging(service="test", level=logging.INFO, enable_relay=False)
    client = openai.OpenAI(api_key="test-key", base_url="https://models.example/v1/route-AAtestTOKENvalue",
                           http_client=openai.DefaultHttpxClient(transport=httpx2.MockTransport(answer)))
    with caplog.at_level(logging.INFO):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    assert "AAtestTOKENvalue" not in caplog.text
    assert "HTTP Request" not in caplog.text
