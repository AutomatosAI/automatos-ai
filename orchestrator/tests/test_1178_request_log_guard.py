"""#1178: the request-log guard holds at the handlers, whatever an SDK sets.

#1051 held httpx and its relatives at WARNING by logger level. A level doesn't hold:
importing the Anthropic SDK with ANTHROPIC_LOG=debug (or composio_client with
COMPOSIO_LOG) sets the SDK's logger and httpx's to DEBUG, whatever setup_logging set
before. The SDK then logs each request's options, its body (the prompt) included, and
httpx logs the URL. And about 20 scripts set up logging with ``logging.basicConfig``,
which never had the guard. ``RequestLogGuard`` sits on the handlers instead, and
every script that configures logging itself puts it on its own.
"""
from __future__ import annotations

import ast
import io
import logging
from pathlib import Path

import httpx
import pytest

from core.monitoring.automatos_logging import setup_logging
from core.monitoring.request_log_guard import RequestLogGuard, guard_request_logs

_ORCH = Path(__file__).resolve().parents[1]
MARKER = "MARKER_PROMPT_7"


@pytest.fixture
def captured():
    """A handler on the root logger, as a script's basicConfig or an SDK's own adds one."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    root.addHandler(handler)
    yield stream
    root.removeHandler(handler)


@pytest.fixture
def anthropic_debug(monkeypatch):
    """ANTHROPIC_LOG=debug, applied as the SDK applies it when it's imported."""
    from anthropic._utils import _logs

    for name in ("anthropic", "httpx"):
        monkeypatch.setattr(logging.getLogger(name), "level", logging.getLogger(name).level)
    monkeypatch.setenv("ANTHROPIC_LOG", "debug")
    _logs.setup_logging()


def _anthropic_call():
    import anthropic

    def answer(request):
        return httpx.Response(200, json={
            "id": "m", "type": "message", "role": "assistant", "model": "claude-x",
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}})

    client = anthropic.Anthropic(api_key="test-key", base_url=f"https://models.example/route-{MARKER}",
                                 http_client=httpx.Client(transport=httpx.MockTransport(answer)), max_retries=0)
    client.messages.create(model="claude-x", max_tokens=8, messages=[{"role": "user", "content": MARKER}])


def test_anthropic_log_debug_set_after_setup_logging_leaks_no_prompt_or_url(captured, anthropic_debug):
    setup_logging(service="test", level=logging.INFO, enable_relay=False)
    from anthropic._utils import _logs

    _logs.setup_logging()   # an import after setup_logging resets the levels
    assert logging.getLogger("anthropic").level == logging.DEBUG
    _anthropic_call()
    assert MARKER not in captured.getvalue()


def test_an_sdk_retry_line_and_a_library_warning_still_show(captured):
    setup_logging(service="test", level=logging.INFO, enable_relay=False)
    logging.getLogger("anthropic._base_client").info("Retrying request to /v1/messages in 0.4 seconds")
    logging.getLogger("httpx2").warning("connection pool is full")
    logging.getLogger("anthropic._base_client").debug("Request options: %s", {"json_data": MARKER})
    logging.getLogger("urllib3.poolmanager").info("Redirecting %s -> %s", f"https://a/{MARKER}", "https://b")
    out = captured.getvalue()
    assert "Retrying request to /v1/messages" in out and "connection pool is full" in out
    assert MARKER not in out


def test_a_scripts_own_handler_is_guarded_once():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        guard_request_logs()
        guard_request_logs()
        assert sum(isinstance(f, RequestLogGuard) for f in handler.filters) == 1
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("httpx").info("HTTP Request: GET https://api.telegram.org/bot%s/getUpdates", MARKER)
        assert MARKER not in stream.getvalue()
    finally:
        root.removeHandler(handler)
        logging.getLogger("httpx").setLevel(logging.NOTSET)


_CONFIGURERS = {"basicConfig", "fileConfig", "dictConfig"}


def _calls(node: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) == name
               for n in ast.walk(node))


def _unguarded_configs(path: Path) -> list:
    """Each logging-config call in ``path`` with no ``guard_request_logs()`` after it in
    the same block of statements, as ``file:line``."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for index, statement in enumerate(body):
            if any(_calls(statement, name) for name in _CONFIGURERS) and not isinstance(
                    statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.With, ast.Try)):
                if not any(_calls(later, "guard_request_logs") for later in body[index:]):
                    found.append(f"{path.relative_to(_ORCH)}:{statement.lineno}")
    return found


def test_every_script_that_configures_logging_puts_the_guard_on_its_handlers():
    """A script, job or migration that sets up logging itself (basicConfig, fileConfig,
    dictConfig) calls guard_request_logs() after it, in the same block. A new one fails here."""
    unguarded = [hit for path in sorted(_ORCH.rglob("*.py"))
                 if "tests" not in path.parts and ".venv" not in path.parts
                 for hit in _unguarded_configs(path)]
    assert unguarded == []
