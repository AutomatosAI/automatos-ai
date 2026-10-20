"""PRD-264 EUAI-03 — a channel message can only leave through a point that adds the AI notice.

Channels have two outbound paths, and each has one point that applies the
Article 50(1) notice:

* the adapter path: ``BaseChannelAdapter._send_reply`` in ``channels/base.py``
  is the only caller of an adapter's ``send_message``;
* the driver path: ``send_to_channel`` in ``channels/sender.py`` is the only
  caller of a driver's ``send``.

This test reads every orchestrator module (tests excluded) with ``ast`` and
fails when a third path appears, so a new caller can't skip the notice.

* An adapter ``send_message`` call is any ``<x>.send_message(...)``. The only
  ones allowed outside ``base.py`` are python-telegram-bot's ``Bot`` calls in
  the Telegram adapter (``<x>.bot.send_message``): the adapter's own transport
  inside its ``send_message``, the /start reply (which says it is an AI itself,
  tested in test_prd264_euai03_channel_notice_paths.py) and the error apology.
* A driver ``send`` call is ``<x>.send(...)`` with no positional argument:
  ``ChannelDriver.send`` is keyword-only. Platform SDK and HTTP sends
  (``channel.send(chunk)``, ``client.send(request)``) pass a positional one.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

ORCH = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"tests", "__pycache__", "node_modules", ".venv", "venv"}
BASE_PY = "channels/base.py"
SENDER_PY = "channels/sender.py"
TELEGRAM_ADAPTER_PY = "channels/telegram_adapter.py"
TELEGRAM_BOT_RECEIVER_SUFFIX = ".bot"
REPLY_POINT = "_send_reply"
SEND_POINT = "send_to_channel"
MODULE_SCOPE = "<module>"


class SendCall(NamedTuple):
    path: str
    line: int
    function: str
    receiver: str
    method: str

    def label(self) -> str:
        return f"{self.path}:{self.line} {self.function}(): {self.receiver}.{self.method}(...)"


def _is_scanned(path: Path) -> bool:
    rel = path.relative_to(ORCH)
    if SKIP_DIRS & set(rel.parts):
        return False
    return not (rel.name.startswith("test_") or rel.name == "conftest.py")


class _SendCallFinder(ast.NodeVisitor):
    """Collects send calls, each under its innermost enclosing function."""

    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.stack: list[str] = [MODULE_SCOPE]
        self.calls: list[SendCall] = []

    def _visit_function(self, node: ast.AST) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method == "send_message" or (method == "send" and not node.args):
                self.calls.append(SendCall(
                    self.rel, node.lineno, self.stack[-1], ast.unparse(node.func.value), method,
                ))
        self.generic_visit(node)


def _calls_in(path: Path) -> list[SendCall]:
    """Every ``.send_message(...)`` and keyword-only ``.send(...)`` call in ``path``."""
    source = path.read_text(encoding="utf-8", errors="replace")
    if ".send_message(" not in source and ".send(" not in source:
        return []
    rel = path.relative_to(ORCH).as_posix()
    finder = _SendCallFinder(rel)
    finder.visit(ast.parse(source, filename=rel))
    return finder.calls


@lru_cache(maxsize=1)
def _all_calls() -> tuple[SendCall, ...]:
    found: list[SendCall] = []
    for path in sorted(ORCH.rglob("*.py")):
        if _is_scanned(path):
            found.extend(_calls_in(path))
    return tuple(found)


def _allowed_adapter_send(call: SendCall) -> bool:
    if call.path == BASE_PY:
        return True
    return call.path == TELEGRAM_ADAPTER_PY and call.receiver.endswith(TELEGRAM_BOT_RECEIVER_SUFFIX)


def test_only_base_py_calls_an_adapters_send_message():
    offenders = [
        call.label() for call in _all_calls()
        if call.method == "send_message" and not _allowed_adapter_send(call)
    ]
    assert not offenders, (
        "These modules call an adapter's send_message directly, skipping the AI notice "
        "(PRD-264 EUAI-03). Reply through BaseChannelAdapter._send_reply instead:\n"
        + "\n".join(offenders)
    )


def test_only_sender_py_calls_a_drivers_send():
    offenders = [
        call.label() for call in _all_calls()
        if call.method == "send" and call.path != SENDER_PY
    ]
    assert not offenders, (
        "These modules call a channel driver's send directly, skipping the AI notice "
        "(PRD-264 EUAI-03). Send through channels.sender.send_to_channel instead:\n"
        + "\n".join(offenders)
    )


def test_base_py_sends_only_from_the_reply_point_that_adds_the_notice():
    calls = [c for c in _all_calls() if c.path == BASE_PY and c.method == "send_message"]
    assert calls, "base.py no longer sends a reply: the adapter path lost its outbound point"
    assert {c.function for c in calls} == {REPLY_POINT}
    assert {c.receiver for c in calls} == {"self"}


def test_sender_py_calls_the_driver_only_from_send_to_channel():
    calls = [c for c in _all_calls() if c.path == SENDER_PY and c.method == "send"]
    assert calls, "sender.py no longer calls a driver: the driver path lost its outbound point"
    assert {c.function for c in calls} == {SEND_POINT}


def _function_source(rel: str, name: str) -> str:
    source = (ORCH / rel).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name}() not found in {rel}")


def test_both_outbound_points_ask_for_the_notice_and_record_it():
    for rel, name in ((BASE_PY, REPLY_POINT), (SENDER_PY, SEND_POINT)):
        body = _function_source(rel, name)
        assert "notice_for(" in body, f"{rel} {name}() must decide the AI notice"
        assert "record_notice_sent(" in body, f"{rel} {name}() must record the notice it sent"


def test_the_scan_sees_the_known_transport_calls():
    """Guard against a scan that silently reads nothing (a wrong root, a bad filter)."""
    sites = {(c.path, c.function) for c in _all_calls()}
    assert (TELEGRAM_ADAPTER_PY, "send_message") in sites
    assert (BASE_PY, REPLY_POINT) in sites
    assert (SENDER_PY, SEND_POINT) in sites
