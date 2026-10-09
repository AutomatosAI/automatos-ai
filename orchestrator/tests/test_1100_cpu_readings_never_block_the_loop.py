"""#1100 — the system/dashboard API routes must never block the event loop.

``psutil.cpu_percent(interval=1)`` sleeps for a full second. Inside the
``async def`` handlers of ``api/system.py``, ``api/statistics.py`` and
``api/analytics_real.py`` that held the loop hostage on every dashboard poll,
stalling chat streams, SSE and every other request on the worker.

Pinned statically (importing the routers builds the whole app): every
``cpu_percent`` call in those three modules must pass ``interval=None`` —
the use since the previous call, primed once at startup in ``main.lifespan``
— and GET ``/api/system/metrics`` must no longer write metric rows to the
database from the request handler.
"""
from __future__ import annotations

import ast
from pathlib import Path

ORCHESTRATOR = Path(__file__).resolve().parents[1]
MODULES = ["api/system.py", "api/statistics.py", "api/analytics_real.py"]


def _parse(rel: str) -> ast.Module:
    return ast.parse((ORCHESTRATOR / rel).read_text(encoding="utf-8"))


def _cpu_percent_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cpu_percent"
    ]


def _interval_of(call: ast.Call) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == "interval":
            return kw.value
    if call.args:
        return call.args[0]
    return None


def test_every_cpu_reading_in_the_api_modules_is_non_blocking():
    for rel in MODULES:
        calls = _cpu_percent_calls(_parse(rel))
        assert calls, f"{rel} must still report the CPU"
        for call in calls:
            interval = _interval_of(call)
            assert isinstance(interval, ast.Constant) and interval.value is None, (
                f"{rel}:{call.lineno}: cpu_percent must use interval=None — "
                "any other interval sleeps on the event loop (see #1100)"
            )


def test_system_metrics_get_no_longer_writes_rows():
    tree = _parse("api/system.py")
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "_store_current_metrics" not in names, (
        "GET /api/system/metrics must not write metric rows to the database (#1100)"
    )


def test_lifespan_primes_the_cpu_reading():
    tree = _parse("main.py")
    lifespans = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    ]
    assert len(lifespans) == 1, f"one lifespan in main.py, found {len(lifespans)}"
    assert _cpu_percent_calls(lifespans[0]), (
        "main.lifespan must prime psutil once at startup so the first "
        "interval=None reading is not 0.0 (#1100)"
    )
