"""F229 — ``GET /health`` must never block the event loop.

Under load on build 4 the route held the loop for 17.9 s (the loop watchdog's stack:
main.py ``health_check`` → ``psutil.cpu_percent(interval=0.1)`` → ``cpu_times`` →
``open_binary``): an ``async def`` that slept 0.1 s, read /proc and ran a synchronous
database probe on the loop, which fed a 99 s /health and scheduler slips of up to 218 s.

Pinned statically (importing main.py builds the whole app): the route is a plain ``def``,
so FastAPI runs it in the threadpool (F105), and no CPU reading in it sleeps.
"""
from __future__ import annotations

import ast
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "main.py"


def _route(name: str) -> ast.AST:
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    found = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    assert len(found) == 1, f"one {name} in main.py, found {len(found)}"
    return found[0]


def test_the_health_route_is_a_plain_def():
    route = _route("health_check")
    assert isinstance(route, ast.FunctionDef), "health_check must be a plain def: it probes the database and /proc synchronously"
    assert not any(isinstance(node, ast.Await) for node in ast.walk(route))


def test_no_cpu_reading_in_the_health_route_sleeps():
    # The route reads the CPU through _health_metrics (a plain def it calls).
    assert isinstance(_route("_health_metrics"), ast.FunctionDef) and isinstance(_route("_health_database"), ast.FunctionDef)
    calls = [
        node for node in ast.walk(_route("_health_metrics"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "cpu_percent"
    ]
    assert calls, "the route reports the CPU"
    for call in calls:
        interval = next((kw.value for kw in call.keywords if kw.arg == "interval"), call.args[0] if call.args else None)
        assert isinstance(interval, ast.Constant) and interval.value is None, (
            "cpu_percent(interval=None) measures since the previous reading; any other interval sleeps"
        )
