"""F229 — ``GET /health`` must never block the event loop.

Under load on build 4 the route held the loop for 17.9 s (the loop watchdog's stack:
main.py ``health_check`` → ``psutil.cpu_percent(interval=0.1)`` → ``cpu_times`` →
``open_binary``): an ``async def`` that slept 0.1 s, read /proc and ran a synchronous
database probe on the loop, which fed a 99 s /health and scheduler slips of up to 218 s.

Pinned statically (importing main.py builds the whole app): the route is a plain ``def``,
so FastAPI runs it in the threadpool (F105), and no CPU reading in it sleeps. Its probes
live in core/health_probe.py (F458); they are plain defs too.
"""
from __future__ import annotations

import ast
from pathlib import Path

ORCH = Path(__file__).resolve().parents[1]
MAIN = ORCH / "main.py"
PROBES = ORCH / "core" / "health_probe.py"


def _function(path: Path, name: str) -> ast.AST:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    assert len(found) == 1, f"one {name} in {path.name}, found {len(found)}"
    return found[0]


def test_the_health_route_is_a_plain_def():
    route = _function(MAIN, "health_check")
    assert isinstance(route, ast.FunctionDef), "health_check must be a plain def: it probes the database and /proc synchronously"
    assert not any(isinstance(node, ast.Await) for node in ast.walk(route))


def test_the_probes_are_plain_defs():
    tree = ast.parse(PROBES.read_text(encoding="utf-8"))
    assert not [node.name for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)]
    for name in ("database_status", "_select_one", "host_metrics"):
        assert isinstance(_function(PROBES, name), ast.FunctionDef)


def test_no_cpu_reading_in_the_health_route_sleeps():
    # The route reads the CPU through health_probe.host_metrics (a plain def it calls).
    calls = [
        node for node in ast.walk(_function(PROBES, "host_metrics"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "cpu_percent"
    ]
    assert calls, "the route reports the CPU"
    for call in calls:
        interval = next((kw.value for kw in call.keywords if kw.arg == "interval"), call.args[0] if call.args else None)
        assert isinstance(interval, ast.Constant) and interval.value is None, (
            "cpu_percent(interval=None) measures since the previous reading; any other interval sleeps"
        )
