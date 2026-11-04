"""F105 (4 Nov): no route runs a synchronous database session on the event loop.

A self-hosted install froze for up to 30 s at a time. Its loop watchdog named
``async def`` routes doing synchronous SQLAlchemy work: while one waited for a pooled
connection (pool_timeout 30 s) or ran a slow query, every other request on that worker
waited too, the phone's included. 479 routes in 82 files had that shape. AGENTS.md's
rule is that a route that touches the database without awaiting anything is a plain
``def``, which FastAPI runs in its threadpool.

Pinned statically over every router: an ``async def`` route that takes
``Depends(get_db)`` awaits something in its own body (``await``, ``async for``,
``async with``). Awaits inside a nested function don't count, because the route's own
body still runs on the loop.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, Tuple

ORCH = Path(__file__).resolve().parents[1]
SKIPPED_DIRS = {"tests", "alembic", "migrations", "__pycache__"}
ROUTE_METHODS = {"get", "post", "put", "patch", "delete", "options", "head", "api_route"}
SESSION_DEPENDENCY = "get_db"
# Fewer than this many session routes means the scan is broken, not the code clean.
MIN_SESSION_ROUTES = 400


def _modules() -> Iterator[Path]:
    for path in ORCH.rglob("*.py"):
        if SKIPPED_DIRS.isdisjoint(path.relative_to(ORCH).parts):
            yield path


def _is_route(node: ast.AST) -> bool:
    return any(
        isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr in ROUTE_METHODS
        for d in node.decorator_list
    )


def _takes_a_session(node: ast.AST) -> bool:
    defaults = [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]
    annotations = [a.annotation for a in (*node.args.args, *node.args.kwonlyargs) if a.annotation is not None]
    return any(SESSION_DEPENDENCY in ast.unparse(n) for n in (*defaults, *annotations))


def _own_body(node: ast.AST) -> Iterator[ast.AST]:
    stack = list(node.body)
    while stack:
        current = stack.pop()
        yield current
        for child in ast.iter_child_nodes(current):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                stack.append(child)


def _awaits(node: ast.AST) -> bool:
    return any(
        isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith))
        or (isinstance(n, ast.comprehension) and n.is_async)
        for n in _own_body(node)
    )


def _session_routes() -> Iterator[Tuple[str, ast.AST]]:
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_route(node) and _takes_a_session(node):
                yield f"{path.relative_to(ORCH)}:{node.lineno} {node.name}", node


def test_the_scan_sees_the_routers():
    assert sum(1 for _ in _session_routes()) >= MIN_SESSION_ROUTES


def test_no_async_route_over_a_session_awaits_nothing():
    offenders = sorted(
        where for where, node in _session_routes()
        if isinstance(node, ast.AsyncFunctionDef) and not _awaits(node)
    )
    assert offenders == [], (
        "async def routes over a synchronous session that await nothing run their database "
        "work on the event loop; make them plain def (AGENTS.md):\n" + "\n".join(offenders)
    )
