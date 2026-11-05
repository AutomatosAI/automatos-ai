"""F105 (4 Nov): a route that reads the request context takes it as a parameter.

Found while sweeping the routers: six ``/api/workflow-templates`` routes (get, create,
update, delete, use, featured) filtered by ``ctx.workspace_id`` but never took ``ctx``,
so every call raised NameError and answered 500. The web app calls five of them
(``lib/api-client.ts``). They now take ``ctx`` from the same dependency as the list route.

Pinned statically over every router: a route whose own body reads the name ``ctx`` has
a parameter called ``ctx``.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, Tuple

ORCH = Path(__file__).resolve().parents[1]
SKIPPED_DIRS = {"tests", "alembic", "migrations", "__pycache__"}
ROUTE_METHODS = {"get", "post", "put", "patch", "delete", "options", "head", "api_route"}
CONTEXT = "ctx"


def _routes() -> Iterator[Tuple[str, ast.AST]]:
    for path in ORCH.rglob("*.py"):
        if not SKIPPED_DIRS.isdisjoint(path.relative_to(ORCH).parts):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr in ROUTE_METHODS
                for d in node.decorator_list
            ):
                yield f"{path.relative_to(ORCH)}:{node.lineno} {node.name}", node


def _params(node: ast.AST) -> set:
    args = node.args
    return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}


def _uses_context(node: ast.AST, use: type) -> bool:
    return any(isinstance(n, ast.Name) and n.id == CONTEXT and isinstance(n.ctx, use)
               for stmt in node.body for n in ast.walk(stmt))


def _reads_an_unbound_context(node: ast.AST) -> bool:
    """Reads ``ctx`` that is neither a parameter nor assigned in the route (``task, ctx = session``)."""
    return (_uses_context(node, ast.Load) and CONTEXT not in _params(node)
            and not _uses_context(node, ast.Store))


def test_every_route_that_reads_ctx_takes_it():
    offenders = sorted(where for where, node in _routes() if _reads_an_unbound_context(node))
    assert offenders == [], "routes that read ctx without taking it answer 500 (NameError):\n" + "\n".join(offenders)


def test_the_template_routes_take_ctx():
    from api import workflow_templates as templates

    for name in ("get_workflow_template", "create_workflow_template", "update_workflow_template",
                 "delete_workflow_template", "record_template_usage", "list_featured_templates"):
        assert CONTEXT in getattr(templates, name).__code__.co_varnames, name
