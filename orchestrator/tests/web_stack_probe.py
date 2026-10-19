"""PRD-262 US-008: what a FRESH process serves (run as a SUBPROCESS, never imported).

F092 hid because it showed only on a cold start: a fresh install served a route
manifest of about a dozen routes while warm interpreters and the frozen image looked
fine. This probe imports the real app in a clean interpreter, DB-free (the
``scripts/dump_routes.py`` precedent: ``create_engine`` is lazy and the lifespan never
runs on import), and prints one JSON object:

- ``collected``: the ``[method, path]`` pairs ``scripts/dump_routes.py`` collects;
- ``answers``: for each ``[method, path]`` asked about, the status the app's routing
  gives a request for that path with ``PROBE_METHOD``, a method no route takes. 405
  means a route serves the path; 404 means none does. ``app.router`` is called
  directly: routing only, so no middleware, no lifespan, and no endpoint or
  dependency runs;
- ``routers``: for every RouterSpec of ``router_manifest.py``, whether it imported,
  its HTTP routes (asked about in ``answers``) and its routes whose endpoint the app
  does not serve.

Usage (from ``orchestrator/``), with the paths of the committed manifest on stdin::

    python3 tests/web_stack_probe.py < reports/route-manifest.json
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

# Script-mode sys.path[0] is tests/, not the orchestrator root (authz_sweep_probe.py).
_ORCH_ROOT = str(Path(__file__).resolve().parents[1])
if _ORCH_ROOT not in sys.path:
    sys.path.insert(0, _ORCH_ROOT)

PROBE_METHOD = "PROPFIND"  # no route of the API takes it, so a served path answers 405
_PARAM = re.compile(r"\{(\w+)(?::(\w+))?\}")
_SAMPLES = {"int": "1", "float": "1.5", "uuid": "00000000-0000-0000-0000-000000000001"}


def concrete(template: str) -> str:
    """``template`` with each path parameter filled with a value its convertor accepts."""
    return _PARAM.sub(lambda m: _SAMPLES.get(m.group(2) or "str", "probe"), template)


async def _status(router, path: str) -> int:
    scope = {
        "type": "http", "http_version": "1.1", "method": PROBE_METHOD, "scheme": "http",
        "path": path, "raw_path": path.encode(), "root_path": "", "query_string": b"",
        "headers": [], "client": ("127.0.0.1", 1), "server": ("testserver", 80),
    }
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await router(scope, receive, send)
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


async def _answers(router, templates) -> dict:
    return {template: await _status(router, concrete(template)) for template in sorted(set(templates))}


def _router_report(spec, router, served_endpoints) -> dict:
    from fastapi.routing import iter_route_contexts

    routes = list(iter_route_contexts(router.routes))
    return {
        "spec": spec.name,
        "imported": True,
        "http_paths": sorted({route.path for route in routes if route.methods}),
        "unserved": sorted(route.path for route in routes if route.endpoint not in served_endpoints),
    }


def main() -> None:
    from fastapi.routing import iter_route_contexts

    from main import app
    from router_manifest import MANIFEST_ROUTERS, load_routers
    from scripts.dump_routes import collect_pairs

    committed = json.load(sys.stdin)["routes"]
    served_endpoints = {route.endpoint for route in iter_route_contexts(app.routes)}
    mounted, failures = load_routers(MANIFEST_ROUTERS, allow_degraded=True)
    routers = [_router_report(spec, router, served_endpoints) for spec, router in mounted]
    routers += [{"spec": spec.name, "imported": False, "optional": spec.optional, "error": repr(exc)}
                for spec, exc in failures]
    asked = [r["path"] for r in committed if r["method"]] + [p for r in routers for p in r.get("http_paths", [])]
    report = {
        "collected": [[pair["method"], pair["path"]] for pair in collect_pairs(app)],
        "answers": asyncio.run(_answers(app.router, asked)),
        "routers": routers,
    }
    print("WEB_STACK_PROBE=" + json.dumps(report))


if __name__ == "__main__":
    main()
