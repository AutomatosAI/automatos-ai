"""PRD-195 S2 — boundary-sweep probe (run as a SUBPROCESS, never imported).

Imports the real FastAPI app (DB-free by construction — the
``scripts/dump_routes.py`` / ``test_route_manifest.py`` precedent: create_engine
is lazy and the lifespan never runs on import) and prints one JSON record per
served route+method with the authorization facts the sweep asserts on:

- ``hybrid``   — ``get_request_context_hybrid`` in the flattened dependency tree
- ``su``       — ``require_super_admin`` present (PRD-143 obs lock)
- ``wsadmin``  — ``require_workspace_admin`` present (PRD-185 S12)
- ``perm``     — the ``require_workspace_permission`` marker string, if gated
- ``admin_in_handler`` — the endpoint body asserts the shared admin check
  (``assert_admin(ctx)`` / ``_assert_admin(ctx)`` / ``_require_admin(ctx…)``)
- ``own_gate_in_handler`` — the endpoint body carries its own explicit
  auth-type gate (today: credentials resolve)

Usage (from ``orchestrator/``)::

    python3 tests/authz_sweep_probe.py > routes.json

The caller (tests/test_p2w2_authz_boundary_sweep.py) provides the fake
POSTGRES_* env so generation is provably DB-free.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

# Script-mode sys.path[0] is tests/, not the orchestrator root — put the root
# first so `import core` / `import main` resolve (the -m lanes get this free).
_ORCH_ROOT = str(Path(__file__).resolve().parents[1])
if _ORCH_ROOT not in sys.path:
    sys.path.insert(0, _ORCH_ROOT)


def _flatten_calls(dependant, acc):
    for sub in dependant.dependencies:
        if sub.call is not None:
            acc.append(sub.call)
        _flatten_calls(sub, acc)


def main() -> None:
    from fastapi.routing import APIRoute

    from core.auth.hybrid import get_request_context_hybrid
    from core.auth.super_admin import require_super_admin
    from core.auth.workspace_admin import require_workspace_admin
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR
    from main import app

    records = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        calls: list = []
        _flatten_calls(route.dependant, calls)

        perm = None
        for call in calls:
            marker = getattr(call, PERMISSION_MARKER_ATTR, None)
            if marker:
                perm = marker
                break

        try:
            src = inspect.getsource(route.endpoint)
        except (OSError, TypeError):
            src = ""

        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            records.append(
                {
                    "method": method,
                    "path": route.path,
                    "module": getattr(route.endpoint, "__module__", ""),
                    "hybrid": get_request_context_hybrid in calls,
                    "su": require_super_admin in calls,
                    "wsadmin": require_workspace_admin in calls,
                    "perm": perm,
                    "admin_in_handler": (
                        "assert_admin(ctx)" in src
                        or "_require_admin(ctx" in src
                        # api/gdpr.py's hand-rolled gate — PRD-196 S7 owns its
                        # consolidation; classified, not touched (PRD-195).
                        or "_require_workspace_admin(ctx)" in src
                        # PRD-207 voice/arm: platform-admin gate in the handler.
                        or "_is_platform_admin(ctx)" in src
                    ),
                    "own_gate_in_handler": (
                        "Admin access required to resolve credentials" in src
                        # PRD-207 voice/web-call: the mint's own signed-in-user gate.
                        or "Live voice requires a signed-in user" in src
                    ),
                }
            )

    json.dump(records, sys.stdout)


if __name__ == "__main__":
    main()
