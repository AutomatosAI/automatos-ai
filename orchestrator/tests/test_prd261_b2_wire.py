"""PRD-261 B2-wire: the W-B2 theme routers, wired once into the app.

Pins (plan §2.4 W-B2 row, §2.6 B2-wire; Gerard D1-a/d/e):

* ``main.py`` mounts each of the five theme routers exactly once, and the live app serves
  every W-B2 route once (a probe in a clean subprocess with an unreachable Postgres, the
  ``test_route_manifest.py`` / authZ-sweep precedent);
* every W-B2 route is in ``reports/route-manifest.json`` and ``route_count == len(routes)``;
* the step-up dependency (A-07c) is on EXACTLY the routes §2.2 marks (import, opt-in,
  default, update, rollback, the settings PUT), each after its permission gate, and NOT on the
  workspace role change (its step-up lands with the client flow, PRD-261 W-A2, #1157), and the app renders its refusal with ``clerk_error`` at the
  top level;
* "who may install" (A-07e) gates import, update and rollback;
* every W-B2 mutating route carries ``require_workspace_permission`` (authZ-sweep bucket a);
  the files and boot routes carry none: they authorise by membership, never by
  ``ctx.workspace_id``;
* the import route's body limit is sized from ``THEME_MAX_PACKAGE_MB``;
* the app's security-headers middleware keeps the files route's ``sandbox`` CSP, and puts
  ``default-src 'none'`` in front of a route policy that names none.

The refusals themselves, on the real routes, are
``test_prd261_b2_step_up_routes.py``.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ORCH = Path(__file__).resolve().parents[1]
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

from fastapi.routing import APIRoute  # noqa: E402

import api.team as team_api  # noqa: E402
import api.themes_files as files_api  # noqa: E402
import api.themes_import as import_api  # noqa: E402
import api.themes_install as install_api  # noqa: E402
import api.themes_looks as looks_api  # noqa: E402
import api.themes_settings as settings_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
from config import config  # noqa: E402
from core.auth import step_up  # noqa: E402

THEME_MODULES = {
    "themes_import_router": import_api, "themes_install_router": install_api, "themes_files_router": files_api,
    "themes_looks_router": looks_api, "themes_settings_router": settings_api,
}
ROLE_CHANGE = ("PATCH", "/api/workspaces/{workspace_id}/team/members/{member_id}/role")
STEP_UP_ROUTES = {
    ("POST", "/api/themes/import"),
    ("POST", "/api/themes/installs/{install_id}/opt-in"),
    ("PUT", "/api/themes/default"),
    ("POST", "/api/themes/installs/{install_id}/update"),
    ("POST", "/api/themes/installs/{install_id}/rollback"),
    ("PUT", "/api/themes/settings"),
}
INSTALL_ROLE_ROUTES = {
    ("POST", "/api/themes/import"),
    ("POST", "/api/themes/installs/{install_id}/update"),
    ("POST", "/api/themes/installs/{install_id}/rollback"),
}
MEMBERSHIP_ROUTES = {("GET", "/api/themes/files/{digest}/{path:path}"), ("GET", "/api/themes/boot/{install_id}")}
MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
MB, MULTIPART_ENVELOPE = 1024 * 1024, 512 * 1024


def _calls(dependant):
    """Every dependency callable of a route, in resolution order (depth first)."""
    found = []
    for sub in dependant.dependencies:
        found += [sub.call] + _calls(sub)
    return found


def _theme_routes():
    """``{(method, path): APIRoute}`` over the five W-B2 routers."""
    table = {}
    for module in THEME_MODULES.values():
        for route in module.router.routes:
            assert isinstance(route, APIRoute)
            for method in route.methods - {"HEAD", "OPTIONS"}:
                table[(method, route.path)] = route
    return table


def _index(calls, predicate):
    return next((n for n, call in enumerate(calls) if predicate(call)), None)


def _is_step_up(call):
    return getattr(call, step_up.STEP_UP_MARKER_ATTR, None) is not None


def _is_gate(call):
    return getattr(call, permission_mod.PERMISSION_MARKER_ATTR, None) is not None


def _is_install_role(call):
    return "require_install_role" in getattr(call, "__qualname__", "")


# ---------------------------------------------------------------------------
# main.py and the route manifest
# ---------------------------------------------------------------------------


def _main_tree():
    return ast.parse((ORCH / "main.py").read_text(encoding="utf-8"))


def test_main_mounts_each_theme_router_exactly_once():
    mounted = [
        node.args[0].id for node in ast.walk(_main_tree())
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "include_router"
        and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id.startswith("themes_")
    ]
    assert sorted(mounted) == sorted(THEME_MODULES)


def test_main_installs_the_step_up_handler_once():
    calls = [node for node in ast.walk(_main_tree())
             if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "install_step_up_handler"]
    assert len(calls) == 1


def test_every_w_b2_route_is_in_the_manifest_and_the_count_holds():
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    assert manifest["route_count"] == len(manifest["routes"])
    served = {(r["method"], r["path"]) for r in manifest["routes"]}
    expected = set(_theme_routes())
    assert len(expected) == 16
    assert not expected - served, sorted(expected - served)
    assert ROLE_CHANGE in served


def test_every_theme_config_name_is_on_config_and_in_the_config_surface():
    names = {"THEMES_PRIVATE_UPLOAD", "THEMES_MARKETPLACE", "THEME_MAX_PACKAGE_MB", "THEME_MAX_CODE_KB",
             "THEME_MAX_ENTRY_KB", "THEME_MAX_CLIP_KB", "THEME_FILE_CACHE_SECONDS", "STEP_UP_MAX_AGE_MINUTES"}
    surface = set(json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"])
    assert names <= surface
    for name in names:
        assert hasattr(config, name), name


def test_the_config_defaults_are_gerards_d1_values():
    defaults = {"THEME_MAX_CODE_KB": 3072, "THEME_MAX_ENTRY_KB": 300, "THEME_MAX_CLIP_KB": 1024,
                "STEP_UP_MAX_AGE_MINUTES": 10}
    source = (ORCH / "config.py").read_text(encoding="utf-8")
    for name, value in defaults.items():
        assert f'os.getenv("{name}", "{value}")' in source, name
    assert 'os.getenv("THEMES_PRIVATE_UPLOAD", "auto")' in source


# ---------------------------------------------------------------------------
# The gates on the routers themselves
# ---------------------------------------------------------------------------


def test_step_up_is_on_exactly_the_marked_theme_routes_after_the_gate():
    marked = set()
    for key, route in _theme_routes().items():
        calls = _calls(route.dependant)
        at = _index(calls, _is_step_up)
        if at is None:
            continue
        marked.add(key)
        assert getattr(calls[at], step_up.STEP_UP_MARKER_ATTR) == config.STEP_UP_MAX_AGE_MINUTES
        assert _index(calls, _is_gate) < at, f"{key}: the permission gate must run before step-up"
    assert marked == STEP_UP_ROUTES


def test_no_team_route_carries_step_up_until_w_a2():
    # The role change's only client cannot show Clerk's reverification prompt yet (W-A2, #1157).
    marked = set()
    for route in team_api.router.routes:
        for method in route.methods - {"HEAD", "OPTIONS"}:
            if _index(_calls(route.dependant), _is_step_up) is not None:
                marked.add((method, route.path))
    assert marked == set()


def test_who_may_install_gates_import_update_and_rollback():
    gated = {key for key, route in _theme_routes().items() if _index(_calls(route.dependant), _is_install_role) is not None}
    assert gated == INSTALL_ROLE_ROUTES


def test_every_mutating_theme_route_is_workspace_gated_and_files_and_boot_are_not():
    for key, route in _theme_routes().items():
        gated = _index(_calls(route.dependant), _is_gate) is not None
        if key in MEMBERSHIP_ROUTES:
            assert not gated, f"{key} authorises by membership, never by ctx.workspace_id"
        elif key[0] in MUTATING:
            assert gated, f"{key} lands outside authZ-sweep bucket (a)"


def test_the_files_and_boot_code_never_reads_ctx_workspace_id():
    for relative in ("api/themes_files.py", "services/themes/membership.py", "services/themes/serve.py"):
        tree = ast.parse((ORCH / relative).read_text(encoding="utf-8"))
        reads = [node.lineno for node in ast.walk(tree)
                 if isinstance(node, ast.Attribute) and node.attr == "workspace_id"
                 and isinstance(node.value, ast.Name) and node.value.id == "ctx"]
        assert reads == [], f"{relative} reads ctx.workspace_id at {reads}"


# ---------------------------------------------------------------------------
# The live app (a clean subprocess; no database is dialled on import)
# ---------------------------------------------------------------------------

_PROBE = """
import asyncio, collections, json
from fastapi import Response
from fastapi.routing import APIRoute, iter_route_contexts
from api import themes_files
from core.auth import step_up
from core.auth.workspace_permission import PERMISSION_MARKER_ATTR
import main

def calls(dependant):
    out = []
    for sub in dependant.dependencies:
        out += [sub.call] + calls(sub)
    return out

def csp_after_app_headers(route_headers):
    async def route(_request):
        return Response(b"x", headers=route_headers)
    response = asyncio.run(main.add_security_headers(None, route))
    return response.headers.get("content-security-policy")

served = collections.Counter()
step_up_routes, gated = [], []
# FastAPI 0.137+ keeps an included router as one node of app.routes (PRD-262 US-008); the
# walk gives each served route with its full path and its inherited dependencies.
for route in iter_route_contexts(main.app.routes):
    if not isinstance(route.original_route, APIRoute):
        continue
    found = calls(route.dependant)
    for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
        served[method + " " + route.path] += 1
        if any(getattr(c, step_up.STEP_UP_MARKER_ATTR, None) for c in found):
            step_up_routes.append(method + " " + route.path)
        if any(getattr(c, PERMISSION_MARKER_ATTR, None) for c in found):
            gated.append(method + " " + route.path)
print(json.dumps({
    "served": dict(served), "step_up": sorted(step_up_routes), "gated": sorted(gated),
    "handler": step_up.StepUpRequired in main.app.exception_handlers,
    "import_limit": main._body_limit_for("/api/themes/import"),
    "file_csp": csp_after_app_headers(themes_files.SERVING_HEADERS),
    "default_csp": csp_after_app_headers({}),
    "sandbox_only_csp": csp_after_app_headers({"Content-Security-Policy": "sandbox"}),
}))
"""


@pytest.fixture(scope="module")
def live_app():
    env = dict(os.environ)
    env.update({
        "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test", "POSTGRES_HOST": "127.0.0.1",
        "POSTGRES_PORT": "59432", "POSTGRES_DB": "test",
        "DATABASE_URL": "postgresql://test:test@127.0.0.1:59432/test",
    })
    proc = subprocess.run([sys.executable, "-c", _PROBE], cwd=str(ORCH), env=env,
                          capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, f"probe failed (rc={proc.returncode})\nSTDERR:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _key(method_path):
    method, path = method_path.split(" ", 1)
    return method, path


def test_the_app_serves_every_w_b2_route_exactly_once(live_app):
    served = {_key(name): count for name, count in live_app["served"].items()}
    for key in _theme_routes():
        assert served.get(key) == 1, f"{key} served {served.get(key, 0)} times"


def test_the_app_carries_step_up_on_exactly_the_planned_routes(live_app):
    # The /api/themes routes only: W-C1's submit and listing install are pinned, with the
    # app-wide set, in test_prd261_c1_wire.py.
    marked = {_key(name) for name in live_app["step_up"]}
    assert {key for key in marked if key[1].startswith("/api/themes/")} == STEP_UP_ROUTES
    assert live_app["handler"] is True


def test_the_app_classifies_every_w_b2_mutating_route_in_bucket_a(live_app):
    gated = {_key(name) for name in live_app["gated"]}
    mutating = {key for key in _theme_routes() if key[0] in MUTATING}
    assert mutating <= gated
    assert ROLE_CHANGE in gated


def test_the_import_body_limit_is_sized_from_the_package_cap(live_app):
    assert live_app["import_limit"] == config.THEME_MAX_PACKAGE_MB * MB + MULTIPART_ENVELOPE


def test_the_app_keeps_the_theme_files_sandbox_csp(live_app):
    # The app's security-headers middleware runs after every route; it must keep the
    # files route's ``sandbox`` (a theme file opened as a page runs nothing) and still
    # refuse framing.
    directives = [part.strip() for part in live_app["file_csp"].split(";")]
    assert "sandbox" in directives
    assert "default-src 'none'" in directives
    assert "frame-ancestors 'none'" in directives


def test_the_app_default_csp_is_unchanged_for_routes_without_one(live_app):
    assert live_app["default_csp"] == "default-src 'none'; frame-ancestors 'none'"
    # A route policy without default-src (the widget attachment's bare ``sandbox``) keeps the
    # app's default-src 'none', so an inline-rendered file still loads nothing remote.
    assert live_app["sandbox_only_csp"] == "default-src 'none'; sandbox; frame-ancestors 'none'"
