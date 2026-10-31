"""PRD-270 W2 (K5, K7, K12, K22) — the wave's routes and settings, wired once.

Pins:

* the hosted router (``api.marketplace_catalog_items``: a bundle by its revision and a plugin's package) and the
  local reads router (``api.marketplace_cloud_items``: the Cloud items, a disclosure and the updates, US-270-14a)
  are ``RouterSpec`` entries in ``router_manifest.py``, and ``main.py`` names neither: the wave leaves it unchanged;
* they declare exactly the five W2 routes, every one a GET (the wave adds no mutating route, K22), each in
  ``reports/route-manifest.json`` as a GET, and ``route_count == len(routes)``;
* the live app (a probe in a clean subprocess with an unreachable Postgres, the ``test_prd261_c3_wire.py``
  precedent, as ``test_prd270w1_wire.py``) serves each W2 route once, each answered by its own handler, none
  shadowed by an earlier route (W1's catalogue files route ``/files/{digest}/{path:path}`` included);
* neither hosted route's dependant tree holds ``get_request_context_hybrid``, on the router and on the live app,
  and neither carries a workspace permission: nothing they answer depends on a caller;
* each local read's first two dependencies are the local-edition check, then ``get_request_context_hybrid``
  (404 on SaaS before the caller is looked at; then any member, K18), on the router and on the live app;
* the wave's three settings are on ``config`` and named in ``reports/config-surface.json``.
"""
from __future__ import annotations

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

import api.marketplace_catalog_items as catalog_items_api  # noqa: E402
import api.marketplace_cloud_items as cloud_items_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
from config import config  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from router_manifest import MANIFEST_ROUTERS, RouterSpec  # noqa: E402

W2_MODULES = {"api.marketplace_catalog_items": catalog_items_api, "api.marketplace_cloud_items": cloud_items_api}
BUNDLE = "/api/marketplace/catalog/bundles/{revision}"
PLUGIN_PACKAGE = "/api/marketplace/catalog/plugins/{slug}/{version}/package.zip"
HOSTED_ROUTES = {("GET", BUNDLE), ("GET", PLUGIN_PACKAGE)}
CLOUD_ITEMS = "/api/marketplace/cloud-sync/items"
DISCLOSURE = "/api/marketplace/cloud-sync/items/{kind}/{key}/disclosure"
UPDATES = "/api/marketplace/cloud-sync/updates"
LOCAL_ROUTES = {("GET", CLOUD_ITEMS), ("GET", DISCLOSURE), ("GET", UPDATES)}
W2_ROUTES = HOSTED_ROUTES | LOCAL_ROUTES
SAMPLES = {
    ("GET", BUNDLE): "/api/marketplace/catalog/bundles/" + "ab" * 32,
    ("GET", PLUGIN_PACKAGE): "/api/marketplace/catalog/plugins/test-research-kit/1.0.0/package.zip",
    ("GET", CLOUD_ITEMS): CLOUD_ITEMS,
    ("GET", DISCLOSURE): "/api/marketplace/cloud-sync/items/agent/12/disclosure",
    ("GET", UPDATES): UPDATES,
}
LOCAL_CHECK = f"{cloud_items_api.require_local_edition.__module__}.{cloud_items_api.require_local_edition.__qualname__}"
SETTINGS = ("AUTOMATOS_CLOUD_BUNDLE_MAX_KB", "AUTOMATOS_CLOUD_PLUGINS_PER_SYNC", "MARKETPLACE_CATALOG_BUNDLE_MAX_KB")
HYBRID = f"{get_request_context_hybrid.__module__}.{get_request_context_hybrid.__qualname__}"


def _calls(dependant):
    """Every dependency callable of a route, in resolution order (depth first)."""
    found = []
    for sub in dependant.dependencies:
        found += [sub.call] + _calls(sub)
    return found


def _w2_routes():
    """``{(method, path): APIRoute}`` over the W2 routers."""
    table = {}
    for module in W2_MODULES.values():
        for route in module.router.routes:
            assert isinstance(route, APIRoute)
            for method in route.methods - {"HEAD", "OPTIONS"}:
                table[(method, route.path)] = route
    return table


# --------------------------------------------------------------------------- the routers, the manifests, main.py


def test_the_w2_routers_are_router_spec_entries_and_main_py_names_none_of_them():
    main_source = (ORCH / "main.py").read_text(encoding="utf-8")
    for name in W2_MODULES:
        assert RouterSpec(name) in MANIFEST_ROUTERS, name
        assert name.split(".")[-1] not in main_source, f"main.py must not mount {name}: RouterSpec does"


def test_the_routers_declare_exactly_the_w2_routes_every_one_a_get():
    routes = _w2_routes()
    assert set(routes) == W2_ROUTES
    assert {method for method, _path in routes} == {"GET"}


def test_every_w2_route_is_in_the_manifest_as_a_get_and_the_count_holds():
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    assert manifest["route_count"] == len(manifest["routes"])
    served = {(route["method"], route["path"]) for route in manifest["routes"]}
    assert not W2_ROUTES - served, sorted(W2_ROUTES - served)
    paths = {path for _method, path in W2_ROUTES}
    assert {method for method, path in served if path in paths} == {"GET"}


def test_the_three_settings_are_on_config_and_in_the_config_surface():
    surface = set(json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"])
    for name in SETTINGS:
        assert hasattr(config, name), name
        assert name in surface, name


def test_no_hosted_route_depends_on_the_caller_or_a_workspace_permission():
    for key in HOSTED_ROUTES:
        calls = _calls(_w2_routes()[key].dependant)
        assert get_request_context_hybrid not in calls, f"{key} must not read a caller"
        assert not [call for call in calls if getattr(call, permission_mod.PERMISSION_MARKER_ATTR, None)], key


def test_every_local_read_checks_the_edition_first_then_reads_the_caller():
    for key in LOCAL_ROUTES:
        calls = [sub.call for sub in _w2_routes()[key].dependant.dependencies]
        assert calls[:2] == [cloud_items_api.require_local_edition, get_request_context_hybrid], key


# --------------------------------------------------------------------------- the live app (a clean subprocess)

_PROBE = """
import collections, json
from fastapi.routing import APIRoute, iter_route_contexts
from core.auth.workspace_permission import PERMISSION_MARKER_ATTR
import main

SAMPLES = %r

def calls(dependant):
    out = []
    for sub in dependant.dependencies:
        out += [sub.call] + calls(sub)
    return out

def name(call):
    return getattr(call, "__module__", "") + "." + getattr(call, "__qualname__", repr(call))

served_routes = [r for r in iter_route_contexts(main.app.routes) if isinstance(r.original_route, APIRoute)]

def first_handler(method, path):
    for route in served_routes:
        if method in route.methods and route.path_regex.match(path):
            return route.endpoint.__module__ + "." + route.endpoint.__name__ + " " + route.path
    return None

served = collections.Counter()
dependencies, perms = {}, {}
for route in served_routes:
    for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
        key = method + " " + route.path
        served[key] += 1
        if key in SAMPLES:
            found = calls(route.dependant)
            dependencies[key] = [name(call) for call in found]
            perms[key] = [getattr(call, PERMISSION_MARKER_ATTR) for call in found
                          if getattr(call, PERMISSION_MARKER_ATTR, None)]
answered = {key: first_handler(key.split(" ", 1)[0], path) for key, path in SAMPLES.items()}
print(json.dumps({"served": dict(served), "answered": answered, "dependencies": dependencies, "perms": perms}))
""" % ({f"{method} {path}": sample for (method, path), sample in SAMPLES.items()},)


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


def test_the_app_serves_every_w2_route_exactly_once(live_app):
    served = {_key(name): count for name, count in live_app["served"].items()}
    for key in W2_ROUTES:
        assert served.get(key) == 1, f"{key} served {served.get(key, 0)} times"


def test_no_earlier_route_shadows_a_w2_route(live_app):
    modules = {key: route.endpoint.__module__ for key, route in _w2_routes().items()}
    assert {_key(name) for name in live_app["answered"]} == W2_ROUTES
    for name, answer in live_app["answered"].items():
        key = _key(name)
        assert answer is not None, f"{key}: nothing answers it"
        handler, path = answer.split(" ", 1)
        assert handler.startswith(modules[key] + ".") and path == key[1], f"{key} is answered by {answer}"


def test_on_the_live_app_no_hosted_route_reads_a_caller_or_a_workspace_permission(live_app):
    for name, found in live_app["dependencies"].items():
        if _key(name) in HOSTED_ROUTES:
            assert HYBRID not in found, f"{name} must not read a caller"
            assert live_app["perms"][name] == [], name


def test_on_the_live_app_every_local_read_checks_the_edition_before_the_caller(live_app):
    for name, found in live_app["dependencies"].items():
        if _key(name) in LOCAL_ROUTES:
            assert found[:2] == [LOCAL_CHECK, HYBRID], f"{name}: {found[:2]}"
