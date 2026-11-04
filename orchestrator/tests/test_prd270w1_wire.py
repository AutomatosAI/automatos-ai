"""PRD-270 W1 (K5, K7, K11, K12) — the wave's routers and settings, wired once.

Pins:

* the three W1 routers (the hosted catalogue and its files, the public reads, the Sync with Automatos
  Cloud routes) are ``RouterSpec`` entries in ``router_manifest.py``, and ``main.py`` names none of
  them: it is unchanged by the wave;
* together they declare exactly the five W1 routes and the one-page Marketplace's two public reads
  (Featured and counts, on the public reads' router), each in ``reports/route-manifest.json``, and
  ``route_count == len(routes)``;
* the live app (a probe in a clean subprocess with an unreachable Postgres, the
  ``test_prd261_c3_wire.py`` precedent) serves each W1 route once, each answered by its own handler,
  none shadowed by an earlier route;
* no W1 public route's dependant tree holds ``get_request_context_hybrid``, on the routers and on the
  live app: nothing in their answers depends on a caller;
* the two Sync routes check the edition first and then ``workspace:manage``, on the live app too (the
  POST is the authZ sweep's class (a) by that gate);
* the five settings are on ``config`` and named in ``reports/config-surface.json``.
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

import api.marketplace_catalog as catalog_api  # noqa: E402
import api.marketplace_cloud_sync as sync_api  # noqa: E402
import api.public_marketplace as public_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
from config import config  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from router_manifest import MANIFEST_ROUTERS, RouterSpec  # noqa: E402

W1_MODULES = {
    "api.marketplace_catalog": catalog_api,
    "api.public_marketplace": public_api,
    "api.marketplace_cloud_sync": sync_api,
}
CATALOG = "/api/marketplace/catalog"
FILES = CATALOG + "/files/{digest}/{path:path}"
PUBLIC_READS = "/api/public/marketplace/{type}"
PUBLIC_FEATURED = "/api/public/marketplace/featured"
PUBLIC_COUNTS = "/api/public/marketplace/counts"
SYNC = "/api/marketplace/cloud-sync"
SYNC_STATUS = SYNC + "/status"
PUBLIC_ROUTES = {
    ("GET", CATALOG), ("GET", FILES), ("GET", PUBLIC_READS), ("GET", PUBLIC_FEATURED), ("GET", PUBLIC_COUNTS),
}
SYNC_ROUTES = {("GET", SYNC_STATUS), ("POST", SYNC)}
W1_ROUTES = PUBLIC_ROUTES | SYNC_ROUTES
SAMPLE_DIGEST = "ab" * 32
SAMPLES = {
    ("GET", CATALOG): CATALOG,
    ("GET", FILES): f"{CATALOG}/files/{SAMPLE_DIGEST}/package.autotheme",
    ("GET", PUBLIC_READS): "/api/public/marketplace/agents",
    ("GET", PUBLIC_FEATURED): PUBLIC_FEATURED,
    ("GET", PUBLIC_COUNTS): PUBLIC_COUNTS,
    ("GET", SYNC_STATUS): SYNC_STATUS,
    ("POST", SYNC): SYNC,
}
SETTINGS = (
    "AUTOMATOS_CLOUD_URL", "AUTOMATOS_CLOUD_FEED_MAX_KB", "AUTOMATOS_CLOUD_TIMEOUT_SECONDS",
    "MARKETPLACE_CATALOG_FEED_MAX_KB", "MARKETPLACE_PUBLIC_RATE_PER_MINUTE",
)
HYBRID = f"{get_request_context_hybrid.__module__}.{get_request_context_hybrid.__qualname__}"
EDITION_CHECK = f"{sync_api.require_local_edition.__module__}.{sync_api.require_local_edition.__qualname__}"


def _calls(dependant):
    """Every dependency callable of a route, in resolution order (depth first)."""
    found = []
    for sub in dependant.dependencies:
        found += [sub.call] + _calls(sub)
    return found


def _w1_routes():
    """``{(method, path): APIRoute}`` over the three W1 routers."""
    table = {}
    for module in W1_MODULES.values():
        for route in module.router.routes:
            assert isinstance(route, APIRoute)
            for method in route.methods - {"HEAD", "OPTIONS"}:
                table[(method, route.path)] = route
    return table


# --------------------------------------------------------------------------- the routers, the manifests, main.py


def test_the_three_routers_are_router_spec_entries_and_main_py_names_none_of_them():
    for name in W1_MODULES:
        assert RouterSpec(name) in MANIFEST_ROUTERS, name
    main_source = (ORCH / "main.py").read_text(encoding="utf-8")
    for name in W1_MODULES:
        assert name.split(".")[-1] not in main_source, f"main.py must not mount {name}: RouterSpec does"


def test_the_routers_declare_exactly_the_w1_routes():
    assert set(_w1_routes()) == W1_ROUTES


def test_every_w1_route_is_in_the_manifest_and_the_count_holds():
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    assert manifest["route_count"] == len(manifest["routes"])
    served = {(r["method"], r["path"]) for r in manifest["routes"]}
    assert not W1_ROUTES - served, sorted(W1_ROUTES - served)


def test_the_five_settings_are_on_config_and_in_the_config_surface():
    surface = set(json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"])
    for name in SETTINGS:
        assert hasattr(config, name), name
        assert name in surface, name


def test_no_w1_public_route_depends_on_the_caller():
    for key, route in _w1_routes().items():
        if key in PUBLIC_ROUTES:
            assert get_request_context_hybrid not in _calls(route.dependant), f"{key} must not read a caller"


def test_the_sync_routes_check_the_edition_first_then_workspace_manage():
    for key in SYNC_ROUTES:
        calls = _calls(_w1_routes()[key].dependant)
        assert calls[0] is sync_api.require_local_edition, key
        perms = [getattr(call, permission_mod.PERMISSION_MARKER_ATTR, None) for call in calls]
        assert [perm for perm in perms if perm] == ["workspace:manage"], key


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


def test_the_app_serves_every_w1_route_exactly_once(live_app):
    served = {_key(name): count for name, count in live_app["served"].items()}
    for key in W1_ROUTES:
        assert served.get(key) == 1, f"{key} served {served.get(key, 0)} times"


def test_no_earlier_route_shadows_a_w1_route(live_app):
    modules = {key: route.endpoint.__module__ for key, route in _w1_routes().items()}
    assert {_key(name) for name in live_app["answered"]} == W1_ROUTES
    for name, answer in live_app["answered"].items():
        key = _key(name)
        assert answer is not None, f"{key}: nothing answers it"
        handler, path = answer.split(" ", 1)
        assert handler.startswith(modules[key] + ".") and path == key[1], f"{key} is answered by {answer}"


def test_on_the_live_app_no_w1_public_route_reads_a_caller_and_the_sync_routes_are_gated(live_app):
    for name, found in live_app["dependencies"].items():
        key = _key(name)
        if key in PUBLIC_ROUTES:
            assert HYBRID not in found, f"{key} must not read a caller"
            assert live_app["perms"][name] == [], key
        else:
            assert found[0] == EDITION_CHECK, f"{key}: the edition is checked first"
            assert live_app["perms"][name] == ["workspace:manage"], key
