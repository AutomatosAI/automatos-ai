"""PRD-261 C3-wire: the W-C3 revocation routers, the pull's scheduler and its settings, wired once.

Pins (plan §2.4 W-C3 row, §2.6 C3-wire):

* ``main.py`` mounts each of the three W-C3 routers exactly once, and the live app serves every
  W-C3 route once, each answered by its own handler (no earlier route shadows one, the admin review
  queue's ``/{package_id}/...`` routes included; a probe in a clean subprocess with an unreachable
  Postgres, the ``test_prd261_c1_wire.py`` precedent);
* every W-C3 route is in ``reports/route-manifest.json`` and ``route_count == len(routes)``;
* create and lift carry a fresh step-up at ``STEP_UP_MAX_AGE_MINUTES``, after the admin check, and
  no other W-C3 route does (the app-wide set is pinned in ``test_prd261_c1_wire.py``);
* the admin routes call ``_require_admin(ctx)`` first and carry no workspace gate; create, lift and
  Check now are in the authZ sweep's ``ADMIN_GATED_IN_HANDLER``; the feed is a public GET;
* the four settings are on ``config`` and in ``reports/config-surface.json``, with the defaults the
  modules fall back to, and none names a key; ``envs/api.defaults`` names the feed URL, no secret;
* the shared scheduler registers the pull when it starts (local edition, a feed URL), registers
  nothing in SaaS, and a failing registration never stops the scheduler.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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

import api.admin_theme_revocation_sync as sync_api  # noqa: E402
import api.admin_theme_revocations as revocations_api  # noqa: E402
import api.themes_revocations_feed as feed_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import services.scheduler as scheduler_mod  # noqa: E402
import services.theme_revocation_scheduler as pull_scheduler  # noqa: E402
from config import config  # noqa: E402
from core.auth import step_up  # noqa: E402
from services.themes import revocation_feed, revocation_pull  # noqa: E402
from tests.test_p2w2_authz_boundary_sweep import ADMIN_GATED_IN_HANDLER  # noqa: E402

C3_MODULES = {
    "theme_revocations_feed_router": feed_api,
    "admin_theme_revocation_sync_router": sync_api,
    "admin_theme_revocations_router": revocations_api,
}
REVOCATIONS = "/api/admin/themes/revocations"
SYNC = REVOCATIONS + "/sync"
FEED = "/api/themes/revocations"
C3_ROUTES = {
    ("GET", FEED),
    ("GET", REVOCATIONS), ("POST", REVOCATIONS), ("POST", REVOCATIONS + "/{revocation_id}/lift"),
    ("GET", SYNC), ("POST", SYNC + "/pull"),
}
C3_STEP_UP_ROUTES = {("POST", REVOCATIONS), ("POST", REVOCATIONS + "/{revocation_id}/lift")}
C3_CONFIG_DEFAULTS = {
    "THEME_REVOCATIONS_URL": revocation_pull.DEFAULT_FEED_URL,
    "THEME_REVOCATIONS_PULL_MINUTES": pull_scheduler.DEFAULT_PULL_MINUTES,
    "THEME_REVOCATIONS_LIFTED_DAYS": revocation_feed.DEFAULT_LIFTED_DAYS,
    "THEME_REVOCATIONS_FEED_MAX_KB": revocation_feed.DEFAULT_FEED_MAX_KB,
}
SETTING_NAMES = {
    revocation_pull.URL_SETTING, pull_scheduler.PULL_MINUTES_SETTING,
    revocation_feed.LIFTED_DAYS_SETTING, revocation_feed.FEED_MAX_KB_SETTING,
}
MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
KEY_NAME = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD)$|SIGN")
SAMPLE_ID = "00000000-0000-4000-8000-000000000000"
FEED_URL = "https://feed.example.test/api/themes/revocations"


def _calls(dependant):
    """Every dependency callable of a route, in resolution order (depth first)."""
    found = []
    for sub in dependant.dependencies:
        found += [sub.call] + _calls(sub)
    return found


def _c3_routes():
    """``{(method, path): APIRoute}`` over the three W-C3 routers."""
    table = {}
    for module in C3_MODULES.values():
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


# ---------------------------------------------------------------------------
# main.py, the manifest, config and the local defaults
# ---------------------------------------------------------------------------


def test_main_mounts_each_w_c3_router_exactly_once():
    tree = ast.parse((ORCH / "main.py").read_text(encoding="utf-8"))
    mounted = [
        node.args[0].id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "include_router"
        and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in C3_MODULES
    ]
    assert sorted(mounted) == sorted(C3_MODULES)


def test_the_routers_declare_exactly_the_planned_routes():
    assert set(_c3_routes()) == C3_ROUTES


def test_every_w_c3_route_is_in_the_manifest_and_the_count_holds():
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    assert manifest["route_count"] == len(manifest["routes"])
    served = {(r["method"], r["path"]) for r in manifest["routes"]}
    assert not C3_ROUTES - served, sorted(C3_ROUTES - served)


def test_every_revocation_setting_is_on_config_with_the_modules_default_and_in_the_surface():
    assert set(C3_CONFIG_DEFAULTS) == SETTING_NAMES  # the names the modules read, exactly
    surface = set(json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"])
    assert set(C3_CONFIG_DEFAULTS) <= surface
    source = (ORCH / "config.py").read_text(encoding="utf-8")
    for name, default in C3_CONFIG_DEFAULTS.items():
        assert hasattr(config, name), name
        assert f'os.getenv("{name}", "{default}")' in source, name


def test_no_revocation_setting_names_a_key():
    # QB-26: the feed is unsigned, so the pull needs no key; it sends no credentials either.
    assert [name for name in C3_CONFIG_DEFAULTS if KEY_NAME.search(name)] == []


def test_the_local_defaults_name_the_feed_url_and_no_secret():
    text = (ORCH.parent / "envs" / "api.defaults").read_text(encoding="utf-8")
    (url,) = re.findall(r"^THEME_REVOCATIONS_URL=(.*)$", text, re.M)
    assert url == revocation_pull.DEFAULT_FEED_URL
    assert "@" not in url and "?" not in url  # no userinfo, no query string: nothing to keep secret


# ---------------------------------------------------------------------------
# The gates on the routers themselves
# ---------------------------------------------------------------------------


def test_step_up_is_on_create_and_lift_only_after_the_admin_check():
    marked = set()
    for key, route in _c3_routes().items():
        calls = _calls(route.dependant)
        at = _index(calls, _is_step_up)
        if at is None:
            continue
        marked.add(key)
        assert getattr(calls[at], step_up.STEP_UP_MARKER_ATTR) == config.STEP_UP_MAX_AGE_MINUTES
        first = calls.index(revocations_api.require_revocation_admin)
        assert first < at, f"{key}: the admin check must run before step-up"
    assert marked == C3_STEP_UP_ROUTES


def test_admin_routes_check_caller_is_admin_first_and_mutating_ones_are_in_the_sweep_bucket():
    for key, route in _c3_routes().items():
        if not key[1].startswith("/api/admin/"):
            continue
        assert _index(_calls(route.dependant), _is_gate) is None, f"{key} is platform-wide, not workspace-gated"
        body = inspect.getsource(route.endpoint).split('"""')[-1]
        first = next(line.strip() for line in body.splitlines() if line.strip() and not line.strip().startswith(")"))
        assert first == "_require_admin(ctx)", f"{key} must check caller_is_admin first, found {first!r}"
        assert (key in ADMIN_GATED_IN_HANDLER) == (key[0] in MUTATING), key


def test_the_feed_is_a_public_read():
    (route,) = [route for key, route in _c3_routes().items() if key == ("GET", FEED)]
    names = {getattr(call, "__name__", "") for call in _calls(route.dependant)}
    assert names == {"get_db"}, names  # no request context, no workspace: nothing depends on the caller
    assert not [key for key in _c3_routes() if key[1] == FEED and key[0] in MUTATING]


# ---------------------------------------------------------------------------
# The shared scheduler registers the pull
# ---------------------------------------------------------------------------


class _FakeAPScheduler:
    """Stands in for AsyncIOScheduler: records every job, starts nothing."""

    def __init__(self, **kwargs):
        self.jobs = {}
        self.running = False

    def start(self):
        self.running = True

    def add_job(self, func, trigger=None, **kwargs):
        self.jobs[kwargs["id"]] = SimpleNamespace(func=func, trigger=trigger, **kwargs)


def _start_unified(monkeypatch):
    monkeypatch.setattr(scheduler_mod, "AsyncIOScheduler", _FakeAPScheduler)
    unified = scheduler_mod.UnifiedScheduler()
    unified.start()
    return unified.apscheduler.jobs


def test_the_shared_scheduler_registers_the_pull_in_the_local_edition(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, revocation_pull.URL_SETTING, FEED_URL)
    job = _start_unified(monkeypatch)[pull_scheduler.JOB_ID]
    assert job.func is pull_scheduler.run_scheduled_pull and job.max_instances == 1


@pytest.mark.parametrize("edition, url", [("saas", FEED_URL), ("local", "")])
def test_the_shared_scheduler_registers_no_pull_in_saas_or_with_no_feed_url(monkeypatch, edition, url):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(config, revocation_pull.URL_SETTING, url)
    assert pull_scheduler.JOB_ID not in _start_unified(monkeypatch)


def test_a_failing_pull_registration_never_stops_the_shared_scheduler(monkeypatch, caplog):
    def broken(self, scheduler):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(pull_scheduler.ThemeRevocationScheduler, "start", broken)
    monkeypatch.setattr(scheduler_mod, "AsyncIOScheduler", _FakeAPScheduler)
    unified = scheduler_mod.UnifiedScheduler()
    with caplog.at_level("ERROR", logger=scheduler_mod.logger.name):
        unified.start()
    assert unified.apscheduler.running is True
    assert any("ThemeRevocationScheduler" in record.getMessage() and record.exc_info for record in caplog.records)


# ---------------------------------------------------------------------------
# The live app (a clean subprocess; no database is dialled on import)
# ---------------------------------------------------------------------------

_PROBE = """
import collections, json
from fastapi.routing import APIRoute, iter_route_contexts
from core.auth import step_up
import main

SAMPLE_ID = %r
SAMPLES = {
    "GET /api/themes/revocations": "/api/themes/revocations",
    "GET /api/admin/themes/revocations": "/api/admin/themes/revocations",
    "POST /api/admin/themes/revocations": "/api/admin/themes/revocations",
    "POST /api/admin/themes/revocations/{revocation_id}/lift": "/api/admin/themes/revocations/" + SAMPLE_ID + "/lift",
    "GET /api/admin/themes/revocations/sync": "/api/admin/themes/revocations/sync",
    "POST /api/admin/themes/revocations/sync/pull": "/api/admin/themes/revocations/sync/pull",
}

def calls(dependant):
    out = []
    for sub in dependant.dependencies:
        out += [sub.call] + calls(sub)
    return out

served_routes = list(iter_route_contexts(main.app.routes))

def first_handler(method, path):
    for route in served_routes:
        if isinstance(route.original_route, APIRoute) and method in route.methods and route.path_regex.match(path):
            return route.endpoint.__module__ + "." + route.endpoint.__name__ + " " + route.path
    return None

served = collections.Counter()
step_up_routes = []
for route in served_routes:
    if not isinstance(route.original_route, APIRoute):
        continue
    marked = any(getattr(c, step_up.STEP_UP_MARKER_ATTR, None) for c in calls(route.dependant))
    for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
        served[method + " " + route.path] += 1
        if marked:
            step_up_routes.append(method + " " + route.path)
answered = {key: first_handler(key.split(" ", 1)[0], path) for key, path in SAMPLES.items()}
print(json.dumps({"served": dict(served), "step_up": sorted(step_up_routes), "answered": answered}))
""" % (SAMPLE_ID,)


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


def test_the_app_serves_every_w_c3_route_exactly_once(live_app):
    served = {_key(name): count for name, count in live_app["served"].items()}
    for key in C3_ROUTES:
        assert served.get(key) == 1, f"{key} served {served.get(key, 0)} times"


def test_no_earlier_route_shadows_a_w_c3_route(live_app):
    modules = {key: route.endpoint.__module__ for key, route in _c3_routes().items()}
    assert {_key(name) for name in live_app["answered"]} == C3_ROUTES
    for name, answer in live_app["answered"].items():
        key = _key(name)
        assert answer is not None, f"{key}: nothing answers it"
        handler, path = answer.split(" ", 1)
        assert handler.startswith(modules[key] + ".") and path == key[1], f"{key} is answered by {answer}"


def test_the_app_carries_step_up_on_create_and_lift_among_the_w_c3_routes(live_app):
    marked = {_key(name) for name in live_app["step_up"]}
    assert marked & C3_ROUTES == C3_STEP_UP_ROUTES
