"""PRD-261 C1-wire: the W-C1 Marketplace routers, wired once into the app.

Pins (plan §2.4 W-C1 row, §2.6 C1-wire; Gerard D3, 19 Oct):

* ``main.py`` mounts each of the three W-C1 routers exactly once, and the live app serves every
  W-C1 route once, each answered by its own handler (no earlier route shadows one; a probe in a
  clean subprocess with an unreachable Postgres, the ``test_prd261_b2_wire.py`` precedent);
* every W-C1 route is in ``reports/route-manifest.json`` and ``route_count == len(routes)``;
* the submit route's body limit is sized from ``THEME_MAX_PACKAGE_MB``;
* step-up (A-07c) is on submit and listing install, after the permission gate, and nowhere else
  among the W-C1 routes; app-wide it is on exactly the W-B2 routes and these two;
* the admin review routes carry no workspace gate, call ``_require_admin(ctx)`` first, and every
  mutating one is in the authZ sweep's ``ADMIN_GATED_IN_HANDLER``; every other W-C1 mutating route
  carries ``require_workspace_permission`` (bucket a);
* the review config names are on ``config`` and in ``reports/config-surface.json``, with the
  defaults the lanes read (Q9 off; the LLM chunking of ``llm_stage.py``), and none names a key;
* the seam: approving a version offers it to the listing installs that run an older one
  (``listing_updates.offer_listed_update`` called from ``review.approve``).

Submit's default review stage (C-02a, wired here) is proven on the route in
``test_prd261_theme_submit.py``; ``sdk-surface.json`` against ``frontend/`` in
``test_prd261_sdk_surface_drift.py``.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import uuid
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

import sqlalchemy as sa  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.admin_themes as admin_api  # noqa: E402
import api.marketplace_themes as listings_api  # noqa: E402
import api.marketplace_themes_submit as submit_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.models  # noqa: E402,F401  (registers every table the FKs name)
from config import config  # noqa: E402
from core.auth import step_up  # noqa: E402
from core.models.themes import ThemeInstallation, ThemePackage, ThemeReviewEvent, ThemeRevocation  # noqa: E402
from services.themes import llm_stage, review, submissions  # noqa: E402
from services.themes.digest import package_digest  # noqa: E402
from tests.test_p2w2_authz_boundary_sweep import ADMIN_GATED_IN_HANDLER  # noqa: E402
from tests.test_prd261_b2_wire import STEP_UP_ROUTES as B2_STEP_UP_ROUTES  # noqa: E402

C1_MODULES = {
    "marketplace_themes_submit_router": submit_api, "marketplace_themes_router": listings_api,
    "admin_themes_router": admin_api,
}
ADMIN = "/api/admin/themes/{package_id}"
LISTING = "/api/marketplace/themes/{listing_id}"
C1_ROUTES = {
    ("POST", "/api/marketplace/themes/submit"),
    ("GET", "/api/marketplace/themes"), ("GET", LISTING), ("POST", LISTING + "/install"),
    ("GET", "/api/admin/themes/pending"), ("GET", ADMIN + "/report"), ("POST", ADMIN + "/report-note"),
    ("POST", ADMIN + "/approve"), ("POST", ADMIN + "/reject"), ("POST", ADMIN + "/deactivate"),
    ("POST", ADMIN + "/signature"),
}
C1_STEP_UP_ROUTES = {("POST", "/api/marketplace/themes/submit"), ("POST", LISTING + "/install")}
C1_CONFIG_DEFAULTS = {
    "THEMES_ONE_OPEN_SUBMISSION": False,
    "THEME_LLM_CHUNK_TOKENS": llm_stage.DEFAULT_CHUNK_TOKENS,
    "THEME_LLM_MAX_CHUNKS": llm_stage.DEFAULT_MAX_CHUNKS,
    "THEME_LLM_TIMEOUT_SECONDS": llm_stage.DEFAULT_TIMEOUT_SECONDS,
}
MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
MB, MULTIPART_ENVELOPE = 1024 * 1024, 512 * 1024
KEY_NAME = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD)$|SIGN")


def _calls(dependant):
    """Every dependency callable of a route, in resolution order (depth first)."""
    found = []
    for sub in dependant.dependencies:
        found += [sub.call] + _calls(sub)
    return found


def _c1_routes():
    """``{(method, path): APIRoute}`` over the three W-C1 routers."""
    table = {}
    for module in C1_MODULES.values():
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
# main.py, the manifest and config
# ---------------------------------------------------------------------------


def test_main_mounts_each_w_c1_router_exactly_once():
    tree = ast.parse((ORCH / "main.py").read_text(encoding="utf-8"))
    mounted = [
        node.args[0].id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "include_router"
        and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in C1_MODULES
    ]
    assert sorted(mounted) == sorted(C1_MODULES)


def test_the_routers_declare_exactly_the_planned_routes():
    assert set(_c1_routes()) == C1_ROUTES


def test_every_w_c1_route_is_in_the_manifest_and_the_count_holds():
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    assert manifest["route_count"] == len(manifest["routes"])
    served = {(r["method"], r["path"]) for r in manifest["routes"]}
    assert not C1_ROUTES - served, sorted(C1_ROUTES - served)


def test_every_review_config_name_is_on_config_and_in_the_config_surface():
    surface = set(json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"])
    assert set(C1_CONFIG_DEFAULTS) <= surface
    source = (ORCH / "config.py").read_text(encoding="utf-8")
    for name, default in C1_CONFIG_DEFAULTS.items():
        assert hasattr(config, name), name
        if isinstance(default, bool):
            assert f'os.getenv("{name}", "{str(default).lower()}")' in source, name
        else:
            assert f'os.getenv("{name}", "{default}")' in source, name
    assert submissions.ONE_OPEN_SUBMISSION_SETTING in C1_CONFIG_DEFAULTS


def test_no_w_c1_config_name_holds_a_key():
    # D3-c: signing is the kit repo's CI step; the orchestrator holds no key and no GitHub token.
    assert [name for name in C1_CONFIG_DEFAULTS if KEY_NAME.search(name)] == []


# ---------------------------------------------------------------------------
# The gates on the routers themselves
# ---------------------------------------------------------------------------


def test_step_up_is_on_submit_and_listing_install_only_after_the_gate():
    marked = set()
    for key, route in _c1_routes().items():
        calls = _calls(route.dependant)
        at = _index(calls, _is_step_up)
        if at is None:
            continue
        marked.add(key)
        assert getattr(calls[at], step_up.STEP_UP_MARKER_ATTR) == config.STEP_UP_MAX_AGE_MINUTES
        assert _index(calls, _is_gate) < at, f"{key}: the permission gate must run before step-up"
    assert marked == C1_STEP_UP_ROUTES


def test_admin_routes_check_caller_is_admin_first_and_are_in_the_sweep_bucket():
    for key, route in _c1_routes().items():
        if not key[1].startswith("/api/admin/"):
            continue
        assert _index(_calls(route.dependant), _is_gate) is None, f"{key} is platform-wide, not workspace-gated"
        body = inspect.getsource(route.endpoint).split('"""')[-1]
        first = next(line.strip() for line in body.splitlines() if line.strip() and not line.strip().startswith(")"))
        assert first == "_require_admin(ctx)", f"{key} must check caller_is_admin first, found {first!r}"
        assert (key in ADMIN_GATED_IN_HANDLER) == (key[0] in MUTATING), key


def test_every_other_mutating_w_c1_route_is_workspace_gated():
    for key, route in _c1_routes().items():
        if key[1].startswith("/api/admin/") or key[0] not in MUTATING:
            continue
        assert _index(_calls(route.dependant), _is_gate) is not None, f"{key} lands outside bucket (a)"
        assert key not in ADMIN_GATED_IN_HANDLER, f"{key} would sit in two sweep buckets"


# ---------------------------------------------------------------------------
# The live app (a clean subprocess; no database is dialled on import)
# ---------------------------------------------------------------------------

_PROBE = """
import collections, json
from fastapi.routing import APIRoute
from core.auth import step_up
import main

SAMPLES = {
    "POST /api/marketplace/themes/submit": "/api/marketplace/themes/submit",
    "GET /api/marketplace/themes": "/api/marketplace/themes",
    "GET /api/marketplace/themes/{listing_id}": "/api/marketplace/themes/acme.ink",
    "POST /api/marketplace/themes/{listing_id}/install": "/api/marketplace/themes/acme.ink/install",
    "GET /api/admin/themes/pending": "/api/admin/themes/pending",
}
for tail in ("report", "report-note", "approve", "reject", "deactivate", "signature"):
    method = "GET" if tail == "report" else "POST"
    SAMPLES[method + " /api/admin/themes/{package_id}/" + tail] = (
        "/api/admin/themes/00000000-0000-4000-8000-000000000000/" + tail)

def calls(dependant):
    out = []
    for sub in dependant.dependencies:
        out += [sub.call] + calls(sub)
    return out

def first_handler(method, path):
    for route in main.app.routes:
        if isinstance(route, APIRoute) and method in route.methods and route.path_regex.match(path):
            return route.endpoint.__module__ + "." + route.endpoint.__name__ + " " + route.path
    return None

served = collections.Counter()
step_up_routes = []
for route in main.app.routes:
    if not isinstance(route, APIRoute):
        continue
    marked = any(getattr(c, step_up.STEP_UP_MARKER_ATTR, None) for c in calls(route.dependant))
    for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
        served[method + " " + route.path] += 1
        if marked:
            step_up_routes.append(method + " " + route.path)
answered = {key: first_handler(key.split(" ", 1)[0], path) for key, path in SAMPLES.items()}
print(json.dumps({
    "served": dict(served), "step_up": sorted(step_up_routes), "answered": answered,
    "submit_limit": main._body_limit_for("/api/marketplace/themes/submit"),
    "listing_limit": main._body_limit_for("/api/marketplace/themes/acme.ink/install"),
    "default_limit": main.MAX_BODY_SIZE,
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


def test_the_app_serves_every_w_c1_route_exactly_once(live_app):
    served = {_key(name): count for name, count in live_app["served"].items()}
    for key in C1_ROUTES:
        assert served.get(key) == 1, f"{key} served {served.get(key, 0)} times"


def test_no_earlier_route_shadows_a_w_c1_route(live_app):
    modules = {key: route.endpoint.__module__ for key, route in _c1_routes().items()}
    for name, answer in live_app["answered"].items():
        key = _key(name)
        assert answer is not None, f"{key}: nothing answers it"
        handler, path = answer.split(" ", 1)
        assert handler.startswith(modules[key] + ".") and path == key[1], f"{key} is answered by {answer}"


def test_the_app_carries_step_up_on_the_w_b2_routes_and_the_two_w_c1_routes_only(live_app):
    assert {_key(name) for name in live_app["step_up"]} == B2_STEP_UP_ROUTES | C1_STEP_UP_ROUTES


def test_the_submit_body_limit_is_sized_from_the_package_cap(live_app):
    assert live_app["submit_limit"] == config.THEME_MAX_PACKAGE_MB * MB + MULTIPART_ENVELOPE
    assert live_app["listing_limit"] == live_app["default_limit"]  # install sends a small JSON body


# ---------------------------------------------------------------------------
# The seam: approve offers the version to older listing installs
# ---------------------------------------------------------------------------

PLAY = "automatos/play"
ENGINES = {"automatos": "^1.0.0", "sdk": "^1.0.0"}
SEAM_TABLES = [m.__table__ for m in (ThemePackage, ThemeInstallation, ThemeReviewEvent, ThemeRevocation)]


@pytest.fixture
def db(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=SEAM_TABLES)
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _version(db, version, status):
    files = {"dist/entry.js": {"type": "text/javascript", "size": 1,
                               "sha256": hashlib.sha256(version.encode()).hexdigest()}}
    row = ThemePackage(
        theme_id=PLAY, version=version, owner_type="marketplace", status=status, reviewed=status == "approved",
        manifest={"id": PLAY, "name": "Play", "version": version, "engines": ENGINES},
        file_manifest=files, storage_prefix=f"themes/{PLAY}/{version}/", scan_verdict="review_required",
        digest=package_digest({path: entry["sha256"] for path, entry in files.items()}),
    )
    db.add(row)
    db.commit()
    return row


def test_approving_a_version_offers_it_to_older_listing_installs(db):
    old = _version(db, "1.0.0", "approved")
    install = ThemeInstallation(workspace_id=uuid.uuid4(), theme_id=PLAY, package_id=old.id)
    db.add(install)
    db.commit()
    new = _version(db, "1.1.0", "pending")

    review.add_report_note(db, new.id, "isolated job: reproducible build, digests equal", "pass", None)
    answer = review.approve(db, new.id, None)
    db.commit()

    db.refresh(install)
    assert answer["updatesOffered"] == 1 and answer["status"] == "approved"
    assert install.pending_package_id == new.id and install.package_id == old.id  # offered, never switched
