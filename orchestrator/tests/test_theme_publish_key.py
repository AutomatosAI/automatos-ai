"""PRD-261 — the theme publish key: its own credential, fail closed, rotatable, never hybrid's, never logged.

Pins (the kit ↔ core publish contract; ``services/themes/publish_key.py``):

* 503 ``PUBLISH_NOT_CONFIGURED`` while any of ``THEME_PUBLISH_KEY_SHA256``, ``THEME_PUBLISH_PUBLISHERS``
  and ``THEME_PUBLISH_ACTOR`` is unset or can't be read (three hashes, a hash that isn't hex, an id
  that isn't one), or the actor's ``users`` row is missing or inactive; even with the right key;
* 401 ``PUBLISH_KEY_REFUSED`` with ``WWW-Authenticate: Bearer`` for no key, a wrong key, the key sent
  as ``X-API-Key`` or under another scheme; the body is never read before the key passed;
* rotation: with two hashes both keys pass; once the old hash is dropped its key is 401;
* hybrid never accepts the key: a hybrid-gated route answers 401 to it as a bearer (it goes to
  the Clerk lane, which refuses it) and as ``X-API-Key`` (only ``ORCHESTRATOR_API_KEY`` matches);
* neither the key nor its hash reaches a log record, refused or published;
* 403 while ``THEMES_MARKETPLACE`` is off, after the key;
* the wiring: the route carries no hybrid dependency and is the sweep's own-auth route, it is in the
  route manifest, ``main.py`` mounts it once with the submit route's body limit, and its settings
  are on ``config`` and in ``reports/config-surface.json``.
"""
from __future__ import annotations

import ast
import json
import logging
import uuid
from types import SimpleNamespace

import pytest
from fastapi.routing import APIRoute

import api.admin_themes as admin_api
import api.marketplace_themes_publish as publish_api
from config import config
from core.auth import hybrid
from core.auth.hybrid import get_request_context_hybrid
from tests import helpers_theme_publish as hp
from tests.test_p2w2_authz_boundary_sweep import OWN_AUTH_ROUTES

ROUTE = ("POST", hp.THEME_PUBLISH_ROUTE)
SETTINGS = ("THEME_PUBLISH_KEY_SHA256", "THEME_PUBLISH_PUBLISHERS", "THEME_PUBLISH_ACTOR", "THEME_PUBLISH_PER_DAY")


@pytest.fixture
def env(monkeypatch, tmp_path):
    yield from hp.publish_env(monkeypatch, tmp_path)


def _not_configured(response):
    hp.refused(response, 503, "PUBLISH_NOT_CONFIGURED")


def _key_refused(response):
    hp.refused(response, 401, "PUBLISH_KEY_REFUSED")
    assert response.headers["www-authenticate"] == "Bearer"


# --- 503: fail closed ------------------------------------------------------------------------


@pytest.mark.parametrize("setting", ["hashes", "publishers", "actor"])
def test_any_unset_setting_is_503_even_with_the_right_key(env, monkeypatch, setting):
    hp.configure(monkeypatch, **{setting: ""})
    _not_configured(hp.post(env.client, hp.package()))
    assert hp.nothing_stored(env)


@pytest.mark.parametrize("changes", [
    {"hashes": ",".join([hp.KEY_HASH, hp.NEXT_KEY_HASH, "0" * 64])},  # more than two
    {"hashes": hp.KEY_HASH[:-1] + "z"},
    {"actor": "fifty"},
    {"actor": "0"},
])
def test_a_setting_that_cant_be_read_is_503(env, monkeypatch, changes):
    hp.configure(monkeypatch, **changes)
    _not_configured(hp.post(env.client, hp.package()))


@pytest.mark.parametrize("actor", ["999", str(hp.PERSON_PK)])
def test_a_missing_or_inactive_actor_row_is_503(env, monkeypatch, actor):
    session = env.sessions()
    session.execute(hp.sa.text("UPDATE users SET is_active = false WHERE id = :id"), {"id": hp.PERSON_PK})
    session.commit()
    session.close()
    hp.configure(monkeypatch, actor=actor)
    _not_configured(hp.post(env.client, hp.package()))
    assert hp.nothing_stored(env)


def test_unset_settings_answer_503_before_any_key_check(env, monkeypatch):
    hp.configure(monkeypatch, hashes="")
    _not_configured(hp.post(env.client, hp.package(), key=None))


# --- 401: the key ----------------------------------------------------------------------------


@pytest.mark.parametrize("key, headers", [
    (None, {}),
    (hp.NEXT_KEY, {}),
    (None, {"X-API-Key": hp.KEY}),
    (None, {"Authorization": f"ApiKey {hp.KEY}"}),
    (hp.KEY_HASH, {}),  # the hash itself is not the key
])
def test_no_key_or_not_the_key_is_401(env, key, headers):
    _key_refused(hp.post(env.client, hp.package(), key=key, headers=headers))
    assert hp.nothing_stored(env) and env.stage.calls == []


def test_the_body_is_not_read_before_the_key_passed(env):
    read = []

    async def parts_spy():
        read.append(True)

    env.client.app.dependency_overrides[publish_api.publish_parts] = parts_spy
    _key_refused(hp.post(env.client, hp.package(), key=hp.NEXT_KEY))
    assert read == []


def test_rotation_both_hashes_pass_then_the_old_key_is_refused(env, monkeypatch):
    data = hp.package()
    hp.configure(monkeypatch, hashes=f"{hp.KEY_HASH}, {hp.NEXT_KEY_HASH.upper()}")
    assert hp.post(env.client, data, key=hp.KEY).json()["status"] == "published"
    assert hp.post(env.client, data, key=hp.NEXT_KEY).json()["status"] == "unchanged"

    hp.configure(monkeypatch, hashes=hp.NEXT_KEY_HASH)
    _key_refused(hp.post(env.client, data, key=hp.KEY))
    assert hp.post(env.client, data, key=hp.NEXT_KEY).json()["status"] == "unchanged"


def test_the_marketplace_switch_off_is_403_after_the_key(env, monkeypatch):
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", "off")
    _key_refused(hp.post(env.client, hp.package(), key=hp.NEXT_KEY))
    response = hp.post(env.client, hp.package())
    assert response.status_code == 403 and hp.nothing_stored(env)


# --- hybrid never accepts the key ------------------------------------------------------------


class _ClerkSpy:
    """Stands in for Clerk: records each token it is asked to verify and accepts none (a hex key isn't a JWT)."""

    def __init__(self):
        self.verified = []

    def verify_token(self, token):
        self.verified.append(token)
        return None


@pytest.fixture
def hybrid_lane(env, monkeypatch):
    spy = _ClerkSpy()
    monkeypatch.setattr(hybrid, "get_clerk_auth", lambda: spy)
    monkeypatch.setattr(hybrid, "SessionLocal", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(config, "ORCHESTRATOR_API_KEY", "an-orchestrator-key-that-is-not-the-publish-key")
    app_client = hp.client(env, routers=(admin_api.router,))
    return SimpleNamespace(client=app_client, clerk=spy)


@pytest.mark.parametrize("headers", [{"Authorization": f"Bearer {hp.KEY}"}, {"X-API-Key": hp.KEY}])
def test_a_hybrid_route_refuses_the_publish_key(hybrid_lane, headers):
    approve = f"/api/admin/themes/{uuid.uuid4()}/approve"
    response = hybrid_lane.client.post(approve, headers=headers)
    assert response.status_code == 401, response.text
    if "Authorization" in headers:
        assert hybrid_lane.clerk.verified == [hp.KEY]  # the Clerk lane saw it, and refused it


def test_the_route_carries_no_hybrid_dependency_and_is_the_sweeps_own_auth_route():
    (route,) = [r for r in publish_api.router.routes if isinstance(r, APIRoute)]
    calls = []

    def walk(dependant):
        for sub in dependant.dependencies:
            calls.append(sub.call)
            walk(sub)

    walk(route.dependant)
    assert get_request_context_hybrid not in calls
    assert calls[:3] == [publish_api.require_publish_key, publish_api.require_marketplace, publish_api.publish_parts]
    assert (route.path, route.methods) == (hp.THEME_PUBLISH_ROUTE, {"POST"}) and ROUTE in OWN_AUTH_ROUTES


# --- never logged ----------------------------------------------------------------------------


def test_neither_the_key_nor_its_hash_reaches_a_log(env, hybrid_lane, caplog):
    caplog.set_level(logging.DEBUG)
    hp.post(env.client, hp.package(), key=hp.NEXT_KEY)
    hp.post(env.client, hp.package(hp.UNLISTED_ID))
    assert hp.post(env.client, hp.package()).json()["status"] == "published"
    hybrid_lane.client.post(f"/api/admin/themes/{uuid.uuid4()}/approve", headers={"Authorization": f"Bearer {hp.KEY}"})

    assert caplog.records
    for record in caplog.records:
        logged = " ".join([record.getMessage(), str(record.args), record.exc_text or ""])
        for secret in (hp.KEY, hp.KEY_HASH, hp.NEXT_KEY):
            assert secret not in logged, record.name
    assert any("PUBLISH_KEY_REFUSED" in record.getMessage() for record in caplog.records)


# --- the wiring ------------------------------------------------------------------------------


def _main_tree():
    return ast.parse((hp.ORCH / "main.py").read_text(encoding="utf-8"))


def test_main_mounts_the_router_once_with_the_submit_routes_body_limit():
    tree = _main_tree()
    mounted = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and getattr(node.func, "attr", None) == "include_router" and node.args
               and getattr(node.args[0], "id", None) == "marketplace_themes_publish_router"]
    assert len(mounted) == 1
    (limits,) = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and any(getattr(target, "id", None) == "PATH_BODY_LIMITS" for target in node.targets)]
    entries = {getattr(key, "id", None): ast.unparse(value) for key, value in zip(limits.keys, limits.values)}
    assert entries["THEME_PUBLISH_ROUTE"] == entries["THEME_SUBMIT_ROUTE"] == "theme_cap_bytes('package') + 512 * 1024"


def test_the_route_is_in_the_manifest():
    manifest = json.loads((hp.ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    assert {"method": ROUTE[0], "path": ROUTE[1]} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])


def test_the_settings_are_on_config_and_in_the_config_surface():
    surface = set(json.loads((hp.ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"])
    assert set(SETTINGS) <= surface
    source = (hp.ORCH / "config.py").read_text(encoding="utf-8")
    for name in SETTINGS[:3]:
        assert f'os.getenv("{name}", "")' in source, name  # unset by default: the endpoint is dark
    assert 'os.getenv("THEME_PUBLISH_PER_DAY", "10")' in source
