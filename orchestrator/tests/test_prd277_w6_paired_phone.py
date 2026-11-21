"""PRD-277 US-601 without a database: the pairing setting, who makes a code, the push decision, the edition check.

* **The setting** (``ENTERPRISE_DEVICE_PAIRING``, §5.10): on when unset; in enterprise ``device_pairing`` follows
  it, a malformed value reading as off; ``local`` and ``saas`` never move whatever it says. Start-up refuses a
  malformed value naming the setting and never the value. With it off a device token is 401 before any lookup.
* **Who makes a code** (``pairing_code_owner``): in enterprise a person signed in on the session lane, a member of
  the workspace, for themselves; a device, a key, the all-workspaces view and a non-member are 403, and the
  console (``scripts/mint_pairing_code.py``) refuses. ``local`` keeps its workspace-admin gate.
* **The push decision** (``PUT /api/v1/mobile/push-consent``): in enterprise an install admin signed in on the
  session lane, after a fresh sign-in, with one ``push:consent_changed`` row; a workspace admin who is not an
  install admin, and a device, are 403. ``local`` is as before: the operator, no step-up, no row.
  ``PUSH_CONSENT_STEP_UP_ROUTES`` is this route's share of the app-wide step-up pin in ``test_prd261_c1_wire.py``.
* **The edition check** (``services.mobile.edition_change``): the lock is taken before the marker is read, so two
  pods starting at once revoke once; a check that cannot run refuses device tokens (503) until it does; every
  worker runs it in ``_boot_phase_1_core`` after the tables, before it serves.
"""
from __future__ import annotations

import ast
import importlib
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import api.mobile_devices as devices_api  # noqa: E402
import api.mobile_push as push_api  # noqa: E402
import core.auth.workspace_admin as admin_mod  # noqa: E402
from config import config  # noqa: E402
from core import edition_start_up, editions  # noqa: E402
from core.auth import device, hybrid, step_up  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.principal import (  # noqa: E402
    AUTH_SESSION_CLAIM_SOURCE,
    LOCAL_OPERATOR_CLAIM_SOURCE,
    MOBILE_DEVICE_CLAIM_SOURCE,
)
from core.auth.session_lane import SESSION_AUTH_TYPE  # noqa: E402
from core.database.database import get_db  # noqa: E402
from services.mobile import device_audit, edition_change  # noqa: E402
from services.push import consent  # noqa: E402
from tests.test_prd277_enterprise_start_up import (  # noqa: E402,F401  (the fixture restores config after a reload)
    ENV_KEYS,
    OIDC_ENV,
    _restore_config_module,
)

CONSENT = "/api/v1/mobile/push-consent"
PUSH_CONSENT_STEP_UP_ROUTES = {("PUT", CONSENT)}
SETTING = editions.DEVICE_PAIRING_SETTING
WS = uuid.uuid4()
PERSON = 601
ON_FORMS = ["true", "1", "yes", "on", " TRUE "]
OFF_FORMS = ["false", "0", "no", "off", ""]
MALFORMED = ["enabled", "maybe", "2", "nope"]
ORCHESTRATOR = Path(__file__).resolve().parents[1]


def _enterprise(monkeypatch, pairing="true"):
    monkeypatch.setattr(config, "AUTH_EDITION", "enterprise")
    monkeypatch.setattr(config, SETTING, pairing)


def _session(system_role="user", minutes_ago=0.0, workspace_id=WS, **extra) -> RequestContext:
    claims = {"source": AUTH_SESSION_CLAIM_SOURCE, "user_id": PERSON, "auth_time": time.time() - minutes_ago * 60}
    user = UserContext(id="person-601", email="person-601@example.test", system_role=system_role, raw_claims=claims)
    return RequestContext(workspace_id=workspace_id, user=user, auth_type=SESSION_AUTH_TYPE, **extra)


def _device_ctx() -> RequestContext:
    claims = {"source": MOBILE_DEVICE_CLAIM_SOURCE, "device_id": str(uuid.uuid4()), "user_id": PERSON}
    user = UserContext(id="person-601@example.test", email="person-601@example.test", system_role="user",
                       raw_claims=claims)
    return RequestContext(workspace_id=WS, user=user, auth_type=device.DEVICE_AUTH_TYPE)


def _key_ctx() -> RequestContext:
    user = UserContext(id="key", email="", system_role="admin", raw_claims={})
    return RequestContext(workspace_id=WS, user=user, auth_type="api_key", api_key_id="key-601")


def _operator() -> RequestContext:
    user = UserContext(id="operator@local", email="operator@local", system_role="super_admin",
                       raw_claims={"source": LOCAL_OPERATOR_CLAIM_SOURCE, "user_id": 1})
    return RequestContext(workspace_id=WS, user=user, auth_type="anonymous")


# --------------------------------------------------------------------------- the setting


@pytest.mark.parametrize("value", ON_FORMS)
def test_enterprise_pairs_while_the_setting_is_on(monkeypatch, value):
    _enterprise(monkeypatch, value)
    assert editions.current().device_pairing is True


@pytest.mark.parametrize("value", OFF_FORMS + MALFORMED + [None])
def test_enterprise_does_not_pair_when_the_setting_is_off_or_malformed(monkeypatch, value):
    _enterprise(monkeypatch, value)
    assert editions.current().device_pairing is False


@pytest.mark.parametrize("edition", ["local", "saas"])
@pytest.mark.parametrize("value", ON_FORMS + OFF_FORMS + MALFORMED)
def test_local_and_saas_never_move_whatever_the_setting_says(monkeypatch, edition, value):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(config, SETTING, value)
    assert editions.current() == editions.CAPABILITIES[edition]


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_the_one_place_the_setting_combines_leaves_other_rows_as_they_are(edition):
    row = editions.CAPABILITIES[edition]
    assert editions.with_device_pairing_setting(row, "false") is row


def test_the_one_place_the_setting_combines_never_switches_on_what_the_row_has_off():
    row = editions.CAPABILITIES["enterprise"]
    assert editions.with_device_pairing_setting(row, "false").device_pairing is False
    assert editions.with_device_pairing_setting(row, "true").device_pairing is row.device_pairing


def _fresh(**env):
    """Reload config.py with only ``env`` among the start-up names and the pairing setting; its singleton."""
    keys = (*ENV_KEYS, SETTING)
    saved = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    os.environ.update({key: value for key, value in env.items() if value is not None})
    try:
        import config as config_mod

        importlib.reload(config_mod)
        return config_mod.config
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_the_setting_is_on_when_unset_and_an_enterprise_install_starts():
    cfg = _fresh(AUTH_EDITION="enterprise", **OIDC_ENV)
    assert getattr(cfg, SETTING) == "true"
    cfg.validate_auth_edition()  # must not raise


@pytest.mark.parametrize("value", ["false", "OFF", "0", "yes"])
def test_an_enterprise_install_starts_with_any_switch_form(value):
    _fresh(AUTH_EDITION="enterprise", **OIDC_ENV, **{SETTING: value}).validate_auth_edition()


@pytest.mark.parametrize("value", MALFORMED)
def test_start_up_refuses_a_malformed_value_naming_the_setting_never_the_value(value):
    cfg = _fresh(AUTH_EDITION="enterprise", **OIDC_ENV, **{SETTING: value})
    with pytest.raises(edition_start_up.EditionNotReady) as refused:
        cfg.validate_auth_edition()
    assert SETTING in str(refused.value) and value not in str(refused.value).replace(SETTING, "")


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_the_check_ignores_the_setting_where_the_edition_does_not_read_it(edition):
    edition_start_up.check_device_pairing_setting(editions.CAPABILITIES[edition], {SETTING: "maybe"})


@pytest.mark.asyncio
async def test_with_pairing_off_a_device_token_is_refused_before_any_lookup(monkeypatch):
    _enterprise(monkeypatch, "false")
    monkeypatch.setattr(device, "SessionLocal", lambda: pytest.fail("no lookup may run"))
    with pytest.raises(HTTPException) as refused:
        await device.resolve_device_context(SimpleNamespace(headers={}), "amd_" + "x" * 43)
    assert refused.value.status_code == 401


# --------------------------------------------------------------------------- who makes a code


@pytest.fixture
def members(monkeypatch):
    """Who is an active member of ``WS``, by ``users.id``: the device lane's own rule, without a database."""
    held = {PERSON}
    monkeypatch.setattr(hybrid, "_user_pk_is_workspace_member",
                        lambda db, workspace_id, user_pk: workspace_id == WS and user_pk in held)
    return held


def _owner_refusal(ctx) -> HTTPException:
    with pytest.raises(HTTPException) as refused:
        devices_api.pairing_code_owner(ctx, db=None)
    return refused.value


def test_in_enterprise_a_signed_in_member_makes_a_code_whatever_their_roles(monkeypatch, members):
    _enterprise(monkeypatch)
    for role in ("user", "admin", "super_admin"):
        assert devices_api.pairing_code_owner(_session(role), db=None).user.raw_claims["user_id"] == PERSON


@pytest.mark.parametrize("caller", [_device_ctx, _key_ctx, _operator])
def test_in_enterprise_no_device_key_or_operator_makes_a_code(monkeypatch, members, caller):
    _enterprise(monkeypatch)
    refused = _owner_refusal(caller())
    assert (refused.status_code, refused.detail) == (403, devices_api.PAIR_FROM_THE_BROWSER)


@pytest.mark.parametrize("ctx", [
    lambda: _session("super_admin", admin_all_workspaces=True),
    lambda: _session(workspace_id=None),
    lambda: _session(workspace_id=uuid.uuid4()),
], ids=["all-workspaces view", "no workspace", "not a member"])
def test_in_enterprise_a_code_is_made_only_in_a_workspace_the_person_belongs_to(monkeypatch, members, ctx):
    _enterprise(monkeypatch)
    refused = _owner_refusal(ctx())
    assert (refused.status_code, refused.detail) == (403, devices_api.PAIR_IN_YOUR_WORKSPACE)


def test_in_enterprise_a_member_removed_from_the_workspace_makes_no_code(monkeypatch, members):
    _enterprise(monkeypatch)
    members.clear()
    assert _owner_refusal(_session()).status_code == 403


def test_local_keeps_the_workspace_admin_gate(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    seen = []
    monkeypatch.setattr(devices_api, "require_workspace_admin", lambda ctx, db: seen.append(ctx) or ctx)
    ctx = _operator()
    assert devices_api.pairing_code_owner(ctx, db=None) is ctx and seen == [ctx]


@pytest.mark.parametrize("pairing", ["true", "false"])
def test_in_enterprise_no_console_makes_a_code(monkeypatch, pairing):
    from scripts import mint_pairing_code as console

    _enterprise(monkeypatch, pairing)
    assert console.refusal() == console.SAAS_REFUSAL
    assert "AUTH_EDITION=local" in console.SAAS_REFUSAL and "signed in as themselves" in console.SAAS_REFUSAL


def test_with_pairing_off_the_code_route_is_absent_in_enterprise(monkeypatch):
    _enterprise(monkeypatch, "false")
    app = FastAPI()
    app.include_router(devices_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: pytest.fail("no caller may be read")
    app.dependency_overrides[get_db] = lambda: None
    assert TestClient(app).post("/api/v1/mobile/pairing-codes").status_code == 404


# --------------------------------------------------------------------------- the push decision


class _Staged:
    """The route's session: what it adds and whether it commits."""

    def __init__(self):
        self.added, self.commits = [], 0

    def add(self, row):
        self.added.append(row)

    def commit(self):
        self.commits += 1

    def rollback(self):
        return None


@pytest.fixture
def push(monkeypatch):
    """The push routes alone; the caller is ``state["ctx"]``, a workspace admin unless ``state["member"]`` is off."""
    state = {"ctx": _operator(), "member": True, "db": _Staged()}
    answered = consent.ConsentState(agreed=True, answered=True, answered_at="2026-11-21T09:00:00+00:00")
    monkeypatch.setattr(admin_mod, "_is_member_in", lambda db, ctx, roles: state["member"])
    monkeypatch.setattr(consent, "set_push_consent", lambda db, agreed: answered)
    monkeypatch.setattr(consent, "consent_state", lambda: answered)
    app = FastAPI()
    app.include_router(push_api.router)
    step_up.install_step_up_handler(app)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state["ctx"]
    app.dependency_overrides[get_db] = lambda: state["db"]

    def call(method, ctx, member=True):
        state.update(ctx=ctx, member=member)
        return TestClient(app).request(method, CONSENT, json={"agreed": True} if method == "PUT" else None)

    return call, state


def _actions(state) -> list:
    return [(row.action, row.user_id, row.resource_type, row.details) for row in state["db"].added]


@pytest.mark.parametrize("method", ["GET", "PUT"])
def test_in_enterprise_a_workspace_admin_who_is_not_an_install_admin_does_not_decide(push, monkeypatch, method):
    call, state = push
    _enterprise(monkeypatch)
    answer = call(method, _session("user"))
    assert (answer.status_code, answer.json()) == (403, {"detail": push_api.INSTALL_ADMINS_ONLY})
    assert state["db"].added == []


@pytest.mark.parametrize("method", ["GET", "PUT"])
def test_in_enterprise_a_device_or_a_key_never_decides(push, monkeypatch, method):
    call, state = push
    _enterprise(monkeypatch)
    for ctx in (_device_ctx(), _key_ctx()):
        assert call(method, ctx).status_code == 403
    assert state["db"].added == []


def test_in_enterprise_an_install_admin_with_an_old_sign_in_is_sent_to_sign_in_again(push, monkeypatch):
    call, state = push
    _enterprise(monkeypatch)
    answer = call("PUT", _session("admin", minutes_ago=config.STEP_UP_MAX_AGE_MINUTES + 5))
    assert answer.status_code == 403 and answer.json()["code"] == step_up.STEP_UP_CODE
    assert state["db"].added == [] and state["db"].commits == 0


@pytest.mark.parametrize("role", ["admin", "super_admin"])
def test_in_enterprise_an_install_admin_with_a_fresh_sign_in_decides_and_it_is_recorded(push, monkeypatch, role):
    call, state = push
    _enterprise(monkeypatch)
    answer = call("PUT", _session(role))
    assert answer.status_code == 200 and answer.json()["agreed"] is True
    assert _actions(state) == [(device_audit.PUSH_CONSENT_CHANGED, PERSON, device_audit.PUSH_CONSENT_RESOURCE,
                                {"agreed": True})]
    assert state["db"].commits == 1


def test_in_enterprise_an_install_admin_must_also_run_the_workspace(push, monkeypatch):
    call, _ = push
    _enterprise(monkeypatch)
    assert call("PUT", _session("admin"), member=False).status_code == 403


def test_in_local_the_operator_decides_as_before_with_no_step_up_and_no_row(push, monkeypatch):
    call, state = push
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    assert call("GET", _operator()).status_code == 200
    assert call("PUT", _operator()).status_code == 200
    assert state["db"].added == [] and state["db"].commits == 1


def test_the_put_carries_step_up_and_its_gate_runs_first():
    route = next(r for r in push_api.router.routes if r.path == CONSENT and "PUT" in r.methods)
    names = [dep.call for dep in route.dependant.dependencies]
    assert names.index(push_api.push_consent_decider) < names.index(push_api.STEP_UP)
    assert getattr(push_api.STEP_UP, step_up.STEP_UP_MARKER_ATTR, None) == config.STEP_UP_MAX_AGE_MINUTES


# --------------------------------------------------------------------------- the edition check


class _Ordered:
    """A session that records the order of what the check does."""

    def __init__(self, events):
        self.events = events

    def execute(self, statement, params=None):
        self.events.append(("execute", str(statement), params))

    def commit(self):
        self.events.append(("commit",))


def test_the_lock_is_taken_before_the_marker_is_read_and_held_to_the_commit(monkeypatch):
    events = []
    _enterprise(monkeypatch)
    monkeypatch.setattr(edition_change, "stored_edition", lambda db: events.append(("read",)) or "enterprise")
    monkeypatch.setattr(edition_change, "revoke_every_paired_device", lambda db, now: pytest.fail("no revoke"))
    assert edition_change.revoke_if_edition_changed(_Ordered(events), None) == 0
    assert [event[0] for event in events] == ["execute", "read", "commit"]
    assert "pg_advisory_xact_lock" in events[0][1] and events[0][2] == {"key": edition_change.LOCK_KEY}


class _Failing:
    def execute(self, *_args, **_kwargs):
        raise OperationalError("SELECT 1", {}, Exception("down"))

    def rollback(self):
        return None

    def close(self):
        return None


@pytest.mark.asyncio
async def test_a_check_that_cannot_run_refuses_device_tokens_until_it_does(monkeypatch):
    _enterprise(monkeypatch)
    monkeypatch.setattr(edition_change, "_status", edition_change.NOT_RUN)
    monkeypatch.setattr(edition_change, "SessionLocal", _Failing)
    monkeypatch.setattr(device, "SessionLocal", lambda: pytest.fail("no lookup before the check"))

    assert edition_change.confirm_edition() is False and edition_change.needs_confirming() is True
    with pytest.raises(HTTPException) as refused:
        await device.resolve_device_context(SimpleNamespace(headers={}), "amd_" + "x" * 43)
    assert (refused.value.status_code, refused.value.detail) == (503, edition_change.NOT_CONFIRMED)
    with pytest.raises(HTTPException):
        devices_api._require_device_pairing()


@pytest.mark.parametrize("earlier, running, changed", [
    ("local", "local", False), ("enterprise", "enterprise", False), ("saas", "saas", False),
    ("local", "enterprise", True), ("enterprise", "local", True), ("saas", "enterprise", True),
])
def test_the_one_edition_comparison_lives_with_the_flag(monkeypatch, earlier, running, changed):
    monkeypatch.setattr(config, "AUTH_EDITION", running)
    assert editions.started_as_another(earlier) is changed


def test_a_check_that_has_run_asks_nothing_more(monkeypatch):
    _enterprise(monkeypatch)
    monkeypatch.setattr(edition_change, "_status", edition_change.CONFIRMED)
    monkeypatch.setattr(edition_change, "SessionLocal", lambda: pytest.fail("the check ran once"))
    edition_change.ensure_confirmed()
    assert edition_change.needs_confirming() is False


def test_every_worker_runs_the_check_after_the_tables_before_it_serves():
    tree = ast.parse((ORCHESTRATOR / "main.py").read_text(encoding="utf-8"))
    boot = next(node for node in tree.body
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "_boot_phase_1_core")
    guarded = next(node for node in ast.walk(boot) if isinstance(node, ast.Try) and node.finalbody)
    called = lambda nodes: {n.func.id for s in nodes for n in ast.walk(s)  # noqa: E731
                            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "create_tables" in called(guarded.body) and "confirm_edition" in called(guarded.finalbody)
