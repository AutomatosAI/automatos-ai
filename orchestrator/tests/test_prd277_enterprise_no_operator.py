"""PRD-277 US-103: in enterprise there is no operator, and a request with no credential is 401.

These tests set the edition directly, without its start-up checks
(``test_prd277_enterprise_start_up.py``), to prove the code paths are safe:

* the real request-context dependency (``hybrid.get_request_context_hybrid``, with the bearer
  registry in front of it) answers 401 to a request with no credential, in every
  ``LOCAL_REMOTE_ACCESS`` mode;
* even with ``REQUIRE_AUTH`` off (which ``config.py`` never allows in enterprise) the
  anonymous lane never binds the operator and never makes anyone ``super_admin``;
* a Clerk token is never verified, so Clerk settings left in the environment trust nobody;
* a paired device whose person has ``LOCAL_OPERATOR_EMAIL`` is a plain person who must still
  be a member, never the operator, through the real dependency too;
* a server key is bound to its own workspace and its own scope, and nothing more: the shared
  dependency refuses it and it holds no workspace permission;
* the platform API key is kept as it is (PRD-277 D-14): an admin-level key, never the operator;
* step-up refuses every context that is not a session (a device, a key, a fresh Clerk one
  included): only a session's person can sign in again (US-406);
* discovery never offers phones a Clerk sign-in (US-602: it offers OIDC, then pairing);
* Auto describes an organisation's install, with the hosted tabs less Billing and Widget SDK.

DB-free: the database seams are stubbed.
"""
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import jwt
import pytest
from fastapi import HTTPException

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core import editions  # noqa: E402
from core.auth import device, hybrid, step_up  # noqa: E402
from core.auth.clerk import ClerkAuth  # noqa: E402
from core.auth.dependencies import UserContext  # noqa: E402
from core.auth.workspace_permission import workspace_permission_granted  # noqa: E402
from core.models.mobile_devices import MobileDeviceStatus  # noqa: E402
from services.mobile import discovery  # noqa: E402

WS = uuid.UUID("00000000-0000-0000-0000-0000000277bb")
# Stand-ins for credentials, built from parts so no literal looks like one.
CLERK_SHAPED_TOKEN = ".".join(("header", "claims", "signature"))
DEVICE_TOKEN = "amd" + "_" + "prd277" + "-stand-in"
SERVER_KEY = "ak" + "_srv_" + "prd277" + "-stand-in"
PLATFORM_KEY = "prd277" + "-" + "platform" + "-stand-in"
KEY_WS = uuid.UUID("00000000-0000-0000-0000-0000000277bc")
MODES = ("off", "open", "paired")
NO_CREDENTIAL = [
    {},
    {"x-workspace-id": str(WS)},  # naming a workspace is not a credential
    {"authorization": "Bearer "},  # an empty bearer is none
    {"authorization": "Basic x"},  # a scheme no lane reads
    {"x-api-key": "   "},  # a blank key
]


def _request(headers=None):
    return SimpleNamespace(method="GET", headers=dict(headers or {}), query_params={}, state=SimpleNamespace())


class _OperatorSpy:
    """Stands in for the operator binder; records a call and returns the operator, so binding shows."""

    def __init__(self):
        self.calls = 0

    def __call__(self, db):
        self.calls += 1
        return UserContext(id=hybrid.config.LOCAL_OPERATOR_EMAIL, email=hybrid.config.LOCAL_OPERATOR_EMAIL,
                           system_role="super_admin")


class _Keys:
    """Stands in for Clerk's key client: records every token it is asked about and trusts none."""

    def __init__(self):
        self.asked = []

    def get_signing_key_from_jwt(self, token):
        self.asked.append(token)
        raise jwt.InvalidTokenError("stand-in key client")


@pytest.fixture
def enterprise(monkeypatch):
    """The live config as an enterprise install's request sees it, its database stubbed."""
    cfg = hybrid.config
    monkeypatch.setattr(cfg, "AUTH_EDITION", "enterprise")
    monkeypatch.setattr(cfg, "REQUIRE_AUTH", True)  # what config.py forces in enterprise
    monkeypatch.setattr(cfg, "LOCAL_REMOTE_ACCESS", "off")
    monkeypatch.setattr(cfg, "DEFAULT_WORKSPACE_ID", str(WS))
    monkeypatch.setattr(cfg, "WORKSPACE_ID", None)
    monkeypatch.setattr(cfg, "AUTH_DEBUG", False)
    monkeypatch.setattr(hybrid, "SessionLocal", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(hybrid, "_workspace_exists", lambda db, workspace_id: True)
    monkeypatch.setattr(hybrid, "_assert_workspace_usable", lambda db, workspace_id, is_admin=False: None)
    spy = _OperatorSpy()
    monkeypatch.setattr(hybrid, "_local_operator_user_context", spy)
    assert editions.current_name() == "enterprise"
    return SimpleNamespace(config=cfg, operator=spy)


# --------------------------------------------------------------------------- no credential: 401


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("headers", NO_CREDENTIAL)
async def test_a_request_with_no_credential_gets_401(enterprise, monkeypatch, mode, headers):
    monkeypatch.setattr(enterprise.config, "LOCAL_REMOTE_ACCESS", mode)

    with pytest.raises(HTTPException) as refused:
        await hybrid.get_request_context_hybrid(_request(headers))

    assert refused.value.status_code == 401
    assert refused.value.detail == "Authentication required"  # hybrid's lane, not the pairing refusal
    assert enterprise.operator.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_no_operator_context_is_ever_returned_even_with_auth_off(enterprise, monkeypatch, mode):
    monkeypatch.setattr(enterprise.config, "REQUIRE_AUTH", False)  # never possible from config.py here
    monkeypatch.setattr(enterprise.config, "LOCAL_REMOTE_ACCESS", mode)

    ctx = await hybrid.get_request_context_hybrid(_request({"x-workspace-id": str(WS)}))

    assert enterprise.operator.calls == 0
    assert ctx.user.system_role != "super_admin"
    assert ctx.user.email != enterprise.config.LOCAL_OPERATOR_EMAIL and ctx.user.id is None


def test_the_edition_has_no_anonymous_operator_and_no_lane_to_close(enterprise):
    from core.auth import bearer_registry

    assert editions.current().anonymous_operator is False
    assert bearer_registry.anonymous_lane_closed() is False  # REQUIRE_AUTH, forced on, refuses instead


# --------------------------------------------------------------------------- Clerk is never trusted


def test_a_clerk_token_is_never_verified(enterprise):
    clerk = ClerkAuth()
    keys = _Keys()
    clerk._jwks_client = keys

    assert clerk.verify_token(CLERK_SHAPED_TOKEN) is None
    assert keys.asked == []


def test_saas_still_verifies_a_clerk_token(enterprise, monkeypatch):
    """The control: the same stand-in is asked on saas, so the enterprise test is not vacuous."""
    monkeypatch.setattr(enterprise.config, "AUTH_EDITION", "saas")
    clerk = ClerkAuth()
    keys = _Keys()
    clerk._jwks_client = keys

    assert clerk.verify_token(CLERK_SHAPED_TOKEN) is None
    assert keys.asked == [CLERK_SHAPED_TOKEN]


@pytest.mark.asyncio
async def test_a_bearer_on_the_request_lane_is_401(enterprise, monkeypatch):
    clerk = ClerkAuth()
    keys = _Keys()
    clerk._jwks_client = keys
    monkeypatch.setattr(hybrid, "get_clerk_auth", lambda: clerk)

    with pytest.raises(HTTPException) as refused:
        await hybrid.get_request_context_hybrid(_request({"authorization": "Bearer " + CLERK_SHAPED_TOKEN}))

    assert refused.value.status_code == 401 and keys.asked == []


# --------------------------------------------------------------------------- a paired device


def test_a_device_with_the_operators_email_is_a_plain_person(enterprise):
    user = device._user_context(SimpleNamespace(id=uuid.uuid4()), (1, enterprise.config.LOCAL_OPERATOR_EMAIL))

    assert user.system_role == device.PERSON_ROLE


def test_the_local_operators_device_is_still_super_admin(enterprise, monkeypatch):
    monkeypatch.setattr(enterprise.config, "AUTH_EDITION", "local")
    user = device._user_context(SimpleNamespace(id=uuid.uuid4()), (1, enterprise.config.LOCAL_OPERATOR_EMAIL))

    assert user.system_role == device.LOCAL_OPERATOR_ROLE


def test_a_device_with_the_operators_email_must_still_be_a_member(enterprise):
    lookups = []
    stand_in = SimpleNamespace(_user_pk_is_workspace_member=lambda db, ws, pk: lookups.append((ws, pk)) or False)
    paired = SimpleNamespace(id=uuid.uuid4(), workspace_id=WS)

    with pytest.raises(HTTPException) as refused:
        device._assert_still_a_member(None, paired, (1, enterprise.config.LOCAL_OPERATOR_EMAIL), stand_in)

    assert refused.value.status_code == 401 and lookups == [(WS, 1)]


@pytest.fixture
def paired_phone(enterprise, monkeypatch):
    """One active paired device in ``WS`` whose person (users.id 7) has the operator's email."""
    # Paired just now: inside the device token's maximum age (PRD-277 US-601).
    phone = SimpleNamespace(id=uuid.uuid4(), workspace_id=WS, user_id=7, paired_at=datetime.now(timezone.utc),
                            status=MobileDeviceStatus.ACTIVE.value, last_seen_at=datetime.now(timezone.utc))
    state = SimpleNamespace(member=True, lookups=[])

    def is_member(db, workspace_id, user_pk):
        state.lookups.append((workspace_id, user_pk))
        return state.member

    monkeypatch.setattr(device, "SessionLocal", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(device, "_find_device", lambda db, token, now: phone if token == DEVICE_TOKEN else None)
    monkeypatch.setattr(device, "_bound_person", lambda db, pk: (pk, enterprise.config.LOCAL_OPERATOR_EMAIL))
    monkeypatch.setattr(device, "_touch_last_seen", lambda *args: None)
    monkeypatch.setattr(hybrid, "_user_pk_is_workspace_member", is_member)
    return state


@pytest.mark.asyncio
async def test_the_operators_email_on_a_phone_is_a_plain_member_through_the_real_dependency(enterprise, paired_phone):
    ctx = await hybrid.get_request_context_hybrid(_request({"authorization": "Bearer " + DEVICE_TOKEN}))

    assert ctx.auth_type == device.DEVICE_AUTH_TYPE and ctx.workspace_id == WS
    assert ctx.user.system_role == device.PERSON_ROLE and ctx.user.system_role != "super_admin"
    assert paired_phone.lookups == [(WS, 7)]  # membership was checked, not skipped
    assert enterprise.operator.calls == 0


@pytest.mark.asyncio
async def test_the_operators_email_on_a_phone_is_refused_once_not_a_member(enterprise, paired_phone):
    paired_phone.member = False

    with pytest.raises(HTTPException) as refused:
        await hybrid.get_request_context_hybrid(_request({"authorization": "Bearer " + DEVICE_TOKEN}))

    assert refused.value.status_code == 401 and paired_phone.lookups == [(WS, 7)]


# --------------------------------------------------------------------------- a server key


@pytest.fixture
def server_key(enterprise, monkeypatch):
    """One server key for ``KEY_WS`` holding the ``tasks:write`` scope only."""
    record = SimpleNamespace(id=uuid.uuid4(), workspace_id=KEY_WS, permissions=["tasks:write"],
                             key_prefix=SERVER_KEY[:11])
    looked_up = []

    def validate_api_key(db, token):
        looked_up.append(token)
        return record if token == SERVER_KEY else None

    monkeypatch.setattr(hybrid, "ApiKeyService",
                        SimpleNamespace(validate_api_key=validate_api_key, check_domain=lambda key, origin: False))
    return SimpleNamespace(record=record, looked_up=looked_up)


@pytest.mark.asyncio
async def test_a_server_key_is_bound_to_its_own_workspace_and_scope(enterprise, server_key):
    ctx = await hybrid.require_task_context("tasks:write")(_request({"authorization": "Bearer " + SERVER_KEY}))

    assert ctx.workspace_id == KEY_WS and ctx.auth_type == "sdk_key"
    assert ctx.user.system_role == "service" and ctx.api_key_id == str(server_key.record.id)
    assert workspace_permission_granted(None, ctx, "members:invite") is False  # a key is never a member
    assert enterprise.operator.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("scope, headers", [
    ("tasks:read", {}),  # a scope the key does not hold
    ("tasks:write", {"x-workspace-id": str(WS)}),  # another workspace
])
async def test_a_server_key_gets_nothing_more(enterprise, server_key, scope, headers):
    with pytest.raises(HTTPException) as refused:
        await hybrid.require_task_context(scope)(_request({"authorization": "Bearer " + SERVER_KEY, **headers}))

    assert refused.value.status_code == 403


@pytest.mark.asyncio
async def test_the_shared_dependency_never_takes_a_server_key(enterprise, server_key):
    with pytest.raises(HTTPException) as refused:
        await hybrid.get_request_context_hybrid(_request({"authorization": "Bearer " + SERVER_KEY}))

    assert refused.value.status_code == 401 and server_key.looked_up == []


# --------------------------------------------------------------------------- the platform API key (D-14)


@pytest.mark.asyncio
async def test_the_platform_key_is_kept_as_it_is(enterprise, monkeypatch):
    """PRD-277 D-14: in enterprise the platform API key stays an admin-level key with no person."""
    monkeypatch.setattr(enterprise.config, "ORCHESTRATOR_API_KEY", PLATFORM_KEY)

    ctx = await hybrid.get_request_context_hybrid(_request({"x-api-key": PLATFORM_KEY, "x-workspace-id": str(WS)}))

    assert ctx.auth_type == "api_key" and ctx.api_key_id == "env" and ctx.workspace_id == WS
    assert ctx.user.system_role == "admin" and ctx.user.email is None
    assert enterprise.operator.calls == 0


@pytest.mark.asyncio
async def test_a_wrong_platform_key_is_401(enterprise, monkeypatch):
    monkeypatch.setattr(enterprise.config, "ORCHESTRATOR_API_KEY", PLATFORM_KEY)

    with pytest.raises(HTTPException) as refused:
        await hybrid.get_request_context_hybrid(_request({"x-api-key": PLATFORM_KEY + "-other"}))

    assert refused.value.status_code == 401 and refused.value.detail == "Invalid API key"


# --------------------------------------------------------------------------- step-up


def _fresh_clerk_context(now):
    return SimpleNamespace(auth_type="clerk", user=SimpleNamespace(raw_claims={"fva": [0, -1], "iat": now}))


@pytest.mark.parametrize("ctx", [
    SimpleNamespace(auth_type="device", user=SimpleNamespace(raw_claims={"source": "mobile_device"})),
    SimpleNamespace(auth_type="api_key", user=SimpleNamespace(raw_claims=None)),
    SimpleNamespace(auth_type="anonymous", user=SimpleNamespace(raw_claims=None)),
])
def test_step_up_refuses_every_context(enterprise, ctx):
    with pytest.raises(HTTPException) as refused:
        step_up.check_step_up(ctx, 10, time.time())

    assert refused.value.status_code == 403
    # PRD-277 US-406: no person who can sign in again here, so never sent to sign in again.
    assert refused.value.detail["code"] == step_up.STEP_UP_UNAVAILABLE_CODE
    assert not isinstance(refused.value, step_up.StepUpRequired)


def test_step_up_never_reads_clerks_freshness(enterprise, monkeypatch):
    now = time.time()
    with pytest.raises(HTTPException) as refused:
        step_up.check_step_up(_fresh_clerk_context(now), 10, now)
    assert refused.value.detail["code"] == step_up.STEP_UP_UNAVAILABLE_CODE

    monkeypatch.setattr(enterprise.config, "AUTH_EDITION", "saas")  # the control: saas passes it
    assert step_up.check_step_up(_fresh_clerk_context(now), 10, now) is None


# --------------------------------------------------------------------------- discovery


def test_discovery_never_offers_clerk(enterprise, monkeypatch):
    # US-602 replaced US-103's refusal: enterprise now says how to sign in, and it is never Clerk.
    monkeypatch.setattr(enterprise.config, "CLERK_PUBLISHABLE_KEY", "pk_live_" + "prd277-stand-in")

    auth, _pairing = discovery._auth_and_pairing("https://automatos.example.org")

    assert discovery.AUTH_MODE_CLERK not in auth["modes"] and auth["clerk"] is None
    assert auth["modes"][0] == discovery.AUTH_MODE_OIDC


# --------------------------------------------------------------------------- what Auto is told


def test_auto_describes_an_organisations_install(enterprise):
    from modules.context.sections import product_facts as facts_module

    facts = facts_module.edition_facts()
    tabs = facts_module.organisation_settings_tabs()

    assert facts.startswith(facts_module.ORGANISATION_EDITION)
    assert "work accounts" in facts and "no plans, credits or billing" in facts
    assert f"Settings has these tabs: {', '.join(tabs)}." in facts
    assert "There is no Billing, Plans or Team Management tab." in facts
    assert "local edition" not in facts and "hosted edition" not in facts


def test_an_organisations_tabs_are_the_hosted_ones_less_billing_and_the_widget_sdk():
    from modules.context.sections import product_facts as facts_module

    tabs = facts_module.organisation_settings_tabs()

    assert "Billing" not in tabs and "Widget SDK" not in tabs and "Profile" not in tabs
    assert {"Webhooks", "API keys", "Model keys"} <= set(tabs)
    assert [tab for tab in facts_module.hosted_settings_tabs() if tab not in {"Billing", "Widget SDK"}] == list(tabs)
