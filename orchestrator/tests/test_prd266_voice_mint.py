"""PRD-266 US-008a (§4.7, D32, D34) — the Auto Live mint for the phone.

``POST /api/voice/web-call`` through the real router (a FastAPI app with the request
context and the session overridden; Retell answered by a recorded body from
``fixtures/voice/``, read by the real ``create_web_call``):

* SaaS: a plan without ``families.voice`` (Basic, Pro, no plan, an unknown plan) is
  403 ``{"code": "voice_live_plan_required", "message": "Auto Live is on the Business
  plan"}`` before Retell is asked; Business mints; enterprise declares no families and is
  unrestricted. The local edition is never plan-gated. The gate is a route dependency.
* The answer carries Retell's ``transport``, ``url``, ``ice_servers`` and ``expires_at``
  when Retell sends them, nothing else of Retell's body, and omits the ones it does not
  send (the v2 call object carries none); a field of another type is dropped.
* The local operator (the anonymous lane) and a paired device mint as their person's
  ``users.id``, read from the lanes' own claims (their ``user.id`` is an email).
* The mint stays on Retell's ``/v2/create-web-call``.
* ``mint_web_call`` is at most 50 code lines, every helper in ``modules/voice/mint.py``
  too, and that module stays under 400 lines.
* Discovery lists ``voice_live``, mapped to the mint.

DB-free: no database, no Retell, no network.
"""
from __future__ import annotations

import ast
import asyncio
import json
import uuid
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
from urllib.parse import urlparse

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.voice_retell as vr
from config import HOSTED_PUBLIC_API_HOST, config, load_plan_tiers
from core.auth import device, hybrid
from core.auth.dependencies import RequestContext
from core.models.core import Chat
from core.models.voice_calls import VoiceCall
from core.models.workspaces import Workspace
from modules.voice import live_settings, mint, retell_api, voice_meter
from modules.voice.live_settings import RetellCredentials
from modules.voice.voice_meter import MeterReading
from services import plan_tiers
from services.mobile import discovery

ORCH = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "voice"
TRANSPORT_FIXTURE = "retell_create_web_call_transport.json"
CALL_OBJECT_FIXTURE = "retell_create_web_call_call_object.json"
TRANSPORT_FIELDS = ("transport", "url", "ice_servers", "expires_at")
MINT = "/api/voice/web-call"
PLAN_REFUSAL = {
    "detail": {"code": "voice_live_plan_required", "message": "Auto Live is on the Business plan"}
}
WS = uuid.UUID("6a0f3c1e-2b4d-4e8f-9a1b-266008a00001")
VOICE_THREAD = uuid.UUID("6a0f3c1e-2b4d-4e8f-9a1b-266008a0c4a7")
DEVICE_ID = uuid.UUID("6a0f3c1e-2b4d-4e8f-9a1b-266008a0d001")
CLERK_PK = 7
OPERATOR_PK = 1
DEVICE_PERSON_PK = 6
DEVICE_EMAIL = "phone@example.com"
DEFAULT_TIERS = load_plan_tiers(env_override="")
ROUTE_CODE_LINE_LIMIT = 50
MINT_MODULE_LINE_LIMIT = 400


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- doubles


@pytest.fixture
def saas(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")


@pytest.fixture
def armed(monkeypatch):
    """Auto Live on, Retell's three credentials present, nothing used this month."""
    monkeypatch.setattr(live_settings, "voice_live_enabled", lambda: True)
    monkeypatch.setattr(
        live_settings, "retell_credentials", lambda: RetellCredentials("key_fx", "sig_fx", "agent_fx")
    )
    monkeypatch.setattr(voice_meter, "monthly_meter", lambda db, workspace_id: MeterReading(0, 0, 10))


@pytest.fixture
def vendor(monkeypatch):
    """Retell answering create-web-call with ``state["body"]``; each URL and payload is kept."""
    state = {"body": _fixture(TRANSPORT_FIXTURE), "posted": [], "payloads": []}

    class _Response:
        status_code = 201
        text = "{}"

        def __init__(self, body: dict) -> None:
            self._body = body

        def json(self) -> dict:
            return self._body

    class _Client:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> bool:
            return False

        async def post(self, url, json, headers):
            state["posted"].append(url)
            state["payloads"].append(json)
            return _Response(state["body"])

    monkeypatch.setattr(retell_api, "httpx", NS(AsyncClient=_Client, HTTPError=httpx.HTTPError))
    return state


def _workspace(plan):
    return NS(id=WS, plan=plan, settings={"voice_live": {"enabled": True}})


def _db(workspace) -> MagicMock:
    """A session double: the workspace by id, the caller's voice thread, no older calls."""
    db = MagicMock()

    def query(model):
        q = MagicMock()
        if model is Workspace:
            q.filter.return_value.first.return_value = workspace
        elif model is Chat:
            q.filter.return_value.order_by.return_value.first.return_value = NS(id=VOICE_THREAD)
        elif model is VoiceCall:
            q.filter.return_value.update.return_value = 0
        return q

    db.query.side_effect = query
    return db


def _minted_rows(db: MagicMock) -> list:
    return [call.args[0] for call in db.add.call_args_list if isinstance(call.args[0], VoiceCall)]


def _clerk_ctx():
    return NS(workspace_id=WS, user=NS(id=CLERK_PK, system_role="user"), auth_type="clerk")


def _local_operator_ctx(monkeypatch) -> RequestContext:
    """The anonymous lane's context, built by hybrid's own binder over a stubbed row read."""
    row = {"id": OPERATOR_PK, "name": "Operator", "username": "operator", "avatar_url": None}
    monkeypatch.setattr(hybrid, "_resolve_local_operator", lambda db, email: {**row, "email": email})
    user = hybrid._local_operator_user_context(MagicMock())
    return RequestContext(workspace_id=WS, user=user, auth_type="anonymous")


def _device_ctx() -> RequestContext:
    """A paired device's context, built by the device lane's own binder."""
    user = device._user_context(NS(id=DEVICE_ID), (DEVICE_PERSON_PK, DEVICE_EMAIL))
    return RequestContext(workspace_id=WS, user=user, auth_type=device.DEVICE_AUTH_TYPE)


def _post(ctx, db) -> httpx.Response:
    app = FastAPI()
    app.include_router(vr.router)
    app.dependency_overrides[vr.get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[vr.get_db] = lambda: db
    return TestClient(app).post(MINT, json={})


# --------------------------------------------------------------------------- the plan (D34)


@pytest.mark.parametrize("plan", ["basic", "pro", None, "no-such-tier"])
def test_saas_without_the_voice_family_is_refused_with_the_code(saas, armed, vendor, plan):
    db = _db(_workspace(plan))

    response = _post(_clerk_ctx(), db)

    assert response.status_code == 403
    assert response.json() == PLAN_REFUSAL
    assert vendor["posted"] == []  # refused before Retell is asked
    assert _minted_rows(db) == []


def test_saas_business_mints_with_retells_transport_fields(saas, armed, vendor):
    recorded = _fixture(TRANSPORT_FIXTURE)
    db = _db(_workspace("business"))

    response = _post(_clerk_ctx(), db)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "call_id": recorded["call_id"],
        "access_token": recorded["access_token"],
        "chat_id": str(VOICE_THREAD),
        **{name: recorded[name] for name in TRANSPORT_FIELDS},
    }
    assert [row.call_id for row in _minted_rows(db)] == [recorded["call_id"]]


def test_saas_enterprise_declares_no_families_and_is_unrestricted(saas, armed, vendor):
    assert "families" not in DEFAULT_TIERS["enterprise"]

    response = _post(_clerk_ctx(), _db(_workspace("enterprise")))

    assert response.status_code == 200, response.text


@pytest.mark.parametrize("plan", ["basic", None])
def test_the_local_edition_is_never_plan_gated(local, armed, vendor, monkeypatch, plan):
    response = _post(_local_operator_ctx(monkeypatch), _db(_workspace(plan)))

    assert response.status_code == 200, response.text


def test_the_plan_gate_is_a_dependency_of_the_route():
    route = next(r for r in vr.router.routes if getattr(r, "path", None) == MINT)

    assert vr._require_voice_live_plan in [dep.call for dep in route.dependant.dependencies]


@pytest.mark.parametrize(
    "plan, expected",
    [("basic", False), ("pro", False), ("business", True), ("enterprise", True), (None, False),
     ("no-such-tier", False)],
)
def test_family_enabled_reads_the_exposure_rules(plan, expected):
    assert plan_tiers.family_enabled(plan, "voice", tiers=DEFAULT_TIERS) is expected


def test_family_enabled_agrees_with_exposure_for_plan_where_a_tier_declares_families():
    for plan in ("basic", "pro", "business"):
        families = plan_tiers.exposure_for_plan(plan, tiers=DEFAULT_TIERS)["families"]
        assert plan_tiers.family_enabled(plan, "voice", tiers=DEFAULT_TIERS) is bool(families.get("voice"))


def test_an_enterprise_tier_that_declares_families_is_gated_by_them():
    override = {**DEFAULT_TIERS, "enterprise": {**DEFAULT_TIERS["enterprise"], "families": {"voice": False}}}

    assert plan_tiers.family_enabled("enterprise", "voice", tiers=override) is False


# --------------------------------------------------------------------------- Retell's fields (D32)


def test_a_response_without_the_transport_fields_omits_them(saas, armed, vendor):
    vendor["body"] = _fixture(CALL_OBJECT_FIXTURE)
    recorded = vendor["body"]

    response = _post(_clerk_ctx(), _db(_workspace("business")))

    assert response.status_code == 200, response.text
    assert response.json() == {
        "call_id": recorded["call_id"],
        "access_token": recorded["access_token"],
        "chat_id": str(VOICE_THREAD),
    }


def test_a_transport_field_of_another_type_is_dropped(vendor):
    vendor["body"] = {**_fixture(TRANSPORT_FIXTURE), "expires_at": True, "ice_servers": {"urls": "x"}}

    web_call = asyncio.run(retell_api.create_web_call("key_fx", {}))

    assert web_call.transport_fields() == {
        "transport": vendor["body"]["transport"],
        "url": vendor["body"]["url"],
    }


def test_the_mint_stays_on_retells_v2_create_web_call(saas, armed, vendor):
    assert urlparse(retell_api.RETELL_CREATE_WEB_CALL_URL).path == "/v2/create-web-call"

    response = _post(_clerk_ctx(), _db(_workspace("business")))

    assert response.status_code == 200, response.text
    assert vendor["posted"] == [retell_api.RETELL_CREATE_WEB_CALL_URL]


# --------------------------------------------------------------------------- the local callers


@pytest.mark.parametrize("lane", ["local_operator", "device"])
def test_the_local_operator_and_a_paired_device_mint_as_their_person(local, armed, vendor, monkeypatch, lane):
    if lane == "local_operator":
        ctx, person, super_admin = _local_operator_ctx(monkeypatch), OPERATOR_PK, True
    else:
        ctx, person, super_admin = _device_ctx(), DEVICE_PERSON_PK, False
    assert ctx.user.id != person  # the lane's user.id is an email; the PK rides its claims
    db = _db(_workspace("basic"))

    response = _post(ctx, db)

    assert response.status_code == 200, response.text
    (row,) = _minted_rows(db)
    assert (row.user_id, row.workspace_id, row.chat_id) == (person, WS, str(VOICE_THREAD))
    assert row.is_super_admin is super_admin
    assert vendor["payloads"][0]["retell_llm_dynamic_variables"]["user_id"] == str(person)


# --------------------------------------------------------------------------- shape and discovery


def _code_lines(path: Path, node: ast.AST) -> int:
    """Code lines of a function as the wave's gate counts them: no blanks, comments or docstring."""
    lines = path.read_text(encoding="utf-8").splitlines()
    first = node.body[0]
    docstring = set()
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        docstring = set(range(first.lineno, first.end_lineno + 1))
    return sum(
        1 for number in range(node.lineno, node.end_lineno + 1)
        if lines[number - 1].strip() and not lines[number - 1].strip().startswith("#") and number not in docstring
    )


def _functions(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_the_mint_route_is_short_over_short_helpers():
    route_file = ORCH / "api" / "voice_retell.py"
    mint_file = ORCH / "modules" / "voice" / "mint.py"

    assert _code_lines(route_file, _functions(route_file)["mint_web_call"]) <= ROUTE_CODE_LINE_LIMIT
    long_helpers = {
        name: _code_lines(mint_file, node) for name, node in _functions(mint_file).items()
        if _code_lines(mint_file, node) > ROUTE_CODE_LINE_LIMIT
    }
    assert long_helpers == {}
    assert len(mint_file.read_text(encoding="utf-8").splitlines()) <= MINT_MODULE_LINE_LIMIT
    assert mint.VOICE_LIVE_PLAN_REQUIRED == PLAN_REFUSAL["detail"]["code"]


def test_discovery_lists_voice_live_on_the_mint_route(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "off")
    monkeypatch.setattr(config, "FRONTEND_URL", "http://localhost:3000")
    monkeypatch.setattr(config, "PUBLIC_API_HOST", HOSTED_PUBLIC_API_HOST)
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text())

    document = discovery.build_discovery("http://192.168.1.20:8000/")

    assert "voice_live" in document["capabilities"]
    assert discovery.CAPABILITY_ROUTES["voice_live"] == ("POST", MINT)
    assert {"method": "POST", "path": MINT} in manifest["routes"]
