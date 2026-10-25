"""Theme voice — ``GET /api/voice/live-status`` says whether Auto Live can start, and why not.

The host voice provider offers Live from this read, so it carries the mint's own gates,
read without refusing (``modules/voice/availability.py``), in the mint's order:

* the plan (SaaS only, D34): a plan without the voice family is ``available: false``,
  ``reason: "voice_live_plan_required"`` (the mint's own code); Business is available;
  the local edition is never plan-gated;
* then the platform switch, the Retell credentials, the workspace's toggle and the
  month's minutes, each with its own code;
* every gate passing is ``available: true``, ``reason: null``; the settings card's fields
  stay as they were.

The route is a plain ``def`` now (every read in it is a synchronous database read).
Through the real router (a FastAPI app with the request context and the session
overridden); no database, no Retell, no network.
"""
from __future__ import annotations

import inspect
import uuid
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.voice_retell as vr
from config import config
from core.models.workspaces import Workspace
from modules.voice import availability, live_settings, mint, voice_meter
from modules.voice.live_settings import RetellCredentials
from modules.voice.voice_meter import MeterReading

STATUS = "/api/voice/live-status"
WS = uuid.UUID("7b1f4d2e-3c5e-4f9a-8b2c-7e1ce0a00001")
CAP_MINUTES = 100
ARMED = RetellCredentials("key_fx", "sig_fx", "agent_fx")
UNARMED = RetellCredentials("key_fx", "", "agent_fx")
NOTHING_USED = MeterReading(0, 0, 10)
CAP_USED = MeterReading(CAP_MINUTES, 0, 10)


@pytest.fixture
def saas(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")


def _gates(monkeypatch, *, platform=True, creds=ARMED, reading=NOTHING_USED):
    monkeypatch.setattr(live_settings, "voice_live_enabled", lambda: platform)
    monkeypatch.setattr(live_settings, "retell_credentials", lambda: creds)
    monkeypatch.setattr(voice_meter, "monthly_meter", lambda db, workspace_id: reading)


def _workspace(plan, *, enabled=True):
    return NS(id=WS, plan=plan, settings={"voice_live": {"enabled": enabled, "monthly_cap_minutes": CAP_MINUTES}})


def _db(workspace) -> MagicMock:
    db = MagicMock()

    def query(model):
        q = MagicMock()
        q.filter.return_value.first.return_value = workspace if model is Workspace else None
        return q

    db.query.side_effect = query
    return db


def _status(workspace) -> dict:
    app = FastAPI()
    app.include_router(vr.router)
    app.dependency_overrides[vr.get_request_context_hybrid] = lambda: NS(
        workspace_id=WS, user=NS(id=7, system_role="user"), auth_type="clerk"
    )
    app.dependency_overrides[vr.get_db] = lambda: _db(workspace)
    response = TestClient(app).get(STATUS)
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------- the plan (D34)


@pytest.mark.parametrize("plan", ["basic", "pro", None, "no-such-tier"])
def test_saas_without_the_voice_family_is_unavailable_with_the_mints_code(saas, monkeypatch, plan):
    _gates(monkeypatch)
    body = _status(_workspace(plan))
    assert body["available"] is False
    assert body["reason"] == mint.VOICE_LIVE_PLAN_REQUIRED == "voice_live_plan_required"


def test_saas_business_is_available(saas, monkeypatch):
    _gates(monkeypatch)
    body = _status(_workspace("business"))
    assert body["available"] is True
    assert body["reason"] is None


@pytest.mark.parametrize("plan", ["basic", None])
def test_the_local_edition_is_never_plan_gated(local, monkeypatch, plan):
    _gates(monkeypatch)
    assert _status(_workspace(plan))["available"] is True


def test_the_plan_is_the_first_reason_on_saas(saas, monkeypatch):
    _gates(monkeypatch, platform=False, creds=UNARMED, reading=CAP_USED)
    assert _status(_workspace("basic", enabled=False))["reason"] == "voice_live_plan_required"


# --------------------------------------------------------------------------- the other gates, in the mint's order


@pytest.mark.parametrize(
    "gates, enabled, reason",
    [
        ({"platform": False, "creds": UNARMED}, False, "voice_live_platform_off"),
        ({"creds": UNARMED}, False, "voice_live_not_armed"),
        ({"reading": CAP_USED}, False, "voice_live_workspace_off"),
        ({"reading": CAP_USED}, True, "voice_live_cap_reached"),
    ],
)
def test_each_gate_names_itself_and_the_first_one_wins(local, monkeypatch, gates, enabled, reason):
    _gates(monkeypatch, **gates)
    body = _status(_workspace("business", enabled=enabled))
    assert body["available"] is False
    assert body["reason"] == reason


def test_the_settings_card_fields_stay(local, monkeypatch):
    _gates(monkeypatch, reading=MeterReading(12, 1, 10))
    body = _status(_workspace("business"))
    assert body["platform_enabled"] is True and body["armed"] is True and body["workspace_enabled"] is True
    assert body["used_minutes"] == 12 and body["cap_minutes"] == CAP_MINUTES and body["active_calls"] == 1
    assert "key_fx" not in str(body) and "sig_fx" not in str(body)


def test_the_route_is_a_plain_def_off_the_event_loop():
    assert not inspect.iscoroutinefunction(vr.voice_live_status)


# --------------------------------------------------------------------------- the rule on its own


def test_live_unavailable_reason_follows_the_mints_order():
    every = dict(plan_allows=True, platform_enabled=True, armed=True, workspace_enabled=True, cap_allows=True)
    assert availability.live_unavailable_reason(availability.LiveGates(**every)) is None
    assert availability.live_unavailable_reason(availability.LiveGates(**{**every, "cap_allows": False})) == (
        availability.LIVE_REASON_CAP_REACHED
    )
    nothing = {name: False for name in every}
    assert availability.live_unavailable_reason(availability.LiveGates(**nothing)) == availability.LIVE_REASON_PLAN


def test_the_plan_rule_reads_without_refusing(saas):
    assert mint.plan_allows_voice_live("business") is True
    assert mint.plan_allows_voice_live("basic") is False
