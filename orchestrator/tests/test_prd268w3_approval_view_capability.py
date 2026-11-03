"""PRD-268 US-C24 — discovery ``approval_view``, and a Mission plan's view by the id it carries.

* Discovery lists ``approval_view`` on ``GET /api/v1/approval-grants/{grant_id}/view`` in both
  editions with no setup. The route's router is a required one (``RouterSpec("api.approval_views")``),
  so a build that lists the capability serves its route; ``test_prd266_discovery.py``'s live probe
  checks every capability's route against the running app.
* A Mission plan has no grant row: its view's ``grant_id`` is the Mission's id (W2 US-C20a), and the
  same route resolves that id (``services.approval_views.view_of``: a number is a grant, a UUID is a
  Mission). The card's view and the route's are one view. The route's offers Refuse and Allow once
  only while the plan awaits approval; another workspace's Mission, and an id that names nothing,
  is 404; a viewer is 403; a grant still resolves by its number.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from config import HOSTED_PUBLIC_API_HOST, config  # noqa: E402
from modules.policy import rule_sentences as rules  # noqa: E402
from services import approval_views as views  # noqa: E402
from services.mobile import discovery  # noqa: E402

VIEW_PATH = "/api/v1/approval-grants/{grant_id}/view"
MANIFEST = _ORCH / "reports" / "route-manifest.json"
CAPABILITY = "approval_view"
GOAL = "Launch the autumn blend"
STEPS = ("Draft the post", "Publish it")
POLICY = "always_ask"


# --------------------------------------------------------------------------- the capability


@pytest.fixture(params=["saas", "local"])
def edition(request, monkeypatch):
    """Either edition, set up as discovery needs it (a local install with no Expo answer stored)."""
    from services.push import consent

    host = "clerk.prd268w3.test"
    monkeypatch.setattr(config, "AUTH_EDITION", request.param)
    monkeypatch.setattr(config, "CLERK_PUBLISHABLE_KEY", "pk_live_" + base64.b64encode(f"{host}$".encode()).decode())
    monkeypatch.setattr(config, "FRONTEND_URL", "https://studio.prd268w3.test")
    monkeypatch.setattr(config, "PUBLIC_API_HOST", HOSTED_PUBLIC_API_HOST)
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "off")
    monkeypatch.setattr(consent, "get_system_setting", lambda category, key, default=None: default)
    return request.param


def test_the_capability_is_listed_with_its_route_in_either_edition(edition):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import well_known

    app = FastAPI()
    app.include_router(well_known.router)
    response = TestClient(app).get("/.well-known/automatos.json")

    assert response.status_code == 200, response.text
    assert CAPABILITY in response.json()["capabilities"]
    assert discovery.CAPABILITY_ROUTES[CAPABILITY] == ("GET", VIEW_PATH)
    assert CAPABILITY not in discovery.CONFIGURED_CAPABILITIES          # nothing to set up: always served


def test_the_capabilitys_route_is_served_by_a_required_router_and_listed():
    from api import approval_views
    from router_manifest import MANIFEST_ROUTERS, RouterSpec

    method, path = discovery.CAPABILITY_ROUTES[CAPABILITY]
    served = {(m, route.path) for route in approval_views.router.routes for m in getattr(route, "methods", ())}
    assert (method, path) in served
    assert RouterSpec("api.approval_views") in MANIFEST_ROUTERS                       # optional=False
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": method, "path": path} in manifest["routes"]


# --------------------------------------------------------------------------- the id a view carries


@pytest.fixture
def resolved(monkeypatch):
    """Which reader ``view_of`` hands an id to, with what."""
    calls = []
    monkeypatch.setattr(views, "grant_view", lambda db, ws, grant_id, language: calls.append(("grant", grant_id)) or {})
    monkeypatch.setattr(views, "mission_plan_view",
                        lambda db, ws, mission_id, language: calls.append(("mission", mission_id)) or {})
    return calls


def test_a_number_names_a_grant_and_a_uuid_names_a_mission(resolved):
    mission_id = uuid.uuid4()

    assert views.view_of(None, "ws", "42", "en") == {}
    assert views.view_of(None, "ws", str(mission_id), "en") == {}
    assert resolved == [("grant", 42), ("mission", mission_id)]


@pytest.mark.parametrize("said", ["", "abc", "4.2", "-1", "²", "42 OR 1=1", str(views.MAX_GRANT_ID + 1),
                                  "1" * 40])
def test_an_id_that_is_neither_names_nothing(resolved, said):
    assert views.view_of(None, "ws", said, "en") is None
    assert resolved == []


def test_an_unreadable_policy_names_none_and_is_logged(monkeypatch, caplog):
    from modules.tools.discovery import mission_approval_rule as rule

    def _unreadable(db, ws, mission_id):
        raise RuntimeError("settings unreadable")

    monkeypatch.setattr(rule, "_holding_policy", _unreadable)
    with caplog.at_level(logging.ERROR):
        assert rule.policy_holding(None, "ws", "m-1") is None
    assert "could not read the policy holding mission m-1" in caplog.text


# --------------------------------------------------------------------------- the route, on a real database


def _person(db, ws, role):
    """A users row with a Clerk id, a ``role`` member of ``ws``."""
    from sqlalchemy import text

    tag = uuid.uuid4().hex[:12]
    pk = db.execute(text("INSERT INTO users (username, email, is_active, clerk_user_id) "
                         "VALUES (:u, :e, true, :c) RETURNING id"),
                    {"u": f"prd268w3-view-{tag}", "e": f"prd268w3-view-{tag}@t.test", "c": f"user_w3view_{tag}"}).scalar()
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :uid, :role, true)"), {"ws": str(ws), "uid": pk, "role": role})
    db.flush()
    return SimpleNamespace(pk=pk, clerk_id=f"user_w3view_{tag}", email=f"prd268w3-view-{tag}@t.test")


def _client(db, ws, person):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import approval_views
    from core.auth.dependencies import RequestContext, UserContext
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    user = UserContext(id=person.clerk_id, email=person.email, system_role="user", clerk_user_id=person.clerk_id)
    app = FastAPI()
    app.include_router(approval_views.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(workspace_id=ws, user=user,
                                                                                  auth_type="clerk")
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def _plan(steps=STEPS):
    return {"tasks": [{"title": title, "sequence_number": n} for n, title in enumerate(steps, start=1)]}


def _mission(db, ws, state="awaiting_approval", plan=None, policy=POLICY, ceiling=None, estimate=1.0):
    """A Mission held for approval under ``policy`` (its plan, by default the two ``STEPS``) and its board card."""
    from core.models.core import BoardTask
    from core.models.orchestration import OrchestrationRun
    from core.services.approval_policy import set_approval_policy
    from services.ticket_numbers import format_number

    set_approval_policy(db, ws, policy=policy, approval_dollar_ceiling=ceiling)
    run = OrchestrationRun(workspace_id=ws, goal=GOAL, state=state, created_by="user_test",
                           plan=_plan() if plan is None else plan, config={"approval_estimated_cost_usd": estimate})
    db.add(run)
    db.flush()
    card = BoardTask(workspace_id=ws, title=f"Mission: {GOAL}", status="review", source_type="orchestration",
                     orchestration_run_id=run.id)
    db.add(card)
    db.flush()
    db.refresh(card)
    return run, format_number(card.workspace_seq)


@pytest.fixture
def ws(seed_workspace):
    return uuid.UUID(seed_workspace())


@pytest.mark.parametrize("role", ["owner", "admin", "editor"])
def test_a_mission_plans_view_resolves_by_the_mission_id(db_session, ws, role):
    run, number = _mission(db_session, ws)

    response = _client(db_session, ws, _person(db_session, ws, role)).get(VIEW_PATH.format(grant_id=run.id))

    assert response.status_code == 200, response.text
    view = response.json()
    rule = rules.approval_policy_rule(POLICY)
    assert number and view["grant_id"] == str(run.id)
    assert view["asked_by"] == {"kind": "auto", "name": "Auto"}
    assert view["action"] == {"verb": "run", "summary": GOAL}
    assert view["target"] == {"label": number}
    assert view["preview"] == {"kind": "text", "body": "1. Draft the post\n2. Publish it"}
    assert view["rule"] == {"id": rule, "sentence": rules.STRINGS["en"][rule]}
    assert view["answers"] == ["refuse", "allow_once"]


@pytest.mark.parametrize("policy, ceiling, estimate", [("always_ask", None, 1.0), ("auto_below_budget", 5.0, 7.0)])
def test_the_cards_view_carries_the_id_the_route_resolves_and_they_are_one_view(db_session, ws, policy, ceiling,
                                                                                  estimate):
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    run, number = _mission(db_session, ws, policy=policy, ceiling=ceiling, estimate=estimate)
    created = {"success": True, "mission_id": run.id, "state": run.state, "awaiting_approval": True, "goal": GOAL,
               "task_count": len(STEPS), "number": number, "approval_policy": policy,
               "tasks": [{"title": title, "sequence": n} for n, title in enumerate(STEPS, start=1)]}
    card_view = ToolResultFormatter.format_for_frontend(created, "platform_create_mission")["mission_approval"][
        "approval_view"]

    route_view = _client(db_session, ws, _person(db_session, ws, "owner")).get(
        VIEW_PATH.format(grant_id=card_view["grant_id"])).json()

    assert route_view == card_view and route_view["rule"]["id"] == rules.approval_policy_rule(policy)


@pytest.mark.parametrize("state", ["running", "paused", "completed", "cancelled"])
def test_a_plan_no_longer_waiting_offers_no_answers_and_names_no_policy(db_session, ws, state):
    run, _number = _mission(db_session, ws, state=state)

    view = _client(db_session, ws, _person(db_session, ws, "owner")).get(VIEW_PATH.format(grant_id=run.id)).json()

    assert view["grant_id"] == str(run.id) and view["answers"] == []
    assert view["rule"] == {"id": rules.FALLBACK, "sentence": rules.STRINGS["en"][rules.FALLBACK]}


def test_the_view_lists_the_plans_first_ten_steps_as_the_card_does(db_session, ws):
    steps = [f"Step {n}" for n in range(1, 13)]
    run, _number = _mission(db_session, ws, plan=_plan(steps))

    view = _client(db_session, ws, _person(db_session, ws, "owner")).get(VIEW_PATH.format(grant_id=run.id)).json()

    assert views.PLAN_STEPS_SHOWN == 10
    assert view["preview"]["body"].splitlines() == [f"{n}. Step {n}" for n in range(1, 11)]


@pytest.mark.parametrize("plan", [{}, {"tasks": None}, {"tasks": {"title": "not a list"}}, {"tasks": ["text"]}])
def test_a_plan_with_no_steps_it_can_read_has_no_preview(db_session, ws, plan):
    run, _number = _mission(db_session, ws, plan=plan)

    response = _client(db_session, ws, _person(db_session, ws, "owner")).get(VIEW_PATH.format(grant_id=run.id))

    assert response.status_code == 200, response.text
    assert "preview" not in response.json() and response.json()["action"]["summary"] == GOAL


def test_a_mission_plans_view_is_in_the_callers_language(db_session, ws):
    run, _number = _mission(db_session, ws)

    view = _client(db_session, ws, _person(db_session, ws, "editor")).get(
        VIEW_PATH.format(grant_id=run.id), headers={"Accept-Language": "pt-BR"}).json()

    rule = rules.approval_policy_rule(POLICY)
    assert view["rule"] == {"id": rule, "sentence": rules.STRINGS["pt-BR"][rule]}


def test_another_workspaces_mission_and_an_id_naming_nothing_are_404(db_session, ws, seed_workspace):
    elsewhere, _number = _mission(db_session, uuid.UUID(seed_workspace()))
    client = _client(db_session, ws, _person(db_session, ws, "owner"))

    for said in (elsewhere.id, uuid.uuid4(), "not-a-view", views.MAX_GRANT_ID + 1):
        response = client.get(VIEW_PATH.format(grant_id=said))
        assert response.status_code == 404 and GOAL not in response.text


def test_a_viewer_is_refused_a_mission_plans_view(db_session, ws):
    from core.auth.workspace_admin import ANSWERER_ONLY

    run, _number = _mission(db_session, ws)
    response = _client(db_session, ws, _person(db_session, ws, "viewer")).get(VIEW_PATH.format(grant_id=run.id))

    assert (response.status_code, response.json()["detail"]) == (403, ANSWERER_ONLY)
    assert GOAL not in response.text


def test_a_grant_still_resolves_by_its_number(db_session, ws):
    from core.models.approval_grants import ApprovalGrant
    from modules.policy.policy_document import RISK_EXTERNAL  # the class the gate stores (tool_grants.py)

    grant = ApprovalGrant(workspace_id=ws, status="pending", kind="approval", subject_type="tool_call",
                          subject_id=f"platform_send_email:{uuid.uuid4().hex[:32]}", tool_name="platform_send_email",
                          risk_tier=RISK_EXTERNAL, expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                          details={"action": "platform_send_email", "action_description": "Send an email.",
                                   "params": {"recipient_email": "aisha@example.com"}})
    db_session.add(grant)
    db_session.flush()

    view = _client(db_session, ws, _person(db_session, ws, "owner")).get(VIEW_PATH.format(grant_id=grant.id)).json()

    assert view["grant_id"] == str(grant.id) and view["action"] == {"verb": "send", "summary": "Send an email."}
    assert view["answers"] == ["refuse", "allow_once"]
