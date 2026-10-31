"""PRD-268 US-C20a (§6.11.4, K10, K16, K21) — ApprovalView@1 and the rule sentences.

* ``modules/policy/rule_sentences.py``: every rule id (``confirm.<risk class>``, each posture ×
  risk-class pair its routing table sends to ask, ``approval_policy.<policy>``) has a sentence in
  English and in Portuguese (Brazil); ``rule_sentence`` never answers machine text;
* the chat's ``tool_approval`` and ``mission_approval`` cards carry ``approval_view`` (the
  decorator ``carries_the_approval_view`` on ``format_for_frontend``): who asked, the action, the
  target, the preview and the rule; nothing past the card digest's cut appears, a newline in a
  model-written value forges no line, and ``answers`` never holds ``always_allow``; a view that
  cannot be built leaves today's card;
* a Mission plan's answer names the policy that held it (``names_the_approval_policy``), against
  Postgres;
* ``GET /api/v1/approval-grants/{grant_id}/view`` (a plain def on ``api/approval_views.py``, in the
  route manifest): an owner, admin or editor reads it, a viewer is 403, another workspace's grant
  and a question are 404; a decided or lapsed grant offers no answers; the words follow
  ``Accept-Language``.
"""
from __future__ import annotations

import asyncio
import inspect
import json
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

from core.services.approval_policy import VALID_POLICIES  # noqa: E402
from modules.policy import rule_sentences as rules  # noqa: E402
from modules.policy.policy_document import (  # noqa: E402
    RISK_DESTRUCTIVE, RISK_EXTERNAL, RISK_INTERNAL_WRITE, RISK_PUBLISH, RISK_READ, VALID_POSTURES, PolicyDocument,
)
from modules.tools.formatting import approval_view  # noqa: E402
from modules.tools.formatting.result_formatter import ToolResultFormatter  # noqa: E402

LANGUAGES = ("en", "pt-BR")
RISKS = (RISK_READ, RISK_INTERNAL_WRITE, RISK_EXTERNAL, RISK_DESTRUCTIVE, RISK_PUBLISH)
TAIL = "PAST-THE-CUT"
LONG = "x" * 200 + TAIL
SHOWN = "x" * 117 + "…"
VIEW_PATH = "/api/v1/approval-grants/{grant_id}/view"
MANIFEST = _ORCH / "reports" / "route-manifest.json"


def _every_rule_id() -> set:
    """The ids the table must hold, read from the policy plane itself."""
    asked = {rules.posture_rule(p, r) for p in VALID_POSTURES for r in RISKS
             if PolicyDocument(posture=p, agents_inherit_admin=False, route_overrides={}).route_for(r) == "ask"}
    return ({rules.confirm_rule(r) for r in RISKS} | asked
            | {rules.approval_policy_rule(policy) for policy in VALID_POLICIES})


def _ask(**overrides) -> dict:
    """A confirmation ask with its grant, as the gate returns it to the formatter."""
    ask = {"success": False, "requires_confirmation": True, "action": "platform_send_email",
           "permission_level": "write", "message": "This action (write) requires confirmation.",
           "params": {"recipient_email": "aisha@example.com", "body": LONG, "_agent_id": 9, "_agent_name": "Scribe",
                      "_secret_plumbing": TAIL},
           "grant_id": 42, "risk_class": RISK_EXTERNAL, "risk_tier": "human_in_the_loop", "requires_approval": True}
    return {**ask, **overrides}


def _plan(**overrides) -> dict:
    """create_mission's answer for a plan awaiting approval."""
    plan = {"success": True, "mission_id": "abc-123", "state": "awaiting_approval", "awaiting_approval": True,
            "goal": "Launch the autumn blend", "task_count": 2, "number": "#0282",
            "tasks": [{"title": "Draft the post", "sequence": 1}, {"title": LONG, "sequence": 2}],
            "approval_policy": "auto_below_budget"}
    return {**plan, **overrides}


@pytest.fixture
def described(monkeypatch):
    """The registry's description of the asked action, without loading every action."""
    monkeypatch.setattr(approval_view, "_described", lambda action: "Send an email to one person.")


# --------------------------------------------------------------------------- the rule sentences


def test_every_rule_id_has_a_sentence_in_both_languages():
    expected = _every_rule_id()

    assert set(rules.RULE_IDS) == expected
    assert set(rules.STRINGS["en"]) == set(rules.STRINGS["pt-BR"])
    for language in LANGUAGES:
        missing = sorted(rule for rule in expected if not rules.STRINGS[language].get(rule, "").strip())
        assert missing == [], f"{language} has no sentence for {missing}"
    assert len({rules.STRINGS["en"][r] for r in rules.RULE_IDS}) == len(rules.RULE_IDS)


@pytest.mark.parametrize("language", LANGUAGES)
def test_a_rule_sentence_is_never_machine_text(language):
    for rule in (*_every_rule_id(), "confirm.unknown", "posture.balanced.read", "nonsense"):
        sentence = rules.rule_sentence(rule, language)
        assert sentence and rule not in sentence
        assert not any(mark in sentence for mark in ("_", "=", "{", "posture", "confirm."))
    assert rules.rule_sentence("confirm.unknown", language) == rules.STRINGS[language][rules.FALLBACK]


def test_a_language_without_a_table_reads_english():
    rule = rules.confirm_rule(RISK_DESTRUCTIVE)

    assert rules.rule_sentence(rule, "fr") == rules.rule_sentence(rule, "en") == rules.STRINGS["en"][rule]
    assert rules.rule_sentence(rule, "pt-BR") != rules.rule_sentence(rule, "en")


# --------------------------------------------------------------------------- the cards


def test_a_tool_card_carries_its_view(described):
    card = ToolResultFormatter.format_for_frontend(_ask(), "platform_send_email")["tool_approval"]
    view = card["approval_view"]

    assert card["grant_id"] == 42 and card["action"] == "platform_send_email"      # today's card stands
    assert view["grant_id"] == "42"
    assert view["asked_by"] == {"kind": "agent", "name": "Scribe"}
    assert view["action"] == {"verb": "send", "summary": "Send an email to one person."}
    assert view["target"] == {"label": "aisha@example.com", "detail": "recipient_email"}
    assert view["preview"] == {"kind": "text", "body": f"recipient_email: aisha@example.com\nbody: {SHOWN}"}
    rule = rules.confirm_rule(RISK_EXTERNAL)
    assert view["rule"] == {"id": rule, "sentence": rules.STRINGS["en"][rule]}
    assert view["answers"] == ["refuse", "allow_once"]


def test_a_mission_card_carries_its_view():
    card = ToolResultFormatter.format_for_frontend(_plan(), "platform_create_mission")["mission_approval"]
    view = card["approval_view"]

    assert card["mission_id"] == "abc-123" and card["task_count"] == 2               # today's card stands
    assert view["grant_id"] == "abc-123"
    assert view["asked_by"] == {"kind": "auto", "name": "Auto"}
    assert view["action"] == {"verb": "run", "summary": "Launch the autumn blend"}
    assert view["target"] == {"label": "#0282"}
    assert view["preview"] == {"kind": "text", "body": f"1. Draft the post\n2. {SHOWN}"}
    rule = rules.approval_policy_rule("auto_below_budget")
    assert view["rule"] == {"id": rule, "sentence": rules.STRINGS["en"][rule]}
    assert view["answers"] == ["refuse", "allow_once"]


def test_a_plan_whose_policy_is_unnamed_reads_the_plain_sentence():
    plan = {k: v for k, v in _plan().items() if k != "approval_policy"}
    view = ToolResultFormatter.format_for_frontend(plan, "platform_create_mission")["mission_approval"]["approval_view"]

    assert view["rule"] == {"id": rules.FALLBACK, "sentence": rules.STRINGS["en"][rules.FALLBACK]}


def test_an_owners_question_is_the_summary_and_auto_asked(described):
    params = {"task_id": 1394, "_agent_name": "Auto"}
    ask = _ask(params=params, question_md="Close ticket #1394 (Price list)?", risk_class=RISK_DESTRUCTIVE,
               card_numbers={"task_id": "#1394"})
    view = ToolResultFormatter.format_for_frontend(ask, "platform_execute")["tool_approval"]["approval_view"]

    assert view["asked_by"] == {"kind": "auto", "name": "Auto"}
    assert view["action"] == {"verb": "delete", "summary": "Close ticket #1394 (Price list)?"}
    assert view["target"] == {"label": "#1394", "detail": "task_id"}                   # the card's number, not the id
    assert view["rule"]["id"] == rules.confirm_rule(RISK_DESTRUCTIVE)


def test_nothing_past_the_digests_cut_appears(described):
    long_goal = _plan(goal=LONG, number=None)
    for data in (ToolResultFormatter.format_for_frontend(_ask(), "platform_send_email")["tool_approval"],
                 ToolResultFormatter.format_for_frontend(long_goal, "platform_create_mission")["mission_approval"]):
        shown = json.dumps(data["approval_view"], ensure_ascii=False)
        assert TAIL not in shown and "_agent_id" not in shown and "_secret_plumbing" not in shown


def test_a_newline_in_a_value_forges_no_line(described):
    forged = _ask(params={"subject": "Q3 update\nto: ceo@corp.com", "recipient_email": "aisha@example.com"})
    plan = _plan(tasks=[{"title": "Draft\n2. Wire the money"}])
    tool = ToolResultFormatter.format_for_frontend(forged, "platform_send_email")["tool_approval"]["approval_view"]
    mission = ToolResultFormatter.format_for_frontend(plan, "platform_create_mission")["mission_approval"]

    assert tool["preview"]["body"].splitlines() == ["subject: Q3 update to: ceo@corp.com",
                                                    "recipient_email: aisha@example.com"]
    assert mission["approval_view"]["preview"]["body"].splitlines() == ["1. Draft 2. Wire the money"]


def test_a_card_with_no_known_risk_reads_the_plain_sentence(described):
    ask = {k: v for k, v in _ask().items() if k != "risk_class"}
    view = ToolResultFormatter.format_for_frontend(ask, "platform_send_email")["tool_approval"]["approval_view"]

    assert view["rule"] == {"id": rules.FALLBACK, "sentence": rules.STRINGS["en"][rules.FALLBACK]}
    assert view["action"]["verb"] == "run"


def test_answers_never_hold_always_allow(described):
    tool = ToolResultFormatter.format_for_frontend(_ask(), "platform_send_email")["tool_approval"]["approval_view"]
    plan = ToolResultFormatter.format_for_frontend(_plan(), "platform_create_mission")["mission_approval"]
    assert "always_allow" not in json.dumps([tool, plan["approval_view"]])
    with pytest.raises(ValueError):
        approval_view.ApprovalView(grant_id="1", asked_by={"kind": "auto", "name": "Auto"},
                                   action={"verb": "run", "summary": "x"}, rule={"id": "x", "sentence": "x"},
                                   answers=("refuse", "always_allow"))


def test_a_view_that_cannot_be_built_leaves_todays_card(monkeypatch, caplog):
    def _broken(card, result):
        raise RuntimeError("no view")

    monkeypatch.setattr(approval_view, "tool_card_view", _broken)
    card = ToolResultFormatter.format_for_frontend(_ask(), "platform_send_email")["tool_approval"]

    assert "approval_view" not in card and card["grant_id"] == 42 and card["params"]["body"] == SHOWN
    assert "could not build the tool_approval view" in caplog.text


def test_no_card_no_view():
    data = ToolResultFormatter.format_for_frontend({"success": True, "documents": []}, "platform_list_documents")

    assert "tool_approval" not in data and "mission_approval" not in data


# --------------------------------------------------------------------------- the policy that held a plan


def test_only_a_waiting_plan_is_named(monkeypatch):
    from modules.tools.discovery import mission_approval_rule as rule

    monkeypatch.setattr(rule, "_holding_policy", lambda db, ws, mission_id: "always_ask")

    async def _made(db, ws, params):
        return dict(params)

    named = rule.names_the_approval_policy(_made)
    waiting = asyncio.run(named(None, "ws", {"success": True, "mission_id": "m", "awaiting_approval": True}))
    running = asyncio.run(named(None, "ws", {"success": True, "mission_id": "m", "awaiting_approval": False}))

    assert waiting["approval_policy"] == "always_ask"
    assert "approval_policy" not in running


def test_an_unreadable_policy_names_none(monkeypatch, caplog):
    from modules.tools.discovery import mission_approval_rule as rule

    def _unreadable(db, ws, mission_id):
        raise RuntimeError("settings unreadable")

    monkeypatch.setattr(rule, "_holding_policy", _unreadable)

    async def _made(db, ws, params):
        return dict(params)

    answer = asyncio.run(rule.names_the_approval_policy(_made)(None, "ws", {"mission_id": "m",
                                                                           "awaiting_approval": True}))
    assert "approval_policy" not in answer and "could not read the policy" in caplog.text


@pytest.mark.parametrize("policy, ceiling, estimate, named", [
    ("always_ask", None, 1.0, "always_ask"),
    ("auto_below_budget", 5.0, 7.0, "auto_below_budget"),
    ("auto_below_budget", 5.0, 3.0, None),                        # under the ceiling: something else held it
    ("full_auto", None, 1.0, "full_auto"),                        # the autonomy gate is off
])
def test_the_policy_named_is_the_one_that_would_hold_the_plan(db_session, seed_workspace, policy, ceiling,
                                                               estimate, named):
    from core.models.orchestration import OrchestrationRun
    from core.services.approval_policy import set_approval_policy
    from modules.tools.discovery.mission_approval_rule import _holding_policy

    ws = uuid.UUID(seed_workspace())
    set_approval_policy(db_session, ws, policy=policy, approval_dollar_ceiling=ceiling)
    run = OrchestrationRun(workspace_id=ws, goal="Launch", state="awaiting_approval", created_by="user_test",
                           config={"approval_estimated_cost_usd": estimate})
    db_session.add(run)
    db_session.flush()

    assert _holding_policy(db_session, ws, run.id) == named


def test_a_plan_asked_for_as_a_plan_names_no_policy(db_session, seed_workspace):
    from core.models.orchestration import OrchestrationRun
    from modules.tools.discovery.mission_approval_rule import _holding_policy

    ws = uuid.UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal="Launch", state="awaiting_approval", created_by="user_test",
                           config={"plan_only": True, "approval_estimated_cost_usd": 9.0})
    db_session.add(run)
    db_session.flush()

    assert _holding_policy(db_session, ws, run.id) is None                 # always_ask, yet the plan was asked for


# --------------------------------------------------------------------------- the view route


def _person(db, ws, role):
    """A users row with a Clerk id, a ``role`` member of ``ws``."""
    from sqlalchemy import text

    tag = uuid.uuid4().hex[:12]
    pk = db.execute(text("INSERT INTO users (username, email, is_active, clerk_user_id) "
                         "VALUES (:u, :e, true, :c) RETURNING id"),
                    {"u": f"prd268w2-view-{tag}", "e": f"prd268w2-view-{tag}@t.test", "c": f"user_view_{tag}"}).scalar()
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :uid, :role, true)"), {"ws": str(ws), "uid": pk, "role": role})
    db.flush()
    return SimpleNamespace(pk=pk, clerk_id=f"user_view_{tag}", email=f"prd268w2-view-{tag}@t.test")


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


def _grant(db, ws, **fields):
    from core.models.approval_grants import ApprovalGrant

    row = {"status": "pending", "kind": "approval", "subject_type": "tool_call",
           "subject_id": f"platform_send_email:{uuid.uuid4().hex[:32]}", "tool_name": "platform_send_email",
           "risk_tier": RISK_EXTERNAL, "expires_at": datetime.now(timezone.utc) + timedelta(hours=1),
           "details": {"action": "platform_send_email", "action_description": "Send an email to one person.",
                       "params": {"recipient_email": "aisha@example.com", "body": LONG}}}
    grant = ApprovalGrant(workspace_id=ws, **{**row, **fields})
    db.add(grant)
    db.flush()
    return grant


@pytest.fixture
def ws(seed_workspace):
    return uuid.UUID(seed_workspace())


@pytest.mark.parametrize("role", ["owner", "admin", "editor"])
def test_an_owner_admin_or_editor_reads_the_grants_view(db_session, ws, role):
    grant = _grant(db_session, ws)

    response = _client(db_session, ws, _person(db_session, ws, role)).get(VIEW_PATH.format(grant_id=grant.id))

    assert response.status_code == 200, response.text
    view = response.json()
    rule = rules.confirm_rule(RISK_EXTERNAL)
    assert view["grant_id"] == str(grant.id) and view["asked_by"] == {"kind": "auto", "name": "Auto"}
    assert view["action"] == {"verb": "send", "summary": "Send an email to one person."}
    assert view["target"] == {"label": "aisha@example.com", "detail": "recipient_email"}
    assert view["preview"]["body"] == f"recipient_email: aisha@example.com\nbody: {SHOWN}" and TAIL not in response.text
    assert view["rule"] == {"id": rule, "sentence": rules.STRINGS["en"][rule]}
    assert view["answers"] == ["refuse", "allow_once"] and view["expires_at"]


def test_a_viewer_is_refused(db_session, ws):
    from core.auth.workspace_admin import ANSWERER_ONLY

    grant = _grant(db_session, ws)
    response = _client(db_session, ws, _person(db_session, ws, "viewer")).get(VIEW_PATH.format(grant_id=grant.id))

    assert (response.status_code, response.json()["detail"]) == (403, ANSWERER_ONLY)


def test_another_workspaces_grant_and_a_question_are_404(db_session, ws, seed_workspace):
    elsewhere = _grant(db_session, uuid.UUID(seed_workspace()))
    question = _grant(db_session, ws, kind="question", question_md="Which café opens first?")
    client = _client(db_session, ws, _person(db_session, ws, "owner"))

    for grant in (elsewhere, question):
        response = client.get(VIEW_PATH.format(grant_id=grant.id))
        assert response.status_code == 404 and TAIL not in response.text


def test_a_decided_or_lapsed_grant_offers_no_answers(db_session, ws):
    decided = _grant(db_session, ws, status="granted")
    lapsed = _grant(db_session, ws, expires_at=datetime.now(timezone.utc) - timedelta(minutes=1))
    client = _client(db_session, ws, _person(db_session, ws, "owner"))

    views = [client.get(VIEW_PATH.format(grant_id=g.id)).json() for g in (decided, lapsed)]

    assert [v["answers"] for v in views] == [[], []]
    assert all(v["rule"]["id"] == rules.confirm_rule(RISK_EXTERNAL) for v in views)


def test_a_board_grant_says_its_policy_in_the_callers_language(db_session, ws):
    old = _grant(db_session, ws, subject_type="board_task", subject_id="999999999", tool_name=None, details={},
                 reason="board task requires approval under 'auto_below_budget' policy")
    worded = _grant(db_session, ws, subject_type="board_task", subject_id="999999998", tool_name=None, details={},
                    reason=rules.STRINGS["en"][rules.approval_policy_rule("full_auto")])
    client = _client(db_session, ws, _person(db_session, ws, "editor"))

    said = [client.get(VIEW_PATH.format(grant_id=g.id), headers={"Accept-Language": "pt-BR"}).json()
            for g in (old, worded)]

    assert [v["rule"]["id"] for v in said] == [rules.approval_policy_rule(p) for p in ("auto_below_budget", "full_auto")]
    assert [v["rule"]["sentence"] for v in said] == [rules.STRINGS["pt-BR"][v["rule"]["id"]] for v in said]
    assert all(v["action"]["verb"] == "run" and "preview" not in v for v in said)


def test_the_route_is_a_plain_def_in_the_route_manifest():
    from api import approval_views
    from router_manifest import MANIFEST_ROUTERS, RouterSpec

    assert not inspect.iscoroutinefunction(approval_views.get_approval_view)
    assert RouterSpec("api.approval_views") in MANIFEST_ROUTERS
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "GET", "path": VIEW_PATH} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])
