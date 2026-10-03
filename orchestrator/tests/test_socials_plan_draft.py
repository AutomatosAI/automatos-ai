"""PRD-251B (3 Oct 2026 pass) — Plan with Auto: a plan drafted from what the person says.

* ``checked_draft``: only connected channels (case aside), a format, days and time a plan can
  take (else the form's default, saying so), dates from today (never in the past, in order,
  a year at most), a starting row when Auto gave none, the suggestions as topics (titled,
  deduplicated, at most ten), and the research sources and notes from the request, never
  the model; with nothing connected it says to connect a channel;
* ``draft``: one call, and one retry when the JSON is unusable; a second failure and a
  timeout each say so;
* ``POST /api/socials/plans/draft``: the draft from the caller's workspace, 504 on a
  timeout, 502 when the answer cannot be read or the model cannot be reached, 422 for an
  empty request; an unknown timezone is UTC; the route is in the manifest.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from datetime import date
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

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.socials_plan_draft as draft_api  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from modules.socials import plan_draft  # noqa: E402

WS = uuid.UUID("00000000-0000-0000-0000-0000000005a1")
TODAY = date(2026, 10, 3)  # a Saturday
REQUEST = "Three posts this week about our autumn colour offer on Instagram. One should be a customer review."
MANIFEST = _ORCH / "reports" / "route-manifest.json"
INSTAGRAM = {"toolkit": "instagram", "label": "Instagram", "kinds": ["image", "reel", "carousel"]}


def _ctx(channels=(INSTAGRAM,), request=REQUEST):
    return plan_draft.DraftContext(request=request, today=TODAY, timezone="Europe/London", channels=list(channels))


def _answer(**over):
    answer = {
        "name": "Autumn colour week",
        "goal": "Fill the colour chairs on quiet weekdays.",
        "audience": "Local regulars",
        "starts_on": "2026-10-05",
        "ends_on": "2026-10-11",
        "cadence": [{"channels": ["instagram"], "format": "image", "days": ["mon", "wed", "fri"], "time": "10:00"}],
        "topics": [
            {"title": "20% off colour, Monday to Thursday", "angle": "The offer, plainly", "formats": ["image"]},
            {"title": "Sarah's review", "angle": "A customer's words", "formats": ["image"]},
        ],
    }
    return {**answer, **over}


# ---------------------------------------------------------------------------
# checked_draft
# ---------------------------------------------------------------------------


def test_a_good_answer_is_the_plan_as_the_plan_page_opens_it():
    out = plan_draft.checked_draft(_answer(), _ctx(), plan_draft.DraftSources(website=False))

    plan = out["plan"]
    assert (plan["name"], plan["goal"], plan["audience"]) == ("Autumn colour week", "Fill the colour chairs on quiet weekdays.", "Local regulars")
    assert (plan["starts_on"], plan["ends_on"], plan["timezone"]) == ("2026-10-05", "2026-10-11", "Europe/London")
    assert plan["cadence"] == [{"channels": ["instagram"], "format": "image", "days": ["mon", "wed", "fri"], "time": "10:00",
                                "template_id": None, "length_seconds": None}]
    # The sources are the person's choice, never the model's; their words go to research.
    assert plan["sources"] == {"knowledge": True, "website": False, "deliverables": True, "github": False,
                               "notes": plan_draft.NOTES_LEAD + REQUEST, "never_say": []}
    assert [t["title"] for t in out["topics"]] == ["20% off colour, Monday to Thursday", "Sarah's review"]
    assert out["warnings"] == []


def test_what_a_plan_cannot_carry_is_dropped_or_defaulted_and_said():
    row = {"channels": ["Instagram", "tiktok"], "format": "gif", "days": ["fri", "someday", "mon"], "time": "9am"}
    out = plan_draft.checked_draft(_answer(cadence=[row]), _ctx(), plan_draft.DraftSources())

    (kept,) = out["plan"]["cadence"]
    assert kept["channels"] == ["instagram"]  # case aside; tiktok is not connected here
    assert (kept["format"], kept["days"], kept["time"]) == ("image", ["mon", "fri"], "09:00")
    assert "Row 1: its format, time could not be used, so the form's default is." in out["warnings"]
    assert "Row 1: tiktok is not connected here, so it was left out." in out["warnings"]


@pytest.mark.parametrize(
    "starts, ends, expected, said",
    [
        ("2026-09-01", None, ("2026-10-03", "2026-10-09"), "cannot start in the past"),
        ("2026-10-10", "2026-10-04", ("2026-10-10", "2026-10-16"), "last day was before its first"),
        (None, "2028-01-01", ("2026-10-03", "2027-10-03"), "at most 366 days"),
        (None, None, ("2026-10-03", "2026-10-09"), None),
    ],
)
def test_the_dates_run_from_today_in_order_and_at_most_a_year(starts, ends, expected, said):
    out = plan_draft.checked_draft(_answer(starts_on=starts, ends_on=ends), _ctx(), plan_draft.DraftSources())
    assert (out["plan"]["starts_on"], out["plan"]["ends_on"]) == expected
    assert any(said in w for w in out["warnings"]) if said else out["warnings"] == []


def test_no_rhythm_gives_a_row_to_start_from_and_no_channel_says_to_connect_one():
    out = plan_draft.checked_draft(_answer(cadence=[]), _ctx(), plan_draft.DraftSources())
    assert out["plan"]["cadence"][0]["channels"] == ["instagram"] and out["plan"]["cadence"][0]["days"] == ["mon", "wed", "fri"]

    bare = plan_draft.checked_draft({}, _ctx(channels=()), plan_draft.DraftSources())
    assert bare["plan"]["cadence"][0]["channels"] == []
    assert bare["warnings"][0].startswith("No social channel is connected yet")
    assert bare["plan"]["name"] == "Plan from 03 Oct" and bare["plan"]["goal"] == REQUEST


def test_the_suggestions_are_titled_deduplicated_and_at_most_ten():
    many = [{"title": f"Idea {n}", "formats": ["image", "gif"]} for n in range(14)]
    raw = _answer(topics=[{"title": "  "}, {"title": "Idea 0"}, *many, "not a topic"])
    topics = plan_draft.checked_draft(raw, _ctx(), plan_draft.DraftSources())["topics"]
    assert [t["title"] for t in topics] == [f"Idea {n}" for n in range(10)]
    assert all(t["formats"] == [] or t["formats"] == ["image"] for t in topics) and topics[0]["formats"] == []


# ---------------------------------------------------------------------------
# draft: the model
# ---------------------------------------------------------------------------


class _Model:
    """The workspace's model: answers from a queue, records what it was asked."""

    def __init__(self, answers, delay=0.0):
        self.answers, self.asked, self.delay = list(answers), [], delay

    async def generate_response(self, messages, tools=None):
        self.asked.append(messages)
        if self.delay:
            await asyncio.sleep(self.delay)
        answer = self.answers.pop(0)
        return SimpleNamespace(content=answer if isinstance(answer, str) else json.dumps(answer))


def _draft(model, timeout=5.0):
    return asyncio.run(plan_draft.draft(_ctx(), plan_draft.DraftSources(), lambda: model, timeout))


def test_one_call_gives_the_draft_and_the_model_is_told_what_it_has():
    model = _Model([_answer()])
    out = _draft(model)
    assert out["plan"]["name"] == "Autumn colour week" and len(model.asked) == 1
    material = json.loads(model.asked[0][1]["content"])
    assert (material["request"], material["today"], material["weekday"]) == (REQUEST, "2026-10-03", "sat")
    assert material["channels"] == [INSTAGRAM]


def test_unusable_json_is_asked_for_once_more_then_fails():
    model = _Model(["not json", _answer()])
    assert _draft(model)["plan"]["name"] == "Autumn colour week" and len(model.asked) == 2
    with pytest.raises(plan_draft.DraftFailed):
        _draft(_Model(["nope", "still nope"]))


def test_a_model_that_does_not_answer_in_time_says_so():
    with pytest.raises(plan_draft.DraftTimedOut, match="did not answer within"):
        _draft(_Model([_answer()], delay=1.0), timeout=0.01)


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    state = SimpleNamespace(model=_Model([_answer()]), bodies=[])

    def context(db, workspace_id, body):
        state.bodies.append((workspace_id, body))
        return _ctx(request=body.request)

    monkeypatch.setattr(draft_api, "draft_context", context)
    monkeypatch.setattr(plan_draft, "llm_factory", lambda workspace_id: (lambda: state.model))
    app = FastAPI()
    app.include_router(draft_api.router, prefix="/api/socials")
    app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(
        workspace_id=WS, user=UserContext(id="owner-1", clerk_user_id="clerk-owner-1", system_role="user"), auth_type="clerk")
    app.dependency_overrides[get_db] = lambda: None
    app.dependency_overrides[draft_api.CAN_CREATE.dependency] = lambda: None
    state.client = TestClient(app)
    return state


def test_the_route_answers_the_draft_from_the_callers_workspace(client):
    body = {"request": REQUEST, "timezone": "Europe/London", "sources": {"knowledge": True, "website": False, "deliverables": True}}
    resp = client.client.post("/api/socials/plans/draft", json=body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["plan"]["name"] == "Autumn colour week" and resp.json()["plan"]["sources"]["website"] is False
    ((workspace_id, seen),) = client.bodies
    assert workspace_id == WS and seen.request == REQUEST


class _Unreachable:
    async def generate_response(self, messages, tools=None):
        raise ConnectionError("provider down")


@pytest.mark.parametrize(
    "model, timeout, status",
    [
        (lambda: _Model(["nope", "still nope"]), 5, 502),
        (lambda: _Unreachable(), 5, 502),
        (lambda: _Model([_answer()], delay=1.0), 0.01, 504),
    ],
)
def test_the_route_says_when_auto_could_not_draft(client, monkeypatch, model, timeout, status):
    client.model = model()
    monkeypatch.setattr(draft_api.config, "SOCIALS_COMPOSE_TIMEOUT_SECONDS", timeout)
    resp = client.client.post("/api/socials/plans/draft", json={"request": REQUEST})
    assert resp.status_code == status
    assert "provider down" not in resp.text  # the detail goes to the log, never to the person


@pytest.mark.parametrize("request_text", ["", "   "])
def test_an_empty_request_is_refused(client, request_text):
    assert client.client.post("/api/socials/plans/draft", json={"request": request_text}).status_code == 422
    assert client.bodies == []


def test_an_unknown_timezone_is_utc():
    assert draft_api.today_in("Mars/Olympus")[1] == "UTC"
    assert draft_api.today_in("Europe/Lisbon")[1] == "Europe/Lisbon"


def test_the_route_is_in_the_manifest():
    routes = json.loads(MANIFEST.read_text())["routes"]
    assert ("POST", "/api/socials/plans/draft") in {(r.get("method"), r["path"]) for r in routes}
