"""PRD-251C Wave 1, US-C104 and US-C106 — no topic repeats by accident, and the bank says so.

On the S0.3b API harness (SQLite). Pinned:

* "too close": the same title once case, punctuation and contractions are folded ("What is a
  Mission?" and "what's a mission"), or content words that overlap at or above
  ``SOCIALS_REPEAT_OVERLAP``; a countdown's next step is not a repeat at the default;
* research's topic is refused when too close to a topic in another plan's bank, to a post of
  the workspace's history within the plan's repeat window (30 days old: refused; 90 days old,
  with a 60-day window: added), or to one the same call added; the refusal names the earlier
  topic or post and its date;
* the threshold is config;
* a person's close topic is added, with a warning naming the earlier post;
* the repeat window is a plan setting: 60 days unless set, 1 to 365;
* the bank names the post each topic is close to, with the post to open; never the topic's own.
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from config import config  # noqa: E402
from core.models.socials import SocialPost, SocialTopic  # noqa: E402
from modules.socials import plans, repeats  # noqa: E402
from modules.tools.discovery import handlers_socials  # noqa: E402
from tests.test_prd251_api import WS_A  # noqa: E402
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402

api = api_harness.api
UTC = timezone.utc
AGENT = {"_agent_id": 7, "_agent_name": "Social Media Director"}
SOURCE = {"kind": "web", "ref": "https://example.com/missions", "label": "Missions page"}


@pytest.fixture
def bank(api, monkeypatch):
    SocialPost.metadata.create_all(api.session.get_bind(), tables=[SocialTopic.__table__])
    monkeypatch.setattr(config, "SOCIALS_REPEAT_OVERLAP", 0.75)
    return api


def _idea(title):
    return {"title": title, "angle": "For founders", "facts": [{"text": "Missions run on the board.", "source": SOURCE}]}


def _research(api, plan, *titles):
    params = {**AGENT, "plan_id": plan["id"], "topics": [_idea(title) for title in titles]}
    answer = asyncio.run(handlers_socials.add_social_topics(api.session, WS_A, params))
    assert answer["success"] is bool(answer["added"]), answer  # F383: a call that adds nothing fails
    return [t["title"] for t in answer["added"]], {r["title"]: r["reason"] for r in answer["refused"]}


def _posted(api, title, days_ago, status="published"):
    post = SocialPost(
        id=uuid.uuid4(), workspace_id=WS_A, created_by="member-1", title=title, status=status, content_hash="0" * 64,
        created_at=datetime.now(UTC) - timedelta(days=days_ago),
    )
    api.session.add(post)
    api.session.commit()
    return post.id


def _day(days_ago):
    moment = datetime.now(UTC) - timedelta(days=days_ago)
    return f"{moment.day} {moment:%b %Y}"


def test_too_close_folds_case_punctuation_and_contractions_and_counts_shared_words():
    assert repeats.fold("What's a Mission?") == repeats.fold("what is a mission") == "what is a mission"
    assert repeats.fold("We can\u2019t wait!") == "we cannot wait"
    assert repeats.words("What is a Mission?") == repeats.words("Missions") == frozenset({"mission"})
    earlier = (repeats.of_topic("What is a Mission?", None, "Countdown"),)
    assert repeats.closest("what's a mission", earlier) is earlier[0]
    assert repeats.closest("Missions", earlier) is earlier[0]  # the same content words
    countdown = (repeats.of_topic("Web Summit in 5 weeks: a countdown", None, "Countdown"),)
    assert repeats.closest("Web Summit in 4 weeks: a countdown", countdown) is None  # 4 of 6 words: a series, not a repeat
    assert repeats.closest("Web Summit in 4 weeks: a countdown", countdown, at_least=0.6) is countdown[0]


def test_research_is_refused_a_topic_another_plans_bank_holds_naming_it_and_its_date(bank):
    other = _create_plan(bank, name="Launch")
    _research(bank, other, "What is a Mission?")
    plan = _create_plan(bank)
    added, refused = _research(bank, plan, "what's a mission", "The roadmap for 2027")
    assert added == ["The roadmap for 2027"]
    assert refused == {"what's a mission": f'too close to "What is a Mission?" in the bank of the plan "Launch" ({_day(0)}): '
                                            "research what neither the history nor the bank covers"}


def test_research_is_refused_a_post_inside_the_plans_repeat_window_only(bank):
    plan = _create_plan(bank)  # research.repeat_after_days: 60
    _posted(bank, "What is a Mission?", days_ago=30)
    _posted(bank, "Playbooks in two minutes", days_ago=90)
    added, refused = _research(bank, plan, "What's a mission?", "Playbooks, in two minutes")
    assert added == ["Playbooks, in two minutes"]  # posted 90 days ago: outside the window
    assert refused["What's a mission?"].startswith(f'too close to "What is a Mission?", posted {_day(30)}: ')


def test_research_is_refused_a_topic_close_to_one_the_same_call_added(bank):
    plan = _create_plan(bank)
    added, refused = _research(bank, plan, "How agents learn from feedback", "How agents learn from your feedback")
    assert added == ["How agents learn from feedback"] and list(refused) == ["How agents learn from your feedback"]


def test_the_threshold_is_config(bank, monkeypatch):
    plan = _create_plan(bank)
    _research(bank, plan, "Web Summit in 5 weeks: a countdown")
    added, _ = _research(bank, plan, "Web Summit in 4 weeks: a countdown")
    assert added == ["Web Summit in 4 weeks: a countdown"]
    monkeypatch.setattr(config, "SOCIALS_REPEAT_OVERLAP", 0.6)
    added, refused = _research(bank, plan, "Web Summit in 3 weeks: a countdown")
    assert added == [] and "Web Summit in 3 weeks: a countdown" in refused


def test_a_persons_close_topic_is_added_with_a_warning_naming_the_post(bank):
    plan = _create_plan(bank)
    _posted(bank, "What is a Mission?", days_ago=9)
    resp = bank.client.post(f"/api/socials/plans/{plan['id']}/topics", json={"title": "what's a mission", "facts": []})
    assert resp.status_code == 201, resp.text
    assert resp.json()["warning"] == f'Posted {_day(9)} as "What is a Mission?".'
    assert resp.json()["origin"] == "person"
    fresh = bank.client.post(f"/api/socials/plans/{plan['id']}/topics", json={"title": "The roadmap for 2027", "facts": []})
    assert fresh.status_code == 201 and fresh.json()["warning"] is None


def test_the_repeat_window_is_a_plan_setting(bank):
    plan = _create_plan(bank)
    assert plan["research"]["repeat_after_days"] == 60
    for days in (0, 366):
        assert bank.client.put(f"/api/socials/plans/{plan['id']}", json={"research": {"repeat_after_days": days}}).status_code == 422
    for days in (True, 0, 366, "60"):
        with pytest.raises(plans.InvalidPlan):
            plans.validate_research({"repeat_after_days": days})
    saved = bank.client.put(f"/api/socials/plans/{plan['id']}", json={"research": {"repeat_after_days": 45}})
    assert saved.status_code == 200 and saved.json()["research"]["repeat_after_days"] == 45
    kept = bank.client.put(f"/api/socials/plans/{plan['id']}", json={"research": {"day": "tue"}})
    assert kept.json()["research"]["repeat_after_days"] == 45  # a save that does not send it keeps it


def _bank(api, plan):
    resp = api.client.get(f"/api/socials/plans/{plan['id']}/topics")
    assert resp.status_code == 200, resp.text
    return {topic["title"]: topic for topic in resp.json()["topics"]}


def test_the_bank_names_the_post_a_topic_is_close_to_never_its_own(bank):
    plan = _create_plan(bank)
    post_id = _posted(bank, "What is a Mission?", days_ago=9)
    for title in ("what's a mission", "The roadmap for 2027"):
        assert bank.client.post(f"/api/socials/plans/{plan['id']}/topics", json={"title": title, "facts": []}).status_code == 201
    listed = _bank(bank, plan)
    assert listed["what's a mission"]["repeat"] == {"post_id": str(post_id), "note": f'Posted {_day(9)} as "What is a Mission?".'}
    assert listed["The roadmap for 2027"]["repeat"] is None

    topic = bank.session.query(SocialTopic).filter(SocialTopic.title == "what's a mission").one()
    topic.used_post_id, topic.used_at = post_id, datetime.now(UTC)  # that very post was made from it
    bank.session.commit()
    assert _bank(bank, plan)["what's a mission"]["repeat"] is None

