"""PRD-251C Wave 1, US-C103 — the workspace's Socials history.

On the S0.3b API harness (SQLite). Pinned:

* what history holds (C5): every post that went out, is approved or scheduled, or waits for
  a person; not a draft, a failed, missed or archived post;
* it spans every plan and the posts people made by hand, and survives a plan's deletion: the
  post stays, and its topic is then its brief's first line (the topic it was made from);
* each item's topic and angle (the bank's topic it used), channels, opening line, date (when
  it went out, else scheduled or planned, else made) and state;
* newest first, ``days`` back (a post scheduled ahead counts) and at most ``limit``;
* another workspace sees nothing;
* ``GET /api/socials/history`` answers it and refuses more than the caps; the agent tool
  ``platform_get_social_history`` reads it (and refuses while Socials is off), and
  ``platform_get_social_plan``'s answer carries it.
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
from core.models.socials import SocialPost, SocialPostTarget, SocialTopic  # noqa: E402
from modules.socials import history  # noqa: E402
from modules.tools.discovery import actions_socials, handlers_socials  # noqa: E402
from tests.test_prd251_api import WS_A, WS_B, WS_OFF  # noqa: E402
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402

api = api_harness.api
UTC = timezone.utc
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
AGENT = {"_agent_id": 7, "_agent_name": "Social Media Director"}
IN_HISTORY = ("rendering", "needs_approval", "changes_requested", "approved", "scheduled", "publishing", "published", "partially_published")
NOT_IN_HISTORY = ("draft", "failed", "missed", "archived")


@pytest.fixture
def posts(api):
    """The harness with the content bank's table (history reads each post's topic)."""
    SocialPost.metadata.create_all(api.session.get_bind(), tables=[SocialTopic.__table__])
    return api


def _naive(moment):
    return moment.astimezone(UTC).replace(tzinfo=None)


def _post(api, title, *, status="published", workspace_id=WS_A, made=NOW - timedelta(days=3), channels=("linkedin",),
          published_at=None, **fields):
    post = SocialPost(id=uuid.uuid4(), workspace_id=workspace_id, created_by="member-1", title=title, status=status,
                      content_hash="0" * 64, created_at=made, format=fields.pop("format", "image"), **fields)
    api.session.add(post)
    for toolkit in channels:
        api.session.add(SocialPostTarget(post_id=post.id, toolkit=toolkit, post_kind="image",
                                         idempotency_key=f"{post.id}:{toolkit}", published_at=published_at))
    api.session.commit()
    return post


def _topic_of(api, plan, post, title, angle):
    topic = SocialTopic(workspace_id=WS_A, campaign_id=uuid.UUID(plan["id"]), title=title, angle=angle, facts=[], formats=["image"],
                        used_post_id=post.id, used_at=NOW, origin="research", created_by="agent:7")
    api.session.add(topic)
    api.session.commit()


def _read(api, workspace_id=WS_A, **window):
    api.session.expire_all()
    return history.history(api.session, workspace_id, now=NOW, **window)


def test_history_holds_what_went_out_is_approved_scheduled_or_waits_for_a_person(posts):
    for status in IN_HISTORY + NOT_IN_HISTORY:
        _post(posts, f"A {status} post", status=status)
    assert sorted(item["state"] for item in _read(posts)) == sorted(IN_HISTORY)
    assert history.HISTORY_STATUSES == IN_HISTORY


def test_history_spans_every_plan_and_hand_made_posts_and_survives_a_plans_deletion(posts):
    first, second = _create_plan(posts), _create_plan(posts, name="A second plan")
    made = _post(posts, "Missions, explained", campaign_id=uuid.UUID(first["id"]), brief="What is a Mission?\nThe board's unit of work",
                 copy={"base": "\nA Mission is how work gets done.\nMore below."}, channels=("linkedin", "instagram"),
                 published_at=NOW - timedelta(days=2))
    _topic_of(posts, first, made, "What is a Mission?", "The board's unit of work")
    other = _post(posts, "The roadmap", campaign_id=uuid.UUID(second["id"]), status="scheduled", scheduled_for=NOW + timedelta(days=2))
    _topic_of(posts, second, other, "The roadmap", None)
    _post(posts, "Behind the scenes", status="needs_approval", brief="Our stand, being built\nPhotos from Lisbon",
          copy={"channels": {"twitter": "Day one at the stand."}}, channels=("twitter",))

    items = {item["title"]: item for item in _read(posts)}
    assert set(items) == {"Missions, explained", "The roadmap", "Behind the scenes"}
    mission = items["Missions, explained"]
    assert (mission["topic"], mission["angle"], mission["plan_id"]) == ("What is a Mission?", "The board's unit of work", first["id"])
    assert (mission["channels"], mission["opening"], mission["state"]) == (["instagram", "linkedin"], "A Mission is how work gets done.", "published")
    assert datetime.fromisoformat(mission["date"]).replace(tzinfo=None) == _naive(NOW - timedelta(days=2))  # when it went out
    by_hand = items["Behind the scenes"]
    assert (by_hand["topic"], by_hand["angle"], by_hand["plan_id"], by_hand["opening"]) == (
        "Our stand, being built", None, None, "Day one at the stand.")

    assert posts.client.delete(f"/api/socials/plans/{first['id']}").status_code == 204
    kept = {item["title"]: item for item in _read(posts)}["Missions, explained"]
    assert (kept["topic"], kept["angle"], kept["plan_id"], kept["state"]) == ("What is a Mission?", None, None, "published")


def test_history_is_newest_first_and_bounded(posts):
    _post(posts, "Forty days ago", made=NOW - timedelta(days=40))
    _post(posts, "Ten days ago", made=NOW - timedelta(days=10))
    _post(posts, "Planned yesterday", status="approved", made=NOW - timedelta(days=20), planned_for=NOW - timedelta(days=1))
    _post(posts, "Scheduled ahead", status="scheduled", made=NOW - timedelta(days=30), scheduled_for=NOW + timedelta(days=3))
    assert [item["title"] for item in _read(posts)] == ["Scheduled ahead", "Planned yesterday", "Ten days ago", "Forty days ago"]
    assert [item["title"] for item in _read(posts, limit=2)] == ["Scheduled ahead", "Planned yesterday"]
    assert [item["title"] for item in _read(posts, days=30)] == ["Scheduled ahead", "Planned yesterday", "Ten days ago"]
    assert len(_read(posts, days=10_000, limit=10_000)) == 4  # past the caps: the caps


def test_another_workspace_sees_nothing(posts):
    _post(posts, "Ours")
    _post(posts, "Theirs", workspace_id=WS_B)
    assert [item["title"] for item in _read(posts)] == ["Ours"]
    assert [item["title"] for item in _read(posts, WS_B)] == ["Theirs"]


def test_the_route_answers_the_workspaces_history_and_refuses_past_the_caps(posts):
    _post(posts, "Recent", made=datetime.now(UTC) - timedelta(days=1))
    _post(posts, "Theirs", workspace_id=WS_B, made=datetime.now(UTC) - timedelta(days=1))
    answer = posts.client.get("/api/socials/history")
    assert answer.status_code == 200, answer.text
    assert [item["title"] for item in answer.json()["posts"]] == ["Recent"] and answer.json()["total"] == 1
    assert posts.client.get(f"/api/socials/history?limit={history.MAX_LIMIT + 1}").status_code == 422
    assert posts.client.get("/api/socials/history?days=0").status_code == 422


def _tool(api, handler, workspace_id=WS_A, **params):
    return asyncio.run(handler(api.session, workspace_id, {**AGENT, **params}))


def test_the_tool_reads_the_history_and_the_plan_answer_carries_it(posts):
    plan = _create_plan(posts)
    _post(posts, "Recent", made=datetime.now(UTC) - timedelta(days=1))
    _post(posts, "Older", made=datetime.now(UTC) - timedelta(days=2))
    read = _tool(posts, handlers_socials.get_social_history, limit=1)
    assert read["success"] is True and [p["title"] for p in read["posts"]] == ["Recent"] and read["count"] == 1
    assert _tool(posts, handlers_socials.get_social_history, days="many")["error"] == "days must be a whole number from 1 to 365."
    assert _tool(posts, handlers_socials.get_social_history, limit=0)["success"] is False
    assert _tool(posts, handlers_socials.get_social_history, WS_OFF)["socials_off"] is True
    answer = _tool(posts, handlers_socials.get_social_plan, plan_id=plan["id"])
    assert [p["title"] for p in answer["history"]] == ["Recent", "Older"]


def test_the_tools_caps_are_the_history_modules():
    assert (actions_socials.HISTORY_MAX_DAYS, actions_socials.HISTORY_MAX_LIMIT) == (history.MAX_DAYS, history.MAX_LIMIT)
