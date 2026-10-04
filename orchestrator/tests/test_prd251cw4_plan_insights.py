"""PRD-251C Wave 4, US-C407 (plan health) and US-C404 (Auto's proposals).

Pinned:

* **Each health check from fixtures:** the bank short of the next two batches' slots; a
  cadence channel not connected; no video in the next seven days; a cap at 80% or more; each
  with its one action; all well, no items.
* **Proposals:** a better time moves the weaker row, a better format turns the weaker row (its
  template left to Auto), the best post's topic becomes research's note (replacing an earlier
  one, never past the notes' limit); too few posts or no lift propose nothing.
* **Through the API** (the S0.3b harness): ``GET /plans/{id}/proposals`` proposes from the
  plan's read posts; a proposal applied through ``PUT /plans/{id}`` changes the plan; a
  proposal ignored changes nothing; ``GET /plans/{id}/health`` lists the items; another
  workspace's plan is a 404.
"""
from __future__ import annotations

import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import core.media_render_quota as render_quota  # noqa: E402
import modules.socials.capabilities as capabilities  # noqa: E402
import modules.socials.media_caps as media_caps  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialCampaign, SocialPost, SocialPostStat, SocialPostTarget  # noqa: E402
from modules.socials import plan_health, plans, proposals  # noqa: E402
from tests.test_prd251_api import WS_A, WS_B, _ctx  # noqa: E402
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402
from tests.test_prd251bw2_plans import bank  # noqa: E402,F401  (the fixture)

api = api_harness.api
UTC = timezone.utc
NOW = datetime(2026, 10, 14, 7, 0, tzinfo=UTC)  # a Wednesday
EVERY_DAY = list(plans.WEEKDAYS)


def _row(row_id, fmt="image", days=EVERY_DAY, time="09:00", channels=("linkedin",)):
    return {"id": row_id, "channels": list(channels), "format": fmt, "days": list(days), "time": time, "template_id": None,
            "length_seconds": None, "kind": None, "visual": None}


def _plan(*rows, rhythm="weekly"):
    return SimpleNamespace(starts_on=date(2026, 10, 1), ends_on=date(2026, 12, 31), timezone="UTC", cadence=list(rows),
                           slot_overrides={}, make={"rhythm": rhythm, "batch_day": "sun", "time": "17:00"},
                           sources=plans.validate_sources({"notes": "Lead with the stand."}))


def _readings(unused=40, connected=("linkedin",), render=None, spend=None):
    return plan_health.Readings(unused_topics=unused, connected=list(connected), render_share=render, spend_share=spend)


# ── health ─────────────────────────────────────────────────────────────────


def test_all_well_is_no_items():
    plan = _plan(_row("r1"), _row("r2", fmt="video", days=["fri"]))
    assert plan_health.health(plan, NOW, _readings()) == []


def test_each_health_check_says_what_and_its_one_action():
    plan = _plan(_row("r1", channels=("linkedin", "tiktok")))  # 7 posts a week, no video
    items = {item.id: item for item in plan_health.health(plan, NOW, _readings(unused=5, render=0.85, spend=0.9))}
    assert list(items) == ["channel_missing", "bank_low", "cap_close", "no_video"]
    assert items["channel_missing"].detail.startswith("tiktok:") and items["channel_missing"].action == plan_health.CONNECT
    assert items["bank_low"].detail == "5 unused topics for the 14 posts of the next two batches." and items["bank_low"].action == plan_health.RESEARCH
    assert "render minutes: 85%" in items["cap_close"].detail and "AI media spend: 90%" in items["cap_close"].detail
    assert items["no_video"].action == plan_health.CADENCE
    assert items["bank_low"].to_dict()["action"] == {"kind": "research", "label": "Research again"}


def test_a_daily_plan_needs_two_days_and_a_cap_below_80_percent_is_fine():
    daily = _plan(_row("r1"), _row("r2", fmt="video", days=["thu"]), rhythm="daily")
    # Wednesday 07:00: the next two days hold three posts (Wednesday's, and Thursday's two).
    assert [item.id for item in plan_health.health(daily, NOW, _readings(unused=3, render=0.79))] == []
    assert [item.id for item in plan_health.health(daily, NOW, _readings(unused=2))] == ["bank_low"]


# ── proposals ──────────────────────────────────────────────────────────────


def _read(title, fmt, row_id, time, engagement, topic=None):
    return proposals.ReadPost(title=title, topic=topic, format=fmt, row_id=row_id, time=time, engagement=engagement)


def test_a_better_time_moves_the_weaker_row():
    plan = _plan(_row("r1", time="09:00"), _row("r2", time="18:00"))
    posts = [_read(f"m{n}", "image", "r1", "09:00", e) for n, e in enumerate((2, 3, 4))]
    posts += [_read(f"e{n}", "image", "r2", "18:00", e) for n, e in enumerate((10, 12, 14))]
    (time,) = [item for item in proposals.proposals(plan, posts) if item.kind == "time"]
    assert time.title == "Post the image row at 18:00"
    assert [row["time"] for row in time.changes["cadence"]] == ["18:00", "18:00"] and len(time.changes) == 1


def test_a_better_format_turns_the_weaker_row_and_the_best_topic_becomes_a_note():
    plan = _plan(_row("r1", fmt="image"), _row("r2", fmt="video", days=["fri"]))
    posts = [_read(f"i{n}", "image", "r1", "09:00", e) for n, e in enumerate((5, 6, 7))]
    posts += [_read(f"v{n}", "video", "r2", "09:00", e, topic="The stand") for n, e in enumerate((20, 25, 30))]
    found = {item.kind: item for item in proposals.proposals(plan, posts)}
    row = found["format"].changes["cadence"][0]
    assert (row["format"], row["template_id"], row["length_seconds"]) == ("video", None, None)
    assert found["angle"].changes["sources"]["notes"] == 'Lead with the stand.\nMore like "The stand": it did best.'


def test_a_new_angle_replaces_the_earlier_one_so_the_notes_stay_bounded():
    plan = _plan(_row("r1"))
    plan.sources = plans.validate_sources({"notes": 'Lead with the stand.\nMore like "Old topic": it did best.'})
    (angle,) = [item for item in proposals.proposals(plan, [_read("x", "image", "r1", "09:00", 9, topic="New topic")]) if item.kind == "angle"]
    assert angle.changes["sources"]["notes"] == 'Lead with the stand.\nMore like "New topic": it did best.'
    plan.sources = plans.validate_sources({"notes": "x" * (plans.NOTES_MAX_CHARS - 5)})
    assert [item.kind for item in proposals.proposals(plan, [_read("x", "image", "r1", "09:00", 9, topic="New topic")])] == []


def test_few_posts_or_no_lift_propose_no_change_of_time_or_format():
    plan = _plan(_row("r1", time="09:00"), _row("r2", time="18:00"))
    few = [_read("a", "image", "r1", "09:00", 1), _read("b", "image", "r2", "18:00", 50)]
    assert [item.kind for item in proposals.proposals(plan, few)] == ["angle"]
    even = [_read(f"x{n}", "image", row, clock, 10) for n, (row, clock) in enumerate([("r1", "09:00")] * 3 + [("r2", "18:00")] * 3)]
    assert [item.kind for item in proposals.proposals(plan, even)] == ["angle"]


# ── through the API ────────────────────────────────────────────────────────


@pytest.fixture
def insights(bank, monkeypatch):  # noqa: F811
    SocialPost.metadata.create_all(bank.session.get_bind(), tables=[SocialPostStat.__table__])
    monkeypatch.setattr(render_quota, "render_quota", lambda db, workspace, now=None: SimpleNamespace(used_minutes=90.0, quota_minutes=100.0))
    monkeypatch.setattr(media_caps, "media_spend", lambda db, workspace, post_id=None, now=None: SimpleNamespace(month_usd=1.0, monthly_cap_usd=30.0))
    monkeypatch.setattr(capabilities, "social_channels", lambda db, ws: [SimpleNamespace(toolkit="linkedin")])
    return bank


def _went_out(api, plan, row_id, clock, engagement, fmt="image", days_ago=3):
    post = SocialPost(id=uuid.uuid4(), workspace_id=WS_A, created_by="u", title=f"{row_id} {clock} {engagement}", content_hash="0" * 64,
                      status="published", format=fmt, campaign_id=uuid.UUID(plan["id"]), slot_key=f"{row_id}|2026-10-{10 + engagement:02d}|{clock}")  # one slot each
    went = datetime.now(UTC) - timedelta(days=days_ago)
    target = SocialPostTarget(id=uuid.uuid4(), post_id=post.id, toolkit="linkedin", post_kind="image", action_plan={},
                              idempotency_key=f"sp:{post.id}:linkedin", status="published", remote_id="urn:li:share:1", published_at=went)
    stat = SocialPostStat(workspace_id=WS_A, post_id=post.id, target_id=target.id, reading=1, read_at=went, numbers={"reactions": engagement},
                          source_action="LINKEDIN_LIST_REACTIONS")
    api.session.add_all([post, target, stat])
    api.session.commit()


def test_a_proposal_applied_changes_the_plan_and_one_ignored_changes_nothing(insights):
    rows = [{"channels": ["linkedin"], "format": "image", "days": EVERY_DAY, "time": "09:00"},
            {"channels": ["linkedin"], "format": "image", "days": EVERY_DAY, "time": "18:00"}]
    plan = _create_plan(insights, cadence=rows)
    for clock, row_id, values in (("09:00", "r1", (2, 3, 4)), ("18:00", "r2", (10, 12, 14))):
        for value in values:
            _went_out(insights, plan, row_id, clock, value)
    answer = insights.client.get(f"/api/socials/plans/{plan['id']}/proposals")
    assert answer.status_code == 200, answer.text
    found = {item["kind"]: item for item in answer.json()["proposals"]}
    assert found["time"]["title"] == "Post the image row at 18:00"
    insights.client.get(f"/api/socials/plans/{plan['id']}/proposals")  # asked again, ignored
    stored = insights.session.get(SocialCampaign, uuid.UUID(plan["id"]))
    insights.session.refresh(stored)
    assert [row["time"] for row in stored.cadence] == ["09:00", "18:00"]  # nothing applied by itself
    applied = insights.client.put(f"/api/socials/plans/{plan['id']}", json=found["time"]["changes"])
    assert applied.status_code == 200, applied.text
    assert [row["time"] for row in applied.json()["cadence"]] == ["18:00", "18:00"]


def test_the_health_route_lists_the_items_and_another_workspace_is_404(insights):
    plan = _create_plan(insights, cadence=[{"channels": ["linkedin", "tiktok"], "format": "image", "days": EVERY_DAY, "time": "09:00"}])
    answer = insights.client.get(f"/api/socials/plans/{plan['id']}/health")
    assert answer.status_code == 200, answer.text
    assert [item["id"] for item in answer.json()["items"]] == ["channel_missing", "bank_low", "cap_close", "no_video"]
    insights.ctx = _ctx(WS_B)
    assert insights.client.get(f"/api/socials/plans/{plan['id']}/health").status_code == 404
    assert insights.client.get(f"/api/socials/plans/{plan['id']}/proposals").status_code == 404
