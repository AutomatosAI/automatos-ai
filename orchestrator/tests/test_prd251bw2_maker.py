"""PRD-251B Wave 2, US-B205, US-B206 and US-B204's schedule — posts made on their day,
slots that pass, and research that comes round.

On the S0.3b API harness (SQLite), the composer, the channel registry and the
notifications faked. Pinned:

* a due slot is made: a draft with the slot's key, planned time and plan, the topic marked
  used, written by the composer with the topic's facts as its sources, its channel set and
  sent for approval (approvers told); "Today's posts are ready" goes out once a day; the
  slot is no longer due;
* a slot is made once: a second tick claiming it finds it taken (the unique key) and
  leaves the next topic unused;
* an empty bank skips the slot and says so once a day; no connected channel posting the
  format, or no render minutes left, skips it too: nothing half-made; ``max_per_day`` holds;
* the late policy: ``next_slot`` moves an unapproved post to the row's next free slot and
  tells the workspace; ``skip`` ends it missed; approved after its slot passed, the post
  is scheduled into the next free slot under ``next_slot`` and stays approved under
  ``skip``;
* research is due once a week from a week before the plan starts; Research again starts
  the workspace's installed playbook now, and is 409 without it.
"""
from __future__ import annotations

import functools
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.socials_targets as socials_targets  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.core import RecipeExecution, WorkflowTemplate  # noqa: E402
from core.models.socials import SocialCampaign, SocialPost, SocialTopic  # noqa: E402
from core.seeds.seed_socials_package import RESEARCH_PLAYBOOK_TEMPLATE_ID  # noqa: E402
from modules.socials import notify, plan_late, plan_notify, schedule_jobs, service  # noqa: E402
from modules.socials.capabilities import ChannelKind, SocialChannel  # noqa: E402
from services import socials_plan_maker as maker_mod  # noqa: E402
from services import socials_plan_research as research_mod  # noqa: E402
from tests.test_prd251_api import WS_A, _post  # noqa: E402
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402

api = api_harness.api
UTC = timezone.utc
NOW = datetime(2026, 10, 14, 7, 30, tzinfo=UTC)  # a Wednesday
EVERY_DAY = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
FACT = {"text": "We demo on stand B12.", "source": {"kind": "web", "ref": "https://example.com/stand", "label": "Stand page"}}


def _kind(kind):
    return ChannelKind(kind=kind, available=True, reason=None, needs_public_storage=False, steps=())


CHANNELS = [SocialChannel(toolkit="linkedin", label="LinkedIn", post_kinds=(_kind("text"), _kind("image")), verified=True, setup_note=None)]


@pytest.fixture
def maker(api, monkeypatch):
    engine = api.session.get_bind()
    SocialPost.metadata.create_all(engine, tables=[SocialTopic.__table__])
    factory = sessionmaker(bind=engine)
    told = SimpleNamespace(plan=[], moved=[], approval=[], outcome=[])
    monkeypatch.setattr(maker_mod, "_session", factory)
    monkeypatch.setattr(plan_notify, "notify_plan", lambda ws, plan_id, event, title: told.plan.append((event[0], title)))
    monkeypatch.setattr(plan_notify, "notify_slot_moved", lambda ws, post_id, title, at: told.moved.append((str(post_id), at)))
    monkeypatch.setattr(notify, "notify_approval_pending", lambda ws, post_id, title: told.approval.append(str(post_id)))
    monkeypatch.setattr(notify, "notify_publish_outcome", lambda ws, post_id, title, status: told.outcome.append((str(post_id), status)))
    monkeypatch.setattr(maker_mod, "social_channels", lambda db, ws: CHANNELS)
    monkeypatch.setattr(socials_targets, "social_channels", lambda db, ws: CHANNELS)
    proposals = []

    def propose(db, plan, slot, topic, now, visual_slots=()):
        proposals.append((slot.key, topic.title, maker_mod.fact_candidates(topic, now)))
        return {"title": f"Post: {topic.title}", "copy": {"base": f"About {topic.title}.", "per_channel": {}}, "variables": {}, "sources": {}}

    monkeypatch.setattr(maker_mod, "_propose", propose)
    api.told, api.proposals, api.factory = told, proposals, factory
    return api


def _text_plan(api, **overrides):
    cadence = [{"channels": ["linkedin"], "format": "text", "days": EVERY_DAY, "time": "09:00"}]
    return _create_plan(api, timezone="UTC", starts_on="2026-10-12", ends_on="2026-11-08", cadence=cadence, **overrides)


def _topic(api, plan, title):
    resp = api.client.post(f"/api/socials/plans/{plan['id']}/topics", json={"title": title, "facts": [FACT]})
    assert resp.status_code == 201, resp.text
    return resp.json()


def _make(plan, key, now=NOW):
    run = functools.partial(anyio.to_thread.run_sync, maker_mod.make_slot, uuid.UUID(plan["id"]), key, now)
    return anyio.run(run)


def _made(api, key):
    api.session.expire_all()
    return api.session.query(SocialPost).filter(SocialPost.slot_key == key).one_or_none()


# ── made on its day ────────────────────────────────────────────────────────


def test_a_due_slot_is_made_from_the_bank_and_sent_for_approval(maker):
    plan = _text_plan(maker)
    topic = _topic(maker, plan, "Three weeks to Lisbon")
    key = "r1|2026-10-14|09:00"
    assert maker_mod.collect_due(NOW) == [(uuid.UUID(plan["id"]), key)]  # tomorrow's is made tomorrow
    assert _make(plan, key) == maker_mod.MADE
    post = _made(maker, key)
    assert (post.status, post.title, str(post.campaign_id)) == ("needs_approval", "Post: Three weeks to Lisbon", plan["id"])
    assert post.planned_for.replace(tzinfo=UTC) == datetime(2026, 10, 14, 9, tzinfo=UTC)
    assert [(t.toolkit, t.post_kind) for t in post.targets] == [("linkedin", "text")]
    used = maker.session.get(SocialTopic, uuid.UUID(topic["id"]))
    assert used.used_post_id == post.id and used.used_at is not None
    ((proposed_key, title, candidates),) = maker.proposals
    assert (proposed_key, title) == (key, "Three weeks to Lisbon")
    assert candidates[0]["kind"] == "url" and candidates[0]["ref"] == FACT["source"]["ref"]
    assert maker.told.approval == [str(post.id)]
    assert maker_mod.collect_due(NOW) == []
    maker_mod.tell_ready({uuid.UUID(plan["id"]): 1}, NOW)
    maker_mod.tell_ready({uuid.UUID(plan["id"]): 1}, NOW)
    assert maker.told.plan == [("social_plan_ready", "Countdown to Lisbon: 1 waiting for approval")]


def test_a_slot_is_made_once_and_the_loser_leaves_its_topic_unused(maker):
    plan = _text_plan(maker)
    _topic(maker, plan, "First")
    second = _topic(maker, plan, "Second")
    key = "r1|2026-10-14|09:00"
    assert _make(plan, key) == maker_mod.MADE
    assert _make(plan, key) == maker_mod.TAKEN
    maker.session.expire_all()
    assert maker.session.get(SocialTopic, uuid.UUID(second["id"])).used_at is None
    assert maker.session.query(SocialPost).filter(SocialPost.slot_key == key).count() == 1


def test_an_empty_bank_skips_the_slot_and_says_so_once_a_day(maker):
    plan = _text_plan(maker)
    key = "r1|2026-10-14|09:00"
    assert _make(plan, key) == maker_mod.NO_TOPIC
    assert _make(plan, key) == maker_mod.NO_TOPIC
    assert maker.told.plan == [("social_plan_bank_empty", "Countdown to Lisbon: research again or add topics")]
    assert _made(maker, key) is None


def test_no_channel_or_no_render_minutes_skips_the_slot_whole(maker, monkeypatch):
    plan = _text_plan(maker, cadence=[{"channels": ["tiktok"], "format": "text", "days": EVERY_DAY, "time": "09:00"}])
    _topic(maker, plan, "Topic")
    key = "r1|2026-10-14|09:00"
    assert _make(plan, key) == maker_mod.SKIPPED and "no connected channel" in maker.told.plan[-1][1]
    connected = _text_plan(maker, name="Second plan")
    _topic(maker, connected, "Topic")
    monkeypatch.setattr(maker_mod, "fits_quota", lambda db, workspace, seconds: False)
    assert _make(connected, key) == maker_mod.SKIPPED and "render minutes" in maker.told.plan[-1][1]
    maker.session.expire_all()
    assert maker.session.query(SocialPost).count() == 0
    assert maker.session.query(SocialTopic).filter(SocialTopic.used_at.isnot(None)).count() == 0


def test_the_render_quota_check():
    workspace = SimpleNamespace(id=WS_A)
    quota = SimpleNamespace(used_seconds=500, quota_minutes=10)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(maker_mod.render_quota, "render_quota", lambda db, ws: quota)
        patch.setattr(maker_mod.render_quota, "render_seconds_reserved", lambda db, ws: 60)
        assert maker_mod.fits_quota(None, workspace, 40) is True
        assert maker_mod.fits_quota(None, workspace, 41) is False
        assert maker_mod.fits_quota(None, workspace, 0) is True  # stills and text use none


def test_max_per_day_holds_per_local_day(maker):
    rows = [{"channels": ["linkedin"], "format": "text", "days": EVERY_DAY, "time": "09:00"},
            {"channels": ["linkedin"], "format": "text", "days": EVERY_DAY, "time": "12:00"}]
    plan = _text_plan(maker, cadence=rows, make={"max_per_day": 1})
    assert maker_mod.collect_due(NOW) == [(uuid.UUID(plan["id"]), "r1|2026-10-14|09:00")]


# ── slots that pass ────────────────────────────────────────────────────────


def _late_post(api, plan, key, planned_for, status="needs_approval"):
    post = SocialPost(
        id=uuid.uuid4(), workspace_id=WS_A, created_by="member-1", title="Late", content_hash="0" * 64,
        campaign_id=uuid.UUID(plan["id"]), slot_key=key, planned_for=planned_for, status=status, format="text",
        copy={"base": "Late."},
    )
    post.content_hash = service.compute_content_hash(post)
    api.session.add(post)
    api.session.commit()
    return post


def test_next_slot_moves_a_late_post_on_and_skip_misses_it(maker):
    moving = _text_plan(maker, late_policy="next_slot")
    skipping = _text_plan(maker, name="Skipping plan")
    passed = datetime(2026, 10, 13, 9, tzinfo=UTC)
    moved = _late_post(maker, moving, "r1|2026-10-13|09:00", passed)
    missed = _late_post(maker, skipping, "r1|2026-10-13|09:00", passed)
    schedule_jobs.pass_planned_slots(maker.session, NOW)
    maker.session.expire_all()
    moved, missed = maker.session.get(SocialPost, moved.id), maker.session.get(SocialPost, missed.id)
    assert (moved.status, moved.slot_key) == ("needs_approval", "r1|2026-10-14|09:00")  # today 09:00: after the lead
    assert moved.review_log[-1]["action"] == plan_late.ACTION_NEXT_SLOT
    assert maker.told.moved == [(str(moved.id), datetime(2026, 10, 14, 9, tzinfo=UTC))]
    assert missed.status == "missed" and maker.told.outcome == [(str(missed.id), "missed")]


def test_approved_after_its_slot_passed_it_is_scheduled_into_the_next_free_slot(maker, monkeypatch):
    monkeypatch.setattr(service, "_utcnow", lambda: NOW)
    moving = _text_plan(maker, late_policy="next_slot")
    skipping = _text_plan(maker, name="Skipping plan")
    passed = datetime(2026, 10, 13, 9, tzinfo=UTC)
    posts = {}
    for name, plan in (("moving", moving), ("skipping", skipping)):
        post = _late_post(maker, plan, "r1|2026-10-13|09:00", passed)
        resp = _post(maker, str(post.id), "approve", {"content_hash": post.content_hash})
        assert resp.status_code == 200 and resp.json()["status"] == "approved", resp.text
        posts[name] = post.id
    _late_post(maker, moving, "r1|2026-10-14|09:00", datetime(2026, 10, 14, 9, tzinfo=UTC))  # today's slot is taken
    assert plan_late.schedule_late_approvals(maker.session, NOW) == 1
    maker.session.expire_all()
    scheduled = maker.session.get(SocialPost, posts["moving"])
    assert (scheduled.status, scheduled.slot_key) == ("scheduled", "r1|2026-10-15|09:00")
    assert scheduled.scheduled_for.replace(tzinfo=UTC) == datetime(2026, 10, 15, 9, tzinfo=UTC)
    assert maker.session.get(SocialPost, posts["skipping"]).status == "approved"


# ── research comes round ───────────────────────────────────────────────────


def _research_plan(**research):
    return SimpleNamespace(
        status="active", research=research or None, starts_on=date(2026, 10, 12), ends_on=date(2026, 11, 8), timezone="UTC",
    )


def test_research_is_due_once_a_week_within_its_window():
    plan = _research_plan()
    assert research_mod.last_weekly_moment(plan, NOW) == datetime(2026, 10, 12, 6, tzinfo=UTC)  # Monday 06:00
    assert research_mod.research_due(plan, NOW) is True
    assert research_mod.research_due(_research_plan(last_run_at="2026-10-12T06:05:00+00:00"), NOW) is False
    assert research_mod.research_due(_research_plan(last_run_at="2026-10-05T06:05:00+00:00"), NOW) is True
    assert research_mod.research_due(_research_plan(enabled=False), NOW) is False
    early = SimpleNamespace(**{**vars(plan), "starts_on": date(2026, 10, 30)})
    assert research_mod.research_due(early, NOW) is False  # more than a week before the plan


@pytest.fixture
def playbooks(maker, monkeypatch):
    copies = sa.MetaData()
    for table in (WorkflowTemplate.__table__, RecipeExecution.__table__):
        api_harness._sqlite_copy(table, copies)
    copies.create_all(maker.session.get_bind())
    launched = []
    engine = SimpleNamespace(launch=lambda **kwargs: launched.append(kwargs))
    monkeypatch.setattr("services.playbook_engine.get_playbook_engine", lambda: engine)
    maker.launched = launched
    return maker


def _install(api, workspace_id):
    table = WorkflowTemplate.__table__
    with api.session.get_bind().begin() as conn:
        conn.execute(sa.insert(table).values(id=1, template_id=RESEARCH_PLAYBOOK_TEMPLATE_ID, name="Content bank research"))
        conn.execute(sa.insert(table).values(id=2, template_id="ws-research", name="Content bank research",
                                             workspace_id=workspace_id, cloned_from_id=1, steps=[{"step_id": "research"}]))


def test_research_again_starts_the_installed_playbook_and_is_409_without_it(playbooks):
    plan = _text_plan(playbooks)
    missing = playbooks.client.post(f"/api/socials/plans/{plan['id']}/research")
    assert missing.status_code == 409 and "Socials package" in missing.text
    _install(playbooks, WS_A)
    started = playbooks.client.post(f"/api/socials/plans/{plan['id']}/research")
    assert started.status_code == 202, started.text
    (launch,) = playbooks.launched
    assert launch["recipe_id"] == 2 and launch["input_data"] == {"plan_id": plan["id"], "plan_name": "Countdown to Lisbon"}
    assert launch["recipe_execution_id"] == started.json()["execution_id"]
    row = playbooks.session.get(SocialCampaign, uuid.UUID(plan["id"]))
    playbooks.session.refresh(row)
    assert row.research["last_run_id"] == started.json()["execution_id"]
