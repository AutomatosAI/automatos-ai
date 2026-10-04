"""PRD-251C Wave 2, US-C205 — approve the week.

``POST /api/socials/plans/{plan_id}/batches/{batch_key}/approve`` on the S0.3b API harness
(SQLite). Pinned:

* the shown posts of the batch are approved, each by its own hash, and scheduled into their
  slots, with the workspace's series approval switch off (O2);
* a post changed since it was shown is left and listed with its current hash;
* a post of another plan, of another batch, or one of the batch's posts not shown is left;
* another workspace's plan is a 404; a viewer cannot approve, an editor can;
* a batch key that is neither a week nor a month is a 422.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialPost  # noqa: E402
from modules.socials import service  # noqa: E402
from tests.test_prd251_api import WS_A, WS_B, _ctx  # noqa: E402
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402
from tests.test_prd251bw2_plans import bank  # noqa: E402,F401  (the fixture)

api = api_harness.api
WEEK = "2026-W43"
AHEAD = datetime.now(timezone.utc) + timedelta(days=3)  # the slots are still ahead


def _post(api, plan, day, *, batch=WEEK, status="needs_approval"):
    post = SocialPost(
        id=uuid.uuid4(), workspace_id=WS_A, created_by="the plan", title=f"Day {day}", content_hash="0" * 64,
        campaign_id=uuid.UUID(plan["id"]), slot_key=f"r1|2026-10-{day}|09:00", batch_key=batch,
        planned_for=AHEAD + timedelta(hours=day), status=status, format="text", copy={"base": f"Day {day}."},
    )
    post.content_hash = service.compute_content_hash(post)
    api.session.add(post)
    api.session.commit()
    return post


def _approve(api, plan, posts, batch=WEEK):
    shown = [{"post_id": str(post.id), "content_hash": post.content_hash} for post in posts]
    return api.client.post(f"/api/socials/plans/{plan['id']}/batches/{batch}/approve", json={"posts": shown})


def _statuses(api, posts):
    api.session.expire_all()
    return [api.session.get(SocialPost, post.id).status for post in posts]


def test_the_week_is_approved_in_one_sitting_and_scheduled_into_its_slots(bank):  # noqa: F811
    plan = _create_plan(bank)
    week = [_post(bank, plan, day) for day in (19, 20, 21)]
    resp = _approve(bank, plan, week)
    assert resp.status_code == 200, resp.text  # the workspace's series switch is off (O2)
    body = resp.json()
    assert (body["batch_key"], len(body["approved"]), body["left"]) == (WEEK, 3, [])
    assert _statuses(bank, week) == ["scheduled", "scheduled", "scheduled"]


def test_a_post_changed_since_it_was_shown_is_left_and_listed(bank):  # noqa: F811
    plan = _create_plan(bank)
    kept, changed = _post(bank, plan, 19), _post(bank, plan, 20)
    shown_hash = changed.content_hash
    changed.copy = {"base": "Edited after the review opened."}
    changed.content_hash = service.compute_content_hash(changed)
    bank.session.commit()
    shown = [{"post_id": str(kept.id), "content_hash": kept.content_hash}, {"post_id": str(changed.id), "content_hash": shown_hash}]
    body = bank.client.post(f"/api/socials/plans/{plan['id']}/batches/{WEEK}/approve", json={"posts": shown}).json()
    assert [row["id"] for row in body["approved"]] == [str(kept.id)]
    (left,) = body["left"]
    assert (left["post_id"], left["reason"], left["content_hash"]) == (str(changed.id), "changed", changed.content_hash)
    assert _statuses(bank, [changed]) == ["needs_approval"]


def test_another_plans_post_another_batchs_and_an_unshown_one_are_left(bank):  # noqa: F811
    plan, other = _create_plan(bank), _create_plan(bank, name="Another plan")
    shown, unshown = _post(bank, plan, 19), _post(bank, plan, 20)
    next_week = _post(bank, plan, 26, batch="2026-W44")
    elsewhere = _post(bank, other, 21)
    body = _approve(bank, plan, [shown, next_week, elsewhere]).json()
    assert [row["id"] for row in body["approved"]] == [str(shown.id)]
    reasons = {row["post_id"]: row["reason"] for row in body["left"]}
    assert reasons == {str(next_week.id): "not_in_batch", str(elsewhere.id): "not_in_campaign", str(unshown.id): "not_shown"}
    assert _statuses(bank, [next_week, elsewhere, unshown]) == ["needs_approval"] * 3


def test_another_workspace_a_viewer_and_a_bad_key_are_refused(bank):  # noqa: F811
    plan = _create_plan(bank)
    post = _post(bank, plan, 19)
    bank.role = "viewer"
    assert _approve(bank, plan, [post]).status_code == 403
    bank.role = "owner"
    assert _approve(bank, plan, [post], batch="week-43").status_code == 422
    bank.ctx = _ctx(WS_B)
    assert _approve(bank, plan, [post]).status_code == 404
    bank.ctx = _ctx(WS_A)
    bank.role = "editor"
    assert _approve(bank, plan, [post]).status_code == 200
    assert _statuses(bank, [post]) == ["scheduled"]
