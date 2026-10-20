"""P256-FIX-SOC-5 (F437, the lesser point): a plan's post that failed before it ever tried to publish
passes its slot like an unapproved one.

A plan's post whose render was refused, or failed, sat ``failed`` on its slot for good: the
planned-slot sweep (``schedule_jobs.pass_planned_slots``, B11) read only drafts and posts waiting
for approval. ``approved_hash`` cannot tell a post that never published apart (it is never
cleared), so the discriminator is the history: ``publish_lifecycle.start_publish`` logs a
``publish`` entry for every attempt. On the plan maker's harness, pinned:

* a failed plan post with no ``publish`` entry ends ``missed`` after its slot and the grace, the
  workspace told; under the plan's ``next_slot`` it moves to the row's next free slot instead;
* a failed post that was approved, published and has a failed channel is left as it is, and can
  still be retried;
* a post approved, then edited, then failed (``approved_hash`` still set, no ``publish`` entry) is
  swept.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialPost  # noqa: E402
from modules.socials import publish_lifecycle, schedule_jobs, service  # noqa: E402
from modules.socials import targets as post_targets  # noqa: E402
from tests.test_prd251bw2_maker import NOW, _late_post, _text_plan  # noqa: E402
from tests.test_prd251bw2_maker import maker  # noqa: E402,F401  (the fixture)

api = api_harness.api
UTC = timezone.utc
PASSED = datetime(2026, 10, 13, 9, tzinfo=UTC)  # yesterday's slot, well past the grace
KEY = "r1|2026-10-13|09:00"


def _entry(action, at="2026-10-12T10:00:00+00:00", **extra):
    return {"at": at, "by": "member-1", "action": action, "comment": None, **extra}


def _failed(env, plan, history):
    post = _late_post(env, plan, KEY, PASSED, status="failed")
    post.review_log = history
    env.session.commit()
    return post


def _status(env, post):
    env.session.expire_all()
    return env.session.get(SocialPost, post.id)


def test_a_failed_plan_post_that_never_published_ends_missed_or_moves_on(maker):  # noqa: F811
    skipping = _text_plan(maker)
    moving = _text_plan(maker, name="Moving plan", late_policy="next_slot")
    refused = _failed(maker, skipping, [_entry("draft"), _entry("render_failed", comment="The renderer cannot be reached right now.")])
    moved = _failed(maker, moving, [_entry("draft"), _entry("render"), _entry("render_failed")])

    assert schedule_jobs.pass_planned_slots(maker.session, NOW) == 1

    refused, moved = _status(maker, refused), _status(maker, moved)
    assert refused.status == "missed" and refused.review_log[-1]["action"] == service.ACTION_SLOT_PASSED
    assert maker.told.outcome == [(str(refused.id), "missed")]
    assert (moved.status, moved.slot_key) == ("failed", "r1|2026-10-14|09:00")  # the row's next free slot


def test_a_failed_post_that_tried_to_publish_stays_retryable(maker):  # noqa: F811
    plan = _text_plan(maker)
    post = _failed(maker, plan, [_entry("draft"), _entry("submit"), _entry("approve"), _entry("publish", channels=["linkedin"]),
                                 _entry("publish_failed")])
    post_targets.replace_targets(post, [{"toolkit": "linkedin", "post_kind": "text", "options": {}}])
    post.targets[0].status = publish_lifecycle.TARGET_FAILED
    post.content_hash = service.compute_content_hash(post)
    post.approved_hash = post.content_hash
    maker.session.commit()

    assert schedule_jobs.pass_planned_slots(maker.session, NOW) == 0

    kept = _status(maker, post)
    assert kept.status == "failed" and maker.told.outcome == []
    publish_lifecycle.assert_retryable(kept)  # raises when it could not be retried


def test_a_post_approved_then_edited_then_failed_is_swept(maker):  # noqa: F811
    plan = _text_plan(maker)
    history = [_entry("draft"), _entry("submit"), _entry("approve"), _entry("approval_voided"), _entry("render"), _entry("render_failed")]
    post = _failed(maker, plan, history)
    post.approved_hash = "f" * 64  # still set after the edit voided it
    maker.session.commit()

    assert schedule_jobs.pass_planned_slots(maker.session, NOW) == 1
    assert _status(maker, post).status == "missed"


def test_a_failed_post_no_plan_made_is_left_alone(maker):  # noqa: F811
    plan = _text_plan(maker)
    post = _failed(maker, plan, [_entry("draft"), _entry("render"), _entry("render_failed")])
    post.campaign_id, post.slot_key = None, None  # a post made by hand, with a planned time of its own
    maker.session.commit()

    assert schedule_jobs.pass_planned_slots(maker.session, NOW) == 0
    assert _status(maker, post).status == "failed"
