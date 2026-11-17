"""One Approve, one Publish now (17 Nov 2026): the approve does what its button says.

The owner found two approve buttons and two publish steps on a post ("are you sure, are you
really sure?"), and a plain Approve that left a post with no time approved and posted nothing.
Now one Approve says what it does, and the approve route (``api/socials.py``) does it:

* the click carries its intent: ``post_now`` (its button said "Approve & post now"). A post with
  no planned slot, no schedule and a channel is published the moment its approval commits only
  with it, through publish-now's claim (``publisher.begin_publish``, then ``publisher.launch``):
  the answer carries it ``publishing``, with the one publish launched, and a publish-now after it
  finds nothing left to claim. Without it (a label gone stale, the phone app) it is approved only;
  with it and a slot ahead, it is scheduled: the server's own check must agree;
* a post whose planned slot is still ahead is scheduled into it, and so is one whose schedule,
  kept from before an edit sent it back, is still ahead (its planned slot, else that schedule),
  on this route only: a series approval leaves that post approved, as before. Nothing is launched;
* a post whose slot passed, or whose kept schedule passed, stays approved for a person to
  reschedule, and so does a post with no channel: nothing is launched;
* the unsourced-claims refusal (D7) still comes first: nothing is approved or launched until the
  approver confirms the claims by name, and then it publishes;
* after the approval commits: an edit another worker commits before the publish claims it wins
  (409, nothing launched); any other claim refused, or a launch that fails, leaves the approval
  standing: logged, answered 200 with ``posting_not_started``, and a post claimed whose run never
  started is ended at once (failed, each channel never tried, so Retry publishes it), never left
  publishing;
* the decision is ``publish_lifecycle.publishes_on_approval``, and only the approve route starts a
  publish from it: no agent tool reaches it (D6, D14; also pinned in the tool tests).

On the S0.3b routes' harness (``tests/test_prd251_api.py``): the real router on SQLite, the
publish launch recorded instead of run.
"""
from __future__ import annotations

import ast
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from sqlalchemy.orm import sessionmaker  # noqa: E402

import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialPost  # noqa: E402
import api.socials_publish as socials_publish  # noqa: E402
from api.socials_publish import POSTING_NOT_STARTED  # noqa: E402
from modules.socials import notify as socials_notify  # noqa: E402
from modules.socials import publish_lifecycle, publisher, workspace_copies  # noqa: E402
from modules.socials import service as socials_service  # noqa: E402
from tests.test_prd251_api import WS_A, _LINKEDIN_TEXT, _approved_with_a_channel, _create, _post  # noqa: E402
from tests.test_prd251w2_series_approval import _add, _campaign, _series_switch, _shown  # noqa: E402
from tests.test_prd251w2_series_approval import _approve as _approve_series  # noqa: E402

api = api_harness.api

AUTHOR = "user-author"
LISBON = "Europe/Lisbon"
CLAIM = {"users": {"value": 1200, "claim": True}}
CODE_DIRS = ("api", "modules", "core", "services")


def _waiting(api, *, channels=(_LINKEDIN_TEXT,), planned_for=None, **body):
    """A post waiting for approval, publishing to ``channels``, planned for ``planned_for``: its row."""
    post = _create(api, **body)
    row = api.session.get(SocialPost, uuid.UUID(post["id"]))
    if channels:
        socials_service.update_post(row, AUTHOR, {"targets": list(channels)})
    socials_service.set_planned_for(row, planned_for, LISBON if planned_for else None)
    api.session.commit()
    assert _post(api, post["id"], "submit").status_code == 200
    return _row(api, row.id)


def _row(api, post_id):
    api.session.expire_all()
    return api.session.get(SocialPost, post_id)


def _approve(api, row, **extra):
    return _post(api, str(row.id), "approve", {"content_hash": row.content_hash, **extra})


def _actions(body):
    return [entry["action"] for entry in body["review_log"]]


# ---------------------------------------------------------------------------
# What the approval does, by the post's slot
# ---------------------------------------------------------------------------


def test_approving_a_post_with_no_slot_publishes_it_now_as_publish_now_does(api, monkeypatch):
    from core.composio.tool_executor import ComposioToolExecutor

    composio = MagicMock(name="ComposioToolExecutor.execute")
    monkeypatch.setattr(ComposioToolExecutor, "execute", composio)
    row = _waiting(api)
    shown = row.content_hash

    resp = _approve(api, row, post_now=True)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "publishing" and POSTING_NOT_STARTED not in body
    assert body["approved_hash"] == body["content_hash"] == shown
    assert _actions(body) == ["submit", "approve", "publish"]
    assert body["review_log"][-1]["channels"] == ["linkedin text"]
    (job,) = api.launched
    assert (job.post_id, job.workspace_id, job.content_hash) == (row.id, WS_A, shown)
    # The request only claims, as publish-now's does; the background publish runs the channels.
    composio.assert_not_called()
    assert [target["status"] for target in body["targets"]] == ["pending"]
    # Claimed once: a publish-now after it finds the post publishing, and nothing more is launched.
    assert _post(api, str(row.id), "publish-now").status_code == 409
    assert len(api.launched) == 1


def test_a_click_without_post_now_only_approves_a_post_with_no_slot(api):
    """A label gone stale (it said "Approve · posts 18:00" when another tab cleared the slot) or the
    phone app sends no ``post_now``: the post is approved, and nothing is launched."""
    row = _waiting(api)

    resp = _approve(api, row)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["status"], body["planned_for"], body["scheduled_for"]) == ("approved", None, None)
    assert _actions(body) == ["submit", "approve"]
    assert api.launched == []


def test_approving_a_post_whose_slot_is_ahead_schedules_it_there_and_publishes_nothing_even_with_post_now(api):
    """The label said "Approve & post now", but a slot was set since: the server's check wins."""
    row = _waiting(api, planned_for=datetime.now(timezone.utc) + timedelta(days=30))

    resp = _approve(api, row, post_now=True)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "scheduled"
    assert body["scheduled_for"] is not None and body["scheduled_for"] == body["planned_for"]
    assert _actions(body) == ["submit", "approve", "schedule"]
    assert api.launched == []


def test_approving_a_post_whose_slot_passed_leaves_it_approved_and_publishes_nothing(api):
    row = _waiting(api, planned_for=datetime.now(timezone.utc) - timedelta(hours=1))

    resp = _approve(api, row)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["status"], body["scheduled_for"]) == ("approved", None)
    assert _actions(body) == ["submit", "approve"]
    assert api.launched == []


def _scheduled_then_edited(api, when):
    """A post with a channel and no planned slot, scheduled at ``when`` through the schedule route,
    then edited through PATCH: back to needs_approval with its schedule kept. Its row."""
    approved = _approved_with_a_channel(api)  # approved through the route, without post_now: approved only
    scheduled = _post(api, approved["id"], "schedule", {"scheduled_for": when.isoformat(), "timezone": LISBON})
    assert scheduled.status_code == 200, scheduled.text
    edited = api.client.patch(f"/api/socials/posts/{approved['id']}", json={"copy": {"base": "Changed after it was scheduled."}})
    assert edited.status_code == 200, edited.text
    assert edited.json()["status"] == "needs_approval" and edited.json()["scheduled_for"] is not None  # the edit kept it
    return _row(api, uuid.UUID(approved["id"]))


def _utc(text):
    when = datetime.fromisoformat(text)
    return when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)


def test_a_schedule_kept_from_before_an_edit_and_still_ahead_is_scheduled_there_and_publishes_nothing(api):
    kept = (datetime.now(timezone.utc) + timedelta(days=2)).replace(microsecond=0)
    row = _scheduled_then_edited(api, kept)

    resp = _approve(api, row)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["status"], body["planned_for"], body["timezone"]) == ("scheduled", None, LISBON)
    assert _utc(body["scheduled_for"]) == kept
    assert _actions(body)[-3:] == ["approval_voided", "approve", "schedule"]
    assert api.launched == []


def test_a_kept_schedule_ahead_is_used_when_the_planned_slot_passed(api):
    """Planned for a time that has since passed, then rescheduled (the schedule route writes only
    ``scheduled_for``) and edited: the approval schedules it into the kept time, not the passed slot."""
    kept = (datetime.now(timezone.utc) + timedelta(days=3)).replace(microsecond=0)
    row = _scheduled_then_edited(api, kept)
    row.planned_for = datetime.now(timezone.utc) - timedelta(hours=1)  # the planned slot, since passed
    api.session.commit()

    resp = _approve(api, _row(api, row.id))

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "scheduled" and _utc(body["scheduled_for"]) == kept
    assert _actions(body)[-2:] == ["approve", "schedule"]
    assert api.launched == []


def test_a_schedule_kept_from_before_an_edit_whose_time_passed_stays_approved_and_publishes_nothing(api):
    row = _scheduled_then_edited(api, datetime.now(timezone.utc) + timedelta(days=2))
    row.scheduled_for = datetime.now(timezone.utc) - timedelta(hours=1)  # its time then passed (the route refuses a past one)
    api.session.commit()

    resp = _approve(api, _row(api, row.id))

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "approved" and body["scheduled_for"] is not None
    assert _actions(body)[-2:] == ["approval_voided", "approve"]
    assert api.launched == []


def test_a_post_with_no_channel_is_approved_and_nothing_is_launched(api):
    row = _waiting(api, channels=())

    resp = _approve(api, row, post_now=True)

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "approved"
    assert api.launched == []


# ---------------------------------------------------------------------------
# D7 first, and a race lost
# ---------------------------------------------------------------------------


def test_unsourced_claims_are_refused_first_and_it_publishes_only_once_they_are_confirmed(api):
    row = _waiting(api, variables=CLAIM)

    refused = _approve(api, row, post_now=True)

    assert refused.status_code == 422
    assert refused.json()["detail"]["claims"] == ["users"]
    after = _row(api, row.id)
    assert (after.status, after.approved_hash) == ("needs_approval", None)
    assert api.launched == []

    confirmed = _approve(api, after, override_unsourced=True, post_now=True)

    assert confirmed.status_code == 200, confirmed.text
    body = confirmed.json()
    assert body["status"] == "publishing" and body["override_unsourced"] is True
    approval = body["review_log"][-2]
    assert approval["action"] == "approve" and approval["overridden_claims"] == ["users"]
    assert [job.post_id for job in api.launched] == [row.id]


def test_an_edit_committed_between_the_approval_and_its_publish_wins_and_nothing_is_launched(api, monkeypatch):
    """Another worker edits the post after the approval committed and before the publish claims
    it (here: while the approved files are copied). The claim's compare-and-set refuses: 409 with
    the current hash, as publish-now answers, and the edit's voiding of the approval stands."""
    row = _waiting(api)
    other_worker = sessionmaker(bind=api.session.get_bind())()
    edits = []

    def another_worker_edits(db, post, now=None):
        other = other_worker.get(SocialPost, post.id)
        socials_service.update_post(other, "editor-2", {"copy": {"base": "Edited by another worker."}})
        other_worker.commit()
        edits.append(other.content_hash)

    monkeypatch.setattr(workspace_copies, "copy_when_approved", another_worker_edits)
    try:
        resp = _approve(api, row, post_now=True)
    finally:
        other_worker.close()

    assert edits, "the concurrent edit never ran"
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["content_hash"] == edits[0]
    assert api.launched == []
    current = _row(api, row.id)
    assert current.status == "needs_approval"
    assert [entry["action"] for entry in current.review_log] == ["submit", "approve", "approval_voided"]


def test_a_claim_refused_by_a_write_that_kept_the_approval_answers_200_saying_posting_did_not_start(api, monkeypatch):
    """Another worker schedules the post after the approval committed and before the publish claims
    it: its content, so its approval, is unchanged, so this is not the edit race. The claim's
    compare-and-set refuses; the refusal is logged, and the answer is 200 with the post as it is
    (scheduled) and ``posting_not_started``. Nothing is launched."""
    row = _waiting(api)
    other_worker = sessionmaker(bind=api.session.get_bind())()
    later = datetime.now(timezone.utc) + timedelta(days=1)
    log = MagicMock(name="socials_publish.logger")
    monkeypatch.setattr(socials_publish, "logger", log)

    def another_worker_schedules(db, post, now=None):
        other = other_worker.get(SocialPost, post.id)
        socials_service.schedule(other, "editor-2", later, LISBON)
        other_worker.commit()

    monkeypatch.setattr(workspace_copies, "copy_when_approved", another_worker_schedules)
    try:
        resp = _approve(api, row, post_now=True)
    finally:
        other_worker.close()

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["status"], body[POSTING_NOT_STARTED]) == ("scheduled", True)
    assert body["approved_hash"] == body["content_hash"] == row.content_hash
    assert api.launched == []
    log.exception.assert_called_once()
    assert "claim was refused" in log.exception.call_args.args[0]


def test_a_launch_that_fails_after_the_claim_ends_the_post_failed_never_left_publishing(api, monkeypatch):
    """The claim committed, then the launch raised. The post is not left publishing with no run: it
    is ended at once as the lost-publish sweep ends one (``publish_records.end_lost``): failed, each
    channel never tried, so Retry publishes it, and the workspace told, as the sweep tells it. Logged,
    and answered 200 with ``posting_not_started``."""
    def launch_fails(job):
        raise RuntimeError("no event loop to run the publish on")

    log = MagicMock(name="socials_publish.logger")
    told = MagicMock(name="notify.notify_publish_outcome")
    monkeypatch.setattr(socials_publish, "logger", log)
    monkeypatch.setattr(socials_notify, "notify_publish_outcome", told)
    monkeypatch.setattr(publisher, "launch", launch_fails)
    row = _waiting(api)

    resp = _approve(api, row, post_now=True)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["status"], body[POSTING_NOT_STARTED]) == ("failed", True)
    assert _actions(body)[-3:] == ["approve", "publish", "publish_failed"]
    assert [(target["status"], target["error"]) for target in body["targets"]] == [("failed", publish_lifecycle.NOT_TRIED)]
    stored = _row(api, row.id)
    assert stored.status == "failed" and stored.approved_hash == stored.content_hash
    publish_lifecycle.assert_retryable(stored)  # the approval stands: Retry publishes it
    log.exception.assert_called_once()
    assert "did not start" in log.exception.call_args.args[0]
    told.assert_called_once_with(WS_A, row.id, row.title or "", "failed")


def test_a_series_approval_leaves_a_post_with_a_kept_schedule_ahead_approved(api):
    """The kept-schedule fallback is the one-post approve route's alone (``keep_schedule``): a series
    approval approves the same post as before One Approve, scheduled nowhere, publishing nothing."""
    _series_switch(api, True)
    campaign = _campaign(api)
    row = _scheduled_then_edited(api, datetime.now(timezone.utc) + timedelta(days=2))
    assert _add(api, campaign["id"], str(row.id)).status_code == 200
    row = _row(api, row.id)

    resp = _approve_series(api, campaign["id"], _shown({"id": str(row.id), "content_hash": row.content_hash}))

    assert resp.status_code == 200, resp.text
    assert [post["id"] for post in resp.json()["approved"]] == [str(row.id)]
    after = _row(api, row.id)
    assert after.status == "approved" and after.scheduled_for is not None
    assert after.review_log[-1]["action"] == "approve"
    assert api.launched == []


# ---------------------------------------------------------------------------
# The decision, and who may reach it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fields, publishes",
    [
        ({}, True),
        ({"planned_for": datetime(2026, 11, 20, 9, tzinfo=timezone.utc)}, False),  # scheduled, or passed
        ({"scheduled_for": datetime(2026, 11, 20, 9, tzinfo=timezone.utc)}, False),
        ({"targets": []}, False),
        ({"status": "scheduled"}, False),
        ({"status": "needs_approval"}, False),
    ],
)
def test_only_an_approved_post_with_no_slot_no_schedule_and_a_channel_publishes_on_approval(fields, publishes):
    post = SimpleNamespace(**{"status": "approved", "planned_for": None, "scheduled_for": None,
                              "targets": [SimpleNamespace(toolkit="linkedin", post_kind="text")], **fields})
    assert publish_lifecycle.publishes_on_approval(post) is publishes


def _names(node):
    """Every name the code under ``node`` uses or defines (not its strings or docstrings)."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    } | {n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)}


def _code_naming(name):
    """The orchestrator's code files (not its tests) whose code names ``name``."""
    found = []
    for folder in CODE_DIRS:
        for path in sorted((_ORCH / folder).rglob("*.py")):
            source = path.read_text(encoding="utf-8", errors="replace")
            if name in source and name in _names(ast.parse(source)):
                found.append(str(path.relative_to(_ORCH)))
    return sorted(found)


def test_only_the_approve_route_starts_a_publish_on_approval():
    assert _code_naming("publish_on_approval") == ["api/socials.py", "api/socials_publish.py"]
    routes = ast.parse((_ORCH / "api" / "socials.py").read_text(encoding="utf-8"))
    callers = [node.name for node in routes.body if isinstance(node, ast.FunctionDef) and "publish_on_approval" in _names(node)]
    assert callers == ["approve_social_post"]
    # The decision itself is read by the approve route's step alone.
    assert _code_naming("publishes_on_approval") == ["api/socials_publish.py", "modules/socials/publish_lifecycle.py"]
