"""Deleting a post (3 Oct 2026, Gerard: "I need to be able to ... delete").

* An owner deletes a post that has not gone out: 204, and it is gone from the list.
* An editor cannot (403: deleting is ``documents:delete``, owners and admins).
* A post that went out (published, partly published, or a channel row with a remote id)
  stays, and so does one rendering or publishing: 409 with the reason.
* A scheduled post's publish job is removed; the files of a post with media leave
  Deliverables after the delete commits.
* A plan's post frees the idea it used and skips its slot, so the plan does not make it
  again; a slot that left the plan is logged, never in the way.
"""
from __future__ import annotations

import os
import sys
import uuid
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

import api.socials_delete as delete_api  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialPost, SocialPostTarget  # noqa: E402
from modules.socials import plans  # noqa: E402
from tests.test_prd251_api import WS_A, _create, _ctx  # noqa: E402

api = api_harness.api


def _delete(api, post_id):
    return api.client.delete(f"/api/socials/posts/{post_id}")


def _set(api, post_id, **fields):
    post = api.session.get(SocialPost, uuid.UUID(post_id))
    for name, value in fields.items():
        setattr(post, name, value)
    api.session.commit()


def _exists(api, post_id):
    api.session.expire_all()
    return api.session.get(SocialPost, uuid.UUID(post_id)) is not None


@pytest.fixture
def cancelled(monkeypatch):
    seen = []
    monkeypatch.setattr(delete_api.schedule_jobs, "cancel_job", lambda post_id: seen.append(str(post_id)))
    return seen


def test_an_owner_deletes_a_post_that_has_not_gone_out(api, cancelled):
    post = _create(api)
    assert _delete(api, post["id"]).status_code == 204
    assert not _exists(api, post["id"])
    assert post["id"] not in [p["id"] for p in api.client.get("/api/socials/posts").json()["posts"]]
    assert cancelled == [post["id"]]  # a scheduled post's job goes; for any other it is a no-op


def test_an_editor_cannot_delete(api, cancelled):
    post = _create(api)
    api.role, api.ctx = "editor", _ctx(WS_A, "editor-1")
    assert _delete(api, post["id"]).status_code == 403
    assert _exists(api, post["id"]) and cancelled == []


@pytest.mark.parametrize("status, refused", [
    ("published", delete_api.WENT_OUT_MESSAGE),
    ("partially_published", delete_api.WENT_OUT_MESSAGE),
    ("rendering", delete_api.BUSY_MESSAGE),
    ("publishing", delete_api.BUSY_MESSAGE),
])
def test_a_post_that_went_out_or_is_busy_stays(api, cancelled, status, refused):
    post = _create(api)
    _set(api, post["id"], status=status)
    resp = _delete(api, post["id"])
    assert resp.status_code == 409 and resp.json()["detail"] == refused
    assert _exists(api, post["id"]) and cancelled == []


def test_a_channel_row_that_went_out_keeps_the_post(api, cancelled):
    post = _create(api)
    api.session.add(SocialPostTarget(
        post_id=uuid.UUID(post["id"]), toolkit="twitter", post_kind="image", action_plan={},
        idempotency_key=f"{post['id']}:twitter", status="published", remote_id="1840000000000000000",
    ))
    api.session.commit()
    assert _delete(api, post["id"]).status_code == 409
    assert _exists(api, post["id"])


def test_the_files_of_a_post_with_media_leave_deliverables_after_the_delete(api, cancelled, monkeypatch):
    retired = []
    monkeypatch.setattr(delete_api, "_retire_its_files", lambda db, ws, pid: retired.append((ws, str(pid))) or 2)
    post = _create(api)
    _set(api, post["id"], media={"4:5": [{"name": "image-4x5.png"}]})
    assert _delete(api, post["id"]).status_code == 204
    assert retired == [(WS_A, post["id"])] and not _exists(api, post["id"])


def test_retiring_soft_deletes_each_of_the_posts_own_files(monkeypatch):
    soft_deleted = []

    class _Deliverables:
        def __init__(self, db, workspace_id):
            self.workspace_id = workspace_id

        def soft_delete(self, deliverable_id):
            soft_deleted.append(deliverable_id)
            return {"success": deliverable_id != "d-locked"}

    db = SimpleNamespace(execute=lambda sql, params: SimpleNamespace(
        fetchall=lambda: [SimpleNamespace(id="d-1"), SimpleNamespace(id="d-locked")]))
    monkeypatch.setattr(delete_api, "DeliverableService", _Deliverables)
    assert delete_api._retire_its_files(db, WS_A, uuid.uuid4()) == 1
    assert soft_deleted == ["d-1", "d-locked"]


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *conditions):
        return self.rows


def _plan_db(topics, plan):
    return SimpleNamespace(query=lambda model: _Query(topics), get=lambda model, ident: plan)


def test_a_plans_post_frees_its_idea_and_skips_its_slot(monkeypatch):
    moved = []
    monkeypatch.setattr(delete_api.plan_store, "move_slot", lambda plan, key, *, to, skip: moved.append((key, to, skip)))
    topic = SimpleNamespace(used_post_id=uuid.uuid4(), used_at="2026-10-03")
    plan = SimpleNamespace(id=uuid.uuid4(), kind=plans.PLAN)
    post = SimpleNamespace(id=topic.used_post_id, slot_key="row-1:2026-10-04", campaign_id=plan.id)

    delete_api._free_its_plan_slot(_plan_db([topic], plan), post)

    assert (topic.used_post_id, topic.used_at) == (None, None)
    assert moved == [("row-1:2026-10-04", None, True)]


def test_a_slot_that_left_the_plan_is_logged_and_a_campaign_post_skips_nothing(monkeypatch, caplog):
    def gone(plan, key, *, to, skip):
        raise plans.InvalidPlan("no planned slot has that key")

    monkeypatch.setattr(delete_api.plan_store, "move_slot", gone)
    plan = SimpleNamespace(id=uuid.uuid4(), kind=plans.PLAN)
    post = SimpleNamespace(id=uuid.uuid4(), slot_key="row-1:2026-10-04", campaign_id=plan.id)
    delete_api._free_its_plan_slot(_plan_db([], plan), post)  # never raises
    assert "was not skipped" in caplog.text

    campaign = SimpleNamespace(id=uuid.uuid4(), kind="campaign")
    delete_api._free_its_plan_slot(_plan_db([], campaign), SimpleNamespace(id=uuid.uuid4(), slot_key="k", campaign_id=campaign.id))
