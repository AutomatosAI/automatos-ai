"""PRD-251 Wave 2, US-206 (S2.3) — approvers are notified when a post needs them.

A post enters ``needs_approval`` three ways, and each sends ONE ``approval_pending``
notification after its commit, linked to the post (``link_type='social_post'``):

* a submit (a person's route, or an agent's tool through the same flow), a first
  submit and a resubmit after changes were requested;
* a render that finishes (Wave 1, S1.1c); a render that fails sends nothing;
* an edit that voids an approval: a PATCH of an approved post's copy, and a change
  of its channels (``set_post_targets``, US-204).

No other transition notifies: approve, request changes, reject, starting a render,
an edit of a draft or of a post already waiting. The helper sends through
``NotificationDispatcher`` on its own session and never raises into the caller.
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.socials as socials_api  # noqa: E402
import api.socials_targets as targets_api  # noqa: E402
from modules.socials import notify  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
import tests.test_prd251w1_render_lifecycle as render_harness  # noqa: E402
from tests.test_prd251_api import WS_A, _create, _post  # noqa: E402
from tests.test_prd251w1_render_lifecycle import FakeStore, Renderer, _renderable, _run, _start  # noqa: E402

# The S0.3b routes on SQLite (api), and the S1.1c render lifecycle (env).
api = api_harness.api
env = render_harness.env

TITLE = "Countdown"
LINKEDIN = [{"toolkit": "linkedin", "post_kind": "text", "options": {},
             "steps": [{"id": "post", "action": "EXAMPLE_CREATE_POST", "class": "publish", "params": {}}]}]


@pytest.fixture
def sent(monkeypatch):
    """Every approval notice the routes send, as (workspace id, post id, title)."""
    calls = []
    monkeypatch.setattr(
        notify, "notify_approval_pending", lambda ws, post_id, title: calls.append((ws, str(post_id), title))
    )
    return calls


def _submitted(api):
    post = _create(api)
    resp = _post(api, post["id"], "submit")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _approved(api):
    post = _submitted(api)
    resp = _post(api, post["id"], "approve", {"content_hash": post["content_hash"]})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# The three ways in
# ---------------------------------------------------------------------------


def test_a_submit_notifies_once_linked_to_the_post(api, sent):
    post = _submitted(api)
    assert post["status"] == "needs_approval"
    assert sent == [(WS_A, post["id"], TITLE)]


def test_a_resubmit_after_changes_were_requested_notifies_again(api, sent):
    post = _submitted(api)
    assert _post(api, post["id"], "request-changes", {"comment": "Tighter"}).status_code == 200
    assert _post(api, post["id"], "submit").status_code == 200
    assert sent == [(WS_A, post["id"], TITLE)] * 2


def test_an_edit_that_voids_an_approval_notifies_once(api, sent):
    post = _approved(api)
    sent.clear()
    resp = api.client.patch(f"/api/socials/posts/{post['id']}", json={"copy": {"base": "A sharper line."}})
    assert resp.status_code == 200 and resp.json()["status"] == "needs_approval"
    assert sent == [(WS_A, post["id"], TITLE)]


def test_a_change_of_channels_that_voids_an_approval_notifies_once(api, sent, monkeypatch):
    post = _approved(api)
    sent.clear()
    monkeypatch.setattr(targets_api, "resolve_targets", lambda db, ws, requested: LINKEDIN)
    row = socials_api.service.get_post(api.session, WS_A, uuid.UUID(post["id"]))
    saved = targets_api.set_post_targets(api.session, row, "member-1", [])
    assert saved["status"] == "needs_approval"
    assert sent == [(WS_A, post["id"], TITLE)]


def test_a_finished_render_notifies_once_and_a_failed_one_never(env, monkeypatch):
    dispatched = []

    async def record(ws, post_id, title, *, session_factory=None):
        dispatched.append((ws, str(post_id), title))

    monkeypatch.setattr(notify, "dispatch_approval_pending", record)
    post = _renderable(env)
    _, job = _start(env, post)
    assert dispatched == []  # starting a render is not a way in
    assert _run(job, Renderer(), FakeStore(), env.factory) is True
    assert dispatched == [(job.workspace_id, post["id"], post["title"])]

    failing = _renderable(env, title="Second")
    _, job = _start(env, failing)
    assert _run(job, Renderer(status="failed", error={"code": "x", "message": "no"}), FakeStore(), env.factory) is False
    assert len(dispatched) == 1


# ---------------------------------------------------------------------------
# Nothing else notifies
# ---------------------------------------------------------------------------


def test_the_review_actions_and_edits_of_waiting_or_draft_posts_send_nothing(api, sent):
    draft = _create(api)
    assert api.client.patch(f"/api/socials/posts/{draft['id']}", json={"copy": {"base": "v2"}}).status_code == 200
    waiting = _submitted(api)
    sent.clear()
    assert api.client.patch(f"/api/socials/posts/{waiting['id']}", json={"copy": {"base": "v3"}}).status_code == 200
    assert _post(api, waiting["id"], "request-changes", {"comment": "No"}).status_code == 200
    rejected = _submitted(api)
    approved = _submitted(api)
    sent.clear()
    assert _post(api, rejected["id"], "reject", {"reason": "Off brand"}).status_code == 200
    assert _post(api, approved["id"], "approve", {"content_hash": approved["content_hash"]}).status_code == 200
    assert sent == []


def test_a_refused_write_sends_nothing(api, sent):
    post = _submitted(api)
    sent.clear()
    assert _post(api, post["id"], "approve", {"content_hash": "0" * 64}).status_code == 409
    assert _post(api, post["id"], "submit").status_code == 409
    assert sent == []


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "before, after, entered",
    [
        ("draft", "needs_approval", True),
        ("changes_requested", "needs_approval", True),
        ("rendering", "needs_approval", True),
        ("approved", "needs_approval", True),
        ("scheduled", "needs_approval", True),
        ("needs_approval", "needs_approval", False),
        ("needs_approval", "approved", False),
        ("needs_approval", "changes_requested", False),
        ("needs_approval", "archived", False),
        ("draft", "rendering", False),
        ("rendering", "failed", False),
    ],
)
def test_only_a_move_into_needs_approval_counts(before, after, entered):
    assert notify.entered_review(before, after) is entered


class _Session:
    closed = False

    def close(self):
        self.closed = True


class _Dispatcher:
    calls: list = []
    fail = False

    def __init__(self, db, workspace_id):
        self.workspace_id = workspace_id

    async def dispatch(self, **kwargs):
        if type(self).fail:
            raise RuntimeError("the bell is down")
        type(self).calls.append({"workspace_id": self.workspace_id, **kwargs})
        return {"dispatched_to": ["in_app"]}


def test_the_notice_is_approval_pending_linked_to_the_post(monkeypatch):
    _Dispatcher.calls, _Dispatcher.fail = [], False
    monkeypatch.setattr(notify, "_dispatcher", _Dispatcher)
    session = _Session()
    post_id = uuid.uuid4()
    asyncio.run(notify.dispatch_approval_pending(WS_A, post_id, "Launch week", session_factory=lambda: session))
    assert _Dispatcher.calls == [{
        "workspace_id": str(WS_A),
        "event_type": "approval_pending",
        "title": "Social post needs approval: Launch week",
        "link_type": "social_post",
        "link_id": str(post_id),
        "status": "action_required",
    }]
    assert session.closed


def test_a_failing_dispatch_never_raises(monkeypatch):
    _Dispatcher.calls, _Dispatcher.fail = [], True
    monkeypatch.setattr(notify, "_dispatcher", _Dispatcher)
    session = _Session()
    asyncio.run(notify.dispatch_approval_pending(WS_A, uuid.uuid4(), "T", session_factory=lambda: session))
    assert session.closed

    def no_session():
        raise RuntimeError("no database")

    asyncio.run(notify.dispatch_approval_pending(WS_A, uuid.uuid4(), "T", session_factory=no_session))


def test_with_no_running_loop_the_notice_is_sent_inline(monkeypatch):
    sent_now = []

    async def record(ws, post_id, title, *, session_factory=None):
        sent_now.append((ws, post_id, title))

    monkeypatch.setattr(notify, "dispatch_approval_pending", record)
    notify.notify_approval_pending(WS_A, "p1", "T")
    assert sent_now == [(WS_A, "p1", "T")]


def test_on_a_running_loop_the_notice_is_a_tracked_task(monkeypatch):
    sent_later = []

    async def record(ws, post_id, title, *, session_factory=None):
        sent_later.append(post_id)

    monkeypatch.setattr(notify, "dispatch_approval_pending", record)

    async def go():
        notify.notify_approval_pending(WS_A, "p2", "T")
        assert len(notify._PENDING) == 1
        await asyncio.gather(*notify._PENDING)

    asyncio.run(go())
    assert sent_later == ["p2"] and not notify._PENDING


def test_approval_pending_is_a_dispatcher_event():
    from core.services.notification_dispatcher import VALID_EVENT_TYPES

    assert notify.EVENT_TYPE in VALID_EVENT_TYPES
