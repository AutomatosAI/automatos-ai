"""PRD-251 Wave 3, US-301 (S3.3, D6, D8, D14) — the publisher, proven with LinkedIn.

Composio is mocked everywhere: a fake executor records every call (the action, its
params, its upload spec, the way through) and answers from a script. Pinned here:

* the LinkedIn sequences (the engine proven with LinkedIn) are in
  ``tests/test_prd251w3_linkedin.py``, on this file's harness;
* the approval guard runs first: a stale approval publishes nothing and makes no
  call; a post claimed once publishes once (two concurrent claims and a double run on
  the CI Postgres);
* a transient error tries again up to ``SOCIALS_MAX_TARGET_ATTEMPTS``; a 4xx does not
  and keeps the platform's message; one target of two failing ends
  ``partially_published`` and notifies; retry runs only the failed target;
* the executor: the deny list and the post gate before any upload, exactly the named
  params converted, ``UPLOAD_ACTIONS`` untouched; the LinkedIn image workaround reads
  the staged files;
* the routes: plain ``def``, in the committed manifest, called by ``apiClient`` with
  POST; the new settings in ``config.py`` and ``config-surface.json``.

A lost or unstartable publish, and the races the review found, are in
``tests/test_prd251w3_recovery.py``, on this file's harness.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import threading
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

import sqlalchemy as sa  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every table the FKs name)
from api import socials_publish  # noqa: E402
from api.socials_targets import step_plan  # noqa: E402
from config import config  # noqa: E402
from core.composio import tool_executor, upload_spec  # noqa: E402
from core.composio.post_gate import PLATFORM_PUBLISHER  # noqa: E402
from core.models.socials import SocialPost, SocialPostTarget  # noqa: E402
from modules.socials import notify, publish_lifecycle, publish_records, publisher, service  # noqa: E402
from modules.socials.capabilities import SEEDED_CHANNELS, parse_channel_adapters  # noqa: E402
from modules.socials.media_urls import MediaFile, NeedsPublicStorage  # noqa: E402
from modules.socials.publish_steps import Runtime  # noqa: E402
from modules.socials.publishing import run_publish  # noqa: E402
from modules.socials.step_results import lookup, poll_state  # noqa: E402

WS = uuid.uuid4()
AUTHOR, REVIEWER = "user-author", "user-reviewer"
MANIFEST = _ORCH / "reports" / "route-manifest.json"
SURFACE = _ORCH / "reports" / "config-surface.json"
API_CLIENT = _ORCH.parent / "frontend" / "lib" / "api-client.ts"
ME_URN = "urn:li:person:abc"
SHARE_URN = "urn:li:share:7001"
VIDEO_URN = "urn:li:video:9001"
IMAGE = MediaFile(aspect="1:1", deliverable_id=str(uuid.uuid4()), name="card.png", key="k/card.png", content_type="image/png", bytes=10)
VIDEO = MediaFile(aspect="9:16", deliverable_id=str(uuid.uuid4()), name="video.mp4", key="k/video.mp4", content_type="video/mp4", bytes=99)


# ---------------------------------------------------------------------------
# The harness: a SQLite file (a connection per session), a recording executor
# ---------------------------------------------------------------------------


def ok(data):
    """An executor result as Composio answers: its own envelope inside ``data``."""
    return {"success": True, "data": {"data": data, "successful": True, "error": None}, "error": None}


def refused(message, error_type=None):
    return {"success": False, "data": None, "error": message, "error_type": error_type}


class FakeExecutor:
    """Records every call; answers each action from its script (a list: one answer per call)."""

    def __init__(self, script):
        self.script = {action: list(answers) for action, answers in script.items()}
        self.calls = []
        self.lock = threading.Lock()

    async def execute_with_uploads(self, action, params, *, agent_id, workspace_id, app_name, upload_params=(), way_through=None):
        with self.lock:
            self.calls.append(SimpleNamespace(
                action=action, params=dict(params), upload_params=tuple(upload_params), way_through=way_through,
                app_name=app_name, workspace_id=workspace_id,
            ))
            answers = self.script.get(action) or [refused(f"unscripted {action}")]
            return answers.pop(0) if len(answers) > 1 else answers[0]

    @property
    def actions(self):
        return [call.action for call in self.calls]


class FakeStager:
    def __init__(self, folder):
        self.folder = folder

    async def stage(self, media, *, jpeg=False):
        path = self.folder / media.name
        path.write_bytes(b"x" * (media.bytes or 1))
        return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = sa.create_engine(
        f"sqlite:///{tmp_path / 'publish.db'}", connect_args={"check_same_thread": False, "timeout": 30}, poolclass=NullPool,
    )
    SocialPost.metadata.create_all(engine, tables=[SocialPost.__table__, SocialPostTarget.__table__])
    factory = sessionmaker(bind=engine)
    state = SimpleNamespace(factory=factory, media=[], notices=[], sleeps=[], clock=[0.0], tmp=tmp_path)
    monkeypatch.setattr(publish_records.media_urls, "resolve_post_media", lambda db, post: list(state.media))

    class Dispatcher:
        async def dispatch(self, **notice):
            state.notices.append(notice)

    monkeypatch.setattr(notify, "_dispatcher", lambda db, ws: Dispatcher())
    return state


def _stored_steps(toolkit, kind):
    return [step_plan(step) for step in SEEDED_CHANNELS[toolkit].kinds[kind]]


def _target(toolkit, kind, **options):
    return {"toolkit": toolkit, "post_kind": kind, "options": options, "steps": _stored_steps(toolkit, kind)}


def _approved_post(env, *targets, copy=None):
    """An approved post with ``targets``, committed; its id."""
    with env.factory() as db:
        post = service.create_draft(db, workspace_id=WS, created_by=AUTHOR, title="Harvest Club", copy=copy or {"base": "Harvest Club opens Friday."})
        db.flush()
        service.update_post(post, AUTHOR, {"targets": list(targets)})
        service.submit(post, AUTHOR)
        service.approve(post, REVIEWER, content_hash=post.content_hash)
        db.commit()
        return post.id


def _claim(env, post_id, begin=publisher.begin_publish):
    with env.factory() as db:
        return begin(db, service.get_post(db, WS, post_id), REVIEWER)


def _runtime(env):
    def build(executor, stager):
        async def sleep(seconds):
            env.sleeps.append(seconds)
            env.clock[0] += seconds

        return Runtime(executor=executor, stager=FakeStager(env.tmp), sleep=sleep, clock=lambda: env.clock[0])

    return build


def _publish(env, post_id, executor, begin=publisher.begin_publish):
    job = _claim(env, post_id, begin)
    return asyncio.run(run_publish(job, executor=executor, session_factory=env.factory, runtime=_runtime(env)))


def _post(env, post_id):
    with env.factory() as db:
        post = service.get_post(db, WS, post_id)
        return SimpleNamespace(**post.to_dict())


def _targets(env, post_id):
    return {(t["toolkit"], t["post_kind"]): t for t in _post(env, post_id).targets}


LINKEDIN = {
    "LINKEDIN_GET_MY_INFO": [ok({"response_dict": {"author_id": ME_URN, "name": "Ana"}})],
    "LINKEDIN_CREATE_LINKED_IN_POST": [ok({"id": SHARE_URN})],
    "LINKEDIN_UPLOAD_VIDEO": [ok({"video_urn": VIDEO_URN})],
    "LINKEDIN_CREATE_VIDEO_POST": [ok({"id": SHARE_URN})],
}


# ---------------------------------------------------------------------------
# D6: the guard first; publish once
# ---------------------------------------------------------------------------


def test_a_stale_approval_claims_nothing_and_calls_nothing(env):
    post_id = _approved_post(env, _target("linkedin", "text"))
    with env.factory() as db:
        post = service.get_post(db, WS, post_id)
        post.approved_hash = "f" * 64
        db.commit()
    with pytest.raises(service.NotPublishable):
        _claim(env, post_id)
    assert _post(env, post_id).status == "approved"


def test_a_run_whose_post_lost_its_approval_publishes_nothing(env):
    post_id = _approved_post(env, _target("linkedin", "text"))
    job = _claim(env, post_id)
    with env.factory() as db:  # the approval no longer matches (a forged write, say)
        db.get(SocialPost, post_id).approved_hash = "f" * 64
        db.commit()
    executor = FakeExecutor(LINKEDIN)
    # Nothing is published, and the post is ended rather than left publishing (review).
    assert asyncio.run(run_publish(job, executor=executor, session_factory=env.factory, runtime=_runtime(env))) == "failed"
    assert executor.calls == []
    assert _post(env, post_id).status == "failed"


def test_a_post_claimed_once_publishes_once(env):
    post_id = _approved_post(env, _target("linkedin", "text"))
    job = _claim(env, post_id)
    with pytest.raises(service.NotPublishable):
        _claim(env, post_id)  # the second click: the post is publishing
    executor = FakeExecutor(LINKEDIN)

    async def twice():
        run = lambda: run_publish(job, executor=executor, session_factory=env.factory, runtime=_runtime(env))  # noqa: E731
        return await asyncio.gather(run(), run())

    ended = asyncio.run(twice())
    assert executor.actions.count("LINKEDIN_CREATE_LINKED_IN_POST") == 1
    # The run that held nothing leaves the ending to the run publishing the target.
    assert sorted(ended, key=str) == [None, "published"] and _post(env, post_id).status == "published"


def test_a_post_with_no_channels_is_not_published():
    post = service.create_draft(SimpleNamespace(add=lambda obj: None), workspace_id=WS, created_by=AUTHOR, title="T")
    service.submit(post, AUTHOR)
    service.approve(post, REVIEWER, content_hash=post.content_hash)
    with pytest.raises(service.NotPublishable, match="no channels"):
        publish_lifecycle.start_publish(post, REVIEWER)


# ---------------------------------------------------------------------------
# D8: retries, the platform's message, partial success, retry
# ---------------------------------------------------------------------------


def test_a_transient_error_tries_again_up_to_the_limit_then_fails_the_target(env, monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_MAX_TARGET_ATTEMPTS", 3)
    monkeypatch.setattr(config, "SOCIALS_PUBLISH_RETRY_BACKOFF_SECONDS", 7)
    post_id = _approved_post(env, _target("linkedin", "text"))
    executor = FakeExecutor({**LINKEDIN, "LINKEDIN_CREATE_LINKED_IN_POST": [refused("429 Too Many Requests")]})

    assert _publish(env, post_id, executor) == "failed"

    # The account lookup ran once: an attempt resumes at the step that failed.
    assert executor.actions == ["LINKEDIN_GET_MY_INFO"] + ["LINKEDIN_CREATE_LINKED_IN_POST"] * 3
    assert env.sleeps == [7, 14]
    target = _targets(env, post_id)["linkedin", "text"]
    assert target["attempts"] == 3 and target["status"] == "failed" and "429" in target["error"]
    assert [n["event_type"] for n in env.notices] == ["social_post_failed"]


def test_a_transient_error_then_success_publishes(env):
    post_id = _approved_post(env, _target("linkedin", "text"))
    executor = FakeExecutor({**LINKEDIN, "LINKEDIN_CREATE_LINKED_IN_POST": [refused("Connection refused"), ok({"id": SHARE_URN})]})
    assert _publish(env, post_id, executor) == "published"
    assert _targets(env, post_id)["linkedin", "text"]["attempts"] == 2


def test_a_4xx_is_not_retried_and_keeps_the_platform_message(env):
    post_id = _approved_post(env, _target("linkedin", "text"))
    message = "422 Unprocessable Entity: commentary is too long"
    executor = FakeExecutor({**LINKEDIN, "LINKEDIN_CREATE_LINKED_IN_POST": [refused(message)]})

    assert _publish(env, post_id, executor) == "failed"

    assert executor.actions.count("LINKEDIN_CREATE_LINKED_IN_POST") == 1 and env.sleeps == []
    target = _targets(env, post_id)["linkedin", "text"]
    assert target["attempts"] == 1 and message in target["error"]


def test_one_target_of_two_failing_ends_partially_published_and_retry_runs_only_it(env):
    env.media = [VIDEO]
    post_id = _approved_post(env, _target("linkedin", "text"), _target("linkedin", "video"))
    executor = FakeExecutor({**LINKEDIN, "LINKEDIN_UPLOAD_VIDEO": [refused("400 Bad Request: the video is too long")]})

    assert _publish(env, post_id, executor) == "partially_published"

    targets = _targets(env, post_id)
    assert targets["linkedin", "text"]["status"] == "published"
    assert targets["linkedin", "video"]["status"] == "failed"
    assert [n["event_type"] for n in env.notices] == ["social_post_failed"]
    assert "partly published" in env.notices[0]["title"]

    retry = FakeExecutor(LINKEDIN)
    assert _publish(env, post_id, retry, begin=publisher.begin_retry) == "published"
    assert retry.actions == ["LINKEDIN_UPLOAD_VIDEO", "LINKEDIN_CREATE_VIDEO_POST"]  # the text post never runs again
    targets = _targets(env, post_id)
    assert targets["linkedin", "text"]["attempts"] == 1 and targets["linkedin", "video"]["attempts"] == 2
    assert [e["action"] for e in _post(env, post_id).review_log[-3:]] == ["partially_published", "retry", "published"]


def test_retry_needs_a_failed_target_and_a_matching_approval(env):
    post_id = _approved_post(env, _target("linkedin", "text"))
    with pytest.raises(service.NotPublishable, match="cannot be retried"):
        _claim(env, post_id, publisher.begin_retry)  # approved: nothing to retry
    assert _publish(env, post_id, FakeExecutor({**LINKEDIN, "LINKEDIN_CREATE_LINKED_IN_POST": [refused("401 Unauthorized")]})) == "failed"
    with env.factory() as db:
        db.get(SocialPost, post_id).approved_hash = "f" * 64
        db.commit()
    with pytest.raises(service.NotPublishable, match="approve it again"):
        _claim(env, post_id, publisher.begin_retry)


def test_a_render_that_failed_is_not_a_publish_to_retry():
    post = service.create_draft(SimpleNamespace(add=lambda obj: None), workspace_id=WS, created_by=AUTHOR, title="T")
    service.start_render(post, AUTHOR)
    service.fail_render(post, AUTHOR, "The check failed.")
    with pytest.raises(service.NotPublishable):
        publish_lifecycle.assert_retryable(post)


@pytest.mark.parametrize("message, transient", [
    ("LinkedIn answered 503 Service Unavailable", True),
    ("Read timed out", True),
    ("429 Too Many Requests", True),
    ("Connection reset by peer", True),
    ("422 Unprocessable Entity: commentary exceeds 3000 characters", False),
    ("the text exceeds 500 characters", False),
    ("400 Bad Request: the timeout param is invalid", False),
    ("401 Unauthorized", False),
])
def test_only_a_transient_failure_is_tried_again(message, transient):
    from modules.socials.publish_steps import _failure

    step = SEEDED_CHANNELS["linkedin"].kinds["text"][0]  # the account lookup: a read
    assert step.step_class != "publish"
    assert _failure(step, refused(message)).transient is transient


@pytest.mark.parametrize("message, transient", [
    ("429 Too Many Requests", True),
    ("Connection refused", True),
    ("LinkedIn answered 503 Service Unavailable", False),
    ("Read timed out", False),
    ("Connection reset by peer", False),
])
def test_a_publish_step_is_tried_again_only_when_nothing_reached_the_platform(message, transient):
    from modules.socials.publish_steps import MAY_BE_LIVE, _failure

    step = SEEDED_CHANNELS["linkedin"].kinds["text"][1]
    assert step.step_class == "publish"
    failure = _failure(step, refused(message))
    assert failure.transient is transient
    assert (MAY_BE_LIVE in failure.message) is not transient


def test_a_publish_that_timed_out_is_not_sent_again_and_says_it_may_be_live(env):
    post_id = _approved_post(env, _target("linkedin", "text"))
    executor = FakeExecutor({**LINKEDIN, "LINKEDIN_CREATE_LINKED_IN_POST": [refused("Read timed out"), ok({"id": SHARE_URN})]})

    assert _publish(env, post_id, executor) == "failed"

    assert executor.actions.count("LINKEDIN_CREATE_LINKED_IN_POST") == 1 and env.sleeps == []
    target = _targets(env, post_id)["linkedin", "text"]
    assert target["attempts"] == 1 and "check the channel before you retry" in target["error"]


def test_a_refused_step_fails_its_target_and_is_never_retried(env):
    post_id = _approved_post(env, _target("linkedin", "text"))
    denial = refused("LINKEDIN_CREATE_LINKED_IN_POST is on the platform deny list (timeout)", "action_denied")
    executor = FakeExecutor({**LINKEDIN, "LINKEDIN_CREATE_LINKED_IN_POST": [denial]})
    assert _publish(env, post_id, executor) == "failed"
    assert executor.actions.count("LINKEDIN_CREATE_LINKED_IN_POST") == 1  # 'timeout' in a refusal is not transient
    assert "deny list" in _targets(env, post_id)["linkedin", "text"]["error"]


def test_an_optional_step_that_cannot_run_is_skipped_and_noted(env):
    env.media = [VIDEO, IMAGE]
    data = {"youtube": {"label": "YouTube", "kinds": {"video": [
        {"id": "upload", "action": "YOUTUBE_UPLOAD_VIDEO", "class": "publish", "params": {"title": "$title", "videoFilePath": "$media"},
         "files": ["videoFilePath"], "returns": {"id": "id"}, "permalink": "https://www.youtube.com/watch?v={id}"},
        {"id": "thumbnail", "action": "YOUTUBE_UPDATE_THUMBNAIL", "class": "upload",
         "params": {"videoId": "$steps.upload", "thumbnailUrl": "$thumbnail"}, "urls": ["thumbnailUrl"], "optional": True},
    ]}}}
    steps = [step_plan(step) for step in parse_channel_adapters(data)["youtube"].kinds["video"]]
    post_id = _approved_post(env, {"toolkit": "youtube", "post_kind": "video", "options": {}, "steps": steps})
    executor = FakeExecutor({"YOUTUBE_UPLOAD_VIDEO": [ok({"id": "yt1"})]})

    def no_public_storage(post, media):
        raise NeedsPublicStorage()

    job = _claim(env, post_id)

    def build(executor, stager):
        runtime = _runtime(env)(executor, stager)
        runtime.public_link = no_public_storage
        return runtime

    assert asyncio.run(run_publish(job, executor=executor, session_factory=env.factory, runtime=build)) == "published"
    assert executor.actions == ["YOUTUBE_UPLOAD_VIDEO"]
    target = _targets(env, post_id)["youtube", "video"]
    assert target["permalink"] == "https://www.youtube.com/watch?v=yt1"
    assert target["notes"] and "Needs public storage" in target["notes"][0]


# ---------------------------------------------------------------------------
# The executor: the call's own upload spec, after the deny list and the gate
# ---------------------------------------------------------------------------


def _executor_harness(monkeypatch, *, denial=None, refusal=None):
    seen = SimpleNamespace(uploads=[], executed=[])

    async def deny(action):
        return denial

    async def gate(action, workspace_id, *, way_through=None):
        seen.way_through = way_through
        return refusal

    def spec(action, params, upload_params, toolkit):
        seen.uploads.append((action, tuple(upload_params), toolkit))
        return {**params, **{name: {"s3key": f"s3/{name}"} for name in upload_params}}

    async def execute(self, **kwargs):
        seen.executed.append(kwargs)
        return ok({"id": "1"})

    monkeypatch.setattr(upload_spec, "composio_action_denial_async", deny)
    monkeypatch.setattr(upload_spec, "post_action_refusal", gate)
    monkeypatch.setattr(upload_spec, "resolve_upload_spec", spec)
    monkeypatch.setattr(tool_executor.ComposioToolExecutor, "execute", execute)
    return seen


def _upload(params, upload_params=("media",), action="TWITTER_UPLOAD_MEDIA"):
    executor = tool_executor.ComposioToolExecutor(db=None)
    return asyncio.run(executor.execute_with_uploads(
        action, params, agent_id=0, workspace_id=WS, app_name="TWITTER", upload_params=upload_params, way_through=PLATFORM_PUBLISHER,
    ))


def test_execute_with_uploads_converts_exactly_the_named_params_then_executes(monkeypatch):
    seen = _executor_harness(monkeypatch)
    before = set(tool_executor.UPLOAD_ACTIONS)

    result = _upload({"media": Path("/tmp/a.png"), "media_type": "image/png"})

    assert result["success"] is True
    assert seen.uploads == [("TWITTER_UPLOAD_MEDIA", ("media",), "twitter")]
    (call,) = seen.executed
    assert call["params"] == {"media": {"s3key": "s3/media"}, "media_type": "image/png"}
    assert call["way_through"] is PLATFORM_PUBLISHER and call["skip_validation"] is True
    assert set(tool_executor.UPLOAD_ACTIONS) == before


@pytest.mark.parametrize("gate", ["denial", "refusal"])
def test_a_denied_or_gated_call_uploads_nothing_and_executes_nothing(monkeypatch, gate):
    seen = _executor_harness(monkeypatch, **{gate: "refused"})
    result = _upload({"media": Path("/tmp/a.png")})
    assert result["success"] is False and seen.uploads == [] and seen.executed == []


def test_a_file_the_spec_cannot_upload_fails_the_call_before_it_executes(monkeypatch):
    seen = _executor_harness(monkeypatch)

    def failing(action, params, upload_params, toolkit):
        raise upload_spec.FileUploadFailed("media could not be uploaded to Composio: 500")

    monkeypatch.setattr(upload_spec, "resolve_upload_spec", failing)
    result = _upload({"media": Path("/tmp/a.png")})
    assert result["error_type"] == upload_spec.ERROR_TYPE_FILE_UPLOAD and seen.executed == []


def test_a_call_with_its_own_upload_spec_gets_no_global_upload_conversion(monkeypatch):
    """A LinkedIn text post whose approved copy is a bare link is an UPLOAD_ACTIONS
    action: the executor's global conversion would fetch the link as a file. The
    publisher's call is sent as approved; an agent's call still gets the conversion."""
    _executor_harness(monkeypatch)
    converted = []

    async def global_conversion(action, params, workspace_id):
        converted.append(action)
        return params, []

    async def execute(self, **kwargs):
        return {"params": (await self._resolve_file_uploads(kwargs["action"], kwargs["params"], kwargs["workspace_id"]))[0]}

    monkeypatch.setattr(tool_executor, "resolve_file_uploads", global_conversion)
    monkeypatch.setattr(tool_executor.ComposioToolExecutor, "execute", execute)
    copy = {"commentary": "https://harvest.example/club"}
    assert "LINKEDIN_CREATE_LINKED_IN_POST" in tool_executor.UPLOAD_ACTIONS

    result = _upload(copy, (), "LINKEDIN_CREATE_LINKED_IN_POST")
    assert result["params"] == copy and converted == []

    agent_call = tool_executor.ComposioToolExecutor(db=None)
    asyncio.run(agent_call.execute(action="LINKEDIN_CREATE_LINKED_IN_POST", params=copy, workspace_id=WS))
    assert converted == ["LINKEDIN_CREATE_LINKED_IN_POST"]  # the flag does not outlive the call


def test_the_linkedin_image_workaround_takes_the_staged_files_itself(monkeypatch):
    seen = _executor_harness(monkeypatch)
    _upload({"images": [Path("/tmp/a.png")], "commentary": "Hi"}, ("images",), "LINKEDIN_CREATE_LINKED_IN_POST")
    assert seen.uploads == [] and seen.executed[0]["params"]["images"] == [Path("/tmp/a.png")]


def test_resolve_upload_spec_takes_only_staged_files(monkeypatch, tmp_path):
    uploaded = []

    class Uploadable:
        @staticmethod
        def from_path(**kwargs):
            uploaded.append(kwargs)
            return SimpleNamespace(model_dump=lambda: {"s3key": kwargs["file"].name})

    monkeypatch.setattr(upload_spec, "_file_uploadable_class", lambda: Uploadable)
    monkeypatch.setattr(upload_spec, "get_composio_client", lambda: SimpleNamespace(composio=SimpleNamespace(client="http")))
    staged = tmp_path / "v.mp4"
    staged.write_bytes(b"v")

    params = upload_spec.resolve_upload_spec("LINKEDIN_UPLOAD_VIDEO", {"file": staged, "note": "/etc/passwd"}, ["file"], "linkedin")
    assert params == {"file": {"s3key": "v.mp4"}, "note": "/etc/passwd"}
    assert uploaded[0]["toolkit"] == "linkedin" and uploaded[0]["tool"] == "linkedin-upload-video"
    with pytest.raises(upload_spec.FileUploadFailed):  # a string never reads the local disk
        upload_spec.resolve_upload_spec("LINKEDIN_UPLOAD_VIDEO", {"file": "/etc/passwd"}, ["file"], "linkedin")
    with pytest.raises(upload_spec.FileUploadFailed):
        upload_spec.resolve_upload_spec("LINKEDIN_UPLOAD_VIDEO", {}, ["file"], "linkedin")


def test_the_linkedin_workaround_reads_a_staged_file_and_never_a_string_path(tmp_path):
    from core.composio import linkedin_image_workaround as workaround

    staged = tmp_path / "card.png"
    staged.write_bytes(b"png")
    assert workaround._extract_image_paths({"images": [staged]}) == [staged]
    got = asyncio.run(workaround._download_image(staged, ws_client=None, http=None))
    assert got == (b"png", None)


# ---------------------------------------------------------------------------
# The data: what a step returns, and when a status step is done
# ---------------------------------------------------------------------------


def test_lookup_finds_a_path_through_composio_envelopes():
    output = {"data": {"response_dict": {"author_id": ME_URN}}}
    assert lookup(output, "author_id|sub") == ME_URN
    assert lookup(output, "missing|response_dict.author_id") == ME_URN
    assert lookup(output, "nothing") is None


def test_poll_state_reads_done_failed_and_pending():
    until = {"path": "status", "done": ("PUBLISH_COMPLETE",), "failed": ("FAILED",), "error": "fail_reason", "absent": "pending"}
    assert poll_state({"status": "PUBLISH_COMPLETE"}, until) == ("done", None)
    assert poll_state({"status": "FAILED", "fail_reason": "spam_risk"}, until) == ("failed", "spam_risk")
    assert poll_state({"status": "PROCESSING_UPLOAD"}, until) == ("pending", None)
    assert poll_state({}, until) == ("pending", None)


def test_a_step_source_must_name_an_earlier_step_that_returns_an_id():
    data = {"linkedin": {"label": "L", "kinds": {"text": [
        {"id": "me", "action": "LINKEDIN_GET_MY_INFO", "class": "read"},
        {"id": "post", "action": "LINKEDIN_CREATE_LINKED_IN_POST", "class": "publish", "params": {"author": "$steps.me"}},
    ]}}}
    with pytest.raises(ValueError, match="returns an id"):
        parse_channel_adapters(data)


def test_a_media_param_must_be_a_file_or_a_link():
    data = {"linkedin": {"label": "L", "kinds": {"video": [
        {"id": "post", "action": "LINKEDIN_CREATE_VIDEO_POST", "class": "publish", "params": {"video": "$media"}},
    ]}}}
    with pytest.raises(ValueError, match="files or urls"):
        parse_channel_adapters(data)


def test_every_seeded_publish_step_says_what_its_receipt_is():
    for toolkit, adapter in SEEDED_CHANNELS.items():
        for kind, steps in adapter.kinds.items():
            publish = [step for step in steps if step.step_class == "publish"]
            assert publish and all("id" in step.returns for step in publish), f"{toolkit} {kind}"


# ---------------------------------------------------------------------------
# Routes, settings, the client, the reaper
# ---------------------------------------------------------------------------


def test_the_publish_routes_are_plain_defs_in_the_manifest_and_the_client_posts_them():
    routes = {route.path: route for route in socials_publish.router.routes if isinstance(route, APIRoute)}
    manifest = {(r["method"], r["path"]) for r in json.loads(MANIFEST.read_text(encoding="utf-8"))["routes"]}
    client = API_CLIENT.read_text(encoding="utf-8")
    for path, name in (("/posts/{post_id}/publish-now", "publishSocialPostNow"), ("/posts/{post_id}/retry", "retrySocialPost")):
        route = routes[path]
        assert route.methods == {"POST"} and route.status_code == 202
        assert not inspect.iscoroutinefunction(route.endpoint)  # F105
        assert ("POST", f"/api/socials{path}") in manifest
        segment = path.rsplit("/", 1)[-1]
        body = client[client.index(f"async {name}("):]
        assert f"/api/socials/posts/${{postId}}/{segment}`, {{ method: 'POST' }}" in body[:400]


def test_the_manifest_route_count_matches():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["route_count"] == len(manifest["routes"])


def test_the_publisher_settings_are_config_and_named_in_the_surface():
    names = ("SOCIALS_PUBLISH_RETRY_BACKOFF_SECONDS", "SOCIALS_PUBLISH_POLL_SECONDS",
             "SOCIALS_PUBLISH_MAX_WAIT_SECONDS", "SOCIALS_PUBLISH_RUN_MAX_SECONDS")
    surface = json.loads(SURFACE.read_text(encoding="utf-8"))["settings"]
    for name in names:
        assert isinstance(getattr(config, name), int) and name in surface
    assert config.SOCIALS_PUBLISH_RUN_MAX_SECONDS < config.BOOT_REAPER_STALE_MINUTES * 60


def test_the_new_notification_events_are_valid():
    from core.services.notification_dispatcher import VALID_EVENT_TYPES

    assert {"social_post_published", "social_post_failed"} <= VALID_EVENT_TYPES


# ---------------------------------------------------------------------------
# @integration: one claim, one sequence, on the CI Postgres
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_engine():
    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1 FROM social_posts LIMIT 1"))
            conn.execute(sa.text("SELECT 1 FROM workspaces LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("CI"):
            raise
        pytest.skip(f"the Postgres check needs the test database: {exc}")
    yield engine
    engine.dispose()


@pytest.mark.integration
def test_two_concurrent_publishes_of_one_post_issue_one_sequence_on_postgres(pg_engine, monkeypatch):
    from core.models.workspaces import Workspace

    monkeypatch.setattr(publish_records.media_urls, "resolve_post_media", lambda db, post: [])
    monkeypatch.setattr(notify, "_dispatcher", lambda db, ws: SimpleNamespace(dispatch=lambda **k: asyncio.sleep(0)))
    factory = sessionmaker(bind=pg_engine)
    workspace_id = uuid.uuid4()
    with factory() as db:
        db.add(Workspace(id=workspace_id, name="w3-publish-once", plan="basic", plan_limits={}, settings={}))
        db.commit()
        post = service.create_draft(db, workspace_id=workspace_id, created_by=AUTHOR, title="Once", copy={"base": "Once."})
        db.flush()
        service.update_post(post, AUTHOR, {"targets": [_target("linkedin", "text")]})
        service.submit(post, AUTHOR)
        service.approve(post, REVIEWER, content_hash=post.content_hash)
        db.commit()
        post_id = post.id
    barrier, outcomes = threading.Barrier(2), []

    def claim():
        with factory() as db:
            loaded = service.get_post(db, workspace_id, post_id)
            barrier.wait(10)
            try:
                outcomes.append(publisher.begin_publish(db, loaded, REVIEWER))
            except service.SocialsError as exc:
                outcomes.append(exc)

    try:
        threads = [threading.Thread(target=claim) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(20)
        jobs = [o for o in outcomes if isinstance(o, publish_records.PublishJob)]
        assert len(jobs) == 1 and isinstance(next(o for o in outcomes if o not in jobs), service.StaleContent)
        executor = FakeExecutor(LINKEDIN)

        async def double_fire():
            run = lambda: run_publish(jobs[0], executor=executor, session_factory=factory)  # noqa: E731
            return await asyncio.gather(run(), run())

        ended = asyncio.run(double_fire())
        assert sorted(ended, key=str) == [None, "published"]
        assert executor.actions.count("LINKEDIN_CREATE_LINKED_IN_POST") == 1
        with factory() as db:
            assert service.get_post(db, workspace_id, post_id).status == "published"
    finally:
        with factory() as db:
            db.execute(sa.text("DELETE FROM social_posts WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": str(workspace_id)})
            db.execute(sa.text("DELETE FROM workspaces WHERE id = CAST(:ws AS uuid)"), {"ws": str(workspace_id)})
            db.commit()
