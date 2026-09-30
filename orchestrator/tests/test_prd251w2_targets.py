"""PRD-251 Wave 2, US-204 (S3.3-prep, D6) — channels are approved content: the
targets and the hash.

Before this story the content hash covered copy, variables, sources, format,
template_id and media, but not where the post goes: a channel added after
approval would have published unapproved. Pinned here:

* the hash covers the post's sorted target set, each target's toolkit, post kind
  and options, and not its action plan; a post with no target hashes exactly as
  before, so no stored hash or approval moved;
* adding, removing or changing a target of an approved or scheduled post moves it
  back to needs_approval, logs approval_voided, and ``assert_publishable`` refuses
  the old approval; the same set again keeps it;
* the targets are written through ``service.update_post``: one row per (toolkit,
  post kind), keyed ``sp:{post_id}:{toolkit}:{post_kind}``, a kept channel keeping
  its row; a malformed set is refused and changes nothing;
* ``PUT /api/socials/posts/{post_id}/targets``, on the US-203 registry's harness
  (connections, the action cache as the bulk sync leaves it, the deny list): it
  refuses a kind the registry marks unavailable (422 with the registry's reason),
  a channel that is not connected, a kind the channel does not post and an option
  the kind does not read, writing nothing; it writes one target per kind with its
  resolved steps; it voids an approval and lets the new version be approved; it is
  the same compare-and-set as a PATCH; a plain ``def``, gated, in the committed
  manifest;
* ``SocialPost.to_dict()`` carries each target's options, status and receipt.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
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

from fastapi.routing import APIRoute  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

import api.socials as socials_api  # noqa: E402
import api.socials_targets as targets_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import modules.socials.service as service  # noqa: E402
import tests.test_prd251w2_channels as channels_harness  # noqa: E402
from core.models.socials import SOCIAL_TARGET_POST_KINDS, SocialPost, SocialPostTarget  # noqa: E402
from modules.socials import capabilities  # noqa: E402
from modules.socials import targets as post_targets  # noqa: E402
from modules.socials.capabilities import SEEDED_CHANNELS  # noqa: E402
from modules.socials.service import APPROVED, DRAFT, NEEDS_APPROVAL, SCHEDULED, InvalidPost, NotPublishable  # noqa: E402
from tests.test_prd251w2_channels import MANIFEST, OTHER_WS, WS, _cache_channel, _connect, _ctx, _deny  # noqa: E402

# The US-203 registry harness: SQLite copies of the Composio tables, the deny list
# and Wave 1's settings as the migrations seed them, both switches on.
channels = channels_harness.channels

AUTHOR = "user-author"
REVIEWER = "user-reviewer"
AGENT = "Social Media Director"
SLOT = datetime.now(timezone.utc) + timedelta(days=30)
ROUTE = "/api/socials/posts/{post_id}/targets"
# A kind's resolved sequence as the api hands it to the service (the registry's shape).
STEPS = [{"id": "post", "action": "EXAMPLE_CREATE_POST", "class": "publish", "params": {"text": "$copy"}}]
OTHER_STEPS = [{"id": "post", "action": "EXAMPLE_CREATE_POST_V2", "class": "publish", "params": {"body": "$copy"}}]
TARGET_FIELDS = {"id", "toolkit", "post_kind", "options", "status", "attempts", "remote_id", "permalink", "error", "published_at", "notes"}


def _t(toolkit, post_kind, steps=STEPS, **options):
    return {"toolkit": toolkit, "post_kind": post_kind, "options": options, "steps": steps}


LINKEDIN_TEXT = _t("linkedin", "text")
TIKTOK_VIDEO = _t("tiktok", "video", privacy_level="SELF_ONLY", is_aigc=True)


# ---------------------------------------------------------------------------
# The service: what the hash covers, and how targets are written
# ---------------------------------------------------------------------------


class _Added:
    """Stands in for the session ``create_draft`` adds to."""

    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)


def _draft():
    return service.create_draft(
        _Added(), workspace_id=WS, created_by=AUTHOR, title="Harvest Club",
        copy={"base": "Harvest Club opens Friday."}, format="image",
    )


def _post_in(status, targets=(LINKEDIN_TEXT,)):
    """A post with ``targets`` that reached ``status`` through the service."""
    post = _draft()
    service.update_post(post, AUTHOR, {"targets": list(targets)})
    if status == DRAFT:
        return post
    service.submit(post, AUTHOR)
    if status == NEEDS_APPROVAL:
        return post
    service.approve(post, REVIEWER, content_hash=post.content_hash)
    if status == APPROVED:
        return post
    if status == SCHEDULED:
        return service.schedule(post, REVIEWER, SLOT, "Europe/Lisbon")
    raise AssertionError(status)


def _sha256(content):
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _six_fields(post):
    return {
        "copy": post.copy, "variables": post.variables or {}, "sources": post.sources or {},
        "format": post.format, "template_id": None, "media": post.media or {},
    }


def test_the_targets_are_content():
    assert service.CONTENT_FIELDS[-1] == "targets" and "targets" in service.EDITABLE_FIELDS
    assert service.RENDER_FIELDS == ("voice", "footage")  # still render settings, never hashed


def test_a_post_with_no_target_hashes_exactly_as_before():
    """No stored hash moved: before US-204 no row of social_post_targets was ever
    written, and a post with none hashes over the six fields as it always did."""
    post = _draft()
    assert post.targets == [] and post.content_hash == _sha256(_six_fields(post))
    assert service.compute_content_hash(post) == post.content_hash


def test_the_hash_covers_the_sorted_target_set_and_not_the_action_plan():
    post = _draft()
    service.update_post(post, AUTHOR, {"targets": [TIKTOK_VIDEO, LINKEDIN_TEXT]})

    expected = _sha256({
        **_six_fields(post),
        "targets": [
            {"toolkit": "linkedin", "post_kind": "text", "options": {}},
            {"toolkit": "tiktok", "post_kind": "video", "options": {"is_aigc": True, "privacy_level": "SELF_ONLY"}},
        ],
    })
    assert post.content_hash == service.compute_content_hash(post) == expected
    before = post.content_hash

    # The same channels, listed the other way round and resolved to other steps: the same content.
    service.update_post(post, AUTHOR, {"targets": [_t("linkedin", "text", OTHER_STEPS), {**TIKTOK_VIDEO, "steps": OTHER_STEPS}]})
    assert post.content_hash == before
    assert {row.toolkit: row.action_plan["steps"] for row in post.targets} == {"linkedin": OTHER_STEPS, "tiktok": OTHER_STEPS}

    # A post as to_dict answers it hashes the same as the row.
    assert service.compute_content_hash(SimpleNamespace(**post.to_dict())) == before


@pytest.mark.parametrize(
    "targets",
    [
        [LINKEDIN_TEXT, TIKTOK_VIDEO],  # a channel added
        [],  # every channel removed
        [_t("linkedin", "text", author="urn:li:organization:42")],  # an option changed
        [_t("linkedin", "image")],  # the kind changed
        [_t("twitter", "text")],  # another channel
    ],
    ids=["added", "removed", "option-changed", "kind-changed", "channel-swapped"],
)
@pytest.mark.parametrize("start", [APPROVED, SCHEDULED])
def test_changing_an_approved_or_scheduled_posts_targets_voids_the_approval(start, targets):
    post = _post_in(start)
    service.assert_publishable(post)  # publishable before the change
    approved_hash = post.approved_hash

    service.update_post(post, AUTHOR, {"targets": targets})

    assert post.status == NEEDS_APPROVAL
    assert post.review_log[-1]["action"] == "approval_voided"
    assert post.approved_hash == approved_hash != post.content_hash
    with pytest.raises(NotPublishable, match="cannot be published"):
        service.assert_publishable(post)
    # Even forced back to its old status, the old approval does not cover the new channels.
    post.status = start
    with pytest.raises(NotPublishable, match="changed after it was approved"):
        service.assert_publishable(post)
    post.status = NEEDS_APPROVAL
    # Approved again, the new channels are what the approval covers.
    service.approve(post, REVIEWER, content_hash=post.content_hash)
    service.assert_publishable(post)
    assert post.approved_hash == service.compute_content_hash(post)


def test_a_target_changed_behind_the_services_back_is_not_publishable():
    post = _post_in(APPROVED)
    post.targets[0].action_plan = {"options": {"author": "urn:li:person:someone-else"}, "steps": STEPS}
    with pytest.raises(NotPublishable, match="changed after it was approved"):
        service.assert_publishable(post)


def test_the_same_channels_again_keep_the_approval():
    post = _post_in(APPROVED)
    log = list(post.review_log)
    service.update_post(post, AUTHOR, {"targets": [_t("linkedin", "text", OTHER_STEPS)]})
    assert post.status == APPROVED and post.review_log == log
    service.assert_publishable(post)
    assert post.targets[0].action_plan["steps"] == OTHER_STEPS  # the new resolution is kept


def test_a_draft_keeps_its_status_and_rehashes():
    post = _draft()
    before = post.content_hash
    service.update_post(post, AUTHOR, {"targets": [LINKEDIN_TEXT]})
    assert post.status == DRAFT and post.content_hash != before
    assert post.review_log == []  # a person's own save is not logged


def test_one_row_per_channel_and_kind_keyed_sp_and_a_kept_channel_keeps_its_row():
    post = _draft()
    service.update_post(post, AUTHOR, {"targets": [LINKEDIN_TEXT, _t("twitter", "text")]})
    rows = {(row.toolkit, row.post_kind): row for row in post.targets}
    linkedin = rows["linkedin", "text"]
    assert {key: row.idempotency_key for key, row in rows.items()} == {
        ("linkedin", "text"): f"sp:{post.id}:linkedin:text",
        ("twitter", "text"): f"sp:{post.id}:twitter:text",
    }
    assert all(row.status == "pending" and row.attempts == 0 and row.id for row in rows.values())
    assert linkedin.action_plan == {"options": {}, "steps": STEPS}

    service.update_post(post, AUTHOR, {"targets": [_t("linkedin", "text", author="urn:li:person:7"), TIKTOK_VIDEO]})

    rows = {(row.toolkit, row.post_kind): row for row in post.targets}
    assert set(rows) == {("linkedin", "text"), ("tiktok", "video")}
    assert rows["linkedin", "text"] is linkedin and linkedin.idempotency_key == f"sp:{post.id}:linkedin:text"
    assert linkedin.options == {"author": "urn:li:person:7"}
    assert rows["tiktok", "video"].idempotency_key == f"sp:{post.id}:tiktok:video"
    assert rows["tiktok", "video"].options == {"privacy_level": "SELF_ONLY", "is_aigc": True}


def test_the_longest_key_fits_its_column():
    longest_kind = max(SOCIAL_TARGET_POST_KINDS, key=len)
    longest = post_targets.target_key(uuid.uuid4(), "t" * post_targets.TOOLKIT_MAX_CHARS, longest_kind)
    assert len(longest) <= SocialPostTarget.__table__.c.idempotency_key.type.length


def test_an_agents_channel_change_is_logged_naming_it():
    post = _post_in(DRAFT)
    service.update_post(post, "agent:7", {"targets": [TIKTOK_VIDEO]}, agent=AGENT)
    entry = post.review_log[-1]
    assert (entry["action"], entry["agent"], entry["fields"]) == ("edit", AGENT, ["targets"])


@pytest.mark.parametrize(
    "targets, message",
    [
        ({"toolkit": "linkedin"}, "must be a list"),
        (["linkedin"], "must be an object"),
        ([{**LINKEDIN_TEXT, "action_plan": {}}], "keys must be"),
        ([_t("Linked In", "text")], "toolkit must name"),
        ([_t("linkedin:text", "text")], "toolkit must name"),
        ([_t("t" * 65, "text")], "toolkit must name"),
        ([_t("linkedin", "tweet")], "post_kind must be one of"),
        ([LINKEDIN_TEXT, _t("LinkedIn", "text")], "lists linkedin text twice"),
        ([{**LINKEDIN_TEXT, "options": {"Privacy": "x"}}], "is not an option name"),
        ([{**LINKEDIN_TEXT, "options": {"author": {"urn": "x"}}}], "must be text"),
        ([{**LINKEDIN_TEXT, "options": {"author": "x" * 501}}], "must be text"),
        ([{**LINKEDIN_TEXT, "options": {"tags": ["ok", 7]}}], "must be text"),
        ([{**LINKEDIN_TEXT, "options": {"score": float("nan")}}], "must be text"),
        ([{**LINKEDIN_TEXT, "steps": "post"}], "list of steps"),
        ([_t(f"toolkit_{i}", "text") for i in range(post_targets.TARGETS_MAX + 1)], "at most"),
    ],
)
def test_a_malformed_set_is_refused_and_changes_nothing(targets, message):
    post = _post_in(APPROVED)
    before = (post.status, post.content_hash, [row.idempotency_key for row in post.targets], list(post.review_log))
    with pytest.raises(InvalidPost, match=message):
        service.update_post(post, AUTHOR, {"targets": targets})
    assert (post.status, post.content_hash, [row.idempotency_key for row in post.targets], list(post.review_log)) == before


def test_an_unset_option_is_dropped_and_toolkits_are_matched_in_lower_case():
    (clean,) = service.validate_targets([{"toolkit": " LinkedIn ", "post_kind": "text", "options": {"author": None}}])
    assert clean == {"toolkit": "linkedin", "post_kind": "text", "options": {}, "steps": []}


def test_an_edit_that_loads_the_targets_mid_way_still_passes_its_compare_and_set(channels):
    """The hash loads the targets. When they were not loaded, an autoflushing
    session flushes the edit's content fields as they load, but never the status
    or the hash the compare-and-set checks: those are set after the hash is
    computed. So the check still passes, and the edit commits."""
    post = _saved_draft(channels, targets=[LINKEDIN_TEXT])
    db = channels.session
    assert db.autoflush  # the harness's session flushes before a query, as a default Session does
    status, content_hash = post.status, post.content_hash
    db.expire(post, ["targets"])

    service.update_post(post, AUTHOR, {"copy": {"base": "Harvest Club opens Saturday."}})

    assert service.claim_unchanged(db, post, status=status, content_hash=content_hash) is True
    db.commit()
    assert _row(channels, post.id).copy == {"base": "Harvest Club opens Saturday."}


# ---------------------------------------------------------------------------
# The route, on the channel registry's harness
# ---------------------------------------------------------------------------


@pytest.fixture
def api(channels, monkeypatch):
    """The US-203 harness, with the caller's workspace role (owner unless a test says otherwise)."""
    channels.role = "owner"
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: channels.role)
    return channels


def _saved_draft(env, targets=()):
    post = service.create_draft(env.session, workspace_id=WS, created_by=AUTHOR, title="Harvest Club", copy={"base": "Soon."})
    if targets:
        service.update_post(post, AUTHOR, {"targets": list(targets)})
    env.session.commit()
    return post


def _row(env, post_id):
    env.session.expire_all()
    return env.session.get(SocialPost, uuid.UUID(str(post_id)))


def _target_rows(env, post_id):
    env.session.expire_all()
    rows = env.session.query(SocialPostTarget).filter(SocialPostTarget.post_id == uuid.UUID(str(post_id))).all()
    return {(row.toolkit, row.post_kind): row for row in rows}


def _create(env):
    resp = env.client.post("/api/socials/posts", json={"title": "Harvest Club", "copy": {"base": "Harvest Club opens Friday."}})
    assert resp.status_code == 201, resp.text
    return resp.json()


def _put(env, post_id, targets):
    return env.client.put(ROUTE.format(post_id=post_id), json={"targets": targets})


def _approved(env, targets):
    post = _create(env)
    assert _put(env, post["id"], targets).status_code == 200
    submitted = env.client.post(f"/api/socials/posts/{post['id']}/submit")
    assert submitted.status_code == 200, submitted.text
    resp = env.client.post(f"/api/socials/posts/{post['id']}/approve", json={"content_hash": submitted.json()["content_hash"]})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _linkedin_and_x(env):
    _cache_channel(env, "linkedin")
    _cache_channel(env, "twitter")
    _connect(env, "LINKEDIN", "TWITTER")


def test_put_writes_one_target_per_kind_with_the_sp_key_and_its_resolved_steps(api):
    _linkedin_and_x(api)
    post = _create(api)

    resp = _put(api, post["id"], [
        {"toolkit": "twitter", "post_kind": "image"},
        {"toolkit": "linkedin", "post_kind": "text", "options": {"author": "urn:li:organization:42"}},
        {"toolkit": "LinkedIn", "post_kind": "video"},
    ])

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "draft" and body["content_hash"] != post["content_hash"]
    assert [(t["toolkit"], t["post_kind"], t["options"]) for t in body["targets"]] == [
        ("linkedin", "text", {"author": "urn:li:organization:42"}),
        ("linkedin", "video", {}),
        ("twitter", "image", {}),
    ]
    rows = _target_rows(api, post["id"])
    assert {key: row.idempotency_key for key, row in rows.items()} == {
        ("linkedin", "text"): f"sp:{post['id']}:linkedin:text",
        ("linkedin", "video"): f"sp:{post['id']}:linkedin:video",
        ("twitter", "image"): f"sp:{post['id']}:twitter:image",
    }
    for (toolkit, kind), row in rows.items():
        assert (row.status, row.attempts) == ("pending", 0)
        assert row.action_plan["steps"] == [targets_api.step_plan(step) for step in SEEDED_CHANNELS[toolkit].kinds[kind]]
    assert rows["linkedin", "text"].action_plan["steps"][1]["params"] == {"author": "$option.author|$steps.me", "commentary": "$copy"}
    assert rows["twitter", "image"].action_plan["steps"][0]["files"] == ["media"]  # the adapter's own upload spec
    assert _row(api, post["id"]).content_hash == body["content_hash"] == service.compute_content_hash(_row(api, post["id"]))


def test_put_replaces_the_set_keeping_a_kept_channels_row(api):
    _linkedin_and_x(api)
    post = _create(api)
    assert _put(api, post["id"], [{"toolkit": "linkedin", "post_kind": "text"}, {"toolkit": "twitter", "post_kind": "text"}]).status_code == 200
    kept_id = _target_rows(api, post["id"])["linkedin", "text"].id

    resp = _put(api, post["id"], [{"toolkit": "linkedin", "post_kind": "text"}, {"toolkit": "linkedin", "post_kind": "image"}])

    assert resp.status_code == 200, resp.text
    rows = _target_rows(api, post["id"])
    assert set(rows) == {("linkedin", "text"), ("linkedin", "image")}  # X's row is deleted
    assert rows["linkedin", "text"].id == kept_id

    cleared = _put(api, post["id"], [])
    assert cleared.status_code == 200 and cleared.json()["targets"] == []
    assert _target_rows(api, post["id"]) == {}
    assert cleared.json()["content_hash"] == post["content_hash"]  # no channel: the hash it started with


def test_put_refuses_a_kind_the_registry_marks_unavailable_with_its_reason(api):
    _cache_channel(api, "linkedin")
    _cache_channel(api, "instagram", without=("INSTAGRAM_CREATE_CAROUSEL_CONTAINER",))
    _cache_channel(api, "tiktok")
    _connect(api, "LINKEDIN", "INSTAGRAM", "TIKTOK")
    _deny(api, "TIKTOK_UPLOAD_VIDEO")
    post = _create(api)
    listed = {c["toolkit"]: {k["kind"]: k for k in c["post_kinds"]} for c in api.client.get("/api/socials/channels").json()}

    missing = _put(api, post["id"], [{"toolkit": "linkedin", "post_kind": "text"}, {"toolkit": "instagram", "post_kind": "carousel"}])
    denied = _put(api, post["id"], [{"toolkit": "tiktok", "post_kind": "video"}])

    assert missing.status_code == 422 and denied.status_code == 422
    assert missing.json()["detail"] == f"Instagram cannot post a carousel now: {listed['instagram']['carousel']['reason']}"
    assert capabilities.MISSING_ACTION.format(slug="INSTAGRAM_CREATE_CAROUSEL_CONTAINER") in missing.json()["detail"]
    assert denied.json()["detail"] == f"TikTok cannot post a video now: {listed['tiktok']['video']['reason']}"
    assert "TIKTOK_UPLOAD_VIDEO" in denied.json()["detail"]
    # All or nothing: the available LinkedIn target beside the refused one was not written either.
    assert _target_rows(api, post["id"]) == {}
    assert _row(api, post["id"]).content_hash == post["content_hash"]


@pytest.mark.parametrize(
    "target, message",
    [
        ({"toolkit": "youtube", "post_kind": "video"}, "youtube is not a channel connected in this workspace"),
        ({"toolkit": "instagram", "post_kind": "text"}, "Instagram does not post a text: it posts image, reel, carousel"),
        ({"toolkit": "linkedin", "post_kind": "text", "options": {"privacy_level": "PUBLIC"}}, "LinkedIn text takes author, not privacy_level"),
        ({"toolkit": "linkedin", "post_kind": "video", "options": {"author": "urn:li:person:1"}}, "LinkedIn video takes no options, not author"),
        ({"toolkit": "linkedin", "post_kind": "carousel"}, "LinkedIn does not post a carousel"),
        ({"toolkit": "linkedin", "post_kind": "tweet"}, "post_kind must be one of"),
    ],
)
def test_put_refuses_what_the_workspace_cannot_post_and_writes_nothing(api, target, message):
    _cache_channel(api, "linkedin")
    _cache_channel(api, "instagram")
    _cache_channel(api, "youtube")
    _connect(api, "LINKEDIN", "INSTAGRAM")
    _connect(api, "YOUTUBE", workspace=OTHER_WS)  # another workspace's connection is not this one's
    post = _create(api)

    resp = _put(api, post["id"], [target])

    assert resp.status_code == 422 and message in resp.json()["detail"], resp.text
    assert _target_rows(api, post["id"]) == {}


def test_the_options_a_kind_takes_are_the_ones_its_steps_read():
    def takes(toolkit, kind):
        steps = SEEDED_CHANNELS[toolkit].kinds[kind]
        return targets_api.kind_options(capabilities.ChannelKind(kind, True, None, False, steps))

    assert takes("linkedin", "text") == takes("linkedin", "image") == {"author"}
    assert takes("tiktok", "video") == {"privacy_level", "is_aigc"}
    assert takes("youtube", "video") == {"category_id", "privacy_status", "tags"}
    assert takes("twitter", "text") == takes("instagram", "reel") == frozenset()


def test_put_on_an_approved_post_voids_the_approval_and_the_new_channels_can_be_approved(api, monkeypatch):
    from modules.socials import publisher

    launched = []
    monkeypatch.setattr(publisher, "launch", launched.append)
    _linkedin_and_x(api)
    approved = _approved(api, [{"toolkit": "linkedin", "post_kind": "text"}])
    assert approved["status"] == "approved" and approved["approved_hash"] == approved["content_hash"]

    resp = _put(api, approved["id"], [{"toolkit": "linkedin", "post_kind": "text"}, {"toolkit": "twitter", "post_kind": "text"}])

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "needs_approval"
    assert body["review_log"][-1]["action"] == "approval_voided"
    assert body["approved_hash"] == approved["approved_hash"] != body["content_hash"]
    # assert_publishable refuses the old approval: publish-now is 409, before any channel.
    assert api.client.post(f"/api/socials/posts/{approved['id']}/publish-now").status_code == 409

    again = api.client.post(f"/api/socials/posts/{approved['id']}/approve", json={"content_hash": body["content_hash"]})
    assert again.status_code == 200 and again.json()["approved_hash"] == body["content_hash"]
    # Approved with its channels, it publishes (Wave 3): 202, and one publish is launched.
    assert api.client.post(f"/api/socials/posts/{approved['id']}/publish-now").status_code == 202
    assert [job.post_id for job in launched] == [uuid.UUID(approved["id"])]


def test_put_on_a_scheduled_post_voids_it_and_the_same_set_again_keeps_an_approval(api):
    _linkedin_and_x(api)
    approved = _approved(api, [{"toolkit": "linkedin", "post_kind": "text"}])
    same = _put(api, approved["id"], [{"toolkit": "LINKEDIN", "post_kind": "text", "options": {}}])
    assert same.status_code == 200 and same.json()["status"] == "approved"
    assert same.json()["review_log"] == approved["review_log"]

    scheduled = api.client.post(f"/api/socials/posts/{approved['id']}/schedule", json={"scheduled_for": SLOT.isoformat()})
    assert scheduled.status_code == 200 and scheduled.json()["status"] == "scheduled"
    resp = _put(api, approved["id"], [])

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "needs_approval" and resp.json()["review_log"][-1]["action"] == "approval_voided"


def test_a_put_racing_a_committed_edit_is_409_and_writes_no_target(api, monkeypatch):
    _linkedin_and_x(api)
    post = _create(api)
    other_worker = sessionmaker(bind=api.session.get_bind())()
    real_get_post = service.get_post
    committed = []

    def load_then_another_worker_edits(db, workspace_id, post_id):
        loaded = real_get_post(db, workspace_id, post_id)
        if not committed:
            row = other_worker.get(SocialPost, post_id)
            service.update_post(row, "editor-2", {"copy": {"base": "Edited by another worker."}})
            committed.append(row.content_hash)
            other_worker.commit()
        return loaded

    monkeypatch.setattr(service, "get_post", load_then_another_worker_edits)
    try:
        resp = _put(api, post["id"], [{"toolkit": "linkedin", "post_kind": "text"}])
    finally:
        monkeypatch.setattr(service, "get_post", real_get_post)
        other_worker.close()

    assert committed, "the concurrent edit never ran"
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["content_hash"] == committed[0]
    assert _target_rows(api, post["id"]) == {}
    assert _row(api, post["id"]).copy == {"base": "Edited by another worker."}


def test_put_is_refused_where_an_edit_is(api):
    _linkedin_and_x(api)
    post = _create(api)
    assert api.client.post(f"/api/socials/posts/{post['id']}/submit").status_code == 200
    assert api.client.post(f"/api/socials/posts/{post['id']}/reject", json={"reason": "Off brand"}).status_code == 200

    resp = _put(api, post["id"], [{"toolkit": "linkedin", "post_kind": "text"}])

    assert resp.status_code == 409 and "archived" in resp.json()["detail"]
    assert _target_rows(api, post["id"]) == {}


def test_put_needs_edit_rights_and_the_callers_own_post(api):
    _linkedin_and_x(api)
    post = _create(api)
    body = [{"toolkit": "linkedin", "post_kind": "text"}]

    api.role = "viewer"
    assert _put(api, post["id"], body).status_code == 403
    api.role = "owner"
    api.ctx = _ctx(OTHER_WS)
    assert _put(api, post["id"], body).status_code == 404
    assert _target_rows(api, post["id"]) == {}


@pytest.mark.parametrize(
    "body",
    [
        {"targets": [{"toolkit": "linkedin", "post_kind": "text", "action_plan": {"steps": []}}]},
        {"targets": [{"toolkit": "linkedin", "post_kind": "text", "steps": [{"action": "ANYTHING"}]}]},
        {"targets": [{"toolkit": "linkedin", "post_kind": "text"}] * (post_targets.TARGETS_MAX + 1)},
        {"channels": []},
    ],
    ids=["action-plan", "steps", "too-many", "no-targets"],
)
def test_a_client_never_writes_the_steps_or_the_plan(api, body):
    _linkedin_and_x(api)
    post = _create(api)
    assert api.client.put(ROUTE.format(post_id=post["id"]), json=body).status_code == 422
    assert _target_rows(api, post["id"]) == {}


def test_the_route_is_a_plain_def_behind_the_gate_and_in_the_committed_manifest(api):
    (route,) = [r for r in socials_api.router.routes if isinstance(r, APIRoute) and r.path == ROUTE]
    assert route.methods == {"PUT"}
    assert not inspect.iscoroutinefunction(route.endpoint)  # F105: FastAPI runs it in the threadpool
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "PUT", "path": ROUTE} in manifest["routes"]

    post = _create(api)
    api.ctx = _ctx(channels_harness.WS_OFF)
    assert _put(api, post["id"], []).status_code == 404


# ---------------------------------------------------------------------------
# to_dict: each target's status and receipt
# ---------------------------------------------------------------------------


def test_to_dict_carries_each_targets_options_status_and_receipt(api):
    _linkedin_and_x(api)
    post = _create(api)
    assert _put(api, post["id"], [
        {"toolkit": "linkedin", "post_kind": "text", "options": {"author": "urn:li:organization:42"}},
        {"toolkit": "twitter", "post_kind": "text"},
    ]).status_code == 200
    published_at = datetime(2026, 11, 9, 9, 30, tzinfo=timezone.utc)
    rows = _target_rows(api, post["id"])
    linkedin, twitter = rows["linkedin", "text"], rows["twitter", "text"]
    linkedin.status, linkedin.attempts, linkedin.published_at = "published", 1, published_at
    linkedin.remote_id, linkedin.permalink = "urn:li:share:7", "https://www.linkedin.com/feed/update/urn:li:share:7"
    twitter.status, twitter.attempts, twitter.error = "failed", 3, "X answered 429: rate limited"
    api.session.commit()

    got = api.client.get(f"/api/socials/posts/{post['id']}").json()["targets"]
    listed = api.client.get("/api/socials/posts").json()["posts"][0]["targets"]

    assert got == listed
    assert [set(target) for target in got] == [TARGET_FIELDS, TARGET_FIELDS]  # no plan, no key: server-side
    assert got[0] == {
        "id": str(linkedin.id), "toolkit": "linkedin", "post_kind": "text", "options": {"author": "urn:li:organization:42"},
        "status": "published", "attempts": 1, "remote_id": "urn:li:share:7",
        "permalink": "https://www.linkedin.com/feed/update/urn:li:share:7", "error": None,
        "published_at": got[0]["published_at"], "notes": [],
    }
    assert got[0]["published_at"].startswith("2026-11-09T09:30")
    assert got[1] == {
        "id": str(twitter.id), "toolkit": "twitter", "post_kind": "text", "options": {},
        "status": "failed", "attempts": 3, "remote_id": None, "permalink": None,
        "error": "X answered 429: rate limited", "published_at": None, "notes": [],
    }
