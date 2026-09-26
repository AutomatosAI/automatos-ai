"""PRD-251 S0.3a — the post lifecycle: status machine, content hash, approval and the publish guard.

Pure tests of ``modules/socials/service.py`` and ``modules/socials/publisher.py``
(transient ORM objects; the two workspace-scoped reads run on in-memory SQLite):

* every allowed transition happens through its action, and illegal ones raise
  ``IllegalTransition`` — the table IS the whole machine: Wave 0's, plus Wave 1's
  render moves (S1.1c; their rules are pinned in test_prd251w1_render_lifecycle.py);
* ``compute_content_hash`` is canonical JSON over copy, variables, sources,
  format, template_id and media: stable across key order, sensitive to each field;
* D6 — any content edit of an approved or scheduled post moves it back to
  needs_approval and the publish guard then refuses it;
* D7 — an unsourced claim blocks approval unless overridden, and the override
  is stored and named in ``review_log``;
* D6 — ``approve`` binds to the version the approver saw: a post whose content
  changed since refuses it (``StaleContent``, with the current hash) and is left
  unchanged; ``claim_unchanged`` is a compare-and-set on status and hash;
* ``publish_post`` runs ``assert_publishable`` before anything else.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

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
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every table the FKs name)
import modules.socials.publisher as publisher  # noqa: E402
import modules.socials.service as service  # noqa: E402
from core.models.socials import SocialPost, SocialPostTarget  # noqa: E402
from modules.socials.service import (  # noqa: E402
    APPROVED,
    ARCHIVED,
    CHANGES_REQUESTED,
    DRAFT,
    FAILED,
    NEEDS_APPROVAL,
    RENDERING,
    SCHEDULED,
    IllegalTransition,
    InvalidPost,
    NotPublishable,
    StaleContent,
    UnsourcedClaims,
)

WS_A = uuid.uuid4()
WS_B = uuid.uuid4()
AUTHOR = "user-author"
REVIEWER = "user-reviewer"
NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
SLOT = NOW + timedelta(days=3)


class _Added:
    """Stands in for the session ``create_draft`` adds to."""

    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)


@pytest.fixture(autouse=True)
def _frozen_clock(monkeypatch):
    monkeypatch.setattr(service, "_utcnow", lambda: NOW)


def _draft(**fields):
    base = {
        "workspace_id": WS_A,
        "created_by": AUTHOR,
        "title": "Web Summit countdown",
        "brief": "Three weeks to Lisbon",
        "copy": {"base": "Three weeks to go.", "channels": {"linkedin": "Three weeks to Web Summit."}},
        "format": "fact_card",
        "variables": {"days": {"value": 21, "claim": False}},
    }
    base.update(fields)
    return service.create_draft(_Added(), **base)


# A finished render's media (S1.1c): one 9:16 file with its Deliverable and digest.
_RENDERED = {"9:16": [{"deliverable_id": "d-1", "name": "video-9x16.mp4", "sha256": "a" * 64, "bytes": 1024}]}


def _post_in(status: str) -> SocialPost:
    """A post that legitimately reached ``status`` through the service."""
    post = _draft()
    if status == DRAFT:
        return post
    if status in (RENDERING, FAILED):
        service.start_render(post, AUTHOR)
        return post if status == RENDERING else service.fail_render(post, AUTHOR, "The check failed.")
    service.submit(post, AUTHOR)
    if status == NEEDS_APPROVAL:
        return post
    if status == CHANGES_REQUESTED:
        return service.request_changes(post, REVIEWER, "Tighten the headline.")
    if status == ARCHIVED:
        return service.reject(post, REVIEWER, "Off brand.")
    service.approve(post, REVIEWER, content_hash=post.content_hash)
    if status == APPROVED:
        return post
    if status == SCHEDULED:
        return service.schedule(post, REVIEWER, SLOT, "Europe/Lisbon")
    raise AssertionError(status)


def _edit_copy(post):
    return service.update_post(post, AUTHOR, {"copy": {"base": "Two weeks to go."}})


# ---------------------------------------------------------------------------
# The status machine
# ---------------------------------------------------------------------------

# Every allowed (from, to) move, and the action that makes it.
ALLOWED = {
    (DRAFT, NEEDS_APPROVAL): lambda p: service.submit(p, AUTHOR),
    (CHANGES_REQUESTED, NEEDS_APPROVAL): lambda p: service.submit(p, AUTHOR),
    (NEEDS_APPROVAL, APPROVED): lambda p: service.approve(p, REVIEWER, content_hash=p.content_hash),
    (NEEDS_APPROVAL, CHANGES_REQUESTED): lambda p: service.request_changes(p, REVIEWER, "Shorter."),
    (NEEDS_APPROVAL, ARCHIVED): lambda p: service.reject(p, REVIEWER, "No."),
    (APPROVED, SCHEDULED): lambda p: service.schedule(p, REVIEWER, SLOT, "UTC"),
    (SCHEDULED, APPROVED): lambda p: service.unschedule(p, REVIEWER),
    (APPROVED, NEEDS_APPROVAL): _edit_copy,
    (SCHEDULED, NEEDS_APPROVAL): _edit_copy,
    # Wave 1 (S1.1c): a post that holds no approval renders; the render ends it
    # in needs_approval or failed.
    (DRAFT, RENDERING): lambda p: service.start_render(p, AUTHOR),
    (CHANGES_REQUESTED, RENDERING): lambda p: service.start_render(p, AUTHOR),
    (NEEDS_APPROVAL, RENDERING): lambda p: service.start_render(p, AUTHOR),
    (FAILED, RENDERING): lambda p: service.start_render(p, AUTHOR),
    (RENDERING, NEEDS_APPROVAL): lambda p: service.finish_render(p, AUTHOR, _RENDERED),
    (RENDERING, FAILED): lambda p: service.fail_render(p, AUTHOR, "The check failed."),
}


def test_the_table_is_exactly_the_wave_0_and_wave_1_machine():
    table = {(src, dst) for src, dsts in service.ALLOWED_TRANSITIONS.items() for dst in dsts}
    assert table == set(ALLOWED)
    assert service.TRANSITIONS == {
        "submit": {DRAFT: NEEDS_APPROVAL, CHANGES_REQUESTED: NEEDS_APPROVAL},
        "approve": {NEEDS_APPROVAL: APPROVED},
        "request_changes": {NEEDS_APPROVAL: CHANGES_REQUESTED},
        "reject": {NEEDS_APPROVAL: ARCHIVED},
        "schedule": {APPROVED: SCHEDULED},
        "unschedule": {SCHEDULED: APPROVED},
        "edit": {APPROVED: NEEDS_APPROVAL, SCHEDULED: NEEDS_APPROVAL},
        "render": {DRAFT: RENDERING, CHANGES_REQUESTED: RENDERING, NEEDS_APPROVAL: RENDERING, FAILED: RENDERING},
        "render_done": {RENDERING: NEEDS_APPROVAL},
        "render_failed": {RENDERING: FAILED},
    }


@pytest.mark.parametrize("move", sorted(ALLOWED), ids=lambda m: f"{m[0]}->{m[1]}")
def test_every_allowed_transition(move):
    src, dst = move
    post = _post_in(src)
    ALLOWED[move](post)
    assert post.status == dst


ACTIONS = {
    "submit": lambda p: service.submit(p, AUTHOR),
    "approve": lambda p: service.approve(p, REVIEWER, content_hash=p.content_hash),
    "request_changes": lambda p: service.request_changes(p, REVIEWER, "Change it."),
    "reject": lambda p: service.reject(p, REVIEWER),
    "schedule": lambda p: service.schedule(p, REVIEWER, SLOT, "UTC"),
    "unschedule": lambda p: service.unschedule(p, REVIEWER),
    "edit": _edit_copy,
}

ILLEGAL = [
    (DRAFT, "approve"),
    (DRAFT, "request_changes"),
    (DRAFT, "reject"),
    (DRAFT, "schedule"),
    (DRAFT, "unschedule"),
    (NEEDS_APPROVAL, "submit"),
    (NEEDS_APPROVAL, "schedule"),
    (NEEDS_APPROVAL, "unschedule"),
    (CHANGES_REQUESTED, "approve"),
    (CHANGES_REQUESTED, "schedule"),
    (APPROVED, "submit"),
    (APPROVED, "approve"),
    (APPROVED, "request_changes"),
    (APPROVED, "reject"),
    (APPROVED, "unschedule"),
    (SCHEDULED, "submit"),
    (SCHEDULED, "approve"),
    (SCHEDULED, "schedule"),
    (SCHEDULED, "reject"),
    (ARCHIVED, "submit"),
    (ARCHIVED, "approve"),
    (ARCHIVED, "edit"),
    (ARCHIVED, "unschedule"),
]


@pytest.mark.parametrize("status, action", ILLEGAL, ids=lambda v: str(v))
def test_illegal_transitions_raise_and_change_nothing(status, action):
    post = _post_in(status)
    log_before = list(post.review_log)
    with pytest.raises(IllegalTransition) as exc:
        ACTIONS[action](post)
    assert post.status == status
    assert post.review_log == log_before
    assert exc.value.current == status


# ``failed`` left this list in Wave 1: a failed render is edited and rendered
# again (S1.1c). Its every other Wave 0 move stays illegal, pinned in
# test_prd251w1_render_lifecycle.py::test_a_failed_post_is_edited_and_rendered_again_and_nothing_else.
@pytest.mark.parametrize("later_status", ["rendering", "publishing", "published", "partially_published", "missed"])
@pytest.mark.parametrize("action", ["submit", "approve", "edit", "schedule"])
def test_statuses_of_later_waves_have_no_wave_0_moves(later_status, action):
    post = _post_in(APPROVED)
    post.status = later_status
    with pytest.raises(IllegalTransition):
        ACTIONS[action](post)


# ---------------------------------------------------------------------------
# The content hash (D6)
# ---------------------------------------------------------------------------

_CONTENT = {
    "copy": {"base": "Hello", "channels": {"x": "Hi", "linkedin": "Hello, LinkedIn"}},
    "variables": {"users": {"value": 1200, "claim": True}, "city": {"value": "Lisbon", "claim": False}},
    "sources": {"users": {"kind": "metric", "ref": "active_users", "as_of": "2026-09-01T00:00:00Z"}},
    "format": "fact_card",
    "template_id": uuid.UUID("11111111-2222-3333-4444-555555555555"),
    "media": {"9:16": ["d-1", "d-2"], "1:1": ["d-3"]},
}


def _with(**content):
    post = SocialPost(workspace_id=WS_A, created_by=AUTHOR, title="t")
    for name, value in content.items():
        setattr(post, name, value)
    return post


def _reordered(value):
    if isinstance(value, dict):
        return {k: _reordered(value[k]) for k in reversed(list(value))}
    return value


def test_hash_is_canonical_sha256_over_the_six_content_fields():
    canonical = json.dumps(
        {
            "copy": _CONTENT["copy"],
            "variables": _CONTENT["variables"],
            "sources": _CONTENT["sources"],
            "format": "fact_card",
            "template_id": "11111111-2222-3333-4444-555555555555",
            "media": _CONTENT["media"],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert service.compute_content_hash(_with(**_CONTENT)) == expected


def test_hash_is_stable_across_key_order_and_identical_content():
    first = service.compute_content_hash(_with(**_CONTENT))
    again = service.compute_content_hash(_with(**_CONTENT))
    reordered = service.compute_content_hash(
        _with(**{k: _reordered(v) for k, v in reversed(list(_CONTENT.items()))})
    )
    assert first == again == reordered
    assert len(first) == 64


@pytest.mark.parametrize(
    "field, changed",
    [
        ("copy", {"base": "Hello!", "channels": {"x": "Hi", "linkedin": "Hello, LinkedIn"}}),
        ("variables", {"users": {"value": 1300, "claim": True}, "city": {"value": "Lisbon", "claim": False}}),
        ("sources", {"users": {"kind": "metric", "ref": "active_users", "as_of": "2026-09-02T00:00:00Z"}}),
        ("format", "infographic"),
        ("template_id", uuid.UUID("99999999-2222-3333-4444-555555555555")),
        ("media", {"9:16": ["d-1", "d-9"], "1:1": ["d-3"]}),
    ],
)
def test_hash_changes_when_any_one_content_field_changes(field, changed):
    before = service.compute_content_hash(_with(**_CONTENT))
    after = service.compute_content_hash(_with(**{**_CONTENT, field: changed}))
    assert before != after


def test_title_and_brief_are_labels_not_content():
    post = _with(**_CONTENT)
    before = service.compute_content_hash(post)
    post.title, post.brief = "Renamed", "A new brief"
    assert service.compute_content_hash(post) == before


def test_a_new_draft_carries_its_hash_and_is_added_to_the_session():
    added = _Added()
    post = service.create_draft(added, workspace_id=WS_A, created_by=AUTHOR, title="  Launch  ")
    assert added.added == [post]
    assert post.status == DRAFT and post.title == "Launch"
    assert post.content_hash == service.compute_content_hash(post)
    assert post.review_log == [] and post.approved_hash is None


# ---------------------------------------------------------------------------
# D6 — an edit voids the approval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("start", [APPROVED, SCHEDULED])
@pytest.mark.parametrize(
    "field, value",
    [
        ("copy", {"base": "Edited after approval"}),
        ("variables", {"days": {"value": 20, "claim": False}}),
        ("sources", {"days": {"kind": "url", "ref": "https://websummit.com"}}),
        ("format", "image"),
        ("template_id", str(uuid.uuid4())),
        ("media", {"1:1": ["deliverable-7"]}),
    ],
)
def test_editing_any_content_field_of_an_approved_post_voids_the_approval(start, field, value):
    post = _post_in(start)
    service.assert_publishable(post)  # publishable before the edit
    approved_hash = post.approved_hash

    service.update_post(post, AUTHOR, {field: value})

    assert post.status == NEEDS_APPROVAL
    assert post.content_hash != approved_hash == post.approved_hash
    assert post.review_log[-1]["action"] == "approval_voided"
    with pytest.raises(NotPublishable):
        service.assert_publishable(post)


def test_a_label_edit_keeps_the_approval():
    post = _post_in(APPROVED)
    service.update_post(post, AUTHOR, {"title": "Renamed", "brief": "Reworded"})
    assert post.status == APPROVED
    service.assert_publishable(post)


def test_an_identical_content_edit_keeps_the_approval():
    post = _post_in(APPROVED)
    service.update_post(post, AUTHOR, {"copy": dict(post.copy)})
    assert post.status == APPROVED
    service.assert_publishable(post)


def test_a_draft_edit_recomputes_the_hash_and_keeps_the_status():
    post = _draft()
    before = post.content_hash
    service.update_post(post, AUTHOR, {"copy": {"base": "New words"}})
    assert post.status == DRAFT and post.content_hash != before
    assert post.content_hash == service.compute_content_hash(post)


def test_reapproval_after_an_edit_binds_to_the_new_content():
    post = _post_in(APPROVED)
    _edit_copy(post)
    service.approve(post, REVIEWER, content_hash=post.content_hash)
    assert post.status == APPROVED and post.approved_hash == post.content_hash
    service.assert_publishable(post)


def _approval_state(post):
    return (
        post.status, post.content_hash, post.approved_hash, post.approved_by, post.approved_at,
        post.override_unsourced, list(post.review_log),
    )


def test_approve_binds_to_the_version_the_approver_saw():
    post = _post_in(NEEDS_APPROVAL)
    seen = post.content_hash
    service.update_post(post, AUTHOR, {"copy": {"base": "Edited after the reviewer opened it."}})
    before = _approval_state(post)

    with pytest.raises(StaleContent) as exc:
        service.approve(post, REVIEWER, content_hash=seen)

    assert exc.value.current_hash == post.content_hash != seen
    assert "changed since you opened it" in str(exc.value)
    assert _approval_state(post) == before
    assert post.status == NEEDS_APPROVAL and post.approved_hash is None

    service.approve(post, REVIEWER, content_hash=post.content_hash)
    assert post.status == APPROVED and post.approved_hash == post.content_hash
    service.assert_publishable(post)


def test_approve_refuses_content_changed_behind_the_services_back():
    post = _post_in(NEEDS_APPROVAL)
    stamped = post.content_hash
    post.copy = {"base": "Changed without the service"}  # the hash is not recomputed
    with pytest.raises(StaleContent) as exc:
        service.approve(post, REVIEWER, content_hash=stamped)
    assert exc.value.current_hash == service.compute_content_hash(post) != stamped
    assert post.status == NEEDS_APPROVAL and post.approved_hash is None


def test_a_stale_version_is_reported_before_its_unsourced_claims():
    post = _with_claims()
    seen = post.content_hash
    service.update_post(post, AUTHOR, {"copy": {"base": "New words"}})
    with pytest.raises(StaleContent):
        service.approve(post, REVIEWER, content_hash=seen, override_unsourced=True)
    assert post.override_unsourced is False and post.approved_hash is None


def test_a_direct_write_that_skips_the_service_still_cannot_publish():
    post = _post_in(APPROVED)
    post.copy = {"base": "Changed behind the service's back"}  # hash not recomputed
    with pytest.raises(NotPublishable):
        service.assert_publishable(post)


@pytest.mark.parametrize("status", [DRAFT, NEEDS_APPROVAL, CHANGES_REQUESTED, ARCHIVED])
def test_only_approved_or_scheduled_posts_are_publishable(status):
    with pytest.raises(NotPublishable):
        service.assert_publishable(_post_in(status))


def test_an_approved_status_without_an_approval_is_not_publishable():
    post = _draft()
    post.status = APPROVED  # forced, never approved
    with pytest.raises(NotPublishable):
        service.assert_publishable(post)


# ---------------------------------------------------------------------------
# D7 — facts carry sources
# ---------------------------------------------------------------------------


def _with_claims():
    post = _draft(
        variables={
            "users": {"value": 1200, "claim": True},
            "cost": {"value": "$3.42", "claim": True},
            "city": {"value": "Lisbon", "claim": False},
        },
        sources={"cost": {"kind": "report", "ref": "report-42", "as_of": "2026-09-20"}},
    )
    return service.submit(post, AUTHOR)


def test_unsourced_claims_names_only_claims_without_a_source():
    assert service.unsourced_claims(_with_claims()) == ["users"]


def test_approve_refuses_an_unsourced_claim_and_names_it():
    post = _with_claims()
    with pytest.raises(UnsourcedClaims) as exc:
        service.approve(post, REVIEWER, content_hash=post.content_hash)
    assert exc.value.names == ["users"]
    assert "users" in str(exc.value)
    assert post.status == NEEDS_APPROVAL and post.approved_hash is None


def test_approve_with_the_override_stores_it_and_names_the_claim():
    post = _with_claims()
    service.approve(post, REVIEWER, content_hash=post.content_hash, override_unsourced=True)

    assert post.status == APPROVED and post.override_unsourced is True
    assert post.approved_by == REVIEWER and post.approved_hash == post.content_hash
    entry = post.review_log[-1]
    assert entry["action"] == "approve" and entry["by"] == REVIEWER
    assert entry["overridden_claims"] == ["users"]
    assert "users" in entry["comment"]


def test_a_sourced_post_needs_no_override():
    post = _with_claims()
    service.update_post(
        post, AUTHOR,
        {"sources": {**post.sources, "users": {"kind": "metric", "ref": "active_users"}}},
    )
    service.approve(post, REVIEWER, content_hash=post.content_hash)
    assert post.status == APPROVED and post.override_unsourced is False


def test_voiding_an_approval_clears_the_override():
    post = _with_claims()
    service.approve(post, REVIEWER, content_hash=post.content_hash, override_unsourced=True)
    _edit_copy(post)
    assert post.override_unsourced is False


# ---------------------------------------------------------------------------
# The review log and the review actions
# ---------------------------------------------------------------------------


def test_every_review_action_appends_to_review_log():
    post = _draft()
    service.submit(post, AUTHOR)
    service.request_changes(post, REVIEWER, "Use the brand colour.")
    service.submit(post, AUTHOR)
    service.approve(post, REVIEWER, content_hash=post.content_hash, comment="Good to go")
    service.schedule(post, REVIEWER, SLOT, "Europe/Lisbon")
    service.unschedule(post, REVIEWER)

    log = post.review_log
    assert [e["action"] for e in log] == [
        "submit", "request_changes", "submit", "approve", "schedule", "unschedule",
    ]
    assert all(set(e) >= {"at", "by", "action", "comment"} for e in log)
    assert log[1] == {
        "at": NOW.isoformat(), "by": REVIEWER, "action": "request_changes",
        "comment": "Use the brand colour.",
    }
    assert log[3]["comment"] == "Good to go"


def test_the_review_log_is_reassigned_never_mutated_in_place():
    post = _draft()
    before = post.review_log
    service.submit(post, AUTHOR)
    assert post.review_log is not before and before == []


def test_request_changes_needs_a_comment():
    post = service.submit(_draft(), AUTHOR)
    for empty in (None, "", "   "):
        with pytest.raises(InvalidPost):
            service.request_changes(post, REVIEWER, empty)
    assert post.status == NEEDS_APPROVAL


def test_reject_archives_with_the_reason_in_the_log():
    post = service.reject(service.submit(_draft(), AUTHOR), REVIEWER, "Off brand")
    assert post.status == ARCHIVED
    assert post.review_log[-1]["comment"] == "Off brand"


def test_schedule_stores_utc_and_the_display_timezone():
    post = _post_in(APPROVED)
    lisbon_evening = datetime(2026, 10, 1, 19, 30, tzinfo=timezone(timedelta(hours=1)))
    service.schedule(post, REVIEWER, lisbon_evening, "Europe/Lisbon")
    assert post.status == SCHEDULED
    assert post.scheduled_for == datetime(2026, 10, 1, 18, 30, tzinfo=timezone.utc)
    assert post.timezone == "Europe/Lisbon"


@pytest.mark.parametrize(
    "when, tz_name",
    [
        (NOW - timedelta(minutes=1), "UTC"),
        (NOW, "UTC"),
        (SLOT, "Mars/Olympus_Mons"),
        (SLOT, "../../etc/passwd"),
        (SLOT, ""),
    ],
)
def test_schedule_refuses_a_past_slot_or_an_unknown_timezone(when, tz_name):
    post = _post_in(APPROVED)
    with pytest.raises(InvalidPost):
        service.schedule(post, REVIEWER, when, tz_name)
    assert post.status == APPROVED and post.scheduled_for is None


def test_unschedule_clears_the_slot_and_keeps_the_approval():
    post = _post_in(SCHEDULED)
    service.unschedule(post, REVIEWER)
    assert post.status == APPROVED and post.scheduled_for is None
    service.assert_publishable(post)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"title": ""},
        {"title": "x" * 501},
        {"format": "gif"},
        {"copy": "just text"},
        {"copy": {"base": 1}},
        {"copy": {"channels": {"x": 5}}},
        {"copy": {"headline": "no such key"}},
        {"variables": {"users": 1200}},
        {"variables": {"users": {"value": 1, "claim": "yes"}}},
        {"sources": {"users": {"kind": "rumour", "ref": "x"}}},
        {"sources": {"users": {"kind": "url"}}},
        {"media": {"9:16": "d-1"}},
        {"template_id": "not-a-uuid"},
    ],
)
def test_create_draft_refuses_malformed_fields(fields):
    with pytest.raises(InvalidPost):
        _draft(**fields)


def test_update_post_refuses_unknown_fields():
    post = _draft()
    with pytest.raises(InvalidPost):
        service.update_post(post, AUTHOR, {"status": "approved"})
    with pytest.raises(InvalidPost):
        service.update_post(post, AUTHOR, {"approved_hash": "0" * 64})
    assert post.status == DRAFT and post.approved_hash is None


# ---------------------------------------------------------------------------
# publish_post — the guard runs first
# ---------------------------------------------------------------------------


def test_publish_post_refuses_a_stale_approval_before_anything_else(monkeypatch):
    downstream = MagicMock(name="_publish_targets")
    monkeypatch.setattr(publisher, "_publish_targets", downstream)
    post = _post_in(APPROVED)
    post.approved_hash = "f" * 64  # approved_hash != content_hash

    with pytest.raises(NotPublishable):
        publisher.publish_post(MagicMock(name="db"), post)
    downstream.assert_not_called()


def test_publish_post_refuses_an_edited_post_before_anything_else(monkeypatch):
    downstream = MagicMock(name="_publish_targets")
    monkeypatch.setattr(publisher, "_publish_targets", downstream)
    post = _post_in(SCHEDULED)
    _edit_copy(post)

    with pytest.raises(NotPublishable):
        publisher.publish_post(MagicMock(name="db"), post)
    downstream.assert_not_called()


def test_publish_post_reaches_the_seam_only_for_a_valid_approval(monkeypatch):
    downstream = MagicMock(name="_publish_targets", return_value="published")
    monkeypatch.setattr(publisher, "_publish_targets", downstream)
    db, post = MagicMock(name="db"), _post_in(APPROVED)

    assert publisher.publish_post(db, post) == "published"
    downstream.assert_called_once_with(db, post)


def test_wave_0_has_no_channel_publishers():
    with pytest.raises(publisher.PublishingUnavailable) as exc:
        publisher.publish_post(MagicMock(name="db"), _post_in(APPROVED))
    assert str(exc.value) == "Channel publishing arrives in Wave 3"


def test_the_service_and_the_publisher_import_no_fastapi_and_no_composio():
    import ast

    for module in (service, publisher):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not {name for name in imported if "fastapi" in name or "composio" in name}, imported


# ---------------------------------------------------------------------------
# Workspace-scoped reads (SQLite)
# ---------------------------------------------------------------------------


@pytest.fixture
def session():
    engine = sa.create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SocialPost.metadata.create_all(engine, tables=[SocialPost.__table__, SocialPostTarget.__table__])
    s = sessionmaker(bind=engine)()
    try:
        yield s
    finally:
        s.close()
        engine.dispose()


def _stored(session, workspace_id, title, created_at, **fields):
    post = service.create_draft(session, workspace_id=workspace_id, created_by=AUTHOR, title=title, **fields)
    post.created_at = created_at
    session.flush()
    return post


def test_get_post_never_returns_another_workspaces_post(session):
    mine = _stored(session, WS_A, "Mine", NOW)
    theirs = _stored(session, WS_B, "Theirs", NOW)
    session.commit()

    assert service.get_post(session, WS_A, mine.id).title == "Mine"
    assert service.get_post(session, WS_A, theirs.id) is None
    assert service.get_post(session, WS_B, mine.id) is None


def test_claim_unchanged_is_a_compare_and_set_on_status_and_hash(session):
    post = _stored(session, WS_A, "Mine", NOW)
    service.submit(post, AUTHOR)
    session.commit()
    committed = post.content_hash

    assert service.claim_unchanged(session, post, status=NEEDS_APPROVAL, content_hash=committed) is True
    assert service.claim_unchanged(session, post, status=DRAFT, content_hash=committed) is False
    assert service.claim_unchanged(session, post, status=NEEDS_APPROVAL, content_hash="0" * 64) is False

    # The caller's pending changes never land before the check (no autoflush).
    service.approve(post, REVIEWER, content_hash=committed)
    assert post.status == APPROVED  # in memory only
    assert service.claim_unchanged(session, post, status=NEEDS_APPROVAL, content_hash=committed) is True
    session.commit()
    session.expire_all()
    assert service.get_post(session, WS_A, post.id).status == APPROVED


def test_list_posts_is_scoped_newest_first_and_filtered(session):
    old = _stored(session, WS_A, "Old", NOW - timedelta(days=2))
    new = _stored(session, WS_A, "New", NOW - timedelta(hours=1))
    _stored(session, WS_B, "Other workspace", NOW)
    service.submit(new, AUTHOR)
    session.commit()

    assert [p.title for p in service.list_posts(session, WS_A)] == ["New", "Old"]
    assert [p.title for p in service.list_posts(session, WS_A, statuses=["draft"])] == ["Old"]
    assert [p.title for p in service.list_posts(session, WS_A, statuses=["needs_approval", "draft"])] == ["New", "Old"]
    window = service.list_posts(session, WS_A, window_from=NOW - timedelta(days=1), window_to=NOW)
    assert [p.title for p in window] == ["New"]
    assert old not in window
