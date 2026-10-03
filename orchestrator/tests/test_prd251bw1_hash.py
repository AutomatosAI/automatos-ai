"""PRD-251B Wave 1, US-B101 — the chosen length is content; the planned slot is not.

* ``compute_content_hash`` changes with ``length_seconds`` once it is set and never
  with ``planned_for`` (B5, B11); a post without a length hashes as before, so no
  existing approval moves when the field arrives.
* An edit of the length on an approved post voids its approval like any content edit.
* ``set_planned_for`` stores the slot in UTC with its zone and leaves the status, the
  hash and the approval fields untouched; it refuses an unknown zone.
* The length is a positive whole number of seconds (the service and the API models);
  ``text`` is a format.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from pydantic import ValidationError  # noqa: E402

from api.socials import CreateSocialPostRequest, UpdateSocialPostRequest  # noqa: E402
from modules.socials import service  # noqa: E402
from modules.socials.service import ACTION_APPROVAL_VOIDED, APPROVED, NEEDS_APPROVAL, InvalidPost  # noqa: E402

AUTHOR, REVIEWER = "user-1", "reviewer-1"
LONDON = ZoneInfo("Europe/London")


class _Db:
    def add(self, obj) -> None:
        pass


def _draft(**fields):
    base = dict(workspace_id=uuid.uuid4(), created_by=AUTHOR, title="A post", format="video", copy={"base": "Hello"})
    base.update(fields)
    return service.create_draft(_Db(), **base)


def _approved(**fields):
    post = _draft(**fields)
    service.submit(post, AUTHOR)
    service.approve(post, REVIEWER, content_hash=post.content_hash)
    assert post.status == APPROVED
    return post


def test_the_hash_covers_the_length_once_set_and_never_the_slot():
    post = _draft()
    unset = service.compute_content_hash(post)
    post.length_seconds = 15
    with_length = service.compute_content_hash(post)
    assert with_length != unset
    post.length_seconds = 30
    assert service.compute_content_hash(post) not in (unset, with_length)
    post.length_seconds = None
    assert service.compute_content_hash(post) == unset
    post.planned_for = datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc)
    assert service.compute_content_hash(post) == unset


def test_a_length_change_on_an_approved_post_voids_the_approval():
    post = _approved(length_seconds=15)
    approved_hash = post.approved_hash
    service.update_post(post, AUTHOR, {"length_seconds": 30})
    assert post.status == NEEDS_APPROVAL
    assert post.length_seconds == 30
    assert post.content_hash != approved_hash and post.approved_hash == approved_hash
    assert post.review_log[-1]["action"] == ACTION_APPROVAL_VOIDED


def test_setting_the_slot_leaves_status_hash_and_approval_untouched():
    post = _approved(length_seconds=15)
    before = (post.status, post.content_hash, post.approved_hash, post.approved_by)
    service.set_planned_for(post, datetime(2026, 10, 14, 12, 0, tzinfo=LONDON), "Europe/London")
    assert post.planned_for == datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc)  # BST → UTC
    assert post.timezone == "Europe/London"
    assert (post.status, post.content_hash, post.approved_hash, post.approved_by) == before
    assert service.compute_content_hash(post) == post.content_hash
    # Naive is UTC (the database convention); None clears; the zone is kept when not given.
    service.set_planned_for(post, datetime(2026, 10, 15, 9, 30))
    assert post.planned_for == datetime(2026, 10, 15, 9, 30, tzinfo=timezone.utc)
    assert post.timezone == "Europe/London"
    service.set_planned_for(post, None)
    assert post.planned_for is None
    with pytest.raises(InvalidPost):
        service.set_planned_for(post, datetime(2026, 10, 15, 9, 30), "Mars/Olympus")
    with pytest.raises(InvalidPost):
        service.set_planned_for(post, "2026-10-15T09:30:00Z")  # type: ignore[arg-type]


def test_the_length_is_a_positive_whole_number_of_seconds():
    for bad in (0, -5, True, 1.5, "15"):
        with pytest.raises(InvalidPost):
            _draft(length_seconds=bad)
    post = _draft(length_seconds=15)
    row = post.to_dict()
    assert (row["length_seconds"], row["planned_for"]) == (15, None)


def test_text_is_a_format_and_bogus_is_not():
    assert _draft(format="text").format == "text"
    with pytest.raises(InvalidPost):
        _draft(format="bogus")


def test_the_api_models_refuse_a_zero_length():
    with pytest.raises(ValidationError):
        CreateSocialPostRequest(title="x", length_seconds=0)
    with pytest.raises(ValidationError):
        UpdateSocialPostRequest(length_seconds=0)
    assert CreateSocialPostRequest(title="x", length_seconds=15).length_seconds == 15
    assert UpdateSocialPostRequest(length_seconds=30).model_dump(exclude_unset=True) == {"length_seconds": 30}
