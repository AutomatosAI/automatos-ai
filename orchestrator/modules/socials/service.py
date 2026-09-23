"""PRD-251 S0.3a: the post lifecycle, as pure logic.

* **The status machine.** ``TRANSITIONS`` is the whole table, keyed by action:
  an action moves a post only from the statuses listed for it, and anything
  else raises :class:`IllegalTransition`. Wave 0 moves posts between draft,
  needs_approval, changes_requested, approved, scheduled and archived.
  Rendering, publishing, missed and failed arrive with their waves.
* **The content hash (D6).** ``compute_content_hash`` is sha256 over canonical
  JSON of what is published: copy, variables, sources, format, template_id and
  media. An approval binds to it. Any content edit changes it, so an approved
  or scheduled post goes back to needs_approval and its approval is void,
  because ``approved_hash`` no longer matches.
* **Facts carry sources (D7).** A variable marked ``claim: true`` needs an entry
  in ``sources``. ``approve`` refuses unsourced claims unless the approver
  overrides, and the override is stored and named in ``review_log``.
* **The publish guard.** ``assert_publishable`` passes only an approved or
  scheduled post whose approval matches its content as it is NOW.

No FastAPI here: the API maps these exceptions to status codes. Every review
action appends to ``review_log``, the history the approval UI shows. JSON
fields are always reassigned, never mutated in place.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func

from core.models.socials import SOCIAL_POST_FORMATS, SocialPost

# ── statuses ────────────────────────────────────────────────────────────────
DRAFT = "draft"
NEEDS_APPROVAL = "needs_approval"
CHANGES_REQUESTED = "changes_requested"
APPROVED = "approved"
SCHEDULED = "scheduled"
ARCHIVED = "archived"

# review_log actions, which are also the status machine's actions
ACTION_SUBMIT = "submit"
ACTION_APPROVE = "approve"
ACTION_REQUEST_CHANGES = "request_changes"
ACTION_REJECT = "reject"
ACTION_SCHEDULE = "schedule"
ACTION_UNSCHEDULE = "unschedule"
ACTION_EDIT = "edit"
ACTION_APPROVAL_VOIDED = "approval_voided"

# Wave 0's whole status machine: action → {from status: to status}. An action
# applies only from the statuses listed for it.
TRANSITIONS: Dict[str, Dict[str, str]] = {
    ACTION_SUBMIT: {DRAFT: NEEDS_APPROVAL, CHANGES_REQUESTED: NEEDS_APPROVAL},
    ACTION_APPROVE: {NEEDS_APPROVAL: APPROVED},
    ACTION_REQUEST_CHANGES: {NEEDS_APPROVAL: CHANGES_REQUESTED},
    ACTION_REJECT: {NEEDS_APPROVAL: ARCHIVED},
    ACTION_SCHEDULE: {APPROVED: SCHEDULED},
    ACTION_UNSCHEDULE: {SCHEDULED: APPROVED},
    # A content edit voids the approval of an approved or scheduled post.
    ACTION_EDIT: {APPROVED: NEEDS_APPROVAL, SCHEDULED: NEEDS_APPROVAL},
}

# The same table seen per status: current status → the statuses it may move to.
ALLOWED_TRANSITIONS: Dict[str, frozenset] = {
    status: frozenset(moves[status] for moves in TRANSITIONS.values() if status in moves)
    for status in {s for moves in TRANSITIONS.values() for s in moves}
}

# The statuses a post's content may be edited in. An edit to an approved or
# scheduled post voids its approval; in the others the post keeps its status.
EDITABLE_STATUSES = frozenset({DRAFT, NEEDS_APPROVAL, CHANGES_REQUESTED, APPROVED, SCHEDULED})
PUBLISHABLE_STATUSES = frozenset({APPROVED, SCHEDULED})

# What the hash covers (D6), and what a post edit may change.
CONTENT_FIELDS = ("copy", "variables", "sources", "format", "template_id", "media")
LABEL_FIELDS = ("title", "brief")
EDITABLE_FIELDS = LABEL_FIELDS + CONTENT_FIELDS

# D7: where a claim's source may come from.
SOURCE_KINDS = ("deliverable", "report", "document", "url", "metric")

TITLE_MAX_CHARS = 500
COMMENT_MAX_CHARS = 2000


# ── errors ──────────────────────────────────────────────────────────────────
class SocialsError(Exception):
    """Base for every error the Socials lifecycle raises."""


class InvalidPost(SocialsError, ValueError):
    """A field value the post cannot carry."""


class IllegalTransition(SocialsError):
    def __init__(self, current: str, action: str):
        self.current = current
        self.action = action
        verb = action.replace("_", " ")
        super().__init__(f"cannot {verb} a post that is {current.replace('_', ' ')}")


class UnsourcedClaims(SocialsError):
    def __init__(self, names: Iterable[str]):
        self.names = sorted(names)
        super().__init__(
            "these claims have no source: " + ", ".join(self.names)
            + " (approve with override_unsourced to publish them anyway)"
        )


class NotPublishable(SocialsError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


# ── time ────────────────────────────────────────────────────────────────────
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """Aware → UTC; naive is taken to be UTC already (the database convention)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# ── the content hash (D6) ───────────────────────────────────────────────────
def _content_of(post: Any) -> Dict[str, Any]:
    template_id = getattr(post, "template_id", None)
    return {
        "copy": getattr(post, "copy", None) or {},
        "variables": getattr(post, "variables", None) or {},
        "sources": getattr(post, "sources", None) or {},
        "format": getattr(post, "format", None),
        "template_id": str(template_id) if template_id is not None else None,
        "media": getattr(post, "media", None) or {},
    }


def compute_content_hash(post: Any) -> str:
    """sha256 over canonical JSON of copy, variables, sources, format, template_id and media.

    Canonical = ``sort_keys=True``, ``separators=(',', ':')``,
    ``ensure_ascii=False``, so key order never changes the hash.
    Wave 1 extends ``media`` with the rendered files' digests, so an approval
    also binds to the exact rendered bytes.
    """
    canonical = json.dumps(
        _content_of(post), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── validation ──────────────────────────────────────────────────────────────
def _require_dict(name: str, value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise InvalidPost(f"{name} must be an object")
    return value


def _validate_title(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidPost("title is required")
    title = value.strip()
    if len(title) > TITLE_MAX_CHARS:
        raise InvalidPost(f"title must be at most {TITLE_MAX_CHARS} characters")
    return title


def _validate_brief(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidPost("brief must be a string")
    return value


def _validate_copy(value: Any) -> Dict[str, Any]:
    """``{"base": text, "channels": {toolkit: text}}``: both keys optional."""
    copy = _require_dict("copy", value)
    unknown = [k for k in copy if k not in ("base", "channels")]
    if unknown:
        raise InvalidPost(f"copy keys must be 'base' and 'channels', got {unknown!r}")
    if "base" in copy and not isinstance(copy["base"], str):
        raise InvalidPost("copy.base must be a string")
    channels = _require_dict("copy.channels", copy.get("channels"))
    for toolkit, text in channels.items():
        if not isinstance(text, str):
            raise InvalidPost(f"copy.channels.{toolkit} must be a string")
    return dict(copy)


def _validate_format(value: Any) -> Optional[str]:
    if value is None:
        return None
    if value not in SOCIAL_POST_FORMATS:
        raise InvalidPost(f"format must be one of {list(SOCIAL_POST_FORMATS)}")
    return value


def _validate_template_id(value: Any) -> Optional[UUID]:
    if value is None:
        return None
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise InvalidPost("template_id must be a UUID") from exc


def _validate_variables(value: Any) -> Dict[str, Any]:
    """``{name: {"value": ..., "claim": bool}}``."""
    variables = _require_dict("variables", value)
    for name, spec in variables.items():
        if not isinstance(spec, dict):
            raise InvalidPost(f"variables.{name} must be an object with 'value' and 'claim'")
        unknown = [k for k in spec if k not in ("value", "claim")]
        if unknown:
            raise InvalidPost(f"variables.{name} keys must be 'value' and 'claim', got {unknown!r}")
        if "claim" in spec and not isinstance(spec["claim"], bool):
            raise InvalidPost(f"variables.{name}.claim must be a boolean")
    return dict(variables)


def _validate_sources(value: Any) -> Dict[str, Any]:
    """``{claim name: {"kind", "ref", "as_of"}}`` (D7)."""
    sources = _require_dict("sources", value)
    for name, source in sources.items():
        if not isinstance(source, dict):
            raise InvalidPost(f"sources.{name} must be an object")
        unknown = [k for k in source if k not in ("kind", "ref", "as_of")]
        if unknown:
            raise InvalidPost(f"sources.{name} keys must be 'kind', 'ref' and 'as_of', got {unknown!r}")
        if source.get("kind") not in SOURCE_KINDS:
            raise InvalidPost(f"sources.{name}.kind must be one of {list(SOURCE_KINDS)}")
        ref = source.get("ref")
        if not isinstance(ref, str) or not ref.strip():
            raise InvalidPost(f"sources.{name}.ref is required")
        as_of = source.get("as_of")
        if as_of is not None and not isinstance(as_of, str):
            raise InvalidPost(f"sources.{name}.as_of must be an ISO date-time string")
    return dict(sources)


def _validate_media(value: Any) -> Dict[str, Any]:
    """``{aspect: [deliverable ids]}``."""
    media = _require_dict("media", value)
    for aspect, ids in media.items():
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise InvalidPost(f"media.{aspect} must be a list of Deliverable ids")
    return dict(media)


_VALIDATORS = {
    "title": _validate_title,
    "brief": _validate_brief,
    "copy": _validate_copy,
    "format": _validate_format,
    "template_id": _validate_template_id,
    "variables": _validate_variables,
    "sources": _validate_sources,
    "media": _validate_media,
}


def _validate_comment(value: Optional[str], *, required: bool, what: str) -> Optional[str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise InvalidPost(f"{what} is required")
        return None
    if not isinstance(value, str):
        raise InvalidPost(f"{what} must be a string")
    if len(value) > COMMENT_MAX_CHARS:
        raise InvalidPost(f"{what} must be at most {COMMENT_MAX_CHARS} characters")
    return value.strip()


# ── the review log ──────────────────────────────────────────────────────────
def _log(post: SocialPost, actor: str, action: str, comment: Optional[str] = None, **extra: Any) -> None:
    entry: Dict[str, Any] = {
        "at": _utcnow().isoformat(),
        "by": actor,
        "action": action,
        "comment": comment,
    }
    entry.update(extra)
    post.review_log = [*(post.review_log or []), entry]


def _target(post: SocialPost, action: str) -> str:
    """The status ``action`` moves this post to, or :class:`IllegalTransition`."""
    moves = TRANSITIONS[action]
    if post.status not in moves:
        raise IllegalTransition(post.status, action)
    return moves[post.status]


def _move(post: SocialPost, action: str) -> None:
    post.status = _target(post, action)


# ── D7 ──────────────────────────────────────────────────────────────────────
def unsourced_claims(post: Any) -> List[str]:
    """Names of the variables marked ``claim: true`` that have no entry in ``sources``."""
    variables = getattr(post, "variables", None) or {}
    sources = getattr(post, "sources", None) or {}
    return sorted(
        name
        for name, spec in variables.items()
        if isinstance(spec, dict) and spec.get("claim") is True and name not in sources
    )


# ── the lifecycle ───────────────────────────────────────────────────────────
def create_draft(
    db: Any,
    *,
    workspace_id: UUID,
    created_by: str,
    title: str,
    brief: Optional[str] = None,
    copy: Optional[Mapping[str, Any]] = None,
    format: Optional[str] = None,
    template_id: Any = None,
    variables: Optional[Mapping[str, Any]] = None,
    sources: Optional[Mapping[str, Any]] = None,
    media: Optional[Mapping[str, Any]] = None,
) -> SocialPost:
    """A new post in ``draft``, added to ``db`` (the caller commits)."""
    fields = {
        "title": title,
        "brief": brief,
        "copy": copy,
        "format": format,
        "template_id": template_id,
        "variables": variables,
        "sources": sources,
        "media": media,
    }
    clean = {name: _VALIDATORS[name](value) for name, value in fields.items()}
    post = SocialPost(
        workspace_id=workspace_id,
        created_by=created_by,
        status=DRAFT,
        review_log=[],
        override_unsourced=False,
        **clean,
    )
    post.content_hash = compute_content_hash(post)
    db.add(post)
    return post


def update_post(post: SocialPost, actor: str, changes: Mapping[str, Any]) -> SocialPost:
    """Apply an edit. A content change recomputes the hash; if the post was
    approved or scheduled, it goes back to ``needs_approval`` and its approval
    is void (``approved_hash`` no longer matches ``content_hash``)."""
    unknown = [k for k in changes if k not in EDITABLE_FIELDS]
    if unknown:
        raise InvalidPost(f"only {list(EDITABLE_FIELDS)} can be edited, got {unknown!r}")
    if post.status not in EDITABLE_STATUSES:
        raise IllegalTransition(post.status, ACTION_EDIT)

    clean = {name: _VALIDATORS[name](value) for name, value in changes.items()}
    for name, value in clean.items():
        setattr(post, name, value)

    new_hash = compute_content_hash(post)
    if new_hash == post.content_hash:
        return post
    post.content_hash = new_hash
    if post.status in TRANSITIONS[ACTION_EDIT]:
        _move(post, ACTION_EDIT)
        post.override_unsourced = False
        _log(post, actor, ACTION_APPROVAL_VOIDED, "The content changed after approval.")
    return post


def submit(post: SocialPost, actor: str) -> SocialPost:
    """draft or changes_requested → needs_approval."""
    _move(post, ACTION_SUBMIT)
    _log(post, actor, ACTION_SUBMIT)
    return post


def approve(
    post: SocialPost,
    actor: str,
    *,
    override_unsourced: bool = False,
    comment: Optional[str] = None,
) -> SocialPost:
    """needs_approval → approved, bound to the current content hash (D6).

    Refuses unsourced claims (D7) unless ``override_unsourced``; an override is
    stored on the post and names the claims in ``review_log``.
    """
    target = _target(post, ACTION_APPROVE)
    comment = _validate_comment(comment, required=False, what="comment")
    unsourced = unsourced_claims(post)
    if unsourced and not override_unsourced:
        raise UnsourcedClaims(unsourced)

    content_hash = compute_content_hash(post)
    post.content_hash = content_hash
    post.approved_hash = content_hash
    post.approved_by = actor
    post.approved_at = _utcnow()
    post.override_unsourced = bool(unsourced)
    post.status = target
    if unsourced:
        note = "Approved with unsourced claims: " + ", ".join(unsourced)
        _log(post, actor, ACTION_APPROVE, comment or note, overridden_claims=unsourced)
    else:
        _log(post, actor, ACTION_APPROVE, comment)
    return post


def request_changes(post: SocialPost, actor: str, comment: str) -> SocialPost:
    """needs_approval → changes_requested, with the reviewer's comment."""
    target = _target(post, ACTION_REQUEST_CHANGES)
    comment = _validate_comment(comment, required=True, what="comment")
    post.status = target
    _log(post, actor, ACTION_REQUEST_CHANGES, comment)
    return post


def reject(post: SocialPost, actor: str, reason: Optional[str] = None) -> SocialPost:
    """needs_approval → archived."""
    target = _target(post, ACTION_REJECT)
    reason = _validate_comment(reason, required=False, what="reason")
    post.status = target
    _log(post, actor, ACTION_REJECT, reason)
    return post


def schedule(post: SocialPost, actor: str, scheduled_for: datetime, tz_name: str) -> SocialPost:
    """approved → scheduled at ``scheduled_for`` (stored in UTC; ``tz_name`` is for display)."""
    target = _target(post, ACTION_SCHEDULE)
    assert_publishable(post)
    if not isinstance(scheduled_for, datetime):
        raise InvalidPost("scheduled_for must be a date-time")
    when = _as_utc(scheduled_for)
    if when <= _utcnow():
        raise InvalidPost("scheduled_for must be in the future")
    try:
        ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError, TypeError, OSError) as exc:
        raise InvalidPost(f"unknown timezone {tz_name!r}") from exc

    post.scheduled_for = when
    post.timezone = tz_name
    post.status = target
    _log(post, actor, ACTION_SCHEDULE, None, scheduled_for=when.isoformat(), timezone=tz_name)
    return post


def unschedule(post: SocialPost, actor: str) -> SocialPost:
    """scheduled → approved; the slot is cleared, the approval stands."""
    _move(post, ACTION_UNSCHEDULE)
    post.scheduled_for = None
    _log(post, actor, ACTION_UNSCHEDULE)
    return post


# ── the publish guard ───────────────────────────────────────────────────────
def assert_publishable(post: Any) -> None:
    """Pass only an approved or scheduled post whose approval matches its
    content as it is NOW. Every publish path calls this first."""
    if post.status not in PUBLISHABLE_STATUSES:
        raise NotPublishable(f"a post that is {post.status} cannot be published")
    approved_hash = getattr(post, "approved_hash", None)
    if not approved_hash:
        raise NotPublishable("the post has no approval")
    if approved_hash != getattr(post, "content_hash", None) or approved_hash != compute_content_hash(post):
        raise NotPublishable("the content changed after it was approved")


# ── workspace-scoped reads ──────────────────────────────────────────────────
def get_post(db: Any, workspace_id: UUID, post_id: UUID) -> Optional[SocialPost]:
    """The caller's post, or ``None`` — another workspace's post is never returned."""
    return (
        db.query(SocialPost)
        .filter(SocialPost.workspace_id == workspace_id, SocialPost.id == post_id)
        .first()
    )


def list_posts(
    db: Any,
    workspace_id: UUID,
    *,
    statuses: Optional[Sequence[str]] = None,
    window_from: Optional[datetime] = None,
    window_to: Optional[datetime] = None,
) -> List[SocialPost]:
    """The caller's posts, newest first.

    ``window_from`` / ``window_to`` bound a post's date: its slot when it is
    scheduled, otherwise when it was created (``[from, to)``, UTC).
    """
    query = db.query(SocialPost).filter(SocialPost.workspace_id == workspace_id)
    if statuses:
        query = query.filter(SocialPost.status.in_(list(statuses)))
    post_date = func.coalesce(SocialPost.scheduled_for, SocialPost.created_at)
    if window_from is not None:
        query = query.filter(post_date >= _as_utc(window_from))
    if window_to is not None:
        query = query.filter(post_date < _as_utc(window_to))
    return query.order_by(SocialPost.created_at.desc(), SocialPost.id.desc()).all()
