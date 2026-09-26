"""PRD-251 S0.3a: the post lifecycle, as pure logic.

* **The status machine.** ``TRANSITIONS`` is the whole table, keyed by action:
  an action moves a post only from the statuses listed for it, and anything
  else raises :class:`IllegalTransition`. Wave 0 moves posts between draft,
  needs_approval, changes_requested, approved, scheduled and archived. Wave 1
  renders (S1.1c): ``render`` moves a post that holds no approval to
  rendering, and the render ends in needs_approval with the rendered files in
  ``media``, or in failed with the report in ``review_log``. A failed post can
  be edited and rendered again. Publishing and missed arrive with Wave 3.
* **The content hash (D6).** ``compute_content_hash`` is sha256 over canonical
  JSON of what is published: copy, variables, sources, format, template_id and
  media. An approval binds to it: the approver sends the hash of the version
  they were shown, and a post whose content has changed since refuses the
  approval (:class:`StaleContent`, carrying the current hash). Any content edit
  changes the hash, so an approved or scheduled post goes back to
  needs_approval and its approval is void, because ``approved_hash`` no longer
  matches.
* **The write guard.** ``claim_unchanged`` is a compare-and-set on the row:
  a write commits only if the post still has the status and hash the request
  checked. So an edit another worker commits mid-request is never approved,
  an approval never lands on copy nobody reviewed, and a stale copy of
  ``review_log`` never overwrites entries another writer committed.
* **Facts carry sources (D7).** A variable marked ``claim: true`` needs an entry
  in ``sources``. ``approve`` refuses unsourced claims unless the approver
  overrides, and the override is stored and named in ``review_log``. Wave 1
  (S1.4) resolves the sources in the workspace (``modules/socials/sources.py``):
  a claim whose source no longer resolves, a deleted Deliverable say, counts as
  unsourced, and the approval record names why.
* **The publish guard.** ``assert_publishable`` passes only an approved or
  scheduled post whose approval matches its content as it is NOW.
* **Rendered media (D6).** A finished render writes ``media`` as
  ``{aspect: [file records]}``, each with its Deliverable id and the sha256 of
  its bytes, so the content hash (and an approval) binds to the exact files.
  Records come only from a render (``finish_render``); an edit may set
  ``media`` only to Deliverable ids, so no client can forge a digest.
* **The voice (D11, Wave 1 S1.5).** ``voice`` is how the next render speaks the
  script: ``None`` is Kokoro, the template's own voice, and a voice toolkit is
  ``{"toolkit", "voice_id", "name"}`` (``validate_voice`` checks the shape;
  ``modules/socials/recipes/voice.py`` whether the workspace can speak with
  it). It is a render setting, not content: it is outside the hash, and what
  it changes reaches the hash through the next render's file digests.
* **Footage (D12, Wave 1 S1.8).** ``footage`` is what the post asks its
  template's slots to be filled with: ``{slot: {"prompt"}}``, footage or a still
  from the workspace's Composio generation toolkit. A render generates it,
  copies the file into our storage and records it on the slot
  (``record_footage``: ``"status": "done"``, its Deliverable, sha256, cost). An
  edit that keeps a slot's prompt keeps what was made for it; a new prompt asks
  again. Like the voice it is a render setting, outside the hash: the rendered
  files' digests carry what it changed.
* **Music credit (Wave 1 S1.6).** A CC BY track asks for credit wherever the
  video is published: ``with_credits`` appends its line to the post's copy,
  the base text and every channel's own text, once. A render appends the line
  of the music it mixed (``finish_render``); a save appends those of the media
  it names (``modules/socials/credits.py``).
* **Agents draft (Wave 1 US-116, S4.1).** An agent's tools save through the
  same lifecycle, and ``review_log`` names the agent that wrote what a person
  approves: ``create_draft(agent=)`` opens it with a ``draft`` entry, and
  ``update_post(agent=)`` logs an ``edit`` entry naming the fields it changed.
  A person's own saves are not logged. No agent tool approves, schedules or
  publishes.

No FastAPI here: the API maps these exceptions to status codes. Every review
action appends to ``review_log``, the history the approval UI shows. JSON
fields are always reassigned, never mutated in place.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func, update

from core.models.socials import SOCIAL_POST_FORMATS, SocialPost
from core.social_templates import MAX_SLOTS, VARIABLE_NAME

# ── statuses ────────────────────────────────────────────────────────────────
DRAFT = "draft"
RENDERING = "rendering"
NEEDS_APPROVAL = "needs_approval"
CHANGES_REQUESTED = "changes_requested"
APPROVED = "approved"
SCHEDULED = "scheduled"
FAILED = "failed"
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
# S1.1c: a render starts, then ends one way or the other.
ACTION_RENDER = "render"
ACTION_RENDER_DONE = "render_done"
ACTION_RENDER_FAILED = "render_failed"
# US-116: an agent drafted the post. Only logged, never a move of the status machine.
ACTION_DRAFT = "draft"

# The whole status machine: action → {from status: to status}. An action
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
    # Wave 1 (S1.1c): only a post that holds no approval renders. An approved
    # or scheduled post is edited first, which voids its approval.
    ACTION_RENDER: {
        DRAFT: RENDERING,
        CHANGES_REQUESTED: RENDERING,
        NEEDS_APPROVAL: RENDERING,
        FAILED: RENDERING,
    },
    ACTION_RENDER_DONE: {RENDERING: NEEDS_APPROVAL},
    ACTION_RENDER_FAILED: {RENDERING: FAILED},
}

# The same table seen per status: current status → the statuses it may move to.
ALLOWED_TRANSITIONS: Dict[str, frozenset] = {
    status: frozenset(moves[status] for moves in TRANSITIONS.values() if status in moves)
    for status in {s for moves in TRANSITIONS.values() for s in moves}
}

# The statuses a post's content may be edited in. An edit to an approved or
# scheduled post voids its approval; in the others the post keeps its status.
# A failed render is fixed by an edit and rendered again. A rendering post is
# not edited: the render is working from its content.
EDITABLE_STATUSES = frozenset({DRAFT, NEEDS_APPROVAL, CHANGES_REQUESTED, APPROVED, SCHEDULED, FAILED})
PUBLISHABLE_STATUSES = frozenset({APPROVED, SCHEDULED})

# What the hash covers (D6), and what a post edit may change. The voice (D11)
# and the footage (D12) are render settings: editable, never hashed.
CONTENT_FIELDS = ("copy", "variables", "sources", "format", "template_id", "media")
LABEL_FIELDS = ("title", "brief")
RENDER_FIELDS = ("voice", "footage")
EDITABLE_FIELDS = LABEL_FIELDS + CONTENT_FIELDS + RENDER_FIELDS

# D11: the default voice, Kokoro inside media-render; any other toolkit is a
# Composio voice toolkit (modules/socials/recipes/voice.py).
KOKORO = "kokoro"
VOICE_KEYS = ("toolkit", "voice_id", "name")
VOICE_TOOLKIT = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
VOICE_TEXT_MAX_CHARS = 200

# D12 (S1.8): footage a post asks for, per slot, and what a render recorded for
# it. A client writes the prompt; the rest is the server's, and a client that
# sends it back has it ignored.
FOOTAGE_REQUEST_KEYS = ("prompt",)
FOOTAGE_RECORD_KEYS = (
    "status", "toolkit", "model", "deliverable_id", "name", "sha256", "bytes", "content_type",
    "estimate_usd", "cost_usd", "generated_at",
)
FOOTAGE_DONE = "done"
FOOTAGE_PROMPT_MAX_CHARS = 1500

# D7: where a claim's source may come from.
SOURCE_KINDS = ("deliverable", "report", "document", "url", "metric")

TITLE_MAX_CHARS = 500
COMMENT_MAX_CHARS = 2000

# A rendered file record in ``media`` (finish_render): its Deliverable, the
# sha256 of its bytes and what the renderer measured.
RENDERED_FILE_REQUIRED = ("deliverable_id", "name", "sha256", "bytes")
RENDERED_FILE_OPTIONAL = ("content_type", "duration", "width", "height")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# compute_content_hash's output: sha256, lowercase hex.
CONTENT_HASH_PATTERN = r"^[0-9a-f]{64}$"


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
    """Claims with no source, or whose source did not resolve (``unresolved``:
    claim → why, S1.4)."""

    def __init__(self, names: Iterable[str], unresolved: Optional[Mapping[str, str]] = None):
        self.names = sorted(names)
        self.unresolved = {name: unresolved[name] for name in self.names if unresolved and name in unresolved}
        missing = [name for name in self.names if name not in self.unresolved]
        reasons = []
        if missing:
            reasons.append("these claims have no source: " + ", ".join(missing))
        if self.unresolved:
            reasons.append(
                "these claims' sources could not be found: "
                + "; ".join(f"{name} ({why})" for name, why in self.unresolved.items())
            )
        super().__init__("; ".join(reasons) + " (approve with override_unsourced to publish them anyway)")


class NotPublishable(SocialsError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class StaleContent(SocialsError):
    """The post is not the version the request was made against (D6): the
    approver was shown other content, or another writer committed first."""

    def __init__(self, current_hash: str):
        self.current_hash = current_hash
        super().__init__(
            "the post changed since you opened it: review the current version, then try again"
        )


class PostNotFound(SocialsError):
    """No such post in the caller's workspace (another workspace's included)."""

    def __init__(self) -> None:
        super().__init__("Post not found")


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


def validate_sources(value: Any) -> Dict[str, Any]:
    """``{claim name: {"kind", "ref", "as_of"}}`` (D7): the shape only.
    ``modules/socials/sources.py`` checks each source exists in the workspace."""
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


def _voice_text(value: Any, where: str, *, required: bool) -> Optional[str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise InvalidPost(f"{where} is required for a voice toolkit")
        return None
    if not isinstance(value, str):
        raise InvalidPost(f"{where} must be a string")
    text = value.strip()
    if len(text) > VOICE_TEXT_MAX_CHARS:
        raise InvalidPost(f"{where} must be at most {VOICE_TEXT_MAX_CHARS} characters")
    return text


def validate_voice(value: Any) -> Optional[Dict[str, Any]]:
    """The post's voice (D11): ``None`` (or ``{}``, or ``{"toolkit": "kokoro"}``)
    is Kokoro, stored as ``None``; a voice toolkit is ``{"toolkit", "voice_id",
    "name"?}``. The shape only: whether the workspace can speak with the
    toolkit now is ``modules/socials/recipes/voice.py``'s to say."""
    if value is None:
        return None
    voice = _require_dict("voice", value)
    if not voice:
        return None
    unknown = [k for k in voice if k not in VOICE_KEYS]
    if unknown:
        raise InvalidPost(f"voice keys must be {list(VOICE_KEYS)}, got {unknown!r}")
    toolkit = voice.get("toolkit")
    toolkit = toolkit.strip().lower() if isinstance(toolkit, str) else ""
    if not VOICE_TOOLKIT.match(toolkit):
        raise InvalidPost("voice.toolkit must name a voice, such as kokoro or fish_audio")
    if toolkit == KOKORO:
        if set(voice) - {"toolkit"}:
            raise InvalidPost("Kokoro speaks with the template's own voice: set only voice.toolkit")
        return None
    clean = {"toolkit": toolkit, "voice_id": _voice_text(voice.get("voice_id"), "voice.voice_id", required=True)}
    name = _voice_text(voice.get("name"), "voice.name", required=False)
    if name:
        clean["name"] = name
    return clean


def validate_footage(value: Any) -> Optional[Dict[str, Dict[str, str]]]:
    """The footage the post asks for (D12): ``None`` (or ``{}``) asks for none, and
    every slot plays the template's own motion graphics; otherwise ``{slot:
    {"prompt"}}``. The shape only: whether the template has the slot, and lets
    a toolkit fill it, is the api's check against the template."""
    if value is None:
        return None
    footage = _require_dict("footage", value)
    if not footage:
        return None
    if len(footage) > MAX_SLOTS:
        raise InvalidPost(f"footage names at most {MAX_SLOTS} slots")
    clean: Dict[str, Dict[str, str]] = {}
    for slot, request in footage.items():
        if not isinstance(slot, str) or not VARIABLE_NAME.match(slot):
            raise InvalidPost(f"footage.{slot} is not a slot name (letters, digits and _, not starting with a digit)")
        if not isinstance(request, dict):
            raise InvalidPost(f'footage.{slot} must be an object such as {{"prompt": "a calm sea at dawn"}}')
        unknown = [k for k in request if k not in FOOTAGE_REQUEST_KEYS + FOOTAGE_RECORD_KEYS]
        if unknown:
            raise InvalidPost(f"footage.{slot} takes a prompt, got {unknown!r}")
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise InvalidPost(f"footage.{slot}.prompt is required")
        text = prompt.strip()
        if len(text) > FOOTAGE_PROMPT_MAX_CHARS:
            raise InvalidPost(f"footage.{slot}.prompt must be at most {FOOTAGE_PROMPT_MAX_CHARS} characters")
        clean[slot] = {"prompt": text}
    return clean


def footage_after_edit(stored: Any, requested: Optional[Mapping[str, Mapping[str, str]]]) -> Optional[Dict[str, Any]]:
    """``requested`` (``validate_footage``'s shape) keeping what a render already
    made for every slot whose prompt did not change: a new prompt asks again."""
    if requested is None:
        return None
    before = stored if isinstance(stored, dict) else {}
    kept: Dict[str, Any] = {}
    for slot, request in requested.items():
        record = before.get(slot)
        same = isinstance(record, dict) and record.get("prompt") == request["prompt"]
        kept[slot] = dict(record) if same else dict(request)
    return kept


def record_footage(post: SocialPost, slot: str, record: Mapping[str, Any]) -> bool:
    """Record what a render made for ``slot`` (S1.8), marked done. Only while the
    post still asks for the slot with the prompt it was made for: ``False``, and
    nothing changes, when the request has changed since."""
    footage = dict(post.footage) if isinstance(post.footage, dict) else {}
    asked = footage.get(slot)
    if not isinstance(asked, dict) or asked.get("prompt") != record.get("prompt"):
        return False
    footage[slot] = {**dict(record), "status": FOOTAGE_DONE}
    post.footage = footage
    return True


_VALIDATORS = {
    "title": _validate_title,
    "brief": _validate_brief,
    "copy": _validate_copy,
    "format": _validate_format,
    "template_id": _validate_template_id,
    "variables": _validate_variables,
    "sources": validate_sources,
    "media": _validate_media,
    "voice": validate_voice,
    "footage": validate_footage,
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
def unsourced_claims(post: Any, unresolved: Optional[Iterable[str]] = None) -> List[str]:
    """Names of the variables marked ``claim: true`` that have no entry in
    ``sources``, or whose source is among ``unresolved`` (it does not resolve in
    the workspace any more, S1.4)."""
    variables = getattr(post, "variables", None) or {}
    sources = getattr(post, "sources", None) or {}
    broken = set(unresolved or ())
    return sorted(
        name
        for name, spec in variables.items()
        if isinstance(spec, dict) and spec.get("claim") is True and (name not in sources or name in broken)
    )


# ── music credit (S1.6) ─────────────────────────────────────────────────────
def _credited(text: Any, lines: Sequence[str]) -> Any:
    if not isinstance(text, str):
        return text  # not copy the validators take (null included): left for them to refuse
    body = text.rstrip()
    missing = [line for line in lines if line not in body]
    if not missing:
        return text
    return "\n\n".join(([body] if body else []) + missing)


def with_credits(copy: Any, lines: Iterable[str]) -> Any:
    """``copy`` with each credit line at the end of its base text (one it lacks
    starts as the lines) and of every channel's own text that lacks it: a new
    object. A line already there is not added again, and copy with nothing to
    add comes back as it was."""
    wanted = [line for line in dict.fromkeys(lines) if isinstance(line, str) and line]
    if not wanted or (copy is not None and not isinstance(copy, dict)):
        return copy
    out = dict(copy or {})
    out["base"] = _credited(out.get("base", ""), wanted)
    channels = out.get("channels")
    if isinstance(channels, dict) and channels:
        out["channels"] = {name: _credited(text, wanted) for name, text in channels.items()}
    return out if out != (copy or {}) else copy


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
    voice: Optional[Mapping[str, Any]] = None,
    footage: Optional[Mapping[str, Any]] = None,
    agent: Optional[str] = None,
) -> SocialPost:
    """A new post in ``draft``, added to ``db`` (the caller commits).

    ``agent`` names the agent drafting it (US-116): ``review_log`` then opens
    with a ``draft`` entry by ``created_by`` that names it.
    """
    fields = {
        "title": title,
        "brief": brief,
        "copy": copy,
        "format": format,
        "template_id": template_id,
        "variables": variables,
        "sources": sources,
        "media": media,
        "voice": voice,
        "footage": footage,
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
    if agent:
        _log(post, created_by, ACTION_DRAFT, f"Drafted by {agent}.", agent=agent)
    db.add(post)
    return post


def update_post(
    post: SocialPost, actor: str, changes: Mapping[str, Any], *, agent: Optional[str] = None
) -> SocialPost:
    """Apply an edit. A content change recomputes the hash; if the post was
    approved or scheduled, it goes back to ``needs_approval`` and its approval
    is void (``approved_hash`` no longer matches ``content_hash``).

    ``agent`` names the agent editing (US-116): an edit that changes a field
    logs an ``edit`` entry by ``actor`` naming the agent and the fields, before
    the approval it voids.
    """
    unknown = [k for k in changes if k not in EDITABLE_FIELDS]
    if unknown:
        raise InvalidPost(f"only {list(EDITABLE_FIELDS)} can be edited, got {unknown!r}")
    if post.status not in EDITABLE_STATUSES:
        raise IllegalTransition(post.status, ACTION_EDIT)

    clean = {name: _VALIDATORS[name](value) for name, value in changes.items()}
    if "footage" in clean:
        clean["footage"] = footage_after_edit(post.footage, clean["footage"])
    changed = [name for name in EDITABLE_FIELDS if name in clean and getattr(post, name) != clean[name]]
    for name, value in clean.items():
        setattr(post, name, value)
    if agent and changed:
        _log(post, actor, ACTION_EDIT, f"Edited by {agent}: {', '.join(changed)}.", agent=agent, fields=changed)

    new_hash = compute_content_hash(post)
    if new_hash == post.content_hash:
        return post
    post.content_hash = new_hash
    if post.status in TRANSITIONS[ACTION_EDIT]:
        _move(post, ACTION_EDIT)
        post.override_unsourced = False
        _log(post, actor, ACTION_APPROVAL_VOIDED, "The content changed after approval.")
    return post


def submit(post: SocialPost, actor: str, note: Optional[str] = None) -> SocialPost:
    """draft or changes_requested → needs_approval. ``note`` tells the reviewer
    what to look at (an agent's, US-116); it is the history entry's comment."""
    target = _target(post, ACTION_SUBMIT)
    note = _validate_comment(note, required=False, what="note")
    post.status = target
    _log(post, actor, ACTION_SUBMIT, note)
    return post


def approve(
    post: SocialPost,
    actor: str,
    *,
    content_hash: str,
    override_unsourced: bool = False,
    comment: Optional[str] = None,
    unresolved_sources: Optional[Mapping[str, str]] = None,
) -> SocialPost:
    """needs_approval → approved, bound to the content the approver saw (D6).

    ``content_hash`` is the hash of the version the approver was shown. When the
    post's content is not that version any more, :class:`StaleContent` (with the
    current hash) and the post is unchanged. Refuses unsourced claims (D7)
    unless ``override_unsourced``; an override is stored on the post and names
    the claims in ``review_log``. ``unresolved_sources`` (claim → why) are the
    sources the caller found missing from the workspace (S1.4): their claims
    count as unsourced, and an override records why.
    """
    target = _target(post, ACTION_APPROVE)
    comment = _validate_comment(comment, required=False, what="comment")
    current = compute_content_hash(post)
    if content_hash != current or content_hash != post.content_hash:
        raise StaleContent(current)
    unsourced = unsourced_claims(post, unresolved_sources)
    if unsourced and not override_unsourced:
        raise UnsourcedClaims(unsourced, unresolved_sources)

    post.approved_hash = current
    post.approved_by = actor
    post.approved_at = _utcnow()
    post.override_unsourced = bool(unsourced)
    post.status = target
    if unsourced:
        note = "Approved with unsourced claims: " + ", ".join(unsourced)
        extra: Dict[str, Any] = {"overridden_claims": unsourced}
        broken = {name: why for name, why in (unresolved_sources or {}).items() if name in unsourced}
        if broken:
            extra["unresolved_sources"] = dict(sorted(broken.items()))
        _log(post, actor, ACTION_APPROVE, comment or note, **extra)
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


# ── rendering (S1.1c) ───────────────────────────────────────────────────────
def assert_can_render(post: Any) -> None:
    """:class:`IllegalTransition` unless ``post`` may start a render now."""
    _target(post, ACTION_RENDER)


def start_render(post: SocialPost, actor: str) -> SocialPost:
    """draft, changes_requested, needs_approval or failed → rendering."""
    _move(post, ACTION_RENDER)
    _log(post, actor, ACTION_RENDER)
    return post


def _optional_number(record: Mapping[str, Any], key: str, where: str) -> Optional[float]:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise InvalidPost(f"{where}.{key} must be a non-negative number")
    return value


def _rendered_file(record: Any, where: str) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise InvalidPost(f"{where} must be a rendered file record")
    unknown = [k for k in record if k not in RENDERED_FILE_REQUIRED + RENDERED_FILE_OPTIONAL]
    missing = [k for k in RENDERED_FILE_REQUIRED if k not in record]
    if unknown or missing:
        raise InvalidPost(f"{where} has unknown keys {unknown!r} or lacks {missing!r}")
    for key in ("deliverable_id", "name"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise InvalidPost(f"{where}.{key} is required")
    if not isinstance(record["sha256"], str) or not SHA256_PATTERN.match(record["sha256"]):
        raise InvalidPost(f"{where}.sha256 must be a lowercase sha256")
    size = record["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise InvalidPost(f"{where}.bytes must be a positive whole number")
    clean: Dict[str, Any] = {key: record[key] for key in RENDERED_FILE_REQUIRED}
    if record.get("content_type") is not None:
        if not isinstance(record["content_type"], str):
            raise InvalidPost(f"{where}.content_type must be a string")
        clean["content_type"] = record["content_type"]
    for key in ("duration", "width", "height"):
        value = _optional_number(record, key, where)
        if value is not None:
            clean[key] = value
    return clean


def _rendered_media(media: Any) -> Dict[str, List[Dict[str, Any]]]:
    """``{aspect: [rendered file records]}``, at least one file."""
    if not isinstance(media, Mapping) or not media:
        raise InvalidPost("a render must produce at least one file")
    clean: Dict[str, List[Dict[str, Any]]] = {}
    for aspect, records in media.items():
        if not isinstance(aspect, str) or not aspect.strip():
            raise InvalidPost("a rendered aspect must be named")
        if not isinstance(records, (list, tuple)) or not records:
            raise InvalidPost(f"media.{aspect} must list the rendered files")
        clean[aspect] = [_rendered_file(r, f"media.{aspect}[{i}]") for i, r in enumerate(records)]
    return clean


def finish_render(
    post: SocialPost,
    actor: str,
    media: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    summary: Optional[str] = None,
    report: Optional[Mapping[str, Any]] = None,
    credits: Sequence[str] = (),
) -> SocialPost:
    """rendering → needs_approval with the rendered files as ``media``.

    ``media`` replaces what the post carried before, and the content hash is
    recomputed over it, so an approval binds to these exact files (D6).
    ``credits`` are the lines the render's music asks for (S1.6, a CC BY
    track): they join the copy the approver reviews, and the history says so.
    """
    target = _target(post, ACTION_RENDER_DONE)
    post.media = _rendered_media(media)
    credited = with_credits(post.copy, credits)
    added = credited is not post.copy
    if added:
        post.copy = credited
    post.content_hash = compute_content_hash(post)
    post.status = target
    extra: Dict[str, Any] = {"report": dict(report)} if report else {}
    if added:
        extra["credits_added"] = [line for line in dict.fromkeys(credits) if line]
    _log(post, actor, ACTION_RENDER_DONE, summary, **extra)
    return post


def fail_render(
    post: SocialPost,
    actor: str,
    message: str,
    *,
    report: Optional[Mapping[str, Any]] = None,
) -> SocialPost:
    """rendering → failed, with the reason and the renderer's report in ``review_log``.

    The post's content is untouched: edit it and render again.
    """
    target = _target(post, ACTION_RENDER_FAILED)
    text = (message or "The render failed.").strip()[:COMMENT_MAX_CHARS]
    post.status = target
    extra = {"report": dict(report)} if report else {}
    _log(post, actor, ACTION_RENDER_FAILED, text, **extra)
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


# ── the write guard ─────────────────────────────────────────────────────────
def claim_unchanged(db: Any, post: SocialPost, *, status: str, content_hash: str) -> bool:
    """Compare-and-set before a write: lock the post's row until this
    transaction ends, but only if it still has ``status`` and ``content_hash``,
    the version the request checked.

    ``False`` means another writer committed since the post was loaded. The
    caller must roll back and write nothing. The no-op UPDATE takes the row lock
    on Postgres (the write lock on SQLite), and its WHERE reads the latest
    committed row. A racing write therefore either lands first, and this returns
    ``False``, or waits for this transaction to commit. Autoflush is off, so the
    caller's pending changes to ``post`` cannot land before the check.
    """
    table = SocialPost.__table__
    with db.no_autoflush:
        result = db.execute(
            update(table)
            .where(
                table.c.id == post.id,
                table.c.workspace_id == post.workspace_id,
                table.c.status == status,
                table.c.content_hash == content_hash,
            )
            .values(status=table.c.status)
        )
    return result.rowcount == 1


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
    limit: Optional[int] = None,
) -> List[SocialPost]:
    """The caller's posts, newest first; the ``limit`` newest when given.

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
    query = query.order_by(SocialPost.created_at.desc(), SocialPost.id.desc())
    return (query.limit(limit) if limit is not None else query).all()
