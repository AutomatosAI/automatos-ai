"""
Socials targets (PRD-251 US-204, S3.3-prep)
===========================================

``PUT /api/socials/posts/{post_id}/targets`` replaces a post's channels with a list of
``{toolkit, post_kind, options}``. Each is checked against what the channel registry
resolves for the workspace NOW (``capabilities.social_channels``): the toolkit is a
connected channel, the kind is one it posts and is available, and each option is one
the kind's steps read (``$option.<name>``). Otherwise 422 with the reason (the
registry's own for an unavailable kind), and nothing is written. The post then has one
target per (toolkit, post kind), keyed ``sp:{post_id}:{toolkit}:{post_kind}``, holding
its options and the kind's resolved action sequence for Wave 3's publisher
(``modules/socials/targets.py``). A channel and kind the post already had keeps its row.

The channels are approved content (D6): the write goes through
``service.update_post``, so adding, removing or changing a target of an approved or
scheduled post voids its approval like any edit, and it commits by the same
compare-and-set as a PATCH (409 with the current ``content_hash`` when another writer
committed first). Nothing publishes here: ``api/socials_publish.py`` does (Wave 3).

This router has no prefix and no gate of its own: ``api/socials.py`` includes it in the
Socials router, whose ``require_socials_enabled`` answers 404 unless both switches are
on (D1). Never mount it in the app directly. It reuses that module's post helpers,
imported when a request runs, because that module includes this one.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.socials import SocialPost
from modules.socials import service
from modules.socials import targets as post_targets
from modules.socials.capabilities import ChannelKind, ChannelStep, SocialChannel, social_channels

router = APIRouter()

CAN_UPDATE = Depends(require_workspace_permission("documents:update"))
OPTION_SOURCE = "$option."  # a step parameter's source naming the target's option


class SocialPostTargetSpec(BaseModel):
    """One channel and post kind the post publishes to, with the kind's options."""

    model_config = ConfigDict(extra="forbid")

    toolkit: str = Field(..., min_length=1, max_length=post_targets.TOOLKIT_MAX_CHARS)
    post_kind: str = Field(..., min_length=1)
    options: Dict[str, Any] = Field(default_factory=dict)


class ReplaceSocialPostTargetsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    targets: List[SocialPostTargetSpec] = Field(..., max_length=post_targets.TARGETS_MAX)


# ---------------------------------------------------------------------------
# The registry check
# ---------------------------------------------------------------------------


def _option_refs(value: Any) -> FrozenSet[str]:
    """The options a step parameter's source reads: ``$option.<name>``, alone, among
    ``a|b`` alternatives, or in a list."""
    if isinstance(value, (list, tuple)):
        return frozenset(name for item in value for name in _option_refs(item))
    if not isinstance(value, str):
        return frozenset()
    return frozenset(ref[len(OPTION_SOURCE):] for ref in value.split("|") if ref.startswith(OPTION_SOURCE))


def kind_options(kind: ChannelKind) -> FrozenSet[str]:
    """The options a post kind takes: every ``$option.<name>`` its steps read."""
    return frozenset(name for step in kind.steps for value in step.params.values() for name in _option_refs(value))


def step_plan(step: ChannelStep) -> Dict[str, Any]:
    """A resolved step as a target's ``action_plan`` keeps it: the adapter data's own keys."""
    params = {name: list(value) if isinstance(value, (list, tuple)) else value for name, value in step.params.items()}
    return {
        "id": step.id,
        "action": step.action,
        "class": step.step_class,
        "params": params,
        "files": list(step.files),
        "urls": list(step.urls),
        "optional": step.optional,
        # US-301: what the publisher reads back from the call.
        "returns": dict(step.returns),
        "until": _until_plan(step.until),
        "permalink": step.permalink,
    }


def _until_plan(until: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """A status step's end condition as JSON keeps it (lists, not tuples)."""
    if until is None:
        return None
    return {name: list(value) if isinstance(value, (list, tuple)) else value for name, value in until.items()}


def _channel_kind(channels: Mapping[str, SocialChannel], toolkit: str, post_kind: str) -> Tuple[SocialChannel, ChannelKind]:
    """The channel and kind a target names, or InvalidPost saying why the workspace
    cannot post it now."""
    channel = channels.get(toolkit)
    if channel is None:
        raise service.InvalidPost(
            f"{toolkit} is not a channel connected in this workspace: connect it in Composio, "
            "then choose it from the channels GET /api/socials/channels lists"
        )
    kind = next((offered for offered in channel.post_kinds if offered.kind == post_kind), None)
    if kind is None:
        posts = ", ".join(offered.kind for offered in channel.post_kinds)
        raise service.InvalidPost(f"{channel.label} does not post a {post_kind}: it posts {posts}")
    if not kind.available:
        raise service.InvalidPost(f"{channel.label} cannot post a {post_kind} now: {kind.reason}")
    return channel, kind


def _resolved(channels: Mapping[str, SocialChannel], target: Mapping[str, Any]) -> Dict[str, Any]:
    channel, kind = _channel_kind(channels, target["toolkit"], target["post_kind"])
    takes = kind_options(kind)
    unknown = sorted(set(target["options"]) - takes)
    if unknown:
        offered = ", ".join(sorted(takes)) or "no options"
        raise service.InvalidPost(f"{channel.label} {kind.kind} takes {offered}, not {', '.join(unknown)}")
    return {**target, post_targets.STEPS: [step_plan(step) for step in kind.steps]}


def resolve_targets(db: Session, workspace_id: Any, requested: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """``requested`` (``{toolkit, post_kind, options}`` each) checked against the
    workspace's registry now, sorted, each with its kind's resolved ``steps``.

    The shape is checked first (``service.validate_targets``), so a malformed list
    reads nothing, and an empty one clears the post's channels without reading the
    registry. Raises ``service.InvalidPost`` (422) naming the first target the
    workspace cannot post, and why. The registry may commit the session (a pending
    Composio connection upgraded): call this before staging any write.
    """
    wanted = service.validate_targets(requested)
    if not wanted:
        return []
    channels = {channel.toolkit: channel for channel in social_channels(db, workspace_id)}
    return [_resolved(channels, target) for target in wanted]


# ---------------------------------------------------------------------------
# The write: one flow, which a route (and an agent's tool) calls
# ---------------------------------------------------------------------------


def _posts_api() -> Any:
    """``api/socials.py``: its router includes this one, so it is imported when a request runs."""
    from api import socials

    return socials


def set_post_targets(
    db: Session, post: SocialPost, actor: str, targets: List[Dict[str, Any]], *, agent: Optional[str] = None
) -> Dict[str, Any]:
    """Replace ``post``'s targets with ``targets`` (``{toolkit, post_kind, options}``
    each), checked against the registry now; the compare-and-set commit.

    A change to an approved or scheduled post's channels voids its approval (D6).
    ``agent`` names the agent a tool acts for (US-116), as ``edit_post``'s does.
    """
    status, content_hash = post.status, post.content_hash
    resolved = resolve_targets(db, post.workspace_id, targets)
    service.update_post(post, actor, {post_targets.TARGETS: resolved}, agent=agent)
    return _posts_api()._commit_unchanged(db, post, status=status, content_hash=content_hash)


@router.put("/posts/{post_id}/targets", dependencies=[CAN_UPDATE])
def replace_social_post_targets(
    post_id: UUID,
    body: ReplaceSocialPostTargetsRequest,
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """Replace the post's channels (``set_post_targets``) and answer the post, with its
    ``targets``. A plain ``def``: FastAPI runs its synchronous database work in the
    threadpool (F105)."""
    posts_api = _posts_api()
    post = posts_api._load(db, ctx, post_id)
    actor = posts_api._actor(ctx)
    try:
        return set_post_targets(db, post, actor, [target.model_dump() for target in body.targets])
    except service.SocialsError as exc:
        posts_api._raise_for(exc)
