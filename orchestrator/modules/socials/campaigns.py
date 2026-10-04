"""
Socials campaigns and series approval (PRD-251 D6, S2.4)
========================================================

A campaign is a named set of posts in one workspace (``social_campaigns``,
US-201). Its ``approval_mode`` is ``per_post`` (every post approved on its own,
the default) or ``series``.

**Series approval (D6).** When the workspace's series approval switch is on
(``workspace.settings['socials'].series_approval``, ``modules/socials/settings.py``)
AND the campaign is in ``series`` mode, an approver may approve the campaign's
posts as one series. The request carries every post the approver was shown,
each with the ``content_hash`` of the version shown. Each shown post that is
still in the campaign, waiting for approval, with that hash NOW, is approved
exactly as a single approval: ``service.approve`` (so an unsourced claim needs
that post's own ``override_unsourced``, D7, with its sources resolved again
first), then the same compare-and-set as a single approve
(``service.claim_unchanged``), committed one post at a time. The approved hash
joins the campaign's ``approved_hash_set`` in the same commit, and the campaign
records who approved and when.

A post whose hash changed between display and approval, one with unsourced
claims and no override, one no longer waiting for approval or no longer in the
campaign, is left as it is and reported with why. So is a post of the campaign
waiting for approval that the approver was not shown. Nothing is approved by
its hash being in the set: a post added to the campaign, or edited, later needs
its own approval (an edit voids an approval through ``service.update_post``).

Every read is scoped to the caller's workspace: another workspace's campaign or
post is never returned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID, uuid4

from sqlalchemy import func

from core.models.socials import SocialCampaign, SocialPost
from modules.socials import service
from modules.socials import sources as post_sources
from modules.socials.settings import parse_workspace_socials

logger = logging.getLogger(__name__)

PER_POST = "per_post"
SERIES = "series"
APPROVAL_MODES = (PER_POST, SERIES)  # core.models.socials.SOCIAL_CAMPAIGN_APPROVAL_MODES
CAMPAIGN_NAME_MAX_CHARS = 200  # social_campaigns.name is String(200)
# The most posts one series approval may carry.
SERIES_MAX_POSTS = 200

# Why a post was left unapproved by a series approval.
LEFT_CHANGED = "changed"
LEFT_UNSOURCED = "unsourced"
LEFT_NOT_WAITING = "not_waiting"
LEFT_NOT_IN_CAMPAIGN = "not_in_campaign"
LEFT_NOT_SHOWN = "not_shown"
LEFT_NOT_IN_BATCH = "not_in_batch"  # PRD-251C (C2): approving a plan's week, a post of another batch

SERIES_OFF_FOR_WORKSPACE = (
    "Series approval is off for this workspace, so each post is approved on its own. "
    "A workspace owner or admin turns it on in the Socials tab."
)
CAMPAIGN_NOT_A_SERIES = (
    "This campaign approves post by post. Set its approval mode to series to approve it as one series."
)


class CampaignNotFound(service.SocialsError):
    """No such campaign in the caller's workspace (another workspace's included)."""

    def __init__(self) -> None:
        super().__init__("Campaign not found")


class SeriesApprovalRefused(service.SocialsError):
    """Series approval is not allowed for this campaign now (409)."""


@dataclass(frozen=True)
class ShownPost:
    """A post the approver was shown: its id, the hash of the version shown, and
    whether its unsourced claims are confirmed (D7's second confirmation)."""

    post_id: UUID
    content_hash: str
    override_unsourced: bool = False


# ── validation ──────────────────────────────────────────────────────────────
def validate_name(value: Any) -> str:
    """A campaign name: text, trimmed, 1 to ``CAMPAIGN_NAME_MAX_CHARS`` characters."""
    if not isinstance(value, str) or not value.strip():
        raise service.InvalidPost("name is required")
    name = value.strip()
    if len(name) > CAMPAIGN_NAME_MAX_CHARS:
        raise service.InvalidPost(f"name must be at most {CAMPAIGN_NAME_MAX_CHARS} characters")
    return name


def validate_mode(value: Any) -> str:
    """An approval mode: ``per_post`` or ``series``."""
    if value not in APPROVAL_MODES:
        raise service.InvalidPost(f"approval_mode must be one of {list(APPROVAL_MODES)}")
    return value


# ── campaigns ───────────────────────────────────────────────────────────────
def create_campaign(
    db: Any, *, workspace_id: UUID, created_by: str, name: str, approval_mode: str = PER_POST
) -> SocialCampaign:
    """A new campaign with no posts and nothing approved, added to ``db`` (the caller commits)."""
    campaign = SocialCampaign(
        id=uuid4(),
        workspace_id=workspace_id,
        name=validate_name(name),
        approval_mode=validate_mode(approval_mode),
        approved_hash_set=[],
        created_by=created_by,
    )
    db.add(campaign)
    return campaign


def update_campaign(campaign: SocialCampaign, changes: Mapping[str, Any]) -> SocialCampaign:
    """Rename the campaign or change its approval mode. The approved set stays: it
    records what a series approval approved, and approves nothing by itself."""
    if "name" in changes:
        campaign.name = validate_name(changes["name"])
    if "approval_mode" in changes:
        campaign.approval_mode = validate_mode(changes["approval_mode"])
    return campaign


def get_campaign(db: Any, workspace_id: UUID, campaign_id: UUID) -> Optional[SocialCampaign]:
    """The caller's campaign, or ``None``: another workspace's is never returned."""
    return (
        db.query(SocialCampaign)
        .filter(SocialCampaign.workspace_id == workspace_id, SocialCampaign.id == campaign_id)
        .first()
    )


def list_campaigns(db: Any, workspace_id: UUID) -> List[SocialCampaign]:
    """The caller's campaigns, newest first."""
    return (
        db.query(SocialCampaign)
        .filter(SocialCampaign.workspace_id == workspace_id)
        .order_by(SocialCampaign.created_at.desc(), SocialCampaign.id.desc())
        .all()
    )


def post_counts(db: Any, workspace_id: UUID) -> Dict[UUID, int]:
    """Campaign id → how many of the caller's posts it holds."""
    rows = (
        db.query(SocialPost.campaign_id, func.count(SocialPost.id))
        .filter(SocialPost.workspace_id == workspace_id, SocialPost.campaign_id.isnot(None))
        .group_by(SocialPost.campaign_id)
        .all()
    )
    return {campaign_id: count for campaign_id, count in rows}


def campaign_posts(db: Any, workspace_id: UUID, campaign_id: UUID) -> List[SocialPost]:
    """The campaign's posts in the caller's workspace, oldest first (the series' order)."""
    return (
        db.query(SocialPost)
        .filter(SocialPost.workspace_id == workspace_id, SocialPost.campaign_id == campaign_id)
        .order_by(SocialPost.created_at.asc(), SocialPost.id.asc())
        .all()
    )


def add_post(campaign: SocialCampaign, post: SocialPost) -> SocialPost:
    """Put ``post`` in ``campaign`` (from another campaign, it moves). Membership is
    not content: the hash and any approval stay as they are."""
    if post.workspace_id != campaign.workspace_id:
        raise service.PostNotFound()
    post.campaign_id = campaign.id
    return post


def remove_post(campaign: SocialCampaign, post: SocialPost) -> SocialPost:
    """Take ``post`` out of ``campaign``; a post the campaign does not hold is not found."""
    if post.campaign_id != campaign.id:
        raise service.PostNotFound()
    post.campaign_id = None
    return post


# ── series approval (D6) ────────────────────────────────────────────────────
def assert_series_allowed(workspace_settings: Optional[Dict[str, Any]], campaign: SocialCampaign) -> None:
    """Series approval needs the workspace switch on and the campaign in series mode."""
    if not parse_workspace_socials(workspace_settings).series_approval:
        raise SeriesApprovalRefused(SERIES_OFF_FOR_WORKSPACE)
    if campaign.approval_mode != SERIES:
        raise SeriesApprovalRefused(CAMPAIGN_NOT_A_SERIES)


def validate_shown(shown: Sequence[ShownPost]) -> List[ShownPost]:
    """The shown posts, each once, at most ``SERIES_MAX_POSTS``."""
    if not shown:
        raise service.InvalidPost("posts: list the posts you were shown, each with its content_hash")
    if len(shown) > SERIES_MAX_POSTS:
        raise service.InvalidPost(f"posts: at most {SERIES_MAX_POSTS} in one series approval")
    ids = [item.post_id for item in shown]
    if len(set(ids)) != len(ids):
        raise service.InvalidPost("posts: list each post once")
    return list(shown)


def _left(post: Any, reason: str, message: str, **extra: Any) -> Dict[str, Any]:
    entry = {
        "post_id": str(post.id),
        "title": post.title,
        "status": post.status,
        "reason": reason,
        "message": message,
    }
    return {**entry, **extra}


def _left_unknown(post_id: UUID) -> Dict[str, Any]:
    return {
        "post_id": str(post_id),
        "title": None,
        "status": None,
        "reason": LEFT_NOT_IN_CAMPAIGN,
        "message": "the post is not in this campaign",
    }


def _locked_campaign(db: Any, workspace_id: UUID, campaign_id: UUID) -> Optional[SocialCampaign]:
    """The campaign's row as committed now, locked until this transaction ends."""
    return (
        db.query(SocialCampaign)
        .filter(SocialCampaign.workspace_id == workspace_id, SocialCampaign.id == campaign_id)
        .populate_existing()
        .with_for_update()
        .first()
    )


def _still_in_campaign(db: Any, post_id: UUID, campaign_id: UUID) -> bool:
    row = db.query(SocialPost.campaign_id).filter(SocialPost.id == post_id).first()
    return row is not None and row.campaign_id == campaign_id


def _stale(db: Any, workspace_id: UUID, post_id: UUID) -> service.SocialsError:
    current = service.get_post(db, workspace_id, post_id)
    if current is None:
        return service.PostNotFound()
    return service.StaleContent(service.compute_content_hash(current))


def _commit_approval(db: Any, post: SocialPost, campaign_id: UUID, content_hash: str) -> Dict[str, Any]:
    """Commit one post's approval, compare-and-set like a single approve: only if the
    row still has ``needs_approval`` and ``content_hash`` and is still in the
    campaign. Its hash joins the campaign's approved set in the same commit.
    Otherwise roll back, write nothing and raise (StaleContent with the current
    hash, PostNotFound when it left the campaign, CampaignNotFound)."""
    post_id, workspace_id = post.id, post.workspace_id
    if not service.claim_unchanged(db, post, status=service.NEEDS_APPROVAL, content_hash=content_hash):
        db.rollback()
        raise _stale(db, workspace_id, post_id)
    if not _still_in_campaign(db, post_id, campaign_id):
        db.rollback()
        raise service.PostNotFound()
    campaign = _locked_campaign(db, workspace_id, campaign_id)
    if campaign is None:
        db.rollback()
        raise CampaignNotFound()
    campaign.approved_hash_set = sorted({*(campaign.approved_hash_set or []), content_hash})
    campaign.approved_by = post.approved_by
    campaign.approved_at = post.approved_at
    db.commit()
    db.refresh(post)
    return post.to_dict()


def _precheck(
    post: Optional[SocialPost], campaign_id: UUID, item: ShownPost, batch_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Why a shown post cannot be approved in this series (or this batch), before anything is tried."""
    if post is None or post.campaign_id != campaign_id:
        return _left_unknown(item.post_id) if post is None else _left(
            post, LEFT_NOT_IN_CAMPAIGN, "the post is not in this campaign"
        )
    if batch_key is not None and post.batch_key != batch_key:
        return _left(post, LEFT_NOT_IN_BATCH, "the post is not in this batch")
    if post.status != service.NEEDS_APPROVAL:
        status = post.status.replace("_", " ")
        return _left(post, LEFT_NOT_WAITING, f"the post is {status}, not waiting for approval")
    return None


def _approve_one(
    db: Any, ids: Tuple[UUID, UUID], actor: str, item: ShownPost, comment: Optional[str],
    batch_key: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """One shown post of the campaign ``ids`` (workspace id, campaign id), approved
    exactly as a single approval: (the approved post, None) or (None, why it was left)."""
    workspace_id, campaign_id = ids
    post = service.get_post(db, workspace_id, item.post_id)
    refused = _precheck(post, campaign_id, item, batch_key)
    if refused is not None:
        return None, refused
    unresolved = post_sources.unresolved(db, workspace_id, post.sources)
    try:
        service.approve(
            post,
            actor,
            content_hash=item.content_hash,
            override_unsourced=item.override_unsourced,
            comment=comment,
            unresolved_sources=unresolved,
        )
        return _commit_approval(db, post, campaign_id, item.content_hash), None
    except service.StaleContent as exc:
        message = "the post changed after you were shown it: review the current version"
        return None, _left(post, LEFT_CHANGED, message, content_hash=exc.current_hash)
    except service.UnsourcedClaims as exc:
        return None, _left(post, LEFT_UNSOURCED, str(exc), claims=exc.names, unresolved=exc.unresolved)
    except service.IllegalTransition:
        # Its status moved after the check (another reviewer acted first): left, never an abort.
        return None, _left(post, LEFT_NOT_WAITING, "the post is no longer waiting for approval")
    except service.PostNotFound:
        return None, _left_unknown(item.post_id)


def _not_shown(
    db: Any, campaign: SocialCampaign, shown: Sequence[ShownPost], batch_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """The campaign's (or the batch's) posts waiting for approval that the approver was not shown."""
    shown_ids = {item.post_id for item in shown}
    return [
        _left(post, LEFT_NOT_SHOWN, "you were not shown this post, so it still needs its own approval")
        for post in campaign_posts(db, campaign.workspace_id, campaign.id)
        if post.status == service.NEEDS_APPROVAL and post.id not in shown_ids
        and (batch_key is None or post.batch_key == batch_key)
    ]


def approve_series(
    db: Any,
    campaign: SocialCampaign,
    actor: str,
    shown: Sequence[ShownPost],
    *,
    comment: Optional[str] = None,
    batch_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Approve the shown posts of ``campaign`` as a series (D6), one committed
    approval each. The caller checks ``assert_series_allowed`` first, except for a
    plan's batch (PRD-251C C2, O2): with ``batch_key``, only the batch's posts are
    approved, a post of another batch is left, and the posts not shown are the batch's.

    Returns ``{"campaign", "approved", "left"}``: the campaign as committed, the
    posts approved, and each post left unapproved with its ``reason`` and
    ``message`` (plus the current ``content_hash`` for a changed post, the
    ``claims`` for an unsourced one).
    """
    items = validate_shown(shown)
    campaign_id, workspace_id = campaign.id, campaign.workspace_id
    approved: List[Dict[str, Any]] = []
    left: List[Dict[str, Any]] = []
    for item in items:
        done, why = _approve_one(db, (workspace_id, campaign_id), actor, item, comment, batch_key)
        if done is not None:
            approved.append(done)
        else:
            left.append(why)
    current = get_campaign(db, workspace_id, campaign_id)
    if current is None:
        raise CampaignNotFound()
    left.extend(_not_shown(db, current, items, batch_key))
    logger.info(
        "[Socials] series approval of campaign %s by %s: %d approved, %d left",
        campaign_id, actor, len(approved), len(left),
    )
    return {"campaign": current.to_dict(), "approved": approved, "left": left}
