"""PRD-251C (US-C303): render a still of the person's own, cropped for each channel.

``render_post`` (``api/socials.py``) hands over a post whose visual is the person's own still
(``modules/socials/upload_crops.own_still``: an upload or a Library picture as the whole post).
The render starts as a template's does, the same refusals first (a post holding an approval,
the month's render minutes, no renderer or storage), and the post is ``rendering`` when this
returns. The crop composition renders at each size its channels need, the picture linked from
storage; the job ends the post in ``needs_approval`` with the crops by aspect, and the original
kept beside them for the next render. A still takes no render minutes.

Not a router: no route of its own. It reuses ``api/socials.py``'s post helpers, imported when
a request runs, because that module imports this one.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict

from sqlalchemy.orm import Session

from core import media_render_quota as render_quota
from core.models.socials import SocialPost
from core.models.workspaces import Workspace
from core.social_templates import SOCIAL_IMAGE
from modules.socials import media_urls, render, service, upload_crops

PICTURE_GONE = "this post's picture is no longer in the workspace's Deliverables: upload it again or pick another"


def _posts_api() -> Any:
    from api import socials

    return socials


def _original_key(db: Session, post: SocialPost) -> str:
    """The storage key of the post's own still; :class:`render.NotRenderable` when it is gone."""
    found = next((f for f in media_urls.resolve_post_media(db, post) if f.aspect == upload_crops.ORIGINAL), None)
    if found is None or not found.key:
        raise render.NotRenderable(PICTURE_GONE)
    return found.key


async def render_crops(db: Session, workspace: Workspace, post: SocialPost, actor: str) -> Dict[str, Any]:
    """Start the crops' render in the background (the module docstring)."""
    posts_api = _posts_api()
    status, content_hash = post.status, post.content_hash
    service.assert_can_render(post)
    key = await asyncio.to_thread(_original_key, db, post)
    bundle, *other_sizes = upload_crops.crop_bundles(post)
    template = SimpleNamespace(format=SOCIAL_IMAGE, blocks=upload_crops.CROP_BLOCKS)
    reservation = await render.reserve_seconds(db, workspace, post, template)
    try:
        await render.ensure_renderer()
        service.start_render(post, actor)
        saved = posts_api._commit_unchanged(db, post, status=status, content_hash=content_hash)
    except BaseException:
        await render_quota.release_render(render_quota.sessions_for(db), reservation)
        raise
    posts_api._launch_render(
        render.RenderJob(
            post_id=post.id, workspace_id=post.workspace_id, actor=actor, content_hash=content_hash, title=post.title,
            format=post.format, bundle=bundle, extra_bundles=tuple(other_sizes), reservation=reservation,
            slot_keys={upload_crops.CROP_PATH: key}, keep_media=(upload_crops.ORIGINAL,),
        )
    )
    return saved
