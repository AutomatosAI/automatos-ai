"""
Socials preview render (PRD-251 S2.2b, US-208)
==============================================

``POST /api/socials/posts/{id}/render`` with ``{"preview": true}``: the composer's
preview of a video, at half resolution, stored as the post's ``preview``
(``modules/socials/preview.py``). The post keeps its status, ``media`` and
content hash; a post left to "Let Auto pick" first gets Auto's template, an edit
like any other (F253). Its seconds are held against the month's render quota before
anything reaches media-render (429 when none are left), and the render runs in
the background. A post whose content can no longer be edited is 409, and so is a
second preview while one renders.

Called by the render route in ``api/socials.py``; its helpers come from there.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from sqlalchemy.orm import Session

from core import media_render_quota as render_quota
from core.models.socials import SocialPost
from core.models.workspaces import Workspace
from core.utils.background_tasks import launch_guarded
from api import socials_compose
from modules.socials import preview, render


def _posts_api() -> Any:
    """``api/socials.py``: it calls this module, so it is imported when a request runs."""
    from api import socials

    return socials


def _launch_preview(job: render.RenderJob) -> None:
    """The preview runs in the background; its end is written by the task itself."""
    launch_guarded(preview.run_preview(job), subsystem="socials", operation="preview", workspace_id=job.workspace_id)


async def preview_post(db: Session, workspace: Workspace, post: SocialPost, actor: str) -> Dict[str, Any]:
    """Start the post's preview render; the post answers with ``preview.status`` rendering.
    A post left to "Let Auto pick" gets Auto's template first, saved on the post (F253)."""
    posts_api = _posts_api()
    await socials_compose.let_auto_pick(db, workspace, post, actor, preview.assert_can_preview)
    preview.assert_can_preview(post)
    template = posts_api._render_template(db, workspace.id, post)
    brand_kit = await asyncio.to_thread(posts_api._render_brand_kit, workspace.settings)
    bundle = render.bundle_for(post, preview.preview_template(template), brand_kit, fallback_name=workspace.name or "")
    status, content_hash = post.status, post.content_hash
    reservation = await render.reserve_seconds(db, workspace, post, template)
    try:
        await render.ensure_renderer()
        preview.start(post)
        saved = posts_api._commit_unchanged(db, post, status=status, content_hash=content_hash)
    except BaseException:
        await render_quota.release_render(render_quota.sessions_for(db), reservation)
        raise
    _launch_preview(
        render.RenderJob(
            post_id=post.id, workspace_id=post.workspace_id, actor=actor, content_hash=content_hash,
            title=post.title, format=post.format, bundle=bundle, reservation=reservation, preview=True,
        )
    )
    return saved
