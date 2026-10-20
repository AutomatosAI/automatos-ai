"""P256-FIX-SOC-4 (F431-c): a submit card is raised only when its click can send the post.

14 Oct, card #2648: Auto called ``platform_submit_social_post`` in the owner's chat for post
7e7c402c, which was already waiting for approval (a render ends there, and an edit keeps it
there). Submit is owner-only (Decision D1), so the card came before anything read the post; the
click re-ran submit, which refused ("cannot submit a post that is needs approval"), and the
click was given back. The click did nothing.

Now ``owner_only._bound_and_checked`` asks this module before the card:

* :func:`bound_to_the_post`: the call names the post by its whole id, so the click runs on the
  post the card showed (``social_post_ref``: a unique prefix of the id is taken too);
* :func:`submit_refused_before_the_card`: a post that cannot be sent for approval is answered in
  plain words (already waiting, rendering, approved, ...), with no card and no grant; an id that
  names no post gets the tool's own words (F091). Every answer is ``success: false``, so nothing
  backs a "sent for approval" claim. The read fails closed: when it raises, the card stands.

``is_owner_only`` stays static: the confirmation gate and ``follows_the_owner`` read it for every
caller. The tool's own refusal for an agent's call is unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from modules.tools.discovery.social_post_ref import post_named

logger = logging.getLogger(__name__)

SUBMIT = "platform_submit_social_post"
POST_ID = "post_id"
NOTHING_DONE = " Nothing was sent and no card was raised."
ALREADY_WAITING = '"{title}" is already waiting for approval in the Socials tab: nothing was sent and no card was raised.'
NO_CHANNELS = (" It has no channels, so an approval would publish it nowhere: set them with "
               "platform_update_social_post channels.")
# Where a post that cannot be submitted stands, in one line each (modules/socials/service.py statuses).
STANDS = {
    "rendering": '"{title}" is rendering: it waits for approval in the Socials tab when the render finishes.',
    "failed": '"{title}" has failed: platform_get_social_post says why, and an edit with render true fixes a render.',
    "approved": '"{title}" is already approved: it publishes from the Socials tab, or at its scheduled time.',
    "scheduled": '"{title}" is already approved and scheduled: it publishes at its time.',
    "published": '"{title}" is already published.',
}
OTHER = '"{title}" is {status}: only a draft, or a post a reviewer asked to change, is sent for approval.'
WAITING = "needs_approval"


def bound_to_the_post(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """The submit call with ``post_id`` as the post's whole id; the call as it is when it names none."""
    if action != SUBMIT or db is None:
        return params
    post = _read(db, workspace_id, params)[0]
    return {**params, POST_ID: str(post.id)} if post is not None else params


def submit_refused_before_the_card(db: Any, workspace_id: Any, action: str,
                                   params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Why the owner's click could not send the post for approval, as the call's answer; None
    when it can (or the post could not be read: the card stands)."""
    from modules.socials import service

    if action != SUBMIT or db is None:
        return None
    post, problem = _read(db, workspace_id, params)
    if problem:
        return {"success": False, "error": problem}
    if post is None or post.status in service.TRANSITIONS[service.ACTION_SUBMIT]:
        return None
    return {"success": False, "error": _where_it_stands(post)}


def _read(db: Any, workspace_id: Any, params: Dict[str, Any]) -> tuple:
    """(the post, None); (None, the tool's own refusal) for an id that names no post, or Socials off;
    (None, None) when the read raised: the card stands, and the handler answers on the click."""
    from modules.socials.settings import socials_off_reason
    from core.models.workspaces import Workspace

    try:
        workspace = db.get(Workspace, workspace_id)
        off = socials_off_reason(workspace)
        if off is not None:
            return None, off
        return post_named(db, workspace_id, params.get(POST_ID))
    except Exception:  # noqa: BLE001 — logged; the card stands, the handler answers on the click
        logger.exception("[card_question_socials] could not read the post a submit names")
        return None, None


def _where_it_stands(post: Any) -> str:
    """A post that cannot be submitted, in plain words: where it is, and that nothing was done."""
    title = str(post.title or "")
    if post.status == WAITING:
        return ALREADY_WAITING.format(title=title) + ("" if post.targets else NO_CHANNELS)
    stands = STANDS.get(post.status)
    line = stands.format(title=title) if stands else OTHER.format(title=title, status=str(post.status).replace("_", " "))
    return line + NOTHING_DONE


__all__ = ["bound_to_the_post", "submit_refused_before_the_card"]
