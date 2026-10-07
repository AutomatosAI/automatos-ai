"""F380 (night 11, 7 Oct): a social ticket is done only when its post exists and has rendered.

The owner: "'done' must mean the post exists and has rendered." Night 11's Social Media
Director tickets closed done with nothing in Socials: #2158's first run answered with a
one-line caption, #2159's with instructions for the owner, and #2160 made a
generate_document PNG and said the draft was "waiting for your approval in the Socials
tab". The posts it did save (2be9c935, 905208ce, 5b99dd4e) had no render: the owner
rendered each one by hand.

When a ticket's brief asks for a Socials post, or its answer says one was made
(``modules.tools.discovery.social_post_asks``), the run must have saved one: a post in
the workspace that the ticket's agent (``agent:<id>``, the actor its Socials tools
write) created, or changed, since the run started. Then:

* no such post: the ticket goes to review, saying no Socials post was saved, and that a
  generate_document image is a Deliverable, not a post, when the run made one;
* a post whose render failed: review, with the render's own reason and how to fix it;
* a post saved without a render (draft, changes requested, or submitted unrendered):
  the platform starts its render now, as ``platform_create_social_post`` does by
  default. It renders within the plan's render minutes and waits for approval when it
  finishes, and the ticket says so. A render that is refused (a field the template
  needs, no render minutes left) sends the ticket to review with the reason instead.
  Submitting the unrendered draft (the Director persona's other path) would hand the
  owner a post with no picture, which is what night 11 complained of;
* a rendered post, a render under way, a text post, or one a person already approved:
  nothing to add.

A result that only asks the owner is left to F183's question, and a ticket whose result
another loop takes up (a mission's, a Playbook's or a chat's) to that loop. With Socials
off for the workspace no post can be made, so nothing is checked. The check runs before
``finalize_board_task_run`` locks the ticket (F210: nothing awaits under that lock), and
its note joins the answer the way F014's named-file note does.
"""
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple
from uuid import UUID

from core.services.ticket_reasons import NOTHING_DONE_NOTE_PREFIX
from modules.tools.discovery.social_post_asks import (
    asks_for_a_social_post, claims_a_social_post, made_a_document_image,
)

logger = logging.getLogger(__name__)

Async = Callable[..., Awaitable[Any]]

AGENT_ACTOR = "agent:{agent_id}"  # handlers_socials._agent: the actor an agent's Socials calls write
# The app's clock (the run's start) and the database's (a post's created_at) may differ a little.
RUN_WINDOW_SLACK_SECONDS = 10
POSTS_SCANNED = 50   # the workspace's posts changed since the run started, newest first
POSTS_CHECKED = 5    # of those, the run's own that are checked
TITLE_CHARS = 120

NO_POST_NOTE = (NOTHING_DONE_NOTE_PREFIX + " no Socials post was saved in this run, so there is nothing to approve "
                "in the Socials tab.{image} Make the post with platform_create_social_post, which saves it and "
                "renders it. Sent to review instead of done.")
IMAGE_IS_NOT_A_POST = (" A generate_document image is a Deliverable, not a Socials post: it cannot be approved or "
                       "published from the Socials tab.")
NOT_RENDERED_NOTE = (NOTHING_DONE_NOTE_PREFIX + ' the post "{title}" was saved but did not render: {reason} Fix '
                     "its fields and render it (platform_update_social_post with render true). Sent to review "
                     "instead of done.")
RENDER_STARTED_NOTE = ('The post "{title}" was saved without its render, so the platform started the render: '
                       "the post waits for approval in the Socials tab when it finishes.")
RENDER_FAILED = "the render failed."
RENDER_COULD_NOT_START = "its render could not be started."

# Where a post stands on its render.
RENDERED, UNDER_WAY, NO_RENDER_NEEDED, FAILED_RENDER, UNRENDERED, LEFT_AS_IT_IS = (
    "rendered", "under_way", "no_render_needed", "failed", "unrendered", "left_as_it_is")
_TEXT_FORMAT = "text"


@dataclass(frozen=True)
class PostCheck:
    """What the ticket's answer gains, and whether it goes to review instead of done."""

    note: str
    review: bool


def answer_of(exec_result: Dict[str, Any]) -> str:
    """The run's answer, read as ``finalize_board_task_run`` reads it."""
    return str(exec_result.get("result") or exec_result.get("response") or exec_result.get("output")
               or exec_result.get("content") or "")


def render_state(post: Any) -> Tuple[str, str]:
    """(where ``post`` stands on its render, the failed render's reason)."""
    from modules.socials import service

    status = getattr(post, "status", None)
    if getattr(post, "media", None):
        return RENDERED, ""
    if status == service.RENDERING:
        return UNDER_WAY, ""
    if getattr(post, "template_id", None) is None and getattr(post, "format", None) == _TEXT_FORMAT:
        return NO_RENDER_NEEDED, ""
    if status == service.FAILED:
        return FAILED_RENDER, _failed_reason(getattr(post, "review_log", None) or [], service.ACTION_RENDER_FAILED)
    if status in (service.DRAFT, service.CHANGES_REQUESTED, service.NEEDS_APPROVAL):
        return UNRENDERED, ""
    return LEFT_AS_IT_IS, ""  # approved, scheduled or published: a person has it


def _failed_reason(review_log: Iterable[Any], action: str) -> str:
    """The comment of the post's last failed render, as the renderer gave it."""
    failed = [entry for entry in review_log if isinstance(entry, dict) and entry.get("action") == action]
    comment = str((failed[-1].get("comment") if failed else "") or "").strip()
    if not comment:
        return RENDER_FAILED
    return comment if comment.endswith((".", "!")) else f"{comment}."


def _as_utc(value: Any) -> Optional[datetime]:
    """A datetime, or an ISO string of one, in UTC (a naive one is read as UTC)."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def made_or_changed_by(post: Any, actor: str, since: datetime) -> bool:
    """``actor`` created ``post`` since ``since``, or wrote an entry of its history since then."""
    created = _as_utc(getattr(post, "created_at", None))
    if getattr(post, "created_by", None) == actor and created is not None and created >= since:
        return True
    for entry in getattr(post, "review_log", None) or []:
        at = _as_utc(entry.get("at")) if isinstance(entry, dict) else None
        if at is not None and at >= since and entry.get("by") == actor:
            return True
    return False


def run_start(task: Any) -> datetime:
    """When this run of the ticket started (a re-run resets it), less the clocks' slack."""
    started = _as_utc(getattr(task, "started_at", None)) or _as_utc(getattr(task, "created_at", None))
    return (started or datetime.now(timezone.utc)) - timedelta(seconds=RUN_WINDOW_SLACK_SECONDS)


def runs_posts(db: Any, workspace_id: UUID, actor: str, since: datetime) -> List[Any]:
    """The posts of the workspace this run's agent made or changed since ``since``."""
    from core.models.socials import SocialPost

    rows = (db.query(SocialPost)
            .filter(SocialPost.workspace_id == workspace_id, SocialPost.updated_at >= since)
            .order_by(SocialPost.updated_at.desc()).limit(POSTS_SCANNED).all())
    return [post for post in rows if made_or_changed_by(post, actor, since)][:POSTS_CHECKED]


def socials_on(db: Any, workspace_id: Any) -> Any:
    """The workspace when Socials is on for it (both switches, D1), else None."""
    from core.models.workspaces import Workspace
    from modules.socials.settings import socials_off_reason

    workspace = db.get(Workspace, UUID(str(workspace_id)))
    return workspace if socials_off_reason(workspace) is None else None


async def start_render(db: Any, workspace: Any, post: Any, actor: str) -> Tuple[bool, str]:
    """Start ``post``'s render as its agent's create would have: (started, why not)."""
    from api import socials as socials_api
    from core import media_render_quota as render_quota
    from modules.socials import service

    try:
        await socials_api.render_post(db, workspace, post, actor)
    except (service.SocialsError, render_quota.RenderQuotaExceeded) as exc:
        db.rollback()
        reason = str(exc).strip() or RENDER_COULD_NOT_START
        return False, reason if reason.endswith(".") else f"{reason}."
    except Exception:  # noqa: BLE001 — logged; the ticket goes to review saying the render didn't start
        logger.exception("[F380] the render of post %s could not be started", getattr(post, "id", None))
        db.rollback()
        return False, RENDER_COULD_NOT_START
    return True, ""


async def _post_note(db: Any, workspace: Any, post: Any, actor: str) -> Optional[PostCheck]:
    """The note one of the run's posts adds, or None when it has rendered (or needs no render)."""
    state, reason = render_state(post)
    title = str(getattr(post, "title", "") or "")[:TITLE_CHARS]
    if state == UNRENDERED:
        started, reason = await start_render(db, workspace, post, actor)
        if started:
            return PostCheck(RENDER_STARTED_NOTE.format(title=title), review=False)
    elif state != FAILED_RENDER:
        return None
    return PostCheck(NOT_RENDERED_NOTE.format(title=title, reason=reason), review=True)


async def _posts_check(db: Any, workspace: Any, posts: List[Any], actor: str) -> Optional[PostCheck]:
    """The notes of the run's posts, joined; review when any did not render."""
    checks = [check for check in [await _post_note(db, workspace, post, actor) for post in posts] if check]
    if not checks:
        return None
    return PostCheck("\n\n".join(check.note for check in checks), review=any(check.review for check in checks))


def _checked_ticket(task: Any, run_id: Optional[str]) -> bool:
    """This run still holds the ticket, and the ticket answers for itself (see the module note)."""
    from services.board_dispatcher import RUN_ID_KEY
    from services.cli_ticket_lane import is_lane_owned
    from services.ticket_owner_ask import OWNED_ELSEWHERE

    if task is None or getattr(task, "status", None) != "in_progress":
        return False
    if run_id is not None and (getattr(task, "runtime_ref", None) or {}).get(RUN_ID_KEY) != run_id:
        return False
    return getattr(task, "source_type", None) not in OWNED_ELSEWHERE and not is_lane_owned(task)


async def social_post_check(
    db: Any, *, task_id: Any, workspace_id: Any, agent_id: Optional[int], exec_result: Dict[str, Any],
    run_id: Optional[str] = None,
) -> Optional[PostCheck]:
    """What a social ticket's answer gains before it closes (see the module note); None for any other."""
    from core.models.core import BoardTask
    from services.ticket_owner_ask import result_only_asks

    task = db.get(BoardTask, task_id)
    if not _checked_ticket(task, run_id):
        return None
    answer = answer_of(exec_result)
    asked = asks_for_a_social_post(getattr(task, "title", None), getattr(task, "description", None))
    if not (asked or claims_a_social_post(answer)) or result_only_asks(task, answer, exec_result):
        return None
    agent = agent_id or getattr(task, "assigned_agent_id", None)
    workspace = socials_on(db, workspace_id) if agent else None
    if workspace is None:
        return None
    actor = AGENT_ACTOR.format(agent_id=agent)
    posts = runs_posts(db, workspace.id, actor, run_start(task))
    if posts:
        return await _posts_check(db, workspace, posts, actor)
    ran = (exec_result.get("execution") or {}).get("actions") or ()
    image = IMAGE_IS_NOT_A_POST if made_a_document_image(answer, ran) else ""
    return PostCheck(NO_POST_NOTE.format(image=image), review=True)


async def _checked_kwargs(db: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """``finalize_board_task_run``'s keywords with the check's note on the answer, and
    ``force_review`` when the post is missing or did not render."""
    exec_result = kwargs.get("exec_result") or {}
    if exec_result.get("status") in ("error", "cancelled"):
        return kwargs
    try:
        check = await social_post_check(
            db, task_id=kwargs.get("task_id"), workspace_id=kwargs.get("workspace_id"),
            agent_id=kwargs.get("agent_id"), exec_result=exec_result, run_id=kwargs.get("run_id"))
    except Exception:  # noqa: BLE001 — logged; the ticket closes on its other checks, as before F380
        logger.exception("[F380] the Socials post check of ticket %s failed", kwargs.get("task_id"))
        return kwargs
    if check is None:
        return kwargs
    answer = f"{answer_of(exec_result)}\n\n{check.note}".strip()
    return {**kwargs, "exec_result": {**exec_result, "result": answer},
            "force_review": bool(kwargs.get("force_review")) or check.review}


def a_social_card_has_its_post(finalize: Async) -> Async:
    """Wrap ``api.board_tasks.finalize_board_task_run`` (keywords after ``db``): a social
    ticket's answer carries what its post check found before any of the writer's own
    checks read it, and goes to review when the post is missing or did not render."""
    @functools.wraps(finalize)
    async def wrapped(db: Any, *args: Any, **kwargs: Any) -> Any:
        return await finalize(db, *args, **await _checked_kwargs(db, kwargs))
    return wrapped


__all__ = ["PostCheck", "a_social_card_has_its_post", "render_state", "social_post_check"]
