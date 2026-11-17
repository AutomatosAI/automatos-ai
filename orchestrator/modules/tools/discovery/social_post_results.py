"""F379 (night 11, 7 Oct): a post tool's answer says first whether the post rendered, in a few lines.

Night 11: "I've drafted the three social media posts" (chat 508b4e05), and none rendered or
waits for approval; Auto told the owner rendering "happens later, usually after approval" (B4,
B10). A render the save started and that was refused came back as ``success: true``, with only
its ``message`` changed, behind the whole post (``post.to_dict()``: its copy, every variable, its
history). The chat keeps about 2,000 tokens of a tool's answer, so the refusal was cut off, and
the reply-claim check (``action_claims``) counted the call as done.

So the answer is compact: the post's id, title, status, format, template and its rendered files'
links. A post whose render was refused is a failed call (``success: false``) whose ``error`` says
first that the post was saved as a draft and did not render, why in plain words (the labels of
the fields left empty), and the call that fixes it, which never makes the post again.

P256-FIX-SOC-4 (F432-b, 14 Oct): ``platform_get_social_post`` answered the whole post, and a
carousel's channels came after its copy, variables, media and a history whose every agent edit
carried a copy of up to 4,000 characters: the chat kept the head, and Auto never saw where the post
went out or when. ``read_view`` puts who it is, its channels with their receipts, its times, its
approval and what it still needs first, then the rest, with the last ``REVIEW_LOG_KEPT`` history
entries and none of their copies (the count of all of them beside). Only the composer's preview and
the hashes are left out. ``SocialPost.to_dict`` itself is unchanged: REST and the app read it.

PRD-251D (E6, US-D106): every answer says the slot the post is planned for in its own timezone, as
``planned`` ("Friday 20 November 2026, 09:00 in Europe/Dublin"; None without one). A call that set
the slot says what it means beside it (``planned_note``): the owner still approves the post, and
approving it schedules it into that slot, so it is not scheduled until then. A slot whose commit was
refused after the post's own save is a failed call that says the rest was saved (``slot_not_set``).

PRD-251D (US-D108): a post records what its file was made from, and says ``file_current`` false once
an edit changed what a render reads (``core/social_file_current.py``); the owner cannot approve it
until it renders again. A save that left the file older than the post's words says so in its answer
(``file_current`` false, and a message with the call that renders it), and the post an agent reads or
lists says the new render among what it still needs (``social_post_list.what_it_needs``), in place of the
owner's approval, both only while a render can start (``renders_again``: an approved post keeps its
approved file). The compact post itself is unchanged.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

RENDER_STARTED = (
    "Rendering now: when the render finishes the post waits for approval in the Socials tab, or is "
    "failed with the reason in its history (platform_get_social_post). A post renders when it is saved, "
    "never after approval."
)
DRAFT_SAVED = "Saved as a draft: send it for approval with platform_submit_social_post when it is ready."
CHANGED = "Saved the change. With render true the post renders again."
NOT_RENDERED = (
    "Saved post {id} ({title}) as a draft, but it did NOT render: {why}. It is not waiting for approval, and the "
    "owner can't see it in the Socials tab yet. {fix} Never create this post again: change it."
)
FIX_FIELDS = 'Fill the empty fields with platform_update_social_post {{"post_id": "{id}", "variables": {{{fields}}}, "render": true}}.'
FIX_TEMPLATE = ('Give it a template with platform_update_social_post {{"post_id": "{id}", "template": one of {names}, '
                '"variables": {{its fields}}, "render": true}}.')
FIX_OTHER = 'When that is sorted, render it with platform_update_social_post {{"post_id": "{id}", "render": true}}.'
NO_TEMPLATE_WORDS = ("no template", "pick a template")
PLANNED = "planned"  # the slot in words, in every answer's post (PRD-251D)
PLANNED_NOTE = ("Planned for {when}: that is the post's slot in the Socials calendar. The owner still approves it, "
                "and approving it schedules it into that slot; it is not scheduled until then.")
SLOT_NOT_SET = ('Saved post {id} ({title}), but its planned slot was NOT set: {why}. Nothing after the save ran. '
                'Send planned_for and timezone again with platform_update_social_post {{"post_id": "{id}"}}.')
# PRD-251D (US-D108): the post's flag (SocialPost.to_dict), on a save's answer only when it is false.
FILE_CURRENT = "file_current"
FILE_OLDER = ("Saved the change, but post {id}'s file was made before it, so the file no longer shows the post as it "
              "is now and the owner cannot approve it. Render it again with platform_update_social_post "
              '{{"post_id": "{id}", "render": true}}.')
NEEDS_NEW_RENDER = "a new render: its file was made before its last change, so the owner cannot approve it yet"
# read_view (P256-FIX-SOC-4): what comes first, in order, and what an agent's read leaves out.
READ_FIRST = ("id", "status", "render", "title", "format", "template_id")  # P256-FIX-SOC-5: the render's state too
TARGET_KEYS = ("toolkit", "post_kind", "options", "status", "published_at", "remote_id", "permalink", "error", "notes")
WHEN_KEYS = ("scheduled_for", "planned_for", "timezone")
APPROVAL_KEYS = ("approved_by", "approved_at", "override_unsourced")
READ_DROPPED = ("preview", "content_hash", "approved_hash", "workspace_id")
REVIEW_LOG = "review_log"
REVIEW_LOG_KEPT = 5


def media_links(post: Mapping[str, Any]) -> List[str]:
    """The app links of the post's rendered files (``media_store.media_route``), in aspect order."""
    from modules.socials.media_store import media_route

    media = post.get("media") if isinstance(post.get("media"), dict) else {}
    return [media_route(post.get("id"), item["name"])
            for aspect in sorted(media) for item in (media[aspect] or [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)]


def planned_words(post: Mapping[str, Any]) -> Optional[str]:
    """The slot the post is planned for, in words in its own timezone (PRD-251D); None without one."""
    from modules.tools.discovery.social_post_list import slot_words

    return slot_words(post.get("planned_for"), post.get("timezone"))


def renders_again(post: Mapping[str, Any]) -> bool:
    """PRD-251D (US-D108): the post's file is older than its words, and a render can make it again now. An
    approved or scheduled post keeps its approved file, and a render is refused while it holds the approval."""
    from modules.socials.service import ACTION_RENDER, TRANSITIONS

    return post.get(FILE_CURRENT) is False and post.get("status") in TRANSITIONS[ACTION_RENDER]


def post_view(post: Mapping[str, Any], template_name: Optional[str]) -> Dict[str, Any]:
    """The post as a tool answers it: who it is, what it shows and when it is planned for, never its whole
    content."""
    return {
        "id": post.get("id"),
        "title": post.get("title"),
        "status": post.get("status"),
        "format": post.get("format"),
        "template": template_name,
        "template_id": post.get("template_id"),
        "media": media_links(post),
        PLANNED: planned_words(post),
    }


def read_view(post: Mapping[str, Any]) -> Dict[str, Any]:
    """The post as an agent reads it (``SocialPost.to_dict``, reordered): who it is, its channels and
    their receipts, when, its approval and what it still needs, then the rest, and its last
    ``REVIEW_LOG_KEPT`` history entries without the copies they keep (see the module note)."""
    from modules.socials.voice_examples import AUTO_COPY_KEY
    from modules.tools.discovery.social_post_list import what_it_needs

    targets = [{key: target.get(key) for key in TARGET_KEYS} for target in post.get("targets") or []
               if isinstance(target, Mapping)]
    log = list(post.get(REVIEW_LOG) or [])
    kept = [{key: value for key, value in entry.items() if key != AUTO_COPY_KEY} if isinstance(entry, Mapping)
            else entry for entry in log[-REVIEW_LOG_KEPT:]]
    placed = {*READ_FIRST, "targets", *WHEN_KEYS, *APPROVAL_KEYS, REVIEW_LOG, *READ_DROPPED}
    return {
        **{key: post.get(key) for key in READ_FIRST}, "targets": targets, PLANNED: planned_words(post),
        **{key: post.get(key) for key in (*WHEN_KEYS, *APPROVAL_KEYS)}, "needs": what_it_needs(post),
        **{key: value for key, value in post.items() if key not in placed},
        REVIEW_LOG: kept, "review_log_total": len(log),
    }


def saved(post: Mapping[str, Any], template_name: Optional[str], message: str) -> Dict[str, Any]:
    """A post saved and not rendered by this call. PRD-251D (US-D108): when the save left the post's file
    older than what it holds, the answer says so (``file_current`` false) and how to render it again."""
    answer = {"success": True, "post": post_view(post, template_name), "message": message}
    if not renders_again(post):
        return answer
    return {**answer, FILE_CURRENT: False, "message": FILE_OLDER.format(id=post.get("id"))}


def rendering(post: Mapping[str, Any], template_name: Optional[str]) -> Dict[str, Any]:
    """A post whose render started."""
    return {"success": True, "post": post_view(post, template_name), "render": {"started": True},
            "message": RENDER_STARTED}


def _fix(post: Mapping[str, Any], error: str, template_names: str) -> str:
    from modules.tools.discovery.social_post_checks import missing_names

    post_id = post.get("id")
    names = missing_names(error)
    if names:
        fields = ", ".join(f"{json.dumps(name)}: …" for name in names)
        return FIX_FIELDS.format(id=post_id, fields=fields)
    if any(words in error.lower() for words in NO_TEMPLATE_WORDS):
        return FIX_TEMPLATE.format(id=post_id, names=template_names)
    return FIX_OTHER.format(id=post_id)


def not_rendered(post: Mapping[str, Any], template: Any, error: str, template_names: str) -> Dict[str, Any]:
    """A post saved as a draft whose render was refused: a failed call, saying why and how to fix it."""
    from modules.tools.discovery.social_post_checks import missing_in_words

    name = getattr(template, "name", None)
    title = json.dumps(str(post.get("title") or ""), ensure_ascii=False)
    message = NOT_RENDERED.format(id=post.get("id"), title=title, why=missing_in_words(error, template),
                                  fix=_fix(post, error, template_names))
    return {"success": False, "error": message, "saved_as_draft": True, "post": post_view(post, name),
            "render": {"started": False, "error": error}}


def planned_note(answer: Dict[str, Any], slot: Any) -> Dict[str, Any]:
    """``answer``, with what the slot this call set means beside it (PRD-251D E6); as it is when the
    call set none."""
    post = answer.get("post") if isinstance(answer.get("post"), Mapping) else {}
    when = post.get(PLANNED)
    return answer if slot is None or not when else {**answer, "slot": PLANNED_NOTE.format(when=when)}


def slot_not_set(post: Mapping[str, Any], template_name: Optional[str], why: str) -> Dict[str, Any]:
    """A post saved by this call whose planned slot was then refused: a failed call, saying the rest was saved."""
    title = json.dumps(str(post.get("title") or ""), ensure_ascii=False)
    return {"success": False, "error": SLOT_NOT_SET.format(id=post.get("id"), title=title, why=why),
            "post": post_view(post, template_name)}


__all__ = ["CHANGED", "DRAFT_SAVED", "FILE_CURRENT", "FILE_OLDER", "NEEDS_NEW_RENDER", "PLANNED", "PLANNED_NOTE",
           "RENDER_STARTED", "SLOT_NOT_SET", "media_links",
           "not_rendered", "planned_note", "planned_words", "post_view", "read_view", "renders_again", "rendering",
           "saved", "slot_not_set"]
