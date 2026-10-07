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


def media_links(post: Mapping[str, Any]) -> List[str]:
    """The app links of the post's rendered files (``media_store.media_route``), in aspect order."""
    from modules.socials.media_store import media_route

    media = post.get("media") if isinstance(post.get("media"), dict) else {}
    return [media_route(post.get("id"), item["name"])
            for aspect in sorted(media) for item in (media[aspect] or [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)]


def post_view(post: Mapping[str, Any], template_name: Optional[str]) -> Dict[str, Any]:
    """The post as a tool answers it: who it is and what it shows, never its whole content."""
    return {
        "id": post.get("id"),
        "title": post.get("title"),
        "status": post.get("status"),
        "format": post.get("format"),
        "template": template_name,
        "template_id": post.get("template_id"),
        "media": media_links(post),
    }


def saved(post: Mapping[str, Any], template_name: Optional[str], message: str) -> Dict[str, Any]:
    """A post saved and not rendered by this call."""
    return {"success": True, "post": post_view(post, template_name), "message": message}


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


__all__ = ["CHANGED", "DRAFT_SAVED", "RENDER_STARTED", "media_links", "not_rendered", "post_view", "rendering",
           "saved"]
