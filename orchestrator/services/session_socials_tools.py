"""PRD-256 F430: the Socials post tools, for a ticket session.

Socials day run (14 Oct): the Social Media Director (#348, runtime cli) refused three
times, "platform_create_social_post isn't available in this session". A session's tool
list had no Socials tool at all, so a CLI agent could never make the post its persona,
its skills and Auto's ticket all asked for; 53 of c1's 64 agents are CLI agents.

Each tool here runs the API agents' own action (``actions_socials``) through
``UnifiedToolExecutor.execute_tool`` via ``platform_execute``, as the ticket's agent in
the ticket's workspace (``session_tools.call_tool``); the action's handler names the
agent on every review-log entry the post gets. The input schemas are the actions' own,
built from ``core.social_post_schema``, and only the fields they declare are forwarded.
Like an API agent, a session drafts, changes, re-renders, reads and submits posts; none
of these approves, schedules or publishes (D6, D14).

Plain specs, like ``session_work_tools``, which lists them last (the Socials group's
place in the catalogue). This module imports nothing under ``modules`` at load time:
``modules.tools`` imports the registry, the executor and the router.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from core.social_post_schema import (
    create_properties,
    get_properties,
    list_properties,
    submit_properties,
    update_properties,
)
from services.session_data_tools import project_answer
from services.session_tool_scope import forward, object_schema

TAGS: Tuple[str, ...] = ("socials",)
NEEDS_A_TITLE = "create_social_post needs a title: what the post is about, as the Socials tab lists it."
NEEDS_THE_POST = "{tool} needs the post_id of the post (list_social_posts lists them)."


def _same(properties: Dict[str, Any]) -> Dict[str, str]:
    """Every field the action declares, under its own name."""
    return {key: key for key in properties}


def _spec(name: str, action: str, description: str, properties: Dict[str, Any],
          required: Tuple[str, ...] = (), needs: str = "", reads_only: bool = True) -> Dict[str, Any]:
    """One Socials session tool: the action's own schema, forwarding only what it declares."""
    return {
        "name": name,
        "action": action,
        "reads_only": reads_only,
        "description": description,
        "input_schema": object_schema(properties, required),
        "scope": forward(_same(properties), required, needs),
        "project": project_answer,
        "tags": TAGS,
    }


CREATE_SOCIAL_POST_SPEC = _spec(
    "create_social_post", "platform_create_social_post",
    "Draft a Socials post for the workspace's channels: a title, its copy (the text for every channel and any "
    "channel's own text), a social template (a video or an image: list_templates with format social_image or "
    "social_video, then get_template_schema for its variables) with its variables, the sources of its facts "
    "and figures, and files already made. By default the template is rendered at once, within the plan's "
    "render minutes: when the render finishes the post waits for approval, or is failed with the reason in "
    "its history; a render refused at once fails the call, which says why and how to fix the saved draft "
    "with update_social_post (never make the post again). With render false it stays a draft until "
    "submit_social_post. A person approves every post in the Socials tab and the platform publishes it; "
    "this tool never approves, schedules or publishes. Put the post's id in your report.",
    create_properties(), ("title",), NEEDS_A_TITLE, reads_only=False,
)

UPDATE_SOCIAL_POST_SPEC = _spec(
    "update_social_post", "platform_update_social_post",
    "Change a Socials post: its post_id and only what changes; copy, variables and sources merge into the "
    "post's own. Changing an approved or scheduled post voids its approval, so a person approves it again "
    "in the Socials tab. With render true the post is rendered again after the change, which is how a "
    "failed render is fixed. Never approves, schedules or publishes.",
    update_properties(), ("post_id",), NEEDS_THE_POST.format(tool="update_social_post"), reads_only=False,
)

SUBMIT_SOCIAL_POST_SPEC = _spec(
    "submit_social_post", "platform_submit_social_post",
    "Send a draft Socials post, or one a reviewer asked to change, for approval, with a note for the "
    "reviewer: it moves to needs_approval, and a person approves it, asks for changes or rejects it in the "
    "Socials tab. A rendered post is already waiting for approval. Never approves, schedules or publishes.",
    submit_properties(), ("post_id",), NEEDS_THE_POST.format(tool="submit_social_post"), reads_only=False,
)

GET_SOCIAL_POST_SPEC = _spec(
    "get_social_post", "platform_get_social_post",
    "Read one Socials post: its status, copy, template and variables, sources, rendered media, voice and "
    "footage, and its history (who drafted or changed it, a reviewer's comments, and why a render failed).",
    get_properties(), ("post_id",), NEEDS_THE_POST.format(tool="get_social_post"),
)

LIST_SOCIAL_POSTS_SPEC = _spec(
    "list_social_posts", "platform_list_social_posts",
    "List the workspace's Socials posts, newest first: how many match, then one row each with its status "
    "and what it still needs (the owner's approval, a render that failed and why, a render not made yet, a "
    "caption). Pick them by status, or by queue: what waits for the owner's approval, what has not "
    "rendered, or every post not yet approved or out.",
    list_properties(),
)

# The Socials group's tools, in its order (``session_tool_groups``).
SOCIAL_TOOL_SPECS: Tuple[Dict[str, Any], ...] = (
    CREATE_SOCIAL_POST_SPEC, UPDATE_SOCIAL_POST_SPEC, SUBMIT_SOCIAL_POST_SPEC,
    GET_SOCIAL_POST_SPEC, LIST_SOCIAL_POSTS_SPEC,
)
