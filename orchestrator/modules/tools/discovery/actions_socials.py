"""Socials ActionDefinitions (PRD-251 US-116, S4.1): agents draft posts; people approve them.

Five tools over the Socials post lifecycle: create, update, submit, get and
list. Their handlers (``handlers_socials.py``) run the flows the /api/socials
routes run. None of them approves, schedules or publishes, and no parameter
does (D6, D14): a person approves in the Socials tab, and the platform
publishes (Wave 3).

The enums are the lifecycle's own vocabulary, spelled out here so the registry
build stays stdlib-light (the utterance-corpus linter leaf-loads this module);
``tests/test_prd251w1_social_post_tools.py`` pins them to ``SOCIAL_POST_FORMATS``,
``SOCIAL_POST_STATUSES``, ``CHART_KINDS`` and ``PARTS``.
"""

from .action_registry import ActionDefinition, ActionRegistry

# core/models/socials.py SOCIAL_POST_FORMATS
POST_FORMATS = ["video", "image", "carousel", "fact_card", "infographic"]
# core/models/socials.py SOCIAL_POST_STATUSES
POST_STATUSES = [
    "draft",
    "rendering",
    "needs_approval",
    "changes_requested",
    "approved",
    "scheduled",
    "publishing",
    "published",
    "partially_published",
    "failed",
    "missed",
    "archived",
]
# core/chart_binding.py CHART_KINDS and core/report_tables.py PARTS
CHART_KINDS = ["bar", "line", "grid"]
CHART_PARTS = ["table", "metrics"]

# platform_list_social_posts: how many posts one call returns.
LIST_DEFAULT_LIMIT = 25
LIST_MAX_LIMIT = 100


def _post_id() -> dict:
    return {"type": "string", "description": "The post's id (platform_list_social_posts lists them)."}


def _post_fields() -> dict:
    """What a post carries: the REST body's fields, plus template (id or name) and chart_report."""
    return {
        "title": {
            "type": "string",
            "description": "What the post is about, as the Socials tab lists it (at most 500 characters).",
        },
        "brief": {"type": "string", "description": "The ask the post answers: who it is for and what it says."},
        "copy": {
            "type": "object",
            "description": (
                "The post's text: {\"base\": the text every channel gets, \"channels\": {channel: "
                "its own text}}, or simply {channel: text}, keyed by toolkit (linkedin, twitter, "
                "instagram, tiktok, youtube). An update keeps what it does not send; a channel "
                "sent as \"\" drops its own text."
            ),
            "properties": {
                "base": {"type": "string"},
                "channels": {"type": "object", "additionalProperties": {"type": "string"}},
            },
        },
        "format": {"type": "string", "enum": POST_FORMATS, "description": "What the post is."},
        "template": {
            "type": "string",
            "description": (
                "The social template: its id or its name (platform_list_templates with format "
                "social_video or social_image; its variables: platform_get_template_schema)."
            ),
        },
        "variables": {
            "type": "object",
            "description": (
                "The template's variables: {name: value}, or {name: {\"value\": ..., \"claim\": true "
                "or false}}. A claim is a fact or a figure and needs a source in sources; one the "
                "template marks as a claim always is. An update keeps what it does not send; null "
                "clears one. A chart template's rows and source chip come from chart_report, never "
                "typed here."
            ),
        },
        "sources": {
            "type": "object",
            "description": (
                "Where each claim comes from, keyed by the variable it backs: {variable: {\"kind\": "
                "deliverable, report, document, url or metric, \"ref\": its id (a url's address, a "
                "metric's name), \"as_of\": ISO date-time}}; a list of {\"claim\": variable, "
                "\"kind\", \"ref\", \"as_of\"} is taken too. Every source must exist in this "
                "workspace; a url is checked by shape only. An update keeps what it does not send; "
                "null removes one."
            ),
        },
        "media": {
            "type": "object",
            "description": (
                "Files already made for the post, as Deliverable ids per aspect ratio, such as "
                "{\"9:16\": [\"<deliverable id>\"]} for a video generate_document rendered. A render "
                "replaces them, so send render false with them."
            ),
        },
        "voice": {
            "type": "object",
            "description": (
                "Who speaks the script: leave it out for Kokoro, the template's own voice, or name a "
                "voice toolkit the workspace has connected: {\"toolkit\": \"fish_audio\" or "
                "\"elevenlabs\", \"voice_id\", \"name\"}."
            ),
            "properties": {
                "toolkit": {"type": "string"},
                "voice_id": {"type": "string"},
                "name": {"type": "string"},
            },
        },
        "footage": {
            "type": "object",
            "description": (
                "AI footage or stills for the template's slots, from the workspace's connected "
                "generation toolkit: {slot: {\"prompt\": \"...\"}}. Priced and held to the post's and "
                "the month's media caps before anything is spent; with no generation toolkit "
                "connected, the slot plays the template's own motion graphics."
            ),
        },
        "chart_report": {
            "type": "object",
            "description": (
                "Fill a chart template (the Infographic) from a report of this workspace: its top "
                "rows, each figure a claim bound to the report, and the chip naming it."
            ),
            "properties": {
                "report_id": {"type": "string", "description": "The report's id (platform_browse_reports)."},
                "chart": {
                    "type": "string",
                    "enum": CHART_KINDS,
                    "description": "The chart kind; the template's own when the figures fit it.",
                },
                "column": {
                    "type": "string",
                    "description": "The figures' column in the report's table, by its header; its first numeric column by default.",
                },
                "part": {
                    "type": "string",
                    "enum": CHART_PARTS,
                    "description": "The report's table (the default when its file has one) or its metrics.",
                },
            },
            "required": ["report_id"],
        },
    }


def register_socials_actions(registry: ActionRegistry) -> None:
    """Register the Socials draft tools (PRD-251 US-116)."""

    registry.register(ActionDefinition(
        name="platform_create_social_post",
        description=(
            "Draft a Socials post for the workspace's channels: a title, its copy (the text for "
            "every channel and any channel's own text), a social template (a video or an image) "
            "with its variables, the sources of its facts and figures, and files already made. "
            "By default the template is rendered at once, within the plan's render minutes: when "
            "the render finishes the post waits for approval, or is failed with the reason in its "
            "history. With render false it stays a draft until platform_submit_social_post. A "
            "person approves every post in the Socials tab and the platform publishes it; this "
            "tool never approves, schedules or publishes."
        ),
        category="socials",
        parameters={
            "type": "object",
            "properties": {
                **_post_fields(),
                "render": {
                    "type": "boolean",
                    "description": (
                        "Render the post's template now (default true). Send false for a post "
                        "without a template, or when media already holds its rendered files."
                    ),
                },
            },
            "required": ["title"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["socials", "social media", "post", "draft", "video", "image", "linkedin", "instagram", "twitter"],
        examples=[
            "draft a LinkedIn post about our launch",
            "make a social video announcing the new feature",
            "create a social post from this week's report",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_update_social_post",
        description=(
            "Change a Socials post: send its post_id and only what changes; copy, variables and "
            "sources merge into the post's own. Changing an approved or scheduled post voids its "
            "approval, so a person approves it again in the Socials tab. With render true the post "
            "is rendered again after the change, which is how a failed render is fixed. Never "
            "approves, schedules or publishes."
        ),
        category="socials",
        parameters={
            "type": "object",
            "properties": {
                "post_id": _post_id(),
                **_post_fields(),
                "render": {
                    "type": "boolean",
                    "description": "Render the post again after the change (default false).",
                },
            },
            "required": ["post_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["socials", "social media", "post", "edit", "copy", "render"],
        examples=[
            "change the copy of that social post",
            "fix the headline on the draft social post and render it again",
            "add a source for the figure in the social post",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_submit_social_post",
        description=(
            "Send a draft Socials post, or one a reviewer asked to change, for approval: it moves to "
            "needs_approval, and a person approves it, asks for changes or rejects it in the Socials "
            "tab. A rendered post is already waiting for approval."
        ),
        category="socials",
        parameters={
            "type": "object",
            "properties": {
                "post_id": _post_id(),
                "note": {
                    "type": "string",
                    "description": "What the reviewer should look at; shown in the post's history.",
                },
            },
            "required": ["post_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["socials", "social media", "post", "review", "submit"],
        examples=[
            "send the social post for approval",
            "submit that draft social post for review",
            "the social post is ready, put it up for approval",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_social_post",
        description=(
            "Read one Socials post: its status, copy, template and variables, sources, rendered "
            "media, voice and footage, and its history (who drafted or changed it, a reviewer's "
            "comments, and why a render failed)."
        ),
        category="socials",
        parameters={
            "type": "object",
            "properties": {"post_id": _post_id()},
            "required": ["post_id"],
        },
        permission_level="read",
        tags=["socials", "social media", "post", "status", "review"],
        examples=[
            "show me that social post",
            "what's the status of the launch video post?",
            "why did the social post fail to render?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_list_social_posts",
        description=(
            "List the workspace's Socials posts, newest first, optionally only those in some "
            "statuses: what waits for approval, what a reviewer sent back, what failed to render, "
            "what is scheduled or published."
        ),
        category="socials",
        parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "array",
                    "items": {"type": "string", "enum": POST_STATUSES},
                    "description": "Only posts in these statuses; every status when left out.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": LIST_MAX_LIMIT,
                    "description": f"How many posts, newest first (default {LIST_DEFAULT_LIMIT}).",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["socials", "social media", "posts", "approval", "queue"],
        examples=[
            "which social posts are waiting for approval?",
            "list our social media drafts",
            "show the social posts a reviewer sent back",
        ],
    ))
