"""PRD-256 F430: the ONE definition of a Socials post's tool parameters, stdlib only.

The API agents' Socials tools (``modules/tools/discovery/actions_socials.py``) and a
CLI session's (``services/session_socials_tools.py``) take the same fields, so both
build their schemas here: a session tool forwards exactly what the action's registry
entry declares, and the parity test (``test_f430_b``) holds the two equal. This module
imports nothing, because a session module may not import ``modules.tools`` at load
time (its package imports the registry, the executor and the router), and the
utterance-corpus linter leaf-loads the actions.

The enums are the lifecycle's own vocabulary, spelled out here;
``tests/test_prd251w1_social_post_tools.py`` pins them to ``SOCIAL_POST_FORMATS``,
``SOCIAL_POST_STATUSES``, ``CHART_KINDS`` and ``PARTS``.
"""

# core/models/socials.py SOCIAL_POST_FORMATS
POST_FORMATS = ["video", "image", "carousel", "fact_card", "infographic", "text"]
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
# modules/tools/discovery/social_post_list.py QUEUES (F379): the posts by what they still need.
POST_QUEUES = ["awaiting_approval", "not_rendered", "open"]


def post_id_property() -> dict:
    return {"type": "string", "description": "The post's id (platform_list_social_posts lists them)."}


def _post_copy_fields() -> dict:
    """What a post says: its title, brief, copy, format and a video's length."""
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
        "format": {
            "type": "string",
            "enum": POST_FORMATS,
            "description": (
                "What the post is: video for a video template, image for a still, or carousel, "
                "fact_card, infographic or text. social_image and social_video are templates' formats, "
                "never a post's."
            ),
        },
        "length_seconds": {
            "type": "integer",
            "description": (
                "A video's length in seconds: one of the lengths its template declares "
                "(platform_get_template_schema). Leave it out for the template's own length."
            ),
        },
    }


def _post_template_fields() -> dict:
    """The template a post fills, its variables and the sources of its claims."""
    return {
        "template": {
            "type": "string",
            "description": (
                "The social template: its id or its exact name, from platform_list_templates (format "
                "social_image or social_video: the templates' formats, not the post's). Its fields are "
                "the variables platform_get_template_schema lists; a post with a template renders as "
                "soon as it is saved."
            ),
        },
        "variables": {
            "type": "object",
            "description": (
                "The template's variables, by its own field names only (a name it doesn't have is "
                "refused): {name: value}, or {name: {\"value\": ..., \"claim\": true or false}}. A claim is a fact or a figure and needs a source in sources; one the "
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
    }


def _post_media_fields() -> dict:
    """What a post shows and sounds like: files already made, its voice and music, its AI footage."""
    return {
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
                "Who speaks the script: leave it out for Kokoro with the template's own voice; one of "
                "Kokoro's own voices, {\"toolkit\": \"kokoro\", \"voice_id\": \"bm_george\"}; or a "
                "voice toolkit the workspace has connected: {\"toolkit\": \"fish_audio\" or "
                "\"elevenlabs\", \"voice_id\", \"name\"}."
            ),
            "properties": {
                "toolkit": {"type": "string"},
                "voice_id": {"type": "string"},
                "name": {"type": "string"},
            },
        },
        "music": {
            "type": "object",
            "description": (
                "The music a video plays: leave it out for its template's own track, {\"track\": null} "
                "for none, or {\"track\": \"<id>\"} for another track from the music library."
            ),
            "properties": {"track": {"type": ["string", "null"]}},
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
    }


def _post_chart_field() -> dict:
    """A chart template filled from a report (the Infographic)."""
    return {
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


def post_fields() -> dict:
    """What a post carries: the REST body's fields, plus template (id or name) and chart_report."""
    return {**_post_copy_fields(), **_post_template_fields(), **_post_media_fields(), **_post_chart_field()}


def create_properties() -> dict:
    """platform_create_social_post: the post's fields, and render (default true)."""
    return {
        **post_fields(),
        "render": {
            "type": "boolean",
            "description": (
                "Render the post's template now (default true). Send false for a post "
                "without a template, or when media already holds its rendered files."
            ),
        },
    }


def update_properties() -> dict:
    """platform_update_social_post: the post, what changes, and render (default false)."""
    return {
        "post_id": post_id_property(),
        **post_fields(),
        "render": {
            "type": "boolean",
            "description": "Render the post again after the change (default false).",
        },
    }


def submit_properties() -> dict:
    """platform_submit_social_post: the post, and a note for the reviewer."""
    return {
        "post_id": post_id_property(),
        "note": {
            "type": "string",
            "description": "What the reviewer should look at; shown in the post's history.",
        },
    }


def get_properties() -> dict:
    """platform_get_social_post: the post."""
    return {"post_id": post_id_property()}


def list_properties() -> dict:
    """platform_list_social_posts: by status, by queue, and how many."""
    return {
        "status": {
            "type": "array",
            "items": {"type": "string", "enum": POST_STATUSES},
            "description": "Only posts in these statuses; every status when left out.",
        },
        "queue": {
            "type": "string",
            "enum": POST_QUEUES,
            "description": (
                "Posts by what they need: awaiting_approval (waiting for the owner), not_rendered "
                "(a failed render, or a draft with nothing rendered yet) or open (every post not "
                "yet approved or out)."
            ),
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": LIST_MAX_LIMIT,
            "description": f"How many posts, newest first (default {LIST_DEFAULT_LIMIT}).",
        },
    }
