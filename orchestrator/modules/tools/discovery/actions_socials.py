"""Socials ActionDefinitions (PRD-251 US-116, S4.1): agents draft posts; people approve them.

Five tools over the Socials post lifecycle: create, update, submit, get and
list; two over a plan's content bank (PRD-251B B8, US-B204): read a plan, and
add researched topics to its bank; one over a plan's slots (P256-FIX-SOC-5, F436):
why a slot is empty and which batch makes it; and one over the workspace's history
(PRD-251C C5, US-C103): what it already posted, so research repeats nothing.
Their handlers (``handlers_socials.py``) run the flows the /api/socials routes
run. None of them approves, schedules or publishes, and no parameter does (D6,
D14): a person approves in the Socials tab, and the platform publishes (Wave 3).
PRD-251D (E6): create and update take ``planned_for`` and ``timezone``, the slot a post
waits in for that approval; approving it is what schedules it there.

The five post tools' parameters come from ``core/social_post_schema.py`` (PRD-256
F430), the one definition a CLI session's Socials tools are built from too; it holds
the lifecycle's enums, and is stdlib-only, so the registry build stays light (the
utterance-corpus linter leaf-loads this module).
"""

from core.social_plan_schema import add_topics_properties, get_plan_properties, history_properties
from core.social_post_schema import (
    create_properties,
    get_properties,
    list_properties,
    submit_properties,
    update_properties,
)

from .action_registry import ActionDefinition, ActionRegistry

# modules/tools/discovery/social_plan_slots.py DEFAULT_DAYS and MAX_DAYS: the days a slots read covers.
SLOTS_DEFAULT_DAYS = 7
SLOTS_MAX_DAYS = 14


def register_socials_actions(registry: ActionRegistry) -> None:
    """Register the Socials draft tools (PRD-251 US-116) and the plan research tools (PRD-251B US-B204)."""
    _register_create_social_post(registry)
    _register_update_social_post(registry)
    _register_submit_social_post(registry)
    _register_get_social_post(registry)
    _register_list_social_posts(registry)
    _register_plan_actions(registry)
    _register_plan_slots_action(registry)
    _register_history_action(registry)


def _register_create_social_post(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_create_social_post",
        description=(
            "Draft a Socials post for the workspace's channels: a title, its channels (where it "
            "publishes), its copy (the text for every channel and any channel's own text), a social "
            "template (a video or an image) with its variables, the sources of its facts and figures, "
            "and files already made. "
            "By default the template is rendered at once, within the plan's render minutes: when "
            "the render finishes the post waits for approval, or is failed with the reason in its "
            "history; a render refused at once fails the call, which says why and how to fix the "
            "saved draft. With render false it stays a draft until platform_submit_social_post. "
            "With planned_for and timezone (\"post it Friday at nine\") the post sits on that day and "
            "time in the Socials calendar while it waits for approval, and approving it schedules it "
            "there. A person approves every post in the Socials tab and the platform publishes it; "
            "this tool never approves, schedules or publishes."
        ),
        category="socials",
        parameters={"type": "object", "properties": create_properties(), "required": ["title"]},
        permission_level="write",
        promoted=True,  # PRD-256 US-006: a first-class tool, pinned
        requires_confirmation=False,
        tags=["socials", "social media", "post", "draft", "video", "image", "linkedin", "instagram", "twitter"],
        examples=[
            "draft a LinkedIn post about our launch",
            "make a social video announcing the new feature",
            "create a social post from this week's report",
        ],
    ))


def _register_update_social_post(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_update_social_post",
        description=(
            "Change a Socials post: send its post_id and only what changes; copy, variables and "
            "sources merge into the post's own, and channels replaces where it publishes. Changing an "
            "approved or scheduled post (its channels included) voids its "
            "approval, so a person approves it again in the Socials tab. With render true the post "
            "is rendered again after the change, which is how a failed render is fixed. planned_for "
            "and timezone move the day and time it is planned for while it is a draft, waits for "
            "approval or was sent back; an approved or scheduled post's slot is the owner's to move. "
            "Never approves, schedules or publishes."
        ),
        category="socials",
        parameters={"type": "object", "properties": update_properties(), "required": ["post_id"]},
        permission_level="write",
        requires_confirmation=False,
        tags=["socials", "social media", "post", "edit", "copy", "render"],
        examples=[
            "change the copy of that social post",
            "fix the headline on the draft social post and render it again",
            "add a source for the figure in the social post",
        ],
    ))


def _register_submit_social_post(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_submit_social_post",
        description=(
            "Send a draft Socials post, or one a reviewer asked to change, for approval: it moves to "
            "needs_approval, and a person approves it, asks for changes or rejects it in the Socials "
            "tab. A rendered post is already waiting for approval."
        ),
        category="socials",
        parameters={"type": "object", "properties": submit_properties(), "required": ["post_id"]},
        permission_level="write",
        requires_confirmation=False,
        tags=["socials", "social media", "post", "review", "submit"],
        examples=[
            "send the social post for approval",
            "submit that draft social post for review",
            "the social post is ready, put it up for approval",
        ],
    ))


def _register_get_social_post(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_get_social_post",
        description=(
            "Read one Socials post: where it publishes and each channel's receipt (when it went out, "
            "the platform's id and the link), its status, the day and time it is planned for (in its own "
            "timezone) and its other times, approval and what it still needs, then "
            "its copy, template and variables, sources, rendered media, voice and footage, and its last "
            "history entries (who drafted or changed it, a reviewer's comments, why a render failed)."
        ),
        category="socials",
        parameters={"type": "object", "properties": get_properties(), "required": ["post_id"]},
        permission_level="read",
        tags=["socials", "social media", "post", "status", "review"],
        examples=[
            "show me that social post",
            "what's the status of the launch video post?",
            "why did the social post fail to render?",
        ],
    ))


def _register_list_social_posts(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_list_social_posts",
        description=(
            "List the workspace's Socials posts, newest first: how many match, then one row each "
            "with its status, the day and time it is planned for, and what it still needs (the owner's "
            "approval, a render that failed and "
            "why, a render not made yet, a caption). Pick them by status, or by queue: what waits for "
            "the owner's approval, what has not rendered, or every post not yet approved or out."
        ),
        category="socials",
        parameters={"type": "object", "properties": list_properties(), "required": []},
        permission_level="read",
        tags=["socials", "social media", "posts", "approval", "queue"],
        examples=[
            "which social posts are waiting for approval?",
            "list our social media drafts",
            "show the social posts a reviewer sent back",
        ],
    ))


def _register_plan_actions(registry: ActionRegistry) -> None:
    """PRD-251B (B8, US-B204): the Content bank research playbook's read and its one write."""
    registry.register(ActionDefinition(
        name="platform_get_social_plan",
        description=(
            "Read a Socials plan: its goal, audience and dates, its cadence (channels, formats and "
            "times), what to research (sources, notes and the never-say list) and its content bank "
            "(each topic's title, formats and whether a post used it)."
        ),
        category="socials",
        parameters={"type": "object", "properties": get_plan_properties(), "required": ["plan_id"]},
        permission_level="read",
        tags=["socials", "plan", "content bank", "research", "cadence"],
        examples=["read the Socials plan", "what's in our content bank?", "show the social media plan's cadence"],
    ))
    registry.register(ActionDefinition(
        name="platform_add_social_topics",
        description=(
            "Add researched topics to a Socials plan's content bank, each with the facts it rests on and "
            "every fact's source (knowledge, deliverable, web, github or note; a reference and a label; a "
            "note needs no reference). The bank refuses a fact without a source, a title it already holds, "
            "and anything on the plan's never-say list; the answer lists what was added and what was "
            "refused, with why, and fails when nothing was added. It only adds topics: posts are made from "
            "them on their day."
        ),
        category="socials",
        parameters={
            "type": "object",
            "properties": add_topics_properties(),
            "required": ["plan_id", "topics"],
        },
        permission_level="write",
        tags=["socials", "plan", "content bank", "research", "topics"],
        examples=["add these topics to the content bank", "save the researched ideas to the Socials plan"],
    ))


def _register_plan_slots_action(registry: ActionRegistry) -> None:
    """P256-FIX-SOC-5 (F436): a plan's slots, one line each: made, waiting for its batch, building,
    stalled or late, and why; with no plan named, every active or paused plan."""
    registry.register(ActionDefinition(
        name="platform_get_social_plan_slots",
        description=(
            "Read a Socials plan's slots, one line each in the plan's timezone, and where its current batch "
            "stands: made (with its post's state, and what it waits for or why its render failed), waiting "
            "(which batch makes it, and when), building, stalled (why: Socials is off, the plan is paused, "
            "the day's limit, no unused topic suits its format), late, or skipped (why). Use it when asked "
            "why a slot is empty or when the plan makes a post. Asked about one day, pass that day as from "
            "with days 1, so its line is never left out. With no plan_id it reads every active or paused "
            "plan."
        ),
        category="socials",
        parameters={
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "The plan's id; leave it out to read every active or paused plan."},
                "from": {"type": "string", "description": "The first day to read, YYYY-MM-DD (today in each plan's timezone when left out)."},
                "days": {
                    "type": "integer", "minimum": 1, "maximum": SLOTS_MAX_DAYS,
                    "description": f"How many days to read from that day ({SLOTS_DEFAULT_DAYS} when left out).",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["socials", "plan", "slots", "empty slot", "batch", "calendar"],
        examples=[
            "why is Thursday empty in the plan?",
            "when will the plan make Friday's post?",
            "which planned posts are still to be made this week?",
        ],
    ))


def _register_history_action(registry: ActionRegistry) -> None:
    """PRD-251C (C5, US-C103): what the workspace already posted, for research and the composer."""
    registry.register(ActionDefinition(
        name="platform_get_social_history",
        description=(
            "Read what this workspace already posted on social media, newest first: every post that "
            "went out, is approved or scheduled, or waits for approval, across every plan and the posts "
            "people made by hand. Each has its title, topic and angle, format, channels, opening line, "
            "date and state, and its numbers and engagement once read. Read it before researching or "
            "writing, so a topic or a hook the workspace already used is not used again, and more like "
            "what did best is found. With went_out true it lists only the posts that went out in the last "
            "days, newest first by when they went out, each with its receipts: the channel, when, the "
            "platform's id and the link."
        ),
        category="socials",
        parameters={
            "type": "object",
            "properties": history_properties(),
            "required": [],
        },
        permission_level="read",
        tags=["socials", "history", "posts", "research", "repeats", "went out", "receipts"],
        examples=[
            "what have we already posted on social media?",
            "show our social posting history",
            "did we post about this before?",
            "what did I post today and when?",
            "which posts went out this week, and where are the links?",
        ],
    ))
