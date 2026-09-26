"""
The Socials marketplace package (PRD-251 US-120)
================================================

All data, on the PRD-230 machinery:

* two marketplace agents, real ``Agent`` rows (``owner_type='marketplace'``) on
  the Shopify pattern (``seed_shopify_agents``): the Social Media Director and
  the Brand Designer, each with a persona and built-in skills linked by name;
* four marketplace Playbooks, ``workflow_recipes`` rows with
  ``owner_type='marketplace'``: Brand kit from your website, Launch video,
  Weekly social posts (S4.2, seeded unscheduled) and Image carousel;
* the package row, ``SOCIALS_PACKAGE``, which ``seed_packages`` upserts with the
  Shopify packages. The installer (``services/package_installer.py``) clones the
  agents and the Playbooks into a workspace, workspace-owned.

``seed_socials_marketplace`` runs at every boot (``main.py``, leader worker),
after ``seed_builtin_skills`` and before ``seed_packages``. A row it finds is
left as it is, so live curation survives a redeploy, except the agents' skill
links: they are reconciled every time, so a built-in skill the owner syncs
(``scripts/sync-skills.py``) after the first boot attaches on the next one.
A skill is linked only from a global row of a name its agent lists, and the
roster lists only the built-in Socials skills: no skill that posts straight to
a channel (D14: agents draft, the Socials tab and the platform publisher are
the only way out).

The Image carousel's render is the post's own (``platform_create_social_post``
with render true): ``generate_document`` refuses a template that renders several
images, since one document is one file (US-107), and a post keeps every slide.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Tuple
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.models.core import PLAYBOOK_DOCUMENT_STEP, Agent, Skill, WorkflowTemplate, agent_skills

logger = logging.getLogger(__name__)

SEEDED_BY = "seed_socials_package"
MARKETPLACE = "marketplace"
DIRECTOR = "social-media-director"
BRAND_DESIGNER = "brand-designer"

CREATED = "created"
PRESENT = "present"
MISSING_AGENT = "missing_agent"
HELD_ELSEWHERE = "held_elsewhere"

# The seeded social templates the Playbooks render (modules/documents/templates/social/).
LAUNCH_VIDEO_TEMPLATE = "UI story promo"
LAUNCH_VIDEO_ASPECT = "9:16"
CAROUSEL_TEMPLATE = "Carousel"

# Seconds. An agent step makes several tool calls; the Launch video's render step
# waits up to SOCIALS_RENDER_MAX_WAIT_SECONDS on media-render, inside the total.
STEP_TIMEOUT_SECONDS = 600
TOTAL_TIMEOUT_SECONDS = 3600
EXECUTION_CONFIG: Mapping[str, Any] = {
    "mode": "sequential",
    "max_retries": 1,
    "timeout_per_step": STEP_TIMEOUT_SECONDS,
    "total_timeout": TOTAL_TIMEOUT_SECONDS,
}
DEFAULT_CHANNELS = "linkedin, twitter, instagram"
WEEKLY_POST_COUNT = 5

_NEVER_PUBLISH = (
    "A person approves every post in the Socials tab and the platform publishes it: never publish, "
    "schedule or approve one."
)
_NEVER_PUBLISH_PERSONA = (
    "You never publish, schedule or approve a post: a person approves every post in the Socials tab, "
    "and the platform publishes it."
)
_SOURCES_RULE = (
    "Every fact or figure is a claim, and every claim has a source in this workspace: a Deliverable, "
    "a report, a document, a metric, or a URL. Leave out what you cannot source."
)
_SAVED_SHAPE = (
    '"variables" (one JSON object, {variable name: value}), "title" (the post\'s title, as the Socials '
    'tab lists it) and "sources" (a JSON list, [{"claim": variable name, "kind": deliverable, report, '
    'document, url or metric, "ref": its id or address, "as_of": ISO date-time}], empty when nothing '
    "is claimed)"
)

# ── The agents ───────────────────────────────────────────────────────────────

_DIRECTOR_PERSONA = f"""You are the Social Media Director for this workspace. You turn what the workspace already has (its Deliverables, reports, blog posts, documents and brand kit) into social posts: short videos, image cards, carousels, fact cards and infographics, with copy written for each channel.

How you work:
- One idea per post. Decide who it is for, the one thing it says, and its channels (linkedin, twitter, instagram, tiktok, youtube).
- Pick the template that fits: platform_list_templates with format social_video or social_image, then platform_get_template_schema for its variables. Every word on screen is a template variable you fill, within its length limit, in the brand's voice (platform_get_brand_kit).
- {_SOURCES_RULE} Find them with search_knowledge, platform_list_deliverables and platform_browse_reports.
- Write each channel's copy for that channel: LinkedIn in short paragraphs, X within 280 characters, Instagram with the hook in its first line.
- Draft with platform_create_social_post. A post with a template renders at once, within the plan's render minutes, and waits for approval when the render finishes. A draft you did not render goes for approval with platform_submit_social_post and a note for the reviewer.
- AI footage and stills come only from the workspace's connected generation toolkit, through a post's footage: describe the subject, scene, light and camera, with no readable text and no logos. With none connected, the template's own motion graphics play.
- The voice is Kokoro unless the workspace connected a voice toolkit (fish_audio or elevenlabs).
- Check your work with platform_get_social_post: its status, media and history, including why a render failed.

{_NEVER_PUBLISH_PERSONA} Nor do you call a social channel's own posting action: with Socials on, the platform refuses it."""

_BRAND_DESIGNER_PERSONA = f"""You are the Brand Designer for this workspace. You build and keep its brand kit, the colours, fonts, logo, name, voice and social handles every Socials template and document renders with, and you check the workspace's posts against it.

Building the kit:
- Start from the brand's own website: fetch its pages with platform_web_fetch, and read the current kit and its suggestions with platform_get_brand_kit.
- Take only what the pages show. Colours are hex values (primary, secondary, accent, text), fonts are the families the site uses, the voice is 3 to 5 tone words, and handles are the accounts the site links to, without the @.
- Save with platform_update_brand_kit, sending only what you found. social_handles replaces the whole map, so send every handle, the kit's current ones included.
- Logo and font files are uploaded by a person on the Brand kit page: say when one is missing.

Checking posts:
- Read a post with platform_get_social_post and compare its copy and its template's variables with the kit: its names, handles and tone. Say what drifts and what to change; when a render's colours, type or logo look wrong to a person, the kit is what you fix.

{_NEVER_PUBLISH_PERSONA}"""

SOCIALS_AGENTS: List[Dict[str, Any]] = [
    {
        "slug": DIRECTOR,
        "name": "Social Media Director",
        "description": (
            "Plans the workspace's social posts, writes each channel's copy, picks the templates, "
            "drafts posts with their sources, renders them and sends them for approval. Never publishes."
        ),
        "agent_type": "specialized",
        "marketplace_category": "marketing",
        "marketplace_icon": "🎬",
        "team": "Marketing",
        "job_title": "Social Media Director",
        "tags": ["socials", "social media", "video", "content", "marketing"],
        "custom_persona_prompt": _DIRECTOR_PERSONA,
        "skills": [
            "social-video-director",
            "social-ops",
            "social-production-workflow",
            "social-template-payloads",
            "social-media-strategist",
            "social-brand-voice",
            "carousel-design-system",
            "image-prompt-engineer",
            "short-video-editing-coach",
        ],
    },
    {
        "slug": BRAND_DESIGNER,
        "name": "Brand Designer",
        "description": (
            "Builds and keeps the workspace's brand kit from its website (colours, fonts, name, voice "
            "and social handles) and checks posts against it."
        ),
        "agent_type": "specialized",
        "marketplace_category": "design",
        "marketplace_icon": "🎨",
        "team": "Marketing",
        "job_title": "Brand Designer",
        "tags": ["socials", "brand", "design", "brand kit"],
        "custom_persona_prompt": _BRAND_DESIGNER_PERSONA,
        "skills": ["brand-kit-builder", "visual-storyteller", "image-prompt-engineer", "social-brand-voice"],
    },
]

# ── The Playbooks ────────────────────────────────────────────────────────────
# Steps carry the agent's slug; the stored rows carry the marketplace agent's id
# and name, which the installer maps to the workspace's clone (by name first).


def _channels_input(default: str) -> Dict[str, Any]:
    return {
        "type": "string",
        "required": False,
        "description": "The channels, by toolkit name: linkedin, twitter, instagram, tiktok, youtube",
        "default": default,
    }


def _agent_step(step_id: str, order: int, agent: str, output_key: str, prompt: str) -> Dict[str, Any]:
    return {
        "step_id": step_id,
        "order": order,
        "agent_slug": agent,
        "error_handling": "stop",
        "output_key": output_key,
        "prompt_template": prompt,
    }


_BRAND_KIT_PROMPT = """Build this workspace's brand kit from its website: {input.website_url}

1. Read the current kit and its suggestions with platform_get_brand_kit. When no address is given above, use the website in the suggestions; when there is none, stop and say this Playbook needs the website's address.
2. Fetch the home page, and one or two pages that show the brand best (about, product), with platform_web_fetch.
3. From what the pages show, never guesses, work out: the primary, secondary and accent colours and the text colour (hex), the body and heading fonts, the company's name and tagline, 3 to 5 tone words, and the social handles the site links to.
4. Save it with platform_update_brand_kit, sending only what you found. social_handles replaces the whole map, so send every handle, the kit's current ones included.
5. Answer with what you changed, what you could not find, and what a person should do next (a logo or a font file is uploaded on the Brand kit page)."""

_LAUNCH_SCRIPT_PROMPT = f"""Write the launch video for: {{input.launch}}

The next step renders it with the "{LAUNCH_VIDEO_TEMPLATE}" video template.
1. Read the template's variables with platform_get_template_schema (template "{LAUNCH_VIDEO_TEMPLATE}"). Fill every variable without a default, and the others where they make the story better, each within its max_chars.
2. Tell the launch as the template's story, in the brand's voice (platform_get_brand_kit).
3. {_SOURCES_RULE} Look for them with search_knowledge, platform_list_deliverables and platform_browse_reports.
4. Save three keys with scratchpad_write: {_SAVED_SHAPE}.
5. Answer with the script, one line per scene."""

_LAUNCH_DRAFT_PROMPT = f"""Draft the launch post for: {{input.launch}}
Channels: {{input.channels}}

The previous step rendered the video and registered it as a Deliverable: its deliverable_id is in that step's result (platform_list_deliverables with artifact_type video lists it too, newest first).
1. Create the post with platform_create_social_post: the title, variables and sources step 1 saved; format "video"; template "{LAUNCH_VIDEO_TEMPLATE}"; copy for each channel above, written for that channel; media {{"{LAUNCH_VIDEO_ASPECT}": ["<deliverable_id>"]}}; render false, because the video is rendered already.
2. Send it for approval with platform_submit_social_post: its post_id, and a note telling the reviewer what the video claims and where each claim comes from.
3. Answer with the post's id and status.
{_NEVER_PUBLISH}"""

_WEEKLY_PLAN_PROMPT = f"""Plan this week's social posts: {{input.count}} posts across these channels: {{input.channels}}, spread over the coming week.

1. Read what the workspace made in the last 7 days: platform_list_deliverables (a blog post is a Deliverable too), platform_list_blog_posts and platform_browse_reports. Use search_knowledge for background only.
2. Pick the {{input.count}} strongest ideas, one per post. For each: its channels; the day and time you propose to post it, in the workspace's timezone; the image template that fits (platform_list_templates with format social_image, then platform_get_template_schema for its variables; a report's table suits the Infographic, filled from the report with chart_report); the template's variables; each channel's copy; and its sources.
3. {_SOURCES_RULE}
4. Save the plan with scratchpad_write, key "plan": a JSON list with one object per post, {{"title", "brief", "proposed_for", "channels", "format", "template", "variables", "copy", "sources", "chart_report" (the Infographic only)}}.
5. Answer with the plan as a table: title, proposed for, channels, template."""

_WEEKLY_DRAFT_PROMPT = f"""Draft every post in the plan the previous step saved (its "plan" key), one platform_create_social_post call per post: its title; its brief, starting with "Proposed for: <day and time>"; its format, template, variables, copy and sources (and chart_report for the Infographic); render true. A post whose render starts waits for approval when it finishes.

When a call is refused, fix what the answer names and try once more; when it is refused again, leave that post out and say why. When a post's render could not start (the answer says why), send the draft for approval anyway with platform_submit_social_post and a note naming the reason.

Answer with a table: title, proposed for, channels, post id, status.
{_NEVER_PUBLISH}"""

_CAROUSEL_SLIDES_PROMPT = f"""Write an image carousel about: {{input.topic}}

It renders with the "{CAROUSEL_TEMPLATE}" image template: a cover, two to six points, and a closing slide, one image each.
1. Read its variables with platform_get_template_schema (template "{CAROUSEL_TEMPLATE}").
2. Build the slides from the workspace's own material (search_knowledge, platform_list_deliverables, platform_browse_reports): a headline that promises one thing, one point per slide (a short title and a line or two of body), and a closing slide that says what to do next, each within its max_chars, in the brand's voice (platform_get_brand_kit).
3. {_SOURCES_RULE}
4. Save three keys with scratchpad_write: {_SAVED_SHAPE}.
5. Answer with the slides, one line each."""

_CAROUSEL_DRAFT_PROMPT = f"""Draft the carousel post about: {{input.topic}}
Channels: {{input.channels}}

1. Create the post with platform_create_social_post: the title, variables and sources step 1 saved; format "carousel"; template "{CAROUSEL_TEMPLATE}"; copy for each channel above, written for that channel; render true. The render makes one image per slide, and the post waits for approval when it finishes.
2. When the render could not start (the answer says why), send the draft for approval anyway with platform_submit_social_post and a note naming the reason.
3. Answer with the post's id and status.
{_NEVER_PUBLISH}"""

SOCIALS_PLAYBOOKS: List[Dict[str, Any]] = [
    {
        "template_id": "marketplace-socials-brand-kit",
        "name": "Brand kit from your website",
        "description": (
            "The Brand Designer reads your website and fills in your brand kit: colours, fonts, name, "
            "tagline, voice and social handles. Every Socials template renders with it."
        ),
        "icon": "🎨",
        "inputs": {
            "website_url": {
                "type": "string",
                "required": True,
                "description": "The brand's website, such as https://example.com",
            },
        },
        "outputs": {"brand_kit": {"type": "string", "description": "What changed in the brand kit, and what is missing"}},
        "steps": (_agent_step("brand-kit", 1, BRAND_DESIGNER, "brand_kit", _BRAND_KIT_PROMPT),),
        "tags": ["socials", "brand kit", "setup"],
    },
    {
        "template_id": "marketplace-socials-launch-video",
        "name": "Launch video",
        "description": (
            "The Social Media Director writes a launch video, the platform renders it with your brand "
            "kit, and the Director drafts the post with the video and sends it for your approval."
        ),
        "icon": "🎬",
        "inputs": {
            "launch": {
                "type": "string",
                "required": True,
                "description": "What is launching, for whom, and the one thing the video must say",
            },
            "channels": _channels_input(DEFAULT_CHANNELS),
        },
        "outputs": {"post": {"type": "string", "description": "The drafted post, waiting for approval"}},
        "steps": (
            _agent_step("script", 1, DIRECTOR, "script", _LAUNCH_SCRIPT_PROMPT),
            {
                "step_id": "render",
                "order": 2,
                "type": PLAYBOOK_DOCUMENT_STEP,
                "error_handling": "stop",
                "output_key": "video",
                "config": {
                    "title": "{{ step_1.title }}",
                    "format": "social_video",
                    "template_name": LAUNCH_VIDEO_TEMPLATE,
                    "data": "{{ step_1.variables }}",
                },
            },
            _agent_step("draft", 3, DIRECTOR, "post", _LAUNCH_DRAFT_PROMPT),
        ),
        "tags": ["socials", "video", "launch"],
    },
    {
        "template_id": "marketplace-socials-weekly-posts",
        "name": "Weekly social posts",
        "description": (
            "Reads the week's Deliverables, blog posts and reports, proposes posts across your channels "
            "spread over the week, drafts them with their sources and leaves them for your approval. "
            "Unscheduled: run it, or schedule it with platform_schedule_playbook."
        ),
        "icon": "🗓️",
        "inputs": {
            "count": {
                "type": "integer",
                "required": False,
                "description": "How many posts to propose",
                "default": WEEKLY_POST_COUNT,
            },
            "channels": _channels_input(DEFAULT_CHANNELS),
        },
        "outputs": {"posts": {"type": "string", "description": "The drafted posts, with the day proposed for each"}},
        "steps": (
            _agent_step("plan", 1, DIRECTOR, "plan", _WEEKLY_PLAN_PROMPT),
            _agent_step("draft", 2, DIRECTOR, "posts", _WEEKLY_DRAFT_PROMPT),
        ),
        "tags": ["socials", "weekly", "content plan"],
    },
    {
        "template_id": "marketplace-socials-image-carousel",
        "name": "Image carousel",
        "description": (
            "The Social Media Director writes the slides of an image carousel, drafts the post, and the "
            "platform renders one image per slide for your approval."
        ),
        "icon": "🖼️",
        "inputs": {
            "topic": {
                "type": "string",
                "required": True,
                "description": "What the carousel is about, and who it is for",
            },
            "channels": _channels_input("linkedin, instagram"),
        },
        "outputs": {"post": {"type": "string", "description": "The drafted carousel post"}},
        "steps": (
            _agent_step("slides", 1, DIRECTOR, "slides", _CAROUSEL_SLIDES_PROMPT),
            _agent_step("draft", 2, DIRECTOR, "post", _CAROUSEL_DRAFT_PROMPT),
        ),
        "tags": ["socials", "carousel", "images"],
    },
]

def playbook_agents(spec: Mapping[str, Any]) -> List[str]:
    """The agent slugs a Playbook's steps run, in step order, once each."""
    slugs: List[str] = []
    for step in spec["steps"]:
        slug = step.get("agent_slug")
        if slug is not None and slug not in slugs:
            slugs.append(slug)
    return slugs


# ── The package ──────────────────────────────────────────────────────────────


def _first_sentence(text: str) -> str:
    dot = text.find(". ")
    return text[: dot + 1] if dot != -1 else text


def _members() -> List[Dict[str, Any]]:
    """Agents first, then Playbooks: a Playbook's install reuses the agents' clones."""
    agents = [
        {"type": "agent", "ref": spec["slug"], "name": spec["name"], "description": _first_sentence(spec["description"])}
        for spec in SOCIALS_AGENTS
    ]
    playbooks = [
        {"type": "playbook", "ref": spec["template_id"], "name": spec["name"], "description": _first_sentence(spec["description"])}
        for spec in SOCIALS_PLAYBOOKS
    ]
    return agents + playbooks


_PUBLISH_LATER = "Optional, for publishing later. Every post waits for your approval in the Socials tab."
_GENERATION = (
    "Optional: one generation toolkit (fal.ai, Kie.ai or Higgsfield) for AI footage and stills, paid "
    "from your own account. Without one, videos play the template's own motion graphics."
)
_VOICE = (
    "Optional: one voice toolkit (Fish Audio or ElevenLabs), paid from your own account. Without one, "
    "the voice is Kokoro, built in and free."
)


def _connect(app_name: str, group: str, note: str, **extra: Any) -> Dict[str, Any]:
    return {"app_name": app_name, "optional": True, "group": group, "note": note, **extra}


SOCIALS_PACKAGE: Dict[str, Any] = {
    "slug": "socials",
    "name": "Socials",
    "description": (
        "On-brand social posts from your own work: short videos, image cards, carousels and "
        "infographics, drafted with their sources and approved by you before anything goes out. A "
        "Social Media Director and a Brand Designer, with four Playbooks to set up and keep posting."
    ),
    "vertical_tags": ["socials", "social-media", "marketing", "content"],
    "matching": {
        "platforms": ["linkedin", "instagram", "twitter", "tiktok", "youtube"],
        "url_patterns": ["linkedin.com", "instagram.com", "twitter.com", "tiktok.com", "youtube.com"],
        "vocabulary": [
            "social", "socials", "posts", "posting", "video", "videos", "reels", "carousel",
            "brand", "campaign", "launch", "followers", "audience", "promo",
        ],
    },
    "members": _members(),
    "setup_manifest": {
        "questions": [
            {"id": "website_url", "prompt": "What's your website? The Brand Designer builds your brand kit from it."},
            {"id": "channels", "prompt": "Which channels do you post on: LinkedIn, X, Instagram, TikTok, YouTube?"},
            {
                "id": "posting_cadence",
                "prompt": f"How often do you want to post? The weekly Playbook proposes {WEEKLY_POST_COUNT} posts a week unless you say otherwise.",
            },
            {
                "id": "media_monthly_cap_usd",
                "prompt": "What's the most your connected media tools may spend on Socials in a month, in US dollars?",
                "setting": "socials.media_monthly_cap_usd",
            },
        ],
        "required_connects": [
            _connect("LINKEDIN", "publish", _PUBLISH_LATER, app_type="SOCIAL", needs_oauth=True),
            _connect("TWITTER", "publish", _PUBLISH_LATER, app_type="SOCIAL", needs_oauth=True),
            _connect("INSTAGRAM", "publish", _PUBLISH_LATER, app_type="SOCIAL", needs_oauth=True),
            _connect("FAL_AI", "generation", _GENERATION),
            _connect("KIEAI", "generation", _GENERATION),
            _connect("HIGGSFIELD_MCP", "generation", _GENERATION),
            _connect("FISH_AUDIO", "voice", _VOICE),
            _connect("ELEVENLABS", "voice", _VOICE),
        ],
        "guide_steps": [
            {
                "step": 1,
                "title": "Turn Socials on",
                "description": (
                    "Deliverables → Socials → Turn on Socials for this workspace. It adds the video and "
                    "image templates your posts render with."
                ),
            },
            {
                "step": 2,
                "title": "Build your brand kit",
                "description": "Run 'Brand kit from your website': the Brand Designer fills in your colours, fonts, name and voice.",
            },
            {
                "step": 3,
                "title": "Make your launch video",
                "description": "Run 'Launch video': the Social Media Director writes it, it renders with your brand kit, and the post is drafted.",
            },
            {
                "step": 4,
                "title": "Approve it in the Socials tab",
                "description": "Every post waits for your approval there. Nothing is published without it.",
            },
        ],
        "report_templates": [
            {
                "name": "weekly-social-report",
                "title": "Weekly Social Report",
                "description": "The week's posts: drafted, waiting for approval and approved, and what each one said.",
            },
        ],
    },
    "showcase": True,
}

# ── Seeding ──────────────────────────────────────────────────────────────────


def seed_socials_marketplace(db: Session) -> Dict[str, Any]:
    """Create the Socials agents and Playbooks that are missing, and link each agent's
    built-in skills that exist now. Returns what happened, by slug; the caller commits."""
    agents: Dict[str, Agent] = {}
    outcome: Dict[str, Any] = {"agents": {}, "skills": {}, "playbooks": {}}
    for spec in SOCIALS_AGENTS:
        agent, created = _ensure_agent(db, spec)
        agents[spec["slug"]] = agent
        outcome["agents"][spec["slug"]] = CREATED if created else PRESENT
        outcome["skills"][spec["slug"]] = _link_skills(db, agent, spec["skills"])
    for spec in SOCIALS_PLAYBOOKS:
        outcome["playbooks"][spec["template_id"]] = _ensure_playbook(db, spec, agents)
    db.flush()
    return outcome


def _ensure_agent(db: Session, spec: Mapping[str, Any]) -> Tuple[Agent, bool]:
    row = db.query(Agent).filter(Agent.slug == spec["slug"], Agent.owner_type == MARKETPLACE).first()
    if row is not None:
        return row, False
    row = Agent(
        public_id=uuid4(),
        name=spec["name"],
        slug=spec["slug"],
        description=spec["description"],
        agent_type=spec["agent_type"],
        marketplace_category=spec["marketplace_category"],
        marketplace_icon=spec["marketplace_icon"],
        team=spec["team"],
        job_title=spec["job_title"],
        tags=list(spec["tags"]),
        # No model_config: the agent runs on the workspace's configured LLM.
        model_config=None,
        custom_persona_prompt=spec["custom_persona_prompt"],
        use_custom_persona=True,
        configuration={},
        status="active",
        owner_type=MARKETPLACE,
        owner_id=MARKETPLACE,
        workspace_id=None,
        is_approved=True,
        is_featured=True,
        version="1.0.0",
        created_by=SEEDED_BY,
    )
    db.add(row)
    db.flush()
    logger.info("Socials package: created marketplace agent %s (id=%s)", spec["slug"], row.id)
    return row, True


def _link_skills(db: Session, agent: Agent, wanted: List[str]) -> Dict[str, List[str]]:
    """Link the global skill rows of ``wanted`` the agent lacks: the first listed is its primary."""
    rows = (
        db.query(Skill.id, Skill.name)
        .filter(Skill.name.in_(wanted), Skill.workspace_id.is_(None), Skill.is_active.is_(True))
        .all()
    )
    found = {name: skill_id for skill_id, name in rows}
    linked = set(db.execute(select(agent_skills.c.skill_id).where(agent_skills.c.agent_id == agent.id)).scalars())
    attached = []
    for index, name in enumerate(wanted):
        skill_id = found.get(name)
        if skill_id is None or skill_id in linked:
            continue
        db.execute(agent_skills.insert().values(agent_id=agent.id, skill_id=skill_id, priority=len(wanted) - index))
        attached.append(name)
    if attached:
        db.expire(agent, ["skills"])
        logger.info("Socials package: linked %s to %s", attached, agent.slug)
    missing = [name for name in wanted if name not in found]
    if missing:
        logger.info("Socials package: %s waits for skills not synced yet: %s", agent.slug, missing)
    return {"attached": attached, "missing": missing}


def _ensure_playbook(db: Session, spec: Mapping[str, Any], agents: Mapping[str, Agent]) -> str:
    # template_id is unique across every row (ix_workflow_recipes_template_id).
    row = db.query(WorkflowTemplate).filter(WorkflowTemplate.template_id == spec["template_id"]).first()
    if row is not None:
        if row.owner_type == MARKETPLACE:
            return PRESENT
        logger.warning("Socials package: template_id %s is held by a %s Playbook", spec["template_id"], row.owner_type)
        return HELD_ELSEWHERE
    if not set(playbook_agents(spec)) <= set(agents):
        return MISSING_AGENT
    recipe = WorkflowTemplate(**playbook_columns(spec, agents))
    _validate(recipe)
    db.add(recipe)
    db.flush()
    logger.info("Socials package: created marketplace Playbook %s (id=%s)", spec["template_id"], recipe.id)
    return CREATED


def playbook_columns(spec: Mapping[str, Any], agents: Mapping[str, Any]) -> Dict[str, Any]:
    """The ONE definition of what a seeded Playbook row holds (``agents``: slug → row)."""
    return {
        "template_id": spec["template_id"],
        "name": spec["name"],
        "description": spec["description"],
        "workspace_id": None,
        "owner_type": MARKETPLACE,
        "owner_id": MARKETPLACE,
        "template_definition": {"steps": [], "agents": [], "config": {}, "variables": []},
        "steps": _stored_steps(spec["steps"], agents),
        "inputs": deepcopy(spec["inputs"]),
        "outputs": deepcopy(spec["outputs"]),
        "execution_config": dict(EXECUTION_CONFIG),
        # Unscheduled: a person runs it, or schedules it (platform_schedule_playbook).
        "schedule_config": {"type": "manual"},
        "tags": list(spec["tags"]),
        # The installer clones these (by name) or reuses the workspace's clone.
        "recommended_agents": [agents[slug].name for slug in playbook_agents(spec)],
        "required_tools": [],
        "is_public": True,
        "is_featured": True,
        "is_system": False,
        "is_approved": True,
        "marketplace_category": "marketing",
        "marketplace_icon": spec["icon"],
        "version": "1.0",
        "created_by": SEEDED_BY,
    }


def _stored_steps(steps, agents: Mapping[str, Any]) -> List[Dict[str, Any]]:
    stored = []
    for step in steps:
        rest = {key: deepcopy(value) for key, value in step.items() if key != "agent_slug"}
        slug = step.get("agent_slug")
        if slug is not None:
            rest["agent_id"] = agents[slug].id
            rest["agent_name"] = agents[slug].name
        stored.append(rest)
    return stored


def _validate(recipe: WorkflowTemplate) -> None:
    """The create route's validators (api/workflow_recipes.create), raised."""
    for check in (recipe.validate_steps, recipe.validate_execution_config, recipe.validate_schedule_config):
        ok, error = check()
        if not ok:
            raise ValueError(f"Socials Playbook {recipe.template_id} is invalid: {error}")
