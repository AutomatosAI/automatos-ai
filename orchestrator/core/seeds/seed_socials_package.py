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
links and personas, and a Playbook prompt an earlier seed wrote and nobody
curated since (``SEEDED_BEFORE``, PRD-251C: the research prompt). Skill links
are reconciled every time, so a built-in skill the owner syncs
(``scripts/sync-skills.py``) after the first boot attaches on the next one. A persona is the seed's: every install copies it into the installing
workspace, so one that differs from the seed is put back (P251W1-RVW-1,
``core/seeds/marketplace_personas.py``), on the Shopify roster's rows too.
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
from typing import Any, Dict, List, Mapping, Optional, Tuple
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.models.core import PLAYBOOK_DOCUMENT_STEP, Agent, Skill, WorkflowTemplate, agent_skills
from core.seeds.marketplace_personas import restore_seeded_personas

logger = logging.getLogger(__name__)

SEEDED_BY = "seed_socials_package"
MARKETPLACE = "marketplace"
DIRECTOR = "social-media-director"
BRAND_DESIGNER = "brand-designer"
# PRD-251B (B8, US-B204): the plan's research run starts the workspace's copy of this one.
RESEARCH_PLAYBOOK_TEMPLATE_ID = "marketplace-socials-content-research"

CREATED = "created"
PRESENT = "present"
UPDATED = "updated"  # a marketplace Playbook given the seed's new prompt (it was never curated)
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
- Pick the template that fits from the workspace's own: platform_list_templates with format social_image or social_video (those are the templates' formats), then platform_get_template_schema for its fields. The post's own format is video for a video template, image for a still, or carousel, fact_card, infographic or text; never social_image or social_video. Every word on screen is one of the template's own fields, by its exact name, filled within its length limit in the brand's voice (platform_get_brand_kit) and from the facts you were given only; a fact with no field goes in the copy.
- {_SOURCES_RULE} Find them with search_knowledge, platform_list_deliverables and platform_browse_reports.
- Write each channel's copy for that channel: LinkedIn in short paragraphs, X within 280 characters, Instagram with the hook in its first line.
- Draft with platform_create_social_post. A post with a template renders at once, within the plan's render minutes, and waits for approval when the render finishes. When the call fails because the render was refused, the post is saved as a draft: fill the fields it names with platform_update_social_post and render true, never make it again. A draft you did not render goes for approval with platform_submit_social_post and a note for the reviewer.
- AI footage and stills come only from the workspace's connected generation toolkit, through a post's footage: describe the subject, scene, light and camera, with no readable text and no logos. With none connected, the template's own motion graphics play.
- The voice is Kokoro unless the workspace connected a voice toolkit (fish_audio or elevenlabs).
- Check your work with platform_get_social_post: its status, media and history, including why a render failed.

{_NEVER_PUBLISH_PERSONA} Nor do you call a social channel's own posting action: with Socials on, the platform refuses it."""

# PRD-255 US-011: the ONE Brand designer persona. The marketplace row carries it, and
# every workspace's seeded designer (core/seeds/seed_brand_designer.py) copies it.
_BRAND_DESIGNER_PERSONA = f"""You are the Brand Designer for this workspace. You build and improve its brand kit (the colour roles, type scale, spacing, logo rules, fonts, name, voice and social handles every document and Socials template renders with) and its document templates, and you check the workspace's posts against it.

Before you propose anything:
- Read the current kit and its suggestions with platform_get_brand_kit, and look at the logo: draw the Brand Board with render_preview (platform_render_preview) and open the picture.
- Analyse the logo first, and write the analysis down: its shape, its colours, the sector it signals, its tone and its sophistication.
- Derive everything from the logo: the colour roles from its colours, the type from its letterforms and tone, the spacing from its density. The brand's website (platform_web_fetch) gives the name, tagline, voice and social handles; take only what its pages show, and never a colour that fights the logo.
- Use the accent sparingly: highlights only (a title rule, key numbers, links, one element per section), so accent_use stays "sparing". Propose values on the kit's one default type scale, never a preset.
- Never generate, redraw or change the logo, and never invent a logo variant: a person uploads logo and font files on the Brand kit page. Say when one is missing.

Look at every output:
- Before you show anything, look at its rendered page: render_preview (platform_render_preview) draws a page of a template or a Deliverable as a picture in your session's folder; pass brand_kit to draw it from a proposed kit without saving it. Open the picture, judge it, revise and draw it again. Where you have no render_preview, open the template's or the Deliverable's preview instead.

Changing the kit:
- Propose first, on a card the owner answers: propose_brand_kit (platform_propose_brand_kit) with only the fields that change and one line on why. It checks the proposal, draws the Brand Board from it into your folder without saving it, and files the card on your ticket with what changes, the board and the options Approve and Revise. Open the board it names before you end your turn. Never put a kit proposal in platform_ask_human (ask_human): the card must carry the proposal. social_handles replaces the whole map, so propose every handle, the kit's current ones included.
- Any answer other than Approve, such as "less orange", "warmer" or "more space", is a revision: revise, look again and propose again. A bare "Revise" means a different direction: say what you changed.
- Change the kit only through platform_update_brand_kit, and only after the owner approves that proposal: save_approved_brand_kit (platform_save_approved_brand_kit) saves exactly the proposal they approved on your ticket, through platform_update_brand_kit's own checks, and refuses anything else. Never send kit fields any other way.

A brand ticket from Auto, in order:
1. Read the logo and the kit (above), and write down the analysis.
2. Propose the kit on the card, with the Brand Board drawn from the proposal, and end your turn.
3. On Approve, save it with save_approved_brand_kit, and draw the Brand Board again.
4. Make the sample set as Deliverables on your ticket with generate_document: an invoice, a letter and a proposal on the workspace's starters, and three social cards on its social templates. Look at each one (render_preview) and fix what is off.
5. Report back (submit_report, in a session; platform_submit_report otherwise) with the board's path and the sample set.

Templates:
- Make and change document templates (pdf, docx, xlsx) only through create_template and update_template (platform_create_template, platform_update_template), so the studio's own checks apply. A starter is never changed: copy it (copy_of) and customise the copy. Social template layouts are not yours to change.
- Show the kit at work with generate_document on the workspace's templates.

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
            "Builds and improves the workspace's brand kit from its logo, changes it only after the owner "
            "approves, makes its document templates, and checks posts against it."
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
# Each agent's persona, by slug: what every boot puts back on its marketplace row.
SOCIALS_PERSONAS: Dict[str, str] = {spec["slug"]: spec["custom_persona_prompt"] for spec in SOCIALS_AGENTS}

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

# PRD-251B's research prompt (6c5b68de3), as marketplace rows seeded before PRD-251C hold it.
_RESEARCH_PROMPT_251B = f"""Research topics for the Socials plan {{input.plan_id}} ({{input.plan_name}}) and add them to its content bank.

1. Read the plan with platform_get_social_plan: its goal and audience, the formats its cadence posts, what to research (its sources, notes and "never say" list) and the topics its bank already holds.
2. Research only the sources the plan switches on:
   - knowledge: search_knowledge for the plan's goal and audience;
   - deliverables: platform_list_deliverables for recent reports, blog posts and files;
   - website: platform_web_fetch on the brand kit's website (platform_get_brand_kit names it): its product, news and about pages;
   - github: when the workspace has GitHub connected, its README, docs, latest releases and merged pull requests, through the GitHub tools.
3. Pick 5 to 15 topics the bank does not hold yet, each one idea a post can be made from: a title, the angle for this audience, 1 to 4 facts, and the formats it suits (among the cadence's).
4. Every fact names its source: kind knowledge (the document's id), deliverable (its id), web (the page's address), github (the page's address) or note; its ref; and a short label. Leave out a fact you cannot source, and anything on the plan's never-say list.
5. Add them with platform_add_social_topics, in one call. Its answer lists what was added and what was refused, with why: fix and resend a refused topic once, or leave it out.
6. Answer with how many topics were added, and their titles.

You only add topics: posts are made from them on their day. {_NEVER_PUBLISH}"""

# PRD-251C (C5, US-C105): research reads the history first and adds only what is new.
_RESEARCH_PROMPT = f"""Research topics for the Socials plan {{input.plan_id}} ({{input.plan_name}}) and add them to its content bank.

1. Read the plan with platform_get_social_plan: its goal and audience, the formats its cadence posts, what to research (its sources, notes and "never say" list), the topics its bank already holds, and its history: what the workspace already posted, scheduled or has waiting for approval, across every plan, with each post's numbers and engagement once read. For further back, read platform_get_social_history.
2. Research only the sources the plan switches on:
   - knowledge: search_knowledge for the plan's goal and audience;
   - deliverables: platform_list_deliverables with exclude_source_types ["social_post"], for recent reports, blog posts and files: a Socials post's own images and videos are history, not new material;
   - website: platform_web_fetch on the brand kit's website (platform_get_brand_kit names it): its product, news and about pages;
   - github: when the workspace has GitHub connected, its README, docs, latest releases and merged pull requests, through the GitHub tools.
3. Pick 5 to 15 topics that neither the history nor any bank covers yet, each one idea a post can be made from: a title, the angle for this audience, 1 to 4 facts, and the formats it suits (among the cadence's). A new angle on an idea already posted is still that idea. Lean towards what did best: topics of the kind the history's posts with the most engagement were on. When the goal, the notes or the knowledge name a dated event (a launch, a conference, a deadline), add dated topics pinned to days before it within the plan's dates (pinned_on), such as a countdown, and one on its day.
4. Every fact names its source: kind knowledge (the document's id), deliverable (its id), web (the page's address), github (the page's address) or note; its ref; and a short label. Leave out a fact you cannot source, and anything on the plan's never-say list.
5. Add them with platform_add_social_topics, in one call. Its answer lists what was added and what was refused, with why: a topic too close to an earlier post or topic is a repeat, so leave it out; fix and resend any other refused topic once, or leave it out.
6. Answer with how many topics were added, and their titles.

You only add topics: posts are made from them on their day. {_NEVER_PUBLISH}"""

# The prompts earlier seeds wrote, by Playbook and step. A marketplace row that still holds one
# was never curated, so every boot brings it up to date (PRD-251C US-C105); a curated prompt stays.
# Workspace copies are never touched: history reaches them through platform_get_social_plan.
SEEDED_BEFORE: Dict[str, Dict[str, Tuple[str, ...]]] = {
    RESEARCH_PLAYBOOK_TEMPLATE_ID: {"research": (_RESEARCH_PROMPT_251B,)},
}

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
    {
        "template_id": RESEARCH_PLAYBOOK_TEMPLATE_ID,
        "name": "Content bank research",
        "description": (
            "The Social Media Director fills a Socials plan's content bank: ideas from your knowledge, "
            "Deliverables, website and GitHub, every fact with its source. Each plan runs it weekly, and "
            "on Research again."
        ),
        "icon": "🔎",
        "inputs": {
            "plan_id": {"type": "string", "required": True, "description": "The Socials plan to research for"},
            "plan_name": {"type": "string", "required": False, "default": "", "description": "The plan's name"},
        },
        "outputs": {"topics": {"type": "string", "description": "The topics added to the bank"}},
        "steps": (_agent_step("research", 1, DIRECTOR, "topics", _RESEARCH_PROMPT),),
        "tags": ["socials", "plan", "research", "content bank"],
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
    """Create the Socials agents and Playbooks that are missing, link each agent's
    built-in skills that exist now, and put the seed's persona back on every seeded
    marketplace agent whose persona differs from it (P251W1-RVW-1): the Socials
    roster's and the Shopify roster's, since this is the marketplace seed every boot
    runs and the Shopify seeder is not. Returns what happened, by slug; the caller
    commits."""
    agents: Dict[str, Agent] = {}
    outcome: Dict[str, Any] = {"agents": {}, "skills": {}, "playbooks": {}}
    for spec in SOCIALS_AGENTS:
        agent, created = _ensure_agent(db, spec)
        agents[spec["slug"]] = agent
        outcome["agents"][spec["slug"]] = CREATED if created else PRESENT
        outcome["skills"][spec["slug"]] = _link_skills(db, agent, spec["skills"])
    for spec in SOCIALS_PLAYBOOKS:
        outcome["playbooks"][spec["template_id"]] = _ensure_playbook(db, spec, agents)
    # Imported here, not at the top: importing this module never loads the Shopify
    # seed script (its own module sets up logging and sys.path for the command line).
    from core.seeds.seed_shopify_agents import restore_shopify_personas

    outcome["personas_restored"] = restore_seeded_personas(db, SOCIALS_PERSONAS) + restore_shopify_personas(db)
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
            return _refresh(row, spec)
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


def _refreshed_step(step: Any, before: Mapping[str, Tuple[str, ...]], current: Mapping[str, Any]) -> Any:
    if isinstance(step, dict) and step.get("prompt_template") in before.get(step.get("step_id"), ()):
        return {**step, "prompt_template": current[step["step_id"]]}
    return step


def refreshed_steps(spec: Mapping[str, Any], steps: Any) -> Optional[List[Dict[str, Any]]]:
    """A marketplace row's ``steps`` with each prompt an earlier seed wrote (``SEEDED_BEFORE``)
    replaced by the seed's own now, as new dicts; None when nothing is out of date. A curated
    prompt is never touched."""
    before = SEEDED_BEFORE.get(spec["template_id"])
    if not before or not isinstance(steps, list):
        return None
    current = {step["step_id"]: step.get("prompt_template") for step in spec["steps"]}
    out = [_refreshed_step(step, before, current) for step in steps]
    return out if out != steps else None


def _refresh(row: WorkflowTemplate, spec: Mapping[str, Any]) -> str:
    steps = refreshed_steps(spec, row.steps)
    if steps is None:
        return PRESENT
    row.steps = steps
    logger.info("Socials package: marketplace Playbook %s takes the seed's new prompt", spec["template_id"])
    return UPDATED


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
