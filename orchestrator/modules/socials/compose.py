"""PRD-251 S2.2a (US-207): the composer turns a brief into a draft proposal.

ONE call to the workspace's model, through the platform's LLM manager (usage is
tracked under ``request_type='socials_compose'``), with what the workspace
already has: the brand kit's voice, its social templates and their variables,
the connected channels and their copy limits, candidate sources for the brief,
and the built-in Socials skills when present. The proposal is NOT saved: the
person reviews it in the composer, then saves the draft.

The model's JSON is never trusted (``compose_checks.py``): a template must be one
of the workspace's, variables must fit its schema, a source must be one of the
candidates (the model cannot invent a figure's source), and each channel's copy
fits its limits. JSON the model gets wrong is asked for once more; a second
failure is :class:`ComposeFailed` (the API answers 502). A call that outlasts
``SOCIALS_COMPOSE_TIMEOUT_SECONDS`` is :class:`ComposeTimedOut` (504).

An answer that leaves its template's required variables empty (a video template has
dozens) is followed up (``_filled``): those variables are asked for by name, at most
``FILL_BATCH`` a call so each answer fits the output budget, and merged in. A post Auto
writes is then one the render can make; what the model still leaves empty stays in
the warnings ("No value yet for: ...").
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from core.social_templates import resolve_variables
from modules.socials import compose_checks, compose_facts
from modules.socials.copy_limits import limits_for

logger = logging.getLogger(__name__)

SERVICE_NAME = "socials"
REQUEST_TYPE = "socials_compose"
ATTEMPTS = 2  # the first answer, and one retry when its JSON is unusable
SKILL_NAMES = ("social-brand-voice", "social-template-payloads")
SKILL_MAX_CHARS = 6000
TEXT_FORMAT = "text"
WORDS_PER_SECOND = 2.5  # the editor's rule of thumb for a spoken line (PRD-251B mockup)
VOICE_EXAMPLES_NOTE = (
    "voice_examples are posts you drafted and the owner rewrote before approving them: write as the "
    "approved versions do (their length, their words, what they leave out), never as your drafts did."
)
RECENT_OPENINGS_NOTE = (
    "recent_openings are how this workspace's last posts began: open this one differently, and never "
    "reuse their hook."
)
VISUAL_PROMPTS_NOTE = (
    'Also answer "visual_prompts": {"<slot>": "..."}, one for each slot in visual_slots: what its image '
    "or footage shows, in one or two sentences, following brand_style when it is given. No words, letters, "
    "numbers or logos in it: every word on screen is template text."
)
_FENCED = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.S)

FILL_BATCH = 25  # the most variables one follow-up asks for: its answer stays inside the output budget
FILL_NOTE = (
    "Your answer left these variables of the template you chose without a value, and the post cannot be "
    'made without them. Answer with ONE JSON object only, shaped {"variables": {"<name>": "<value>"}}, giving '
    "each of them a value that fits its schema below, from the brief and in the brand voice. The rules above "
    "still hold: never invent a source, a URL or a number."
)
# F378 (night 11): a placeholder left in the copy is asked about once, with the reason.
COPY_FIX_NOTE = (
    "Your copy holds template placeholders, which a post never shows (listed below). Answer with ONE JSON "
    'object only, shaped {"copy": {"base": "...", "channels": {"<toolkit>": "..."}}}: the same copy with each '
    "placeholder replaced by the real words the brief gives. Where the brief does not give the fact a placeholder "
    "stands for, rewrite the sentence without it: never make the fact up."
)
# F378: the skills write tool calls with {braces} where a value goes; a post never does.
SKILLS_NOTE = (
    "The skills below write examples with {braces} or [brackets] where a value goes. A post never carries "
    "braces, brackets, field names or the words true, false or null as text: write the real words."
)
RETRY_NOTE = (
    "Your answer was not the JSON object asked for. Answer again with ONLY that "
    "JSON object, no prose and no code fence."
)


class ComposeFailed(Exception):
    """The model did not give a usable proposal (502)."""


class ComposeTimedOut(Exception):
    """The model did not answer in time (504)."""


@dataclass(frozen=True)
class ComposeContext:
    """Everything the composer knows, gathered from the caller's workspace."""

    brief: str
    format: Optional[str]
    channels: Sequence[Mapping[str, Any]]  # {toolkit, label}: the selected, connected channels
    templates: Sequence[Mapping[str, Any]]  # {id, name, format, sizes, variables_schema}
    candidates: Sequence[Mapping[str, Any]]  # sources.search() candidates
    voice: Mapping[str, Any] = field(default_factory=dict)  # the brand kit's voice
    skills: Mapping[str, str] = field(default_factory=dict)  # built-in skill name → its text
    warnings: Sequence[str] = ()  # what gathering already had to say
    # PRD-251B (B5, US-B103): the editor's choices. A chosen template is the ONLY one
    # listed and is used whatever the model answers; a chosen length is one the template
    # declares, and the model gets its spoken-word budget. A text post has no template.
    template_id: Optional[str] = None
    length_seconds: Optional[int] = None
    # PRD-251B (B9, US-B303): the brand kit's style profile, as a paragraph; empty without one.
    style: str = ""
    # PRD-251B (US-B305): the template slots an AI tool fills for this post ({slot, kind, label}),
    # each needing a prompt from the model (``visual_prompts``); none for most posts.
    visual_slots: Tuple[Mapping[str, str], ...] = ()
    # PRD-251C (C5, US-C105): how the workspace's last posts began, newest first.
    recent_openings: Tuple[str, ...] = ()
    # PRD-251C (C8, US-C406): copy Auto drafted and the copy the owner approved instead, newest first.
    voice_examples: Tuple[Mapping[str, str], ...] = ()


# ── the prompt ──────────────────────────────────────────────────────────────
_ANSWER_SHAPE = {
    "title": "a short working title",
    "copy": {"base": "the text every channel starts from", "channels": {"<toolkit>": "that channel's text"}},
    "format": "one of video, image, carousel, fact_card, infographic, text",
    "template_id": "the id of ONE template listed (null for a text post)",
    "variables": {"<variable name>": "its value"},
    "sources": {"<claim variable name>": {"kind": "<candidate kind>", "ref": "<candidate ref>"}},
}


def _channel_lines(ctx: ComposeContext) -> List[Dict[str, Any]]:
    return [{**dict(c), "limits": limits_for(str(c.get("toolkit")))} for c in ctx.channels]


def _candidate_lines(ctx: ComposeContext) -> List[Dict[str, Any]]:
    keys = ("kind", "ref", "title", "detail", "value", "as_of")
    return [{k: c.get(k) for k in keys if c.get(k) is not None} for c in ctx.candidates]


def _system(ctx: ComposeContext) -> str:
    parts = [
        "You draft social media posts for this workspace. Answer with ONE JSON object only, shaped:",
        json.dumps(_ANSWER_SHAPE),
        "Write every channel listed a text of its own within its limits. Choose a template from the list "
        "and give each of its variables a value that fits its schema. A variable marked claim is a fact: "
        "bind it to one of the candidate sources by kind and ref, or leave it out of sources. Never invent "
        "a source, a URL or a number. Follow the brand voice: use its tone and never its banned phrases.",
    ]
    if ctx.template_id:
        parts.append("The template is chosen: use the one template listed, and no other.")
    if ctx.length_seconds:
        parts.append(
            f"The video is {ctx.length_seconds} seconds long: fit the spoken words into the budget given "
            "(spoken_words_budget) and keep every on-screen line short."
        )
    if ctx.format == TEXT_FORMAT:
        parts.append("This is a text-only post: no template, no variables and no image; write the copy only.")
    if ctx.visual_slots:
        parts.append(VISUAL_PROMPTS_NOTE)
    if ctx.recent_openings:
        parts.append(RECENT_OPENINGS_NOTE)
    if ctx.voice_examples:
        parts.append(VOICE_EXAMPLES_NOTE)
    if ctx.skills:
        parts.append(SKILLS_NOTE)
    for name, text in ctx.skills.items():
        parts.append(f"## Skill: {name}\n{text[:SKILL_MAX_CHARS]}")
    return "\n\n".join(parts)


def build_messages(ctx: ComposeContext) -> List[Dict[str, str]]:
    """The one conversation the model is asked (system, then the workspace's material)."""
    material = {
        "brief": ctx.brief,
        "format": ctx.format,
        "brand_voice": dict(ctx.voice),
        "channels": _channel_lines(ctx),
        "templates": [dict(t) for t in ctx.templates],
        "candidate_sources": _candidate_lines(ctx),
    }
    if ctx.template_id:
        material["template_id"] = ctx.template_id
    if ctx.length_seconds:
        material["length_seconds"] = ctx.length_seconds
        material["spoken_words_budget"] = int(round(ctx.length_seconds * WORDS_PER_SECOND))
    if ctx.style:
        material["brand_style"] = ctx.style  # every image and footage prompt follows it (US-B303)
    if ctx.visual_slots:
        material["visual_slots"] = [dict(slot) for slot in ctx.visual_slots]
    if ctx.recent_openings:
        material["recent_openings"] = list(ctx.recent_openings)
    if ctx.voice_examples:
        material["voice_examples"] = [dict(example) for example in ctx.voice_examples]
    return [
        {"role": "system", "content": _system(ctx)},
        {"role": "user", "content": json.dumps(material, default=str, ensure_ascii=False)},
    ]


# ── the model ───────────────────────────────────────────────────────────────
def llm_factory(workspace_id: Any) -> Callable[[], Any]:
    """The workspace's model, through the platform's LLM manager: usage is tracked
    as ``socials_compose`` for the workspace."""

    def build() -> Any:
        from core.llm import create_llm_manager

        return create_llm_manager(service_name=SERVICE_NAME, workspace_id=workspace_id, request_type=REQUEST_TYPE)

    return build


# ── the answer ──────────────────────────────────────────────────────────────
def parse_answer(text: Any) -> Optional[Dict[str, Any]]:
    """The JSON object in the model's answer (bare, or in a code fence), or ``None``."""
    if not isinstance(text, str) or not text.strip():
        return None
    body = text.strip()
    fenced = _FENCED.search(body)
    if fenced:
        body = fenced.group(1)
    elif not body.startswith("{"):
        start, end = body.find("{"), body.rfind("}")
        body = body[start:end + 1] if 0 <= start < end else body
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _ask(llm: Any, messages: List[Dict[str, str]], timeout: float) -> Optional[Dict[str, Any]]:
    try:
        response = await asyncio.wait_for(llm.generate_response(messages), timeout=timeout)
    except asyncio.TimeoutError:
        raise ComposeTimedOut(f"The model did not answer within {timeout:g} seconds. Try again.") from None
    return parse_answer(getattr(response, "content", None))


def _missing(proposal: Mapping[str, Any]) -> List[str]:
    """The chosen template's variables with neither a value nor a default: the render's own test."""
    schema = (proposal.get("template") or {}).get("variables_schema") or {}
    supplied = {name: spec.get("value") for name, spec in (proposal.get("variables") or {}).items()}
    return resolve_variables(schema, supplied).missing if schema else []


async def _fills(llm: Any, history: List[Dict[str, str]], schema: Mapping[str, Any], names: Sequence[str],
                 timeout: float) -> Dict[str, Any]:
    """The values one follow-up gives for ``names``; nothing when its answer is unusable or late."""
    ask = FILL_NOTE + "\n" + json.dumps({name: schema[name] for name in names}, ensure_ascii=False, default=str)
    try:
        raw = await _ask(llm, [*history, {"role": "user", "content": ask}], timeout)
    except ComposeTimedOut:
        logger.warning("[Socials] compose follow-up for %d variables timed out", len(names))
        return {}
    values = raw.get("variables") if isinstance(raw, dict) else None
    return {name: values[name] for name in names if name in values} if isinstance(values, dict) else {}


async def _filled(raw: Mapping[str, Any], proposal: Dict[str, Any], ctx: ComposeContext, llm: Any,
                  messages: List[Dict[str, str]], timeout: float) -> Dict[str, Any]:
    """The proposal with the required variables its answer left empty asked for by name
    (``FILL_BATCH`` a call), merged over the answer (a value it gave that holds is never
    replaced), and the whole checked again."""
    missing = _missing(proposal)
    if not missing:
        return proposal
    schema = proposal["template"]["variables_schema"]
    history = [*messages, {"role": "assistant", "content": json.dumps(raw, ensure_ascii=False, default=str)}]
    fills: Dict[str, Any] = {}
    for start in range(0, len(missing), FILL_BATCH):
        fills.update(await _fills(llm, history, schema, missing[start:start + FILL_BATCH], timeout))
    if not fills:
        return proposal
    given = raw.get("variables") if isinstance(raw.get("variables"), dict) else {}
    merged = {**raw, "template_id": proposal["template_id"], "variables": {**given, **fills}}
    return compose_checks.checked_proposal(merged, ctx)


async def _copy_fixed(raw: Dict[str, Any], proposal: Dict[str, Any], ctx: ComposeContext, llm: Any,
                      messages: List[Dict[str, str]], timeout: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """F378: the answer and its proposal, with the copy asked for once more when it holds a
    template placeholder; an unusable or late answer leaves both as they were (the placeholder
    stays a warning and an owner's question)."""
    found = compose_facts.copy_placeholders(proposal["copy"])
    if not found:
        return raw, proposal
    listed = {"copy": proposal["copy"], "placeholders": [f"{where}: {placeholder}" for where, placeholder in found]}
    history = [*messages, {"role": "assistant", "content": json.dumps(raw, ensure_ascii=False, default=str)}]
    ask = COPY_FIX_NOTE + "\n" + json.dumps(listed, ensure_ascii=False)
    try:
        answer = await _ask(llm, [*history, {"role": "user", "content": ask}], timeout)
    except ComposeTimedOut:
        logger.warning("[Socials] compose follow-up for the copy's placeholders timed out")
        return raw, proposal
    copy = answer.get("copy") if isinstance(answer, dict) else None
    if not isinstance(copy, dict):
        return raw, proposal
    fixed = {**raw, "copy": copy}
    return fixed, compose_checks.checked_proposal(fixed, ctx)


async def propose(ctx: ComposeContext, llm_factory: Callable[[], Any], timeout: float) -> Dict[str, Any]:
    """The proposal for ``ctx``: one call, one retry when the JSON is unusable, the copy asked
    for again when it holds a placeholder (F378), and a follow-up for the template's required
    variables the answer left empty."""
    llm = llm_factory()
    messages = build_messages(ctx)
    for attempt in range(ATTEMPTS):
        raw = await _ask(llm, messages, timeout)
        if raw is not None:
            raw, proposal = await _copy_fixed(raw, compose_checks.checked_proposal(raw, ctx), ctx, llm, messages, timeout)
            return await _filled(raw, proposal, ctx, llm, messages, timeout)
        logger.warning("[Socials] compose answer %d was not JSON", attempt + 1)
        messages = [*messages, {"role": "user", "content": RETRY_NOTE}]
    raise ComposeFailed("The model's answer could not be read as a proposal. Try again, or start from a blank draft.")
