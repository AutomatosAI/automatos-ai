"""PRD-251 S2.2a (US-207): the model's proposal, checked. Its JSON is never trusted.

* **The template** must be one of the workspace's social templates. Another id
  (another workspace's, or made up) is replaced by the workspace's first template
  of the post's kind, or dropped when it has none, with a warning.
* **Variables** keep only the template's own names, each with a value its
  ``variables_schema`` accepts, in the post's shape ``{name: {value, claim}}``;
  ``claim`` comes from the schema, never from the model.
* **Sources** keep only a claim bound to one of the candidates the composer was
  given (D7): the model cannot invent a URL or a figure's source. An unmatched
  claim stays unsourced, and the approval UI shows it so.
* **Copy**: every selected channel gets its own text (the base when the model
  left one out), fitted to the channel's limits at a word boundary. Its shape is the
  one a save takes, ``{"base", "channels"}`` (F378: one shape, no hand translation).
* **Visual prompts** (PRD-251B US-B305): only for the slots the composer was asked
  about, each one line of at most VISUAL_PROMPT_MAX_CHARS.
* **Facts** (F378): what the proposal says is checked last (``compose_facts.py``): no
  placeholder left in the copy, and ``questions`` lists what Auto needs from the owner.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from core.models.socials import SOCIAL_POST_FORMATS
from core.social_templates import SOCIAL_IMAGE, SOCIAL_VIDEO, claim_names, resolve_variables
from modules.socials import compose_facts
from modules.socials.copy_limits import fit_copy, fit_title

VIDEO_FORMAT = "video"
TEXT_FORMAT = "text"
TITLE_FALLBACK_CHARS = 80
VISUAL_PROMPT_MAX_CHARS = 600


def template_kind(post_format: Optional[str]) -> Optional[str]:
    """The template format a post format renders with (video, else an image)."""
    if post_format is None:
        return None
    return SOCIAL_VIDEO if post_format == VIDEO_FORMAT else SOCIAL_IMAGE


def _format(raw: Any, ctx: Any) -> Optional[str]:
    if ctx.format in SOCIAL_POST_FORMATS:
        return ctx.format
    return raw if raw in SOCIAL_POST_FORMATS else None


def _template(raw_id: Any, ctx: Any, post_format: Optional[str], warnings: List[str]) -> Optional[Mapping[str, Any]]:
    by_id = {str(t["id"]): t for t in ctx.templates}
    chosen = getattr(ctx, "template_id", None)
    if chosen and str(chosen) in by_id:
        return by_id[str(chosen)]  # PRD-251B B5: the editor's choice, whatever the model answered
    if raw_id is not None and str(raw_id) in by_id:
        return by_id[str(raw_id)]
    kind = template_kind(post_format)
    fallback = next((t for t in ctx.templates if kind in (None, t.get("format"))), None)
    if fallback is None:
        warnings.append("No social template of this workspace fits: choose one before rendering")
        return None
    if raw_id is not None:
        warnings.append(f"The model named a template that is not this workspace's; {fallback['name']} is used instead")
    else:
        warnings.append(f"No template was chosen; {fallback['name']} is used")
    return fallback


def _supplied(raw: Any) -> Dict[str, Any]:
    """The model's values, taking ``{"value": x}`` as ``x``."""
    if not isinstance(raw, dict):
        return {}
    return {name: (v.get("value") if isinstance(v, dict) else v) for name, v in raw.items()}


def _variables(raw: Any, template: Optional[Mapping[str, Any]], warnings: List[str]) -> Dict[str, Dict[str, Any]]:
    schema = (template or {}).get("variables_schema") or {}
    supplied = _supplied(raw)
    unknown = sorted(name for name in supplied if name not in schema)
    if unknown:
        warnings.append(f"Variables the template does not have were dropped: {', '.join(unknown)}")
    resolved = resolve_variables(schema, {n: v for n, v in supplied.items() if n in schema})
    warnings.extend(f"The value of {problem}; it was dropped" for problem in resolved.invalid)
    if resolved.missing:
        warnings.append(f"No value yet for: {', '.join(resolved.missing)}")
    claims = set(claim_names(schema))
    return {
        name: {"value": value, "claim": name in claims}
        for name, value in resolved.values.items()
        if name in supplied
    }


def _candidate_index(ctx: Any) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    return {(str(c.get("kind")), str(c.get("ref"))): c for c in ctx.candidates}


def _sources(raw: Any, variables: Mapping[str, Any], ctx: Any, warnings: List[str]) -> Dict[str, Dict[str, Any]]:
    claims = sorted(name for name, spec in variables.items() if spec.get("claim"))
    given = raw if isinstance(raw, dict) else {}
    index = _candidate_index(ctx)
    sources: Dict[str, Dict[str, Any]] = {}
    for name in claims:
        source = given.get(name)
        key = (str(source.get("kind")), str(source.get("ref"))) if isinstance(source, dict) else None
        found = index.get(key) if key else None
        if found is not None:
            sources[name] = {"kind": found["kind"], "ref": found["ref"], "as_of": found.get("as_of")}
        elif source is not None:
            warnings.append(f"The source given for {name} is not one found in this workspace; {name} is unsourced")
    unsourced = [name for name in claims if name not in sources]
    if unsourced:
        warnings.append(f"Claims without a source (bind one before approval): {', '.join(unsourced)}")
    return sources


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _copy(raw: Any, ctx: Any, warnings: List[str]) -> Dict[str, Any]:
    copy = raw if isinstance(raw, dict) else {}
    base = _text(copy.get("base"))
    own = copy.get("channels") if isinstance(copy.get("channels"), dict) else {}
    channels: Dict[str, str] = {}
    for channel in ctx.channels:
        toolkit = str(channel["toolkit"])
        text = _text(own.get(toolkit))
        if not text:
            text = base
            warnings.append(f"{toolkit}: no text of its own was written; it starts from the base copy")
        channels[toolkit], fixes = fit_copy(toolkit, text)
        warnings.extend(fixes)
    return {"base": base, "channels": channels}


def _title(raw: Any, ctx: Any, warnings: List[str]) -> str:
    title = _text(raw) or ctx.brief.strip()[:TITLE_FALLBACK_CHARS] or "New post"
    fitted, fixes = fit_title([str(c["toolkit"]) for c in ctx.channels], title)
    warnings.extend(fixes)
    return fitted


def _visual_prompts(value: Any, ctx: Any) -> Dict[str, str]:
    """The model's prompt per slot it was asked about (``ctx.visual_slots``); nothing else."""
    wanted = {str(slot.get("slot")) for slot in getattr(ctx, "visual_slots", ())}
    if not wanted or not isinstance(value, Mapping):
        return {}
    return {
        slot: " ".join(text.split())[:VISUAL_PROMPT_MAX_CHARS]
        for slot, text in value.items()
        if slot in wanted and isinstance(text, str) and text.strip()
    }


def checked_proposal(raw: Mapping[str, Any], ctx: Any) -> Dict[str, Any]:
    """The proposal as the composer shows it, every field checked against ``ctx``."""
    warnings: List[str] = list(ctx.warnings)
    post_format = _format(raw.get("format"), ctx)
    if post_format == TEXT_FORMAT:
        # PRD-251B (US-B103): a text post is copy alone: no template, no variables, no claims.
        template, variables = None, {}
    else:
        template = _template(raw.get("template_id"), ctx, post_format, warnings)
        if post_format is None and template is not None:
            post_format = VIDEO_FORMAT if template.get("format") == SOCIAL_VIDEO else "image"
        variables = _variables(raw.get("variables"), template, warnings)
    proposal = {
        "title": _title(raw.get("title"), ctx, warnings),
        "copy": _copy(raw.get("copy"), ctx, warnings),
        "format": post_format,
        "template_id": str(template["id"]) if template is not None else None,
        # US-208: the template's variables and sizes, which the composer's fields come from.
        "template": dict(template) if template is not None else None,
        "variables": variables,
        "sources": _sources(raw.get("sources"), variables, ctx, warnings),
        "channels": [str(c["toolkit"]) for c in ctx.channels],
        # PRD-251B (B5): the chosen length rides along to the saved post.
        "length_seconds": getattr(ctx, "length_seconds", None),
        "visual_prompts": _visual_prompts(raw.get("visual_prompts"), ctx),
        "warnings": warnings,
        # F378: what Auto needs from the owner before the post can be made ("Auto needs: …").
        "questions": [],
    }
    return compose_facts.checked(proposal, ctx)  # F378: no fact the brief does not give
