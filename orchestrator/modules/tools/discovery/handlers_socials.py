"""Socials post handlers (PRD-251 US-116, S4.1): agents draft posts; people approve them.

Thin wrappers over what the /api/socials routes run: the flows in
``api/socials.py`` (``create_post``, ``edit_post``, ``submit_post``,
``render_post``) and the service's workspace-scoped reads. An agent's post
therefore passes every check a person's does (its template, voice and footage,
its sources resolved in the workspace, the music credit), moves only by the
status machine, and is committed by the same compare-and-set. The fields are
the REST bodies' own, validated by the same models, plus three:

* ``template``: the social template by id, or by name (its latest version);
* ``chart_report``: a chart template's rows filled from a report of the
  workspace (``report_charts.bind_report``, S1.7). An agent never types a
  chart's rows or its source chip;
* ``render``: render the post's template now: the US-104 render, within the
  plan's render minutes. Create renders by default, update when asked.

They also take the simpler shapes the Socials skills write (copy as
``{channel: text}``, a variable as its bare value, sources as a list of
``{"claim", "kind", "ref"}``) and store the post's own. An update merges what
it sends into what the post has, so an agent sends only what changes. A
variable the template marks as a claim is one whenever it has a value (D7).
Submit takes a ``note`` for the reviewer.

Nothing here approves, schedules or publishes (D6, D14). With Socials off for
the workspace (either switch, D1), every tool refuses and reads or writes
nothing. Each tool acts as its agent (``_agent_id`` / ``_agent_name``, minted
by the server): ``agent:<id>`` is the actor of every ``review_log`` entry it
causes, and a draft or an edit names the agent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from modules.tools.discovery.actions_socials import LIST_DEFAULT_LIMIT, LIST_MAX_LIMIT

logger = logging.getLogger(__name__)

AGENT_ACTOR = "agent"
CHART_REPORT_KEYS = ("report_id", "chart", "column", "part")
SUBMIT_FIELDS = frozenset({"post_id", "note"})

RENDER_STARTED = (
    "Rendering in the background: when the render finishes the post waits for approval in the "
    "Socials tab, or is failed with the reason in its history (platform_get_social_post)."
)
DRAFT_SAVED = "Saved as a draft: send it for approval with platform_submit_social_post when it is ready."
SENT_FOR_APPROVAL = (
    "Sent for approval: a person approves it, asks for changes or rejects it in the Socials tab, "
    "and the platform publishes an approved post."
)


class _Refused(Exception):
    """A call refused before anything is saved; its message is the answer."""


def _refused(message: str, **extra: Any) -> Dict[str, Any]:
    return {"success": False, "error": message, **extra}


def _refusal(exc: Exception) -> Dict[str, Any]:
    """A lifecycle error as the tool's answer, with what the REST 4xx carries."""
    extra: Dict[str, Any] = {}
    if getattr(exc, "names", None):
        extra["claims"] = exc.names
    if getattr(exc, "unresolved", None):
        extra["unresolved"] = exc.unresolved
    if getattr(exc, "current_hash", None):
        extra["content_hash"] = exc.current_hash
    return _refused(str(exc), **extra)


def _agent(params: Dict[str, Any]) -> Tuple[str, str]:
    """The acting agent, as the server minted it: the review_log actor
    ``agent:<id>``, and the name a draft or an edit shows."""
    agent_id = params.get("_agent_id")
    name = params.get("_agent_name")
    actor = f"{AGENT_ACTOR}:{agent_id}" if agent_id else AGENT_ACTOR
    return actor, str(name or (f"agent {agent_id}" if agent_id else "an agent"))


def _fields(params: Dict[str, Any]) -> Dict[str, Any]:
    """The model's own fields: keys starting with "_" are the server's."""
    return {key: value for key, value in params.items() if not key.startswith("_")}


def _open(db: Session, workspace_id: UUID) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """The workspace, or the refusal when Socials is off for it (D1)."""
    from core.models.workspaces import Workspace
    from modules.socials.settings import socials_off_reason

    workspace = db.get(Workspace, workspace_id)
    reason = socials_off_reason(workspace)
    return (workspace, None) if reason is None else (None, _refused(reason, socials_off=True))


def _post(db: Session, workspace_id: UUID, post_id: Any) -> Any:
    from modules.socials import service

    if post_id in (None, ""):
        raise _Refused("Missing required parameter: post_id")
    try:
        key = UUID(str(post_id))
    except (TypeError, ValueError, AttributeError):
        raise _Refused(f"post_id {post_id!r} is not a post's id: platform_list_social_posts lists them.") from None
    post = service.get_post(db, workspace_id, key)
    if post is None:
        raise _Refused("Post not found in this workspace: platform_list_social_posts lists them.")
    return post


def _flag(fields: Dict[str, Any], name: str, default: bool) -> bool:
    value = fields.pop(name, None)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise _Refused(f"{name} must be true or false.")
    return value


def _template(db: Session, workspace_id: UUID, ref: Any) -> Any:
    """The social template ``ref`` names in the workspace: its id, or its name
    (the latest active version), as ``generate_document`` resolves one."""
    from core.social_templates import is_social_format
    from modules.documents.template_service import DocumentTemplateService

    templates = DocumentTemplateService(db)
    text = str(ref).strip()
    try:
        template = templates.get_template(UUID(text), workspace_id)
    except ValueError:
        template = templates.get_template_by_name(workspace_id, text)
    if template is None:
        raise _Refused(
            f"No template {text!r} in this workspace: platform_list_templates lists them "
            "(format social_video or social_image)."
        )
    if not is_social_format(template.format):
        raise _Refused(
            f"{template.name!r} is a {template.format} template: a post takes a social_video or "
            "social_image template."
        )
    return template


def _template_ref(fields: Dict[str, Any]) -> Any:
    """``template``, else ``template_id`` (the name other tools give it)."""
    ref = fields.pop("template", None)
    alias = fields.pop("template_id", None)
    return ref if ref not in (None, "") else alias


def _value(variable: Any) -> str:
    value = variable.get("value") if isinstance(variable, dict) else None
    return str(value).strip() if value is not None else ""


# ---------------------------------------------------------------------------
# An agent's fields as the post keeps them. The tools take the post's own shapes
# and the simpler ones the Socials skills write: copy as {channel: text}, a
# variable as its bare value, sources as a list of {"claim", "kind", "ref"}.
# An update merges what it sends into what the post has, key by key, so an
# agent sends only what changes (null drops a key).
# ---------------------------------------------------------------------------


def _schema_claims(template: Any) -> set:
    """The variables the template marks as claims: its facts and figures (D7)."""
    from core.social_templates import claim_names

    blocks = template.blocks if template is not None and isinstance(template.blocks, dict) else {}
    schema = blocks.get("variables_schema")
    return set(claim_names(schema)) if isinstance(schema, dict) else set()


def _copy_of(value: Any) -> Any:
    """``{"base", "channels"}`` from that shape or from ``{channel: text}``."""
    if not isinstance(value, dict):
        return value  # the service's validator says what is wrong with it
    copy: Dict[str, Any] = {}
    channels: Dict[str, Any] = {}
    for key, text in value.items():
        if key == "base":
            copy["base"] = text
        elif key == "channels" and isinstance(text, dict):
            channels.update(text)
        else:
            channels[key] = text
    if channels:
        copy["channels"] = channels
    return copy


def _variables_of(value: Any) -> Any:
    """``{name: {"value", "claim"}}`` from that shape or from ``{name: value}``
    (a variable's value is text, a number or a boolean, never an object)."""
    if not isinstance(value, dict):
        return value
    return {
        name: spec if spec is None or (isinstance(spec, dict) and "value" in spec) else {"value": spec, "claim": False}
        for name, spec in value.items()
    }


def _sources_of(value: Any) -> Any:
    """``{claim: {"kind", "ref", "as_of"}}`` from that shape or from a list of
    ``{"claim", "kind", "ref", "as_of"}``, each keyed by the variable it backs."""
    if not isinstance(value, list):
        return value
    sources: Dict[str, Any] = {}
    for n, item in enumerate(value):
        claim = item.get("claim") if isinstance(item, dict) else None
        if not isinstance(claim, str) or not claim.strip():
            raise _Refused(f"sources[{n}] needs claim: the name of the variable it backs.")
        sources[claim.strip()] = {key: part for key, part in item.items() if key != "claim"}
    return sources


def _merged(stored: Any, sent: Any) -> Any:
    """``sent`` merged into ``stored`` key by key; a key sent as null is dropped."""
    if not isinstance(sent, dict):
        return sent
    merged = dict(stored) if isinstance(stored, dict) else {}
    for key, value in sent.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def _merged_copy(stored: Any, sent: Any) -> Any:
    """``stored`` copy with ``sent`` merged in: its base when sent, its channels
    key by key; a channel sent as "" or null drops its own text."""
    if not isinstance(sent, dict):
        return sent
    before = stored if isinstance(stored, dict) else {}
    copy = {"base": sent["base"]} if "base" in sent else ({"base": before["base"]} if "base" in before else {})
    channels = _merged(before.get("channels"), sent.get("channels") or {})
    channels = {name: text for name, text in channels.items() if text != ""}
    if channels:
        copy["channels"] = channels
    return copy


def _claimed(variables: Any, claims: set) -> Any:
    """``variables`` with each one the template marks as a claim, and that has a
    value, marked a claim (D7): an agent never un-claims a figure."""
    if not isinstance(variables, dict) or not claims:
        return variables
    return {
        name: {**spec, "claim": True} if name in claims and isinstance(spec, dict) and _value(spec) else spec
        for name, spec in variables.items()
    }


def _typed_chart_rows(template: Any, variables: Any, stored: Any) -> List[str]:
    """The chart variables (rows and source chip) ``variables`` sets to text other
    than the post already has: an agent never types a chart's rows (S1.7)."""
    from core.chart_binding import spec_of

    spec = spec_of(template.blocks) if template is not None else None
    if spec is None or not isinstance(variables, dict):
        return []
    names = {spec.source}
    for n in range(1, spec.rows + 1):
        names.update((spec.label_name(n), spec.value_name(n)))
    kept = stored if isinstance(stored, dict) else {}
    return sorted(
        name for name in names & set(variables)
        if _value(variables[name]) and _value(variables[name]) != _value(kept.get(name))
    )


def _refuse_typed_rows(template: Any, variables: Any, stored: Any) -> None:
    typed = _typed_chart_rows(template, variables, stored)
    if typed:
        raise _Refused(
            f"{', '.join(typed)}: a chart's rows and source chip come from a report of this "
            "workspace, never typed. Send chart_report with the report's id instead."
        )


async def _charted(db: Session, workspace_id: UUID, template: Any, chart: Any, variables: Any, sources: Any):
    """``variables`` and ``sources`` with the chart of ``template`` filled from a
    report (``report_charts.bind_report``): its rows, each figure bound to it."""
    from core.chart_binding import spec_of
    from core.social_templates import SocialTemplateError, validate_social_blocks
    from modules.socials import report_charts

    if not isinstance(chart, dict) or not chart.get("report_id"):
        raise _Refused("chart_report needs report_id: the id of a report of this workspace (platform_browse_reports).")
    unknown = sorted(set(chart) - set(CHART_REPORT_KEYS))
    if unknown:
        raise _Refused(f"chart_report takes {', '.join(CHART_REPORT_KEYS)}; not {', '.join(unknown)}.")
    if template is None:
        raise _Refused("chart_report fills a chart template: name the template too.")
    try:
        blocks = validate_social_blocks(template.blocks, template.format)
    except SocialTemplateError as exc:
        raise _Refused(f"The template {template.name!r} cannot be used: {exc}") from exc
    binding = await report_charts.bind_report(
        db, workspace_id, chart["report_id"], blocks,
        chart=chart.get("chart"), column=chart.get("column"), part=chart.get("part"),
    )
    figures = set(spec_of(blocks).value_names())
    kept_variables = {k: v for k, v in (variables or {}).items() if k not in binding.variables}
    kept_sources = {k: v for k, v in (sources or {}).items() if k not in figures}
    return {**kept_variables, **binding.variables}, {**kept_sources, **binding.sources}


def _invalid(exc: Exception) -> str:
    """A request model's ValidationError, read as one line (no ctx: it may hold the raised error)."""
    problems = "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'post'}: {error['msg']}"
        for error in exc.errors(include_context=False, include_url=False)
    )
    return f"Invalid post, nothing saved. {problems}"


async def _render(db: Session, workspace: Any, post: Any, actor: str) -> Dict[str, Any]:
    """The US-104 render of ``post``, started now; a refusal changes nothing."""
    from api import socials as socials_api
    from core import media_render_quota as render_quota
    from modules.socials import service

    try:
        saved = await socials_api.render_post(db, workspace, post, actor)
    except (service.SocialsError, render_quota.RenderQuotaExceeded) as exc:
        db.rollback()
        return {"render": {"started": False, "error": str(exc)}}
    return {"post": saved, "render": {"started": True}, "message": RENDER_STARTED}


def _not_rendered(result: Dict[str, Any]) -> Dict[str, Any]:
    render = result.get("render") or {}
    if render.get("started") is False:
        return {**result, "message": f"The post was saved but not rendered: {render['error']}"}
    return result


async def create_social_post(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """A new draft, through ``create_post``; then its render unless ``render`` is false."""
    from pydantic import ValidationError

    from api import socials as socials_api
    from modules.socials import service

    workspace, refusal = _open(db, workspace_id)
    if refusal:
        return refusal
    actor, agent = _agent(params)
    fields = _fields(params)
    try:
        render_now = _flag(fields, "render", True)
        chart = fields.pop("chart_report", None)
        ref = _template_ref(fields)
        template = _template(db, workspace_id, ref) if ref not in (None, "") else None
        if "copy" in fields:
            fields["copy"] = _merged_copy({}, _copy_of(fields["copy"]))
        if "variables" in fields:
            fields["variables"] = _merged({}, _variables_of(fields["variables"]))
        if "sources" in fields:
            fields["sources"] = _merged({}, _sources_of(fields["sources"]))
        _refuse_typed_rows(template, fields.get("variables"), None)
        if chart is not None:
            fields["variables"], fields["sources"] = await _charted(
                db, workspace_id, template, chart, fields.get("variables"), fields.get("sources")
            )
        if "variables" in fields:
            fields["variables"] = _claimed(fields["variables"], _schema_claims(template))
        body = socials_api.CreateSocialPostRequest.model_validate(
            {**fields, "template_id": template.id if template is not None else None}
        )
        post = await socials_api.create_post(
            db, workspace_id, actor, body.model_dump(by_alias=True), agent=agent
        )
    except _Refused as exc:
        return _refused(str(exc))
    except ValidationError as exc:
        return _refused(_invalid(exc))
    except service.SocialsError as exc:
        db.rollback()
        return _refusal(exc)
    result: Dict[str, Any] = {"success": True, "post": post.to_dict(), "message": DRAFT_SAVED}
    if render_now:
        result = _not_rendered({**result, **await _render(db, workspace, post, actor)})
    return result


async def update_social_post(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Change a post, through ``edit_post``; then render it again when ``render`` is true."""
    from pydantic import ValidationError

    from api import socials as socials_api
    from modules.documents.template_service import DocumentTemplateService
    from modules.socials import service

    workspace, refusal = _open(db, workspace_id)
    if refusal:
        return refusal
    actor, agent = _agent(params)
    fields = _fields(params)
    try:
        post = _post(db, workspace_id, fields.pop("post_id", None))
        render_now = _flag(fields, "render", False)
        chart = fields.pop("chart_report", None)
        ref = _template_ref(fields)
        if ref not in (None, ""):
            template = _template(db, workspace_id, ref)
            fields["template_id"] = template.id
        elif post.template_id is not None:
            template = DocumentTemplateService(db).get_template(post.template_id, workspace_id)
        else:
            template = None
        if "copy" in fields:
            fields["copy"] = _merged_copy(post.copy, _copy_of(fields["copy"]))
        if "variables" in fields:
            sent = _variables_of(fields["variables"])
            _refuse_typed_rows(template, sent, post.variables)
            fields["variables"] = _merged(post.variables, sent)
        if "sources" in fields:
            fields["sources"] = _merged(post.sources, _sources_of(fields["sources"]))
        if chart is not None:
            fields["variables"], fields["sources"] = await _charted(
                db, workspace_id, template, chart,
                fields.get("variables", post.variables), fields.get("sources", post.sources),
            )
        if "variables" in fields or "template_id" in fields:
            fields["variables"] = _claimed(fields.get("variables", post.variables), _schema_claims(template))
        changes = socials_api.UpdateSocialPostRequest.model_validate(fields).model_dump(
            exclude_unset=True, by_alias=True
        )
        if not changes and not render_now:
            raise _Refused("Nothing to change: send the fields to change, or render true.")
        saved = await socials_api.edit_post(db, post, actor, changes, agent=agent) if changes else post.to_dict()
    except _Refused as exc:
        return _refused(str(exc))
    except ValidationError as exc:
        return _refused(_invalid(exc))
    except service.SocialsError as exc:
        db.rollback()
        return _refusal(exc)
    result: Dict[str, Any] = {"success": True, "post": saved}
    if render_now:
        result = _not_rendered({**result, **await _render(db, workspace, post, actor)})
    return result


async def submit_social_post(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """draft or changes_requested → needs_approval, through ``submit_post``; the
    note, when sent, tells the reviewer what to look at."""
    from api import socials as socials_api
    from modules.socials import service

    _, refusal = _open(db, workspace_id)
    if refusal:
        return refusal
    actor, _ = _agent(params)
    unknown = sorted(set(_fields(params)) - SUBMIT_FIELDS)
    if unknown:
        return _refused(f"platform_submit_social_post takes post_id and note; not {', '.join(unknown)}. Nothing was sent.")
    try:
        post = _post(db, workspace_id, params.get("post_id"))
        saved = socials_api.submit_post(db, post, actor, note=params.get("note"))
    except _Refused as exc:
        return _refused(str(exc))
    except service.SocialsError as exc:
        db.rollback()
        return _refusal(exc)
    return {"success": True, "post": saved, "message": SENT_FOR_APPROVAL}


async def get_social_post(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """One post of the workspace, as GET /api/socials/posts/{id} returns it."""
    _, refusal = _open(db, workspace_id)
    if refusal:
        return refusal
    try:
        post = _post(db, workspace_id, params.get("post_id"))
    except _Refused as exc:
        return _refused(str(exc))
    return {"success": True, "post": post.to_dict()}


def _statuses(value: Any) -> Optional[List[str]]:
    """One status, several (a list, or comma-separated), or none (every status)."""
    from core.models.socials import SOCIAL_POST_STATUSES

    if value in (None, "", []):
        return None
    items = value.split(",") if isinstance(value, str) else value
    if not isinstance(items, list):
        raise _Refused("status must be a list of statuses.")
    statuses = [str(item).strip() for item in items if str(item).strip()]
    unknown = [status for status in statuses if status not in SOCIAL_POST_STATUSES]
    if unknown:
        raise _Refused(f"Unknown status {', '.join(unknown)}: the statuses are {', '.join(SOCIAL_POST_STATUSES)}.")
    return statuses or None


def _limit(value: Any) -> int:
    if value is None:
        return LIST_DEFAULT_LIMIT
    number = int(value) if isinstance(value, str) and value.strip().isdigit() else value
    if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= LIST_MAX_LIMIT:
        raise _Refused(f"limit must be a whole number from 1 to {LIST_MAX_LIMIT}.")
    return number


async def list_social_posts(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """The workspace's posts, newest first, in the statuses asked for."""
    from modules.socials import service

    _, refusal = _open(db, workspace_id)
    if refusal:
        return refusal
    try:
        statuses = _statuses(params.get("status"))
        limit = _limit(params.get("limit"))
    except _Refused as exc:
        return _refused(str(exc))
    posts = service.list_posts(db, workspace_id, statuses=statuses, limit=limit)
    return {"success": True, "posts": [post.to_dict() for post in posts], "count": len(posts), "limit": limit}
