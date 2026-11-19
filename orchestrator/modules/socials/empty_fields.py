"""A field left empty is Auto's to write; a field that holds a value stays as it is (16 Nov 2026).

Choosing a template handed the person its whole form: a render with a required field empty
was refused ("fill in … before rendering"), so every field had to be typed by hand, and
"Redraft with Auto" replaced whatever had been typed. The rule is now one line, the same at
a redraft and at a render: an empty field is written by Auto from the brief, and a field with
a value is never rewritten.

* :func:`own_values`: the fields of the post that hold a value.
* :func:`empty_required`: the chosen template's required fields that hold none, at the post's
  length: the render's own test (``core.social_templates.resolve_variables``).
* :func:`fields_edit`: Auto's proposal as the edit: its value for each field the post holds
  none for, the post's own everywhere else (a variable the template does not have, a
  carousel's slide count, stays), and a source only for a field that holds its value.
* :func:`newly_written`: the fields an edit gives a value that the post did not hold.
* :func:`render_refuses`: the required fields with no value at all, the render's own test: a
  field stored blank on purpose still renders blank, as it always did.
* :func:`record_written`: Auto's values applied as an edit, with a line in the history naming
  what Auto wrote (:func:`written_line`).

PRD-251D (US-D103, E2): an empty caption is Auto's to write too, in the same composer call.

* :func:`holds_a_caption`: the caption has words, in its base text or in any channel's own text;
* :func:`caption_wanted`: the caption is Auto's to write: it is empty, the post has a brief to
  write it from, and it holds no approval a new caption would void (a preview of an approved post);
* :func:`caption_edit`: the caption Auto wrote, as the edit's ``copy``, only for a post whose
  caption is empty: a caption that holds words is never rewritten;
* :func:`record_pick`: "Let Auto pick" applied as one edit, its history line naming the caption
  when Auto wrote one.

Pure but for the post it edits; ``api/socials_compose.py`` asks the composer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from core.social_cuts import schema_for_length
from core.social_templates import claim_names, resolve_variables
from modules.socials import service
from modules.socials.template_claims import with_template_claims

GIVEN_TYPES = (str, int, float, bool)
WROTE_FIELDS = "Auto wrote the fields left empty: {fields}."
WROTE_CAPTION = "Auto wrote the caption."
PICKED_WITH_CAPTION = "Auto picked the template {template} and wrote its fields and the caption."
STILL_NEEDED = "Auto could not write these from the brief: {fields}. Say them in the brief, or fill them in yourself."
COPY = "copy"


def _holds_a_value(value: Any) -> bool:
    """Text with words in it, a number or a switch; nothing and blank text are empty."""
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, GIVEN_TYPES)


def unfit_names(schema: Mapping[str, Any], values: Mapping[str, Any]) -> Set[str]:
    """The fields whose value ``schema`` refuses (longer than its limit, the wrong type): a
    render refuses each of them."""
    return {problem.split(" ", 1)[0] for problem in resolve_variables(schema, values).invalid}


def own_values(post: Any, schema: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The post's fields that hold a value, ``{name: value}``: what Auto never rewrites. With the
    chosen template's ``schema``, a value it refuses is not one of them (19 Nov 2026: a post moved
    to another template kept the old one's words, which the new one refused, and every render of
    it was refused): Auto writes that field again."""
    variables = post.variables if isinstance(post.variables, Mapping) else {}
    held = {
        name: spec.get("value") for name, spec in variables.items()
        if isinstance(spec, Mapping) and _holds_a_value(spec.get("value"))
    }
    if not schema:
        return held
    unfit = unfit_names(schema, held)
    return {name: value for name, value in held.items() if name not in unfit}


def empty_required(template: Mapping[str, Any], post: Any) -> List[str]:
    """The required fields of ``template`` (a composer's entry) the post leaves empty, at its
    length, or holds a value the template refuses: the ones a render refuses to start without."""
    schema = schema_for_length(template, getattr(post, "length_seconds", None))
    if not schema:
        return []
    fitting = own_values(post, schema)
    missing = resolve_variables(schema, fitting).missing
    # A refused value in a field with a default is not missing, but the render refuses it all the same.
    unfit = set(own_values(post)) - set(fitting)
    return [*missing, *(name for name in schema if name in unfit and name not in missing)]


def render_refuses(template: Mapping[str, Any], post: Any) -> List[str]:
    """The required fields a render of the post would refuse: those with no value at all. A
    field stored blank is a value to the render (it shows nothing), so it is not one of them."""
    schema = schema_for_length(template, getattr(post, "length_seconds", None))
    variables = post.variables if isinstance(post.variables, Mapping) else {}
    stored = {name: spec.get("value") for name, spec in variables.items() if isinstance(spec, Mapping)}
    return resolve_variables(schema, stored).missing if schema else []


def fields_edit(proposal: Mapping[str, Any], post: Any) -> Dict[str, Any]:
    """Auto's ``proposal`` as the edit that fills the post's empty fields. Auto's value goes
    only where the post holds none (a blank one never hides it); everything else the post has
    stays, a variable the template does not have included. A figure in a field the template
    marks a claim is one (F433). A source goes with the value it was given for: the post's own
    stays with a field that holds the post's value, Auto's goes only with a value Auto wrote,
    and an empty field's old source never rides on to the value Auto wrote. A value the template
    refuses at the post's length is not kept: Auto's goes in its place, or the field is left
    empty for the render to ask for by its label."""
    template = proposal.get("template") or {}
    schema = template.get("variables_schema") or {}
    held = own_values(post, schema_for_length(template, proposal.get("length_seconds")))
    unfit = set(own_values(post)) - set(held)
    stored = {name: spec for name, spec in (post.variables if isinstance(post.variables, Mapping) else {}).items()
              if name not in unfit}
    written = {name: spec for name, spec in (proposal.get("variables") or {}).items() if name not in held}
    own_sources = {name: source for name, source in (post.sources or {}).items() if name in held}
    auto_sources = {name: source for name, source in (proposal.get("sources") or {}).items() if name in written}
    return {
        "variables": with_template_claims({**stored, **written}, set(claim_names(schema))),
        "sources": {**auto_sources, **own_sources},
    }


def newly_written(changes: Mapping[str, Any], post: Any, schema: Optional[Mapping[str, Any]] = None) -> List[str]:
    """The fields ``changes`` gives a value that the post held none for (or one ``schema`` refuses),
    in the edit's order."""
    held = own_values(post, schema)
    return [
        name for name, spec in (changes.get("variables") or {}).items()
        if name not in held and isinstance(spec, Mapping) and _holds_a_value(spec.get("value"))
    ]


def _holds_words(text: Any) -> bool:
    return isinstance(text, str) and bool(text.strip())


def holds_a_caption(copy: Any) -> bool:
    """Whether a post's caption (``{"base", "channels"}``) holds words: its base text, or any
    channel's own text. Nothing, blank text and a shape that is not a caption are empty."""
    copy = copy if isinstance(copy, Mapping) else {}
    channels = copy.get("channels") if isinstance(copy.get("channels"), Mapping) else {}
    return _holds_words(copy.get("base")) or any(_holds_words(text) for text in channels.values())


def holds_an_approval(post: Any) -> bool:
    """Whether the post holds an approval an edit would void (approved, scheduled, or missed with
    its approval): Auto writes nothing into such a post. A preview is allowed for it and keeps its
    status and hash, so words written there, which are content, would send it back for approval."""
    return getattr(post, "status", None) in service.TRANSITIONS[service.ACTION_EDIT]


def caption_wanted(post: Any) -> bool:
    """Whether the post's caption is Auto's to write at a render: it is empty, the post has a
    brief to write it from, and it holds no approval. With no brief, Auto would have only the
    title (often "Untitled post") and would make a caption up, so the caption stays the person's
    to write (F378). A preview of an approved or scheduled post keeps its status and its hash, so
    a caption written there, which is content, would void the approval: it is not written."""
    return (
        _holds_words(getattr(post, "brief", None)) and not holds_a_caption(getattr(post, COPY, None))
        and not holds_an_approval(post)
    )


def caption_edit(proposal: Mapping[str, Any], post: Any) -> Dict[str, Any]:
    """The caption Auto wrote (the proposal's ``copy``, the shape a post saves) as the edit's
    ``copy``, for a post whose caption is empty. Nothing for a caption that holds words, which is
    never rewritten, and nothing when Auto wrote no words either."""
    copy = proposal.get(COPY)
    if holds_a_caption(getattr(post, COPY, None)) or not holds_a_caption(copy):
        return {}
    base = copy.get("base") if isinstance(copy.get("base"), str) else ""
    channels = copy.get("channels") if isinstance(copy.get("channels"), Mapping) else {}
    return {COPY: {"base": base, "channels": dict(channels)}}


def only_caption(changes: Mapping[str, Any]) -> Dict[str, Any]:
    """The caption alone of an edit (its ``copy``, when it has one): what is written at a render
    whose required fields all hold a value, so an optional field left empty stays empty."""
    return {COPY: changes[COPY]} if COPY in changes else {}


def written_line(labels: Sequence[str], caption: bool) -> str:
    """The history line for what Auto wrote: the fields left empty by their ``labels``, then
    the caption when Auto wrote it."""
    parts = [WROTE_FIELDS.format(fields=", ".join(labels))] if labels else []
    return " ".join([*parts, WROTE_CAPTION] if caption else parts)


def record_written(post: Any, actor: str, changes: Mapping[str, Any], comment: str) -> None:
    """Apply what Auto wrote as an edit; the history line (``comment``) says what it wrote, and
    names Auto as the one who wrote it."""
    service.update_post(post, actor, changes)
    service._log(  # the module's own history writer, as ``retakes.py`` uses it
        post, actor, service.ACTION_EDIT, comment, agent=service.AUTO, fields=sorted(changes),
    )


def record_pick(post: Any, actor: str, changes: Mapping[str, Any], template_name: str) -> None:
    """F253's pick applied as one edit. With the caption Auto wrote in it (US-D103), the one
    history line names the template and the caption; without, it is the service's own line."""
    if COPY in changes:
        record_written(post, actor, changes, PICKED_WITH_CAPTION.format(template=template_name))
    else:
        service.record_auto_pick(post, actor, changes, template_name)
