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
  the fields by their labels.

Pure but for the post it edits; ``api/socials_compose.py`` asks the composer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from core.social_cuts import schema_for_length
from core.social_templates import claim_names, resolve_variables
from modules.socials import service
from modules.socials.template_claims import with_template_claims

GIVEN_TYPES = (str, int, float, bool)
WROTE_FIELDS = "Auto wrote the fields left empty: {fields}."
STILL_NEEDED = "Auto could not write these from the brief: {fields}. Say them in the brief, or fill them in yourself."


def _holds_a_value(value: Any) -> bool:
    """Text with words in it, a number or a switch; nothing and blank text are empty."""
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, GIVEN_TYPES)


def own_values(post: Any) -> Dict[str, Any]:
    """The post's fields that hold a value, ``{name: value}``: what Auto never rewrites."""
    variables = post.variables if isinstance(post.variables, Mapping) else {}
    return {
        name: spec.get("value") for name, spec in variables.items()
        if isinstance(spec, Mapping) and _holds_a_value(spec.get("value"))
    }


def empty_required(template: Mapping[str, Any], post: Any) -> List[str]:
    """The required fields of ``template`` (a composer's entry) the post leaves empty, at its
    length: a field with no value and no default, the ones a render refuses to start without."""
    schema = schema_for_length(template, getattr(post, "length_seconds", None))
    return resolve_variables(schema, own_values(post)).missing if schema else []


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
    marks a claim is one (F433). The post's own source stays with a field that holds its own
    value; an empty field's old source never rides on to the value Auto wrote."""
    schema = (proposal.get("template") or {}).get("variables_schema") or {}
    held = own_values(post)
    stored = dict(post.variables) if isinstance(post.variables, Mapping) else {}
    written = {name: spec for name, spec in (proposal.get("variables") or {}).items() if name not in held}
    own_sources = {name: source for name, source in (post.sources or {}).items() if name in held}
    return {
        "variables": with_template_claims({**stored, **written}, set(claim_names(schema))),
        "sources": {**(proposal.get("sources") or {}), **own_sources},
    }


def newly_written(changes: Mapping[str, Any], post: Any) -> List[str]:
    """The fields ``changes`` gives a value that the post held none for, in the edit's order."""
    held = own_values(post)
    return [
        name for name, spec in (changes.get("variables") or {}).items()
        if name not in held and isinstance(spec, Mapping) and _holds_a_value(spec.get("value"))
    ]


def record_written(post: Any, actor: str, changes: Mapping[str, Any], labels: Sequence[str]) -> None:
    """Apply Auto's values for the empty fields as an edit; the history names the fields
    (``labels``) and Auto as the one who wrote them."""
    service.update_post(post, actor, changes)
    service._log(  # the module's own history writer, as ``retakes.py`` uses it
        post, actor, service.ACTION_EDIT, WROTE_FIELDS.format(fields=", ".join(labels)),
        agent=service.AUTO, fields=sorted(changes),
    )
