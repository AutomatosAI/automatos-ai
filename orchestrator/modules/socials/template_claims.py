"""P256-FIX-SOC-2 (F433, socials day 14 Oct): one provenance rule for every post.

The approval gate (D7) used to count only the variables a stored post marked ``claim: true``.
A plan's posts arrived marked; a post made through the REST routes or by Let Auto pick
arrived as its caller sent it, so an invented figure with no ``claim`` key (post b142ac9a:
987/500/300/187, no sources) reached ``approved``. Now the template decides what is a claim:

* :func:`with_template_claims` marks, in a new dict, every variable the template marks a
  claim and that has a value. Nothing is ever un-claimed, and a blank value claims nothing.
* :func:`names` reads which variables the workspace's template marks (``claim_names`` over
  its ``variables_schema``); a post with no template, or a template that is gone, has none.

The REST create and edit routes, Let Auto pick and the agents' tools mark on write, so the
stored record and the review screen say what the gate checks; both approval sites (one post,
and a series or a plan's batch) also ask the template, so a post stored before this rule is
gated too. Quotes and review ratings are not claims here, and numbers inside the copy, or a
text post's, are outside this rule: nothing structured marks them.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional
from uuid import UUID

from core.models.core import DocumentTemplate
from core.social_templates import claim_names

CLAIM, VALUE, VARIABLES, TEMPLATE_ID = "claim", "value", "variables", "template_id"


def has_value(spec: Any) -> bool:
    """Whether a variable holds something a reader sees: a value that is not blank."""
    value = spec.get(VALUE) if isinstance(spec, Mapping) else None
    return value is not None and str(value).strip() != ""


def with_template_claims(variables: Any, claims: Iterable[str]) -> Any:
    """``variables`` as a new dict, each one ``claims`` names that has a value marked
    ``claim: true``; every other variable exactly as given. ``None``, ``{}`` and anything
    that is not a dict (left for the validators to refuse) come back unchanged."""
    if not isinstance(variables, dict) or not variables:
        return variables
    marked = frozenset(claims)
    return {
        name: {**spec, CLAIM: True} if name in marked and isinstance(spec, dict) and has_value(spec) else spec
        for name, spec in variables.items()
    }


def _template_uuid(template_id: Any) -> Optional[UUID]:
    if isinstance(template_id, UUID):
        return template_id
    try:
        return UUID(str(template_id))
    except (TypeError, ValueError):
        return None


def names(db: Any, workspace_id: UUID, template_id: Any) -> frozenset:
    """The variables the workspace's template ``template_id`` marks as claims. Empty, without
    a query, when there is no template; empty for a template that is not the workspace's, is
    gone, or has no schema."""
    if template_id is None:
        return frozenset()
    wanted = _template_uuid(template_id)
    if wanted is None:
        return frozenset()
    row = (
        db.query(DocumentTemplate.blocks)
        .filter(DocumentTemplate.id == wanted, DocumentTemplate.workspace_id == workspace_id)
        .first()
    )
    blocks = row.blocks if row is not None and isinstance(row.blocks, dict) else {}
    schema = blocks.get("variables_schema")
    return frozenset(claim_names(schema)) if isinstance(schema, dict) else frozenset()


def marked_fields(db: Any, workspace_id: UUID, fields: Mapping[str, Any]) -> Dict[str, Any]:
    """A new post's fields with its variables marked by its template's claims."""
    claims = names(db, workspace_id, fields.get(TEMPLATE_ID))
    return {**fields, VARIABLES: with_template_claims(fields.get(VARIABLES), claims)}


def marked_changes(db: Any, post: Any, changes: Mapping[str, Any]) -> Mapping[str, Any]:
    """An edit's changes with the variables marked by the template the post will have. Only
    an edit that sends variables or a template is looked at; a template change that marks
    nothing new is left as sent, so it neither re-checks the stored variables nor changes
    the hash."""
    if VARIABLES not in changes and TEMPLATE_ID not in changes:
        return changes
    stored = getattr(post, VARIABLES, None)
    template_id = changes.get(TEMPLATE_ID, getattr(post, TEMPLATE_ID, None))
    marked = with_template_claims(changes.get(VARIABLES, stored), names(db, post.workspace_id, template_id))
    if VARIABLES not in changes and (marked or {}) == (stored or {}):
        return changes
    return {**changes, VARIABLES: marked}


__all__ = ["has_value", "marked_changes", "marked_fields", "names", "with_template_claims"]
