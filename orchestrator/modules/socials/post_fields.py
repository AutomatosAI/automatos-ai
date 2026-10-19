"""A post's template fields, as its render resolves them (``render.bundle_for``).

P256-FIX-SOC-4 (F431-a): a ticket that names an existing post says which of its template's
fields are still empty (``services.social_ticket_post``), from the same resolution the render
refuses on, so the two never disagree. Pure: the caller reads the template.
"""
from __future__ import annotations

from typing import Any, List, Mapping

from core.social_templates import ResolvedVariables, resolve_variables


def resolved_fields(blocks: Mapping[str, Any], post: Any) -> ResolvedVariables:
    """The template's fields resolved from the post's variables (``resolve_variables``): each
    one's value, the ones left empty with no default (``missing``) and those that do not fit."""
    supplied = {
        name: spec.get("value")
        for name, spec in (getattr(post, "variables", None) or {}).items()
        if isinstance(spec, dict)
    }
    return resolve_variables(blocks["variables_schema"], supplied)


def unfilled_fields(blocks: Any, post: Any) -> List[str]:
    """The fields of a template's composition (``render.composition_of``) that ``post`` leaves
    empty and that have no default, by name; [] when the blocks declare no fields."""
    schema = blocks.get("variables_schema") if isinstance(blocks, Mapping) else None
    return list(resolved_fields(blocks, post).missing) if isinstance(schema, Mapping) else []


__all__ = ["resolved_fields", "unfilled_fields"]
