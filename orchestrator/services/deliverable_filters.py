"""The WHERE clause of the Deliverables list (``DeliverableService.list_deliverables``).

Moved out of deliverable_service.py (over 800 lines) when the list learned the ``tag``
filter (Gerard, 7 Oct): a Deliverable's tags live in ``extra.tags`` (see
services/deliverable_tags.py), and a blog post's tags reach the same key through the
``v_workspace_outputs`` view. Every value is a bound parameter; the clause names only
fixed columns and placeholders.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from services.deliverable_tags import clean_tag

Clause = Tuple[List[str], Dict[str, Any]]

# The filters matched as they are, on the view's column of the same name.
EQUAL_FILTERS = ("artifact_type", "source_type", "source_id")
# A tag matches whatever case it was stored in (a blog post's tags are as written).
TAG_CONDITION = (
    "EXISTS (SELECT 1 FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(o.extra -> 'tags') = 'array' "
    "THEN o.extra -> 'tags' ELSE '[]'::jsonb END) AS tag(value) WHERE lower(tag.value) = :tag)"
)


def _equal(filters: Mapping[str, Any]) -> Clause:
    named = [name for name in EQUAL_FILTERS if filters.get(name)]
    params: Dict[str, Any] = {name: str(filters[name]) for name in named}
    if filters.get("agent_id") is not None:
        named = [*named, "agent_id"]
        params = {**params, "agent_id": filters["agent_id"]}
    return [f"o.{name} = :{name}" for name in named], params


def _excluded_sources(raw: Any) -> Clause:
    excluded = [s.strip() for s in str(raw or "").split(",") if s.strip()]
    if not excluded:
        return [], {}
    names = [f"excl_src_{i}" for i in range(len(excluded))]
    placeholders = ", ".join(f":{name}" for name in names)
    return [f"o.source_type NOT IN ({placeholders})"], dict(zip(names, excluded))


def _dates(filters: Mapping[str, Any]) -> Clause:
    bounds = (("date_from", ">="), ("date_to", "<="))
    given = [(name, op) for name, op in bounds if filters.get(name)]
    return [f"o.created_at {op} :{name}" for name, op in given], {name: filters[name] for name, _ in given}


def _search(raw: Any) -> Clause:
    if not raw:
        return [], {}
    return ["(o.title ILIKE :search OR o.summary ILIKE :search OR o.file_path ILIKE :search)"], {"search": f"%{raw}%"}


def _tag(raw: Any) -> Clause:
    tag = clean_tag(raw) if isinstance(raw, str) else ""
    return ([TAG_CONDITION], {"tag": tag}) if tag else ([], {})


def list_filter(workspace_id: Any, filters: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """``(where, params)`` for the list's filters: the workspace's rows not deleted, then
    each filter given. Pure."""
    clauses = [
        (["o.workspace_id = :workspace_id", "o.deleted_at IS NULL"], {"workspace_id": str(workspace_id)}),
        _equal(filters),
        _excluded_sources(filters.get("source_type_exclude")),
        _dates(filters),
        _search(filters.get("search")),
        _tag(filters.get("tag")),
    ]
    conditions = [condition for found, _ in clauses for condition in found]
    params = {key: value for _, found in clauses for key, value in found.items()}
    return " AND ".join(conditions), params


__all__ = ["TAG_CONDITION", "list_filter"]
