"""
Centralized Retrieval Filter Builder (PRD-157 S1)
=================================================

ONE grep-able choke point that derives the tenancy + team scope for EVERY search
surface on the platform: RAG ``retrieve``, the UI semantic search
(``api/documents.py``), the RAG-test endpoint, the multimodal knowledge tools,
Knowledge-Graph retrieval, and any future NL2SQL path.

Two guarantees:

* **Fail-closed.** A search that cannot resolve a ``workspace_id`` raises
  :class:`RetrievalScopeError` instead of silently running unscoped. Helpers that
  apply the scope return *nothing* on error rather than passing candidates
  through (the opposite of the old fail-open ``_filter_by_team``).
* **Canonical team normalization.** Team names go through
  :func:`core.team_access.normalize_team` here, so ``"Support"`` and ``"support"``
  are one security domain on every path.

Team policy (unchanged from PRD-124, now centralized):

* an agent **with** a team sees public docs (``team_access = '{}'``) plus docs
  whose ``team_access`` overlaps its team(s);
* an agent **without** a team is workspace-scoped (sees every doc in its
  workspace) — empty ``team_terms`` means "no team restriction", NOT "public
  only".

Grep ``build_retrieval_filters`` to enumerate every scoped search path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Set

from core.team_access import normalize_teams


class RetrievalScopeError(ValueError):
    """Raised when a retrieval scope cannot be resolved (fail-closed)."""


@dataclass(frozen=True)
class RetrievalFilters:
    """Resolved scope for a single search.

    ``team_terms == []`` means "no team restriction" (workspace-wide), which is
    distinct from a team agent that happens to match only public docs.
    """

    workspace_id: str
    team_terms: List[str] = field(default_factory=list)

    @property
    def team(self) -> Optional[str]:
        """First team term — back-compat with the single-team ``retrieve(team=...)`` API."""
        return self.team_terms[0] if self.team_terms else None

    @property
    def has_team_restriction(self) -> bool:
        return bool(self.team_terms)

    def sql_params(self) -> dict:
        """Bind params for :data:`SCOPE_WHERE_TEAM` / :func:`scope_where_clause`."""
        return {
            "workspace_id": self.workspace_id,
            "team_terms": list(self.team_terms),
        }


def _coerce_workspace(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_retrieval_filters(
    *,
    agent: Any = None,
    workspace_id: Any = None,
    team: Any = None,
    teams: Optional[Sequence[str]] = None,
    context: Optional[dict] = None,
    require_workspace: bool = True,
) -> RetrievalFilters:
    """Resolve ``{workspace_id, team_terms}`` from an agent, explicit args, or a context dict.

    Precedence (workspace): explicit ``workspace_id`` > ``agent.workspace_id`` >
    ``context['workspace_id']``.
    Precedence (team): explicit ``teams``/``team`` > ``agent.team`` >
    ``context['team']``.

    :param require_workspace: when True (default) raise :class:`RetrievalScopeError`
        if no workspace can be resolved — the fail-closed default for every real
        search path. Pass False only for diagnostics that intentionally run
        unscoped.
    """
    ws = _coerce_workspace(workspace_id)
    if ws is None and agent is not None:
        ws = _coerce_workspace(getattr(agent, "workspace_id", None))
    if ws is None and context:
        ws = _coerce_workspace(context.get("workspace_id"))

    if ws is None and require_workspace:
        raise RetrievalScopeError(
            "Retrieval scope has no workspace_id — refusing to search unscoped (fail-closed)."
        )

    raw_terms: List[str] = []
    if teams:
        raw_terms = list(teams)
    elif team is not None:
        raw_terms = [team]
    elif agent is not None and getattr(agent, "team", None):
        raw_terms = [getattr(agent, "team")]
    elif context and context.get("team"):
        raw_terms = [context["team"]]

    return RetrievalFilters(workspace_id=ws or "", team_terms=normalize_teams(raw_terms))


# --- Scope application ------------------------------------------------------
#
# A single SQL predicate every path can reuse against the ``documents`` table.
# Bind :workspace_id (uuid) and :team_terms (text[]). ``team_access = '{}'`` is
# the public sentinel; ``&&`` is PostgreSQL array overlap.

SCOPE_WHERE_TEAM = (
    "workspace_id = :workspace_id::uuid "
    "AND (team_access = '{}' OR team_access && :team_terms::text[])"
)
SCOPE_WHERE_WORKSPACE_ONLY = "workspace_id = :workspace_id::uuid"


def scope_where_clause(filters: RetrievalFilters) -> str:
    """Return the ``documents`` WHERE predicate for ``filters`` (no leading AND)."""
    if filters.has_team_restriction:
        return SCOPE_WHERE_TEAM
    return SCOPE_WHERE_WORKSPACE_ONLY


def allowed_document_ids(
    db: Any,
    doc_ids: Iterable[Any],
    filters: RetrievalFilters,
) -> Set[str]:
    """Subset of ``doc_ids`` visible under ``filters``.

    Fail-closed: any error (or an unset workspace) yields an **empty** set, so a
    broken scope query can never leak cross-tenant or cross-team documents.
    """
    from sqlalchemy import text as sa_text

    ids = {str(d) for d in doc_ids if d is not None and str(d).strip()}
    if not ids or not filters.workspace_id:
        return set()

    int_ids: List[int] = []
    for raw in ids:
        try:
            int_ids.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not int_ids:
        return set()

    sql = (
        "SELECT id::text FROM documents "
        f"WHERE id = ANY(:ids::int[]) AND {scope_where_clause(filters)}"
    )
    params = {"ids": int_ids, **filters.sql_params()}
    try:
        rows = db.execute(sa_text(sql), params).fetchall()
    except Exception:  # fail-closed — never pass candidates through on error
        import logging

        logging.getLogger(__name__).warning(
            "allowed_document_ids scope query failed; returning empty (fail-closed)",
            exc_info=True,
        )
        return set()
    return {str(r[0]) for r in rows}
