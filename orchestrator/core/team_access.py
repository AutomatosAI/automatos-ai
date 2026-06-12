"""
Team Access Helpers (PRD-124)
=============================

Canonical normalization and filtering for team-based document scoping.

Every function that touches team names MUST go through ``normalize_team()``
so "Support", "support", and " support " are the same security domain.
"""

from typing import List, Optional


def normalize_team(team: str) -> str:
    """Canonical form: stripped and lowercased.

    >>> normalize_team("  Support ")
    'support'
    """
    return team.strip().lower()


def normalize_teams(teams: List[str]) -> List[str]:
    """Normalize a list of team names, dropping blanks.

    >>> normalize_teams(["Support", " sales ", "", "  "])
    ['support', 'sales']
    """
    return [normalize_team(t) for t in teams if t and t.strip()]


def effective_team(
    auth_team: Optional[str],
    request_team: Optional[str] = None,
) -> Optional[str]:
    """Resolve the effective team: auth (API key) > request > None.

    Always returns a normalized value or None.
    """
    raw = auth_team or request_team
    if not raw or not raw.strip():
        return None
    return normalize_team(raw)


# SQL clause reusable across endpoints (bind :team parameter)
TEAM_FILTER_CLAUSE = "AND (team_access = '{}' OR :team = ANY(team_access))"


def metadata_team_filter_clause(
    json_col: str = "metadata",
    key: str = "team_access",
) -> str:
    """Team-filter clause for tables that store ``team_access`` inside a JSONB
    column instead of a top-level ``ARRAY`` column.

    ``knowledge_items`` (multimodal KB) has no ``team_access`` column — its
    scoping lives in ``metadata->'team_access'`` — so the array-column
    ``TEAM_FILTER_CLAUSE`` does not apply. Semantics are identical: an empty /
    absent list is visible to all; otherwise the bound ``:team`` must be a
    member. Bind the ``:team`` parameter (already normalized via
    ``normalize_team``).

    >>> metadata_team_filter_clause()
    "AND (COALESCE(metadata->'team_access', '[]'::jsonb) = '[]'::jsonb OR metadata->'team_access' @> to_jsonb(:team::text))"
    """
    col = f"{json_col}->'{key}'"
    return (
        f"AND (COALESCE({col}, '[]'::jsonb) = '[]'::jsonb "
        f"OR {col} @> to_jsonb(:team::text))"
    )


# --------------------------------------------------------------------------- #
# PRD-158 S1: table-backed Teams entity helpers.
# Every team WRITE goes through here so normalization stays single-source
# (``normalize_team``) — there is no second normalizer.
# --------------------------------------------------------------------------- #

def list_teams(db, workspace_id):
    """All Teams in a workspace, ordered by canonical name."""
    from core.models import Team

    return (
        db.query(Team)
        .filter(Team.workspace_id == workspace_id)
        .order_by(Team.normalized_name)
        .all()
    )


def get_or_create_team(db, workspace_id, name: str):
    """Idempotently ensure a Team exists for ``(workspace, normalize_team(name))``.

    Returns the :class:`~core.models.Team` (existing or newly added & flushed),
    or ``None`` when ``name`` normalizes to empty. This is the single entry point
    for creating teams, so 'Support'/'support' can never become two rows.
    """
    from core.models import Team

    normalized = normalize_team(name or "")
    if not normalized:
        return None

    existing = (
        db.query(Team)
        .filter(Team.workspace_id == workspace_id, Team.normalized_name == normalized)
        .first()
    )
    if existing:
        return existing

    team = Team(workspace_id=workspace_id, name=name.strip(), normalized_name=normalized)
    db.add(team)
    db.flush()
    return team


def ensure_teams(db, workspace_id, names) -> list:
    """Ensure each name in ``names`` has a Team row; return the normalized names.

    Used by write paths (upload, team-access edits) so any team referenced is a
    real row. Blanks are dropped. The returned list is the normalized
    ``team_access`` to persist.
    """
    normalized = []
    for raw in names or []:
        team = get_or_create_team(db, workspace_id, raw)
        if team is not None:
            normalized.append(team.normalized_name)
    return normalized
