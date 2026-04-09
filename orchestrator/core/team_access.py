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
