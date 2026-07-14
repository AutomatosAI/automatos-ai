"""
Canonical ``skill_source`` provenance scheme (PRD-202 S1)
========================================================

The Agent Skills open standard has no tenancy/provenance model — that is
Automatos's addition. Historically the ``Skill.skill_source`` column carried an
*incoherent* mix of values (dossier agents-skills J4/D.1):

    git-imported      -> ``str(source_id)``      (a bare numeric id, e.g. "5")
    plugin-materialized -> ``plugin:<slug>``     (already scheme-shaped)
    builtin-seeded    -> ``builtin-core`` / ``builtin-seeds``
    workspace-authored -> ``workspace-user`` / ``workspace-fork``

This module is the single source of truth for that provenance: one canonical
``scheme:ref`` vocabulary over the four origins the platform actually has —
``git`` / ``plugin`` / ``builtin`` / ``workspace``.

Two pure functions:

* ``canonicalize_skill_source(...)`` — produce a canonical ``scheme:ref`` string
  for a NEW write (import/export/create paths use this).
* ``parse_skill_source(value)`` — resolve ANY value (canonical OR legacy) to a
  ``(scheme, ref)`` pair, so provenance *resolves* regardless of whether the row
  has been backfilled yet. Callers that branch on origin (e.g. the loader's
  builtin freshness check) go through this, never a bare string compare.

No DB, no I/O — safe to import anywhere.
"""

from __future__ import annotations

from typing import Optional, Tuple

# The fixed provenance vocabulary. Every skill originates from exactly one.
CANONICAL_SKILL_SOURCE_SCHEMES: Tuple[str, ...] = ("git", "plugin", "builtin", "workspace")

_SCHEME_SEP = ":"

# Legacy (pre-PRD-202) provenance strings mapped to their canonical (scheme, ref).
# Kept so ``parse_skill_source`` resolves un-backfilled rows without a reshape.
_LEGACY_EXACT = {
    "builtin-core": ("builtin", "core"),
    "builtin-seeds": ("builtin", "seeds"),
    "workspace-user": ("workspace", "user"),
    "workspace-fork": ("workspace", "fork"),
}


def _norm(value: Optional[str]) -> str:
    return (value or "").strip()


def parse_skill_source(value: Optional[str]) -> Tuple[str, str]:
    """Resolve any ``skill_source`` value to a canonical ``(scheme, ref)``.

    Handles three shapes:

    * canonical ``scheme:ref`` (e.g. ``git:anthropic-official``) — returned as-is;
    * legacy exact tags (``builtin-core`` etc.) via the compatibility map;
    * a bare numeric id (the legacy git shape ``"5"``) → ``("git", "5")``.

    Unknown / empty values resolve to ``("unknown", "")`` — callers treat that as
    "no recognised origin" (never as a match for a real scheme).
    """
    raw = _norm(value)
    if not raw:
        return ("unknown", "")

    # Canonical scheme:ref already?
    if _SCHEME_SEP in raw:
        scheme, ref = raw.split(_SCHEME_SEP, 1)
        scheme = scheme.strip().lower()
        ref = ref.strip()
        if scheme in CANONICAL_SKILL_SOURCE_SCHEMES:
            return (scheme, ref)
        return ("unknown", raw)

    # Legacy exact tag?
    lowered = raw.lower()
    if lowered in _LEGACY_EXACT:
        return _LEGACY_EXACT[lowered]

    # Legacy git shape: a bare numeric SkillSource id.
    if raw.isdigit():
        return ("git", raw)

    return ("unknown", raw)


def canonicalize_skill_source(
    scheme: str,
    ref: Optional[str] = None,
) -> str:
    """Build a canonical ``scheme:ref`` provenance string for a NEW write.

    ``scheme`` must be one of :data:`CANONICAL_SKILL_SOURCE_SCHEMES`. ``ref`` is
    the origin identifier (a git source name, plugin slug, builtin name, or
    workspace kind). Whitespace is trimmed; a missing ref yields a bare
    ``scheme:`` (still resolvable).
    """
    s = _norm(scheme).lower()
    if s not in CANONICAL_SKILL_SOURCE_SCHEMES:
        raise ValueError(
            f"Unknown skill_source scheme '{scheme}'. "
            f"Expected one of {CANONICAL_SKILL_SOURCE_SCHEMES}."
        )
    r = _norm(ref)
    return f"{s}{_SCHEME_SEP}{r}" if r else f"{s}{_SCHEME_SEP}"


def scheme_of(value: Optional[str]) -> str:
    """Convenience: the canonical scheme of any ``skill_source`` value."""
    return parse_skill_source(value)[0]


def is_external_source(value: Optional[str]) -> bool:
    """True for third-party / external provenance (git, plugin).

    ``builtin`` (the platform's own) and ``workspace`` (the user's own) are
    trusted; ``git`` and ``plugin`` arrive from outside and get the LLM scan
    stage ON at import (S4, dossier ClawHub incident D.3).
    """
    return scheme_of(value) in ("git", "plugin")
