"""Variable catalog for document templates (PRD-167 S3).

The static catalog of resolvable variable paths. It drives:
  - the editor's variable-chip picker (``GET /api/documents/variables``), and
  - the resolver's notion of which paths are *known* (an unknown path is a template
    authoring error, distinct from a known-but-empty path).

Categories: ``user.*`` (the requesting user's profile), ``company.*`` (the workspace
business profile), ``brand.*`` (the workspace brand kit — PRD-167 S4), ``date.*``
(computed at render time).
"""

from __future__ import annotations

from typing import Dict, List, TypedDict


class VariableEntry(TypedDict):
    path: str
    category: str
    label: str
    sample: str


CATALOG: List[VariableEntry] = [
    # --- user.* (requesting user, auth-provider-agnostic — PRD-150) ---
    {"path": "user.name", "category": "user", "label": "Your full name", "sample": "Jane Doe"},
    {"path": "user.first_name", "category": "user", "label": "Your first name", "sample": "Jane"},
    {"path": "user.last_name", "category": "user", "label": "Your last name", "sample": "Doe"},
    {"path": "user.email", "category": "user", "label": "Your email", "sample": "jane@acme.com"},
    {"path": "user.username", "category": "user", "label": "Your username", "sample": "jane"},
    # --- company.* (workspace business profile + brand-kit contact details) ---
    {"path": "company.name", "category": "company", "label": "Company name", "sample": "Acme Corp"},
    {"path": "company.website", "category": "company", "label": "Company website", "sample": "acme.com"},
    {"path": "company.address", "category": "company", "label": "Company address", "sample": "123 Main St"},
    {"path": "company.email", "category": "company", "label": "Company email", "sample": "hello@acme.com"},
    {"path": "company.phone", "category": "company", "label": "Company phone", "sample": "+1 555 0100"},
    # --- brand.* (workspace brand kit — PRD-167 S4) ---
    {"path": "brand.name", "category": "brand", "label": "Brand name", "sample": "Acme"},
    {"path": "brand.tagline", "category": "brand", "label": "Brand tagline", "sample": "Build better"},
    {"path": "brand.logo_url", "category": "brand", "label": "Brand logo URL", "sample": "/logo.png"},
    {"path": "brand.primary_color", "category": "brand", "label": "Primary color", "sample": "#1a1a2e"},
    {"path": "brand.secondary_color", "category": "brand", "label": "Secondary color", "sample": "#16213e"},
    {"path": "brand.accent_color", "category": "brand", "label": "Accent color", "sample": "#0f3460"},
    {"path": "brand.font_family", "category": "brand", "label": "Font family", "sample": "Inter"},
    # --- date.* (computed at render time) ---
    {"path": "date.today", "category": "date", "label": "Today (YYYY-MM-DD)", "sample": "2026-06-12"},
    {"path": "date.long", "category": "date", "label": "Today (long form)", "sample": "June 12, 2026"},
    {"path": "date.year", "category": "date", "label": "Current year", "sample": "2026"},
    {"path": "date.month", "category": "date", "label": "Current month", "sample": "06"},
    {"path": "date.day", "category": "date", "label": "Current day", "sample": "12"},
    {"path": "date.iso", "category": "date", "label": "ISO timestamp", "sample": "2026-06-12T09:30:00Z"},
]

# Fast membership / lookup by path.
CATALOG_BY_PATH: Dict[str, VariableEntry] = {e["path"]: e for e in CATALOG}

KNOWN_PATHS = frozenset(CATALOG_BY_PATH)

# Dynamic namespace: ``data.*`` paths are filled from the caller's per-generation data
# dict (e.g. an agent calling generate_document), not from the static catalog. They are
# *valid* paths (not authoring errors) but resolve to empty unless data supplies them.
DYNAMIC_PREFIX = "data."


def is_known_path(path: str) -> bool:
    return path in KNOWN_PATHS


def is_dynamic_path(path: str) -> bool:
    return path.startswith(DYNAMIC_PREFIX) and len(path) > len(DYNAMIC_PREFIX)


def is_valid_path(path: str) -> bool:
    """A path is valid if it's in the static catalog or the dynamic ``data.*`` namespace."""
    return is_known_path(path) or is_dynamic_path(path)


__all__ = [
    "CATALOG",
    "CATALOG_BY_PATH",
    "KNOWN_PATHS",
    "DYNAMIC_PREFIX",
    "VariableEntry",
    "is_known_path",
    "is_dynamic_path",
    "is_valid_path",
]
