"""Template variable catalog + resolution (PRD-167 S3)."""

from .catalog import CATALOG, CATALOG_BY_PATH, is_dynamic_path, is_known_path, is_valid_path
from .resolver import (
    ResolvedVariables,
    VariableResolver,
    build_context,
    resolve_paths,
)

__all__ = [
    "CATALOG",
    "CATALOG_BY_PATH",
    "is_known_path",
    "is_dynamic_path",
    "is_valid_path",
    "ResolvedVariables",
    "VariableResolver",
    "build_context",
    "resolve_paths",
]
