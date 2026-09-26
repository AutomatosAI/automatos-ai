"""
Built-in skills: the manifest and the generated seed files (PRD-251 US-119)
===========================================================================

A built-in skill is platform-owned content (a global ``skills`` row) authored in
the automatos-skills repo. Its copy in this repo is GENERATED: only
``scripts/sync-skills.py <name> ...`` writes it, frontmatter intact, with a
banner as the first body line. ``core/seeds/skills/manifest.json`` lists every
built-in skill (name → seed file, relative to the manifest → its
automatos-skills source), and three places read it through this module, so
they agree on where a seed lives:

* ``core/seeds/seed_builtin_skills.py`` creates a missing row (at boot for every
  entry, and for ``platform-management`` when a workspace's Auto is seeded);
* ``SkillLoader._refresh_builtin_if_stale`` refreshes a builtin row whose content
  hash differs from its seed file when the skill is loaded;
* ``scripts/sync-skills.py`` writes the seed files. It runs with a bare
  ``python3``, so this module uses the standard library only.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

MANIFEST_PATH = Path(__file__).resolve().parent / "seeds" / "skills" / "manifest.json"

# Columns an entry may pin on its row; the rest come from the seed's frontmatter.
ROW_FIELDS = ("description", "skill_type", "category", "tags")
ENTRY_KEYS = frozenset({"seed", "source", "skill_source", "row"})
SOURCE_FILE = "SKILL.md"
SEED_SUFFIX = ".md"
# A new built-in row's provenance, in PRD-202's canonical scheme:ref form.
BUILTIN_SOURCE_PREFIX = "builtin:"

_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VERSION = re.compile(r'^version:\s*"?([\d.]+)"?', re.M)
_FENCE = "---"


class ManifestError(ValueError):
    """The built-in skills manifest is malformed: a build defect, so it fails loud."""


@dataclass(frozen=True)
class BuiltinSkill:
    name: str
    seed_path: Path          # the generated copy in this repo
    source: str              # its SKILL.md inside the automatos-skills repo
    skill_source: str        # the provenance written on the row
    row: Mapping[str, Any]   # pinned columns (platform-management keeps its old row)


@dataclass(frozen=True)
class SeedFile:
    frontmatter: str         # the YAML between the fences, as written
    body: str                # what the platform stores as the prompt, and hashes
    content_hash: str
    version: Optional[str]


def seeds_root(manifest_path: Path) -> Path:
    """Every seed file lives under ``core/seeds``: the folder above the manifest's."""
    return manifest_path.resolve().parent.parent


def load_manifest(path: Path = MANIFEST_PATH) -> Mapping[str, BuiltinSkill]:
    """The built-in skills, by name, in manifest order (read-only)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"{path}: {exc}") from exc
    skills = data.get("skills") if isinstance(data, dict) else None
    if not isinstance(skills, dict) or not skills:
        raise ManifestError(f"{path}: 'skills' must be a non-empty object")
    base = path.resolve().parent
    root = seeds_root(path)
    return MappingProxyType({name: _entry(name, spec, base, root) for name, spec in skills.items()})


def builtin_skill_paths(path: Path = MANIFEST_PATH) -> Dict[str, Path]:
    """Built-in skill name → its seed file."""
    return {name: entry.seed_path for name, entry in load_manifest(path).items()}


def read_seed(path: Path) -> Optional[SeedFile]:
    """A seed file as the platform stores it: frontmatter stripped, body hashed.

    ``None`` when the file is absent: a manifest entry the owner has not synced yet.
    """
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    frontmatter, body = split_frontmatter(raw)
    version = _VERSION.search(raw)
    return SeedFile(
        frontmatter=frontmatter,
        body=body,
        content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        version=version.group(1) if version else None,
    )


def split_frontmatter(raw: str) -> Tuple[str, str]:
    """``(frontmatter, body)`` of a stripped SKILL.md; no fences means all body."""
    if raw.startswith(_FENCE):
        parts = raw.split(_FENCE, 2)
        if len(parts) > 2:
            return parts[1], parts[2].strip()
    return "", raw


def _entry(name: Any, spec: Any, base: Path, root: Path) -> BuiltinSkill:
    if not isinstance(name, str) or not _NAME.match(name):
        raise ManifestError(f"built-in skill name {name!r} is not a lowercase slug")
    if not isinstance(spec, dict):
        raise ManifestError(f"{name}: the entry must be an object")
    unknown = sorted(set(spec) - ENTRY_KEYS)
    if unknown:
        raise ManifestError(f"{name}: unknown keys {unknown}")
    return BuiltinSkill(
        name=name,
        seed_path=_seed_path(name, spec.get("seed"), base, root),
        source=_source(name, spec.get("source")),
        skill_source=_skill_source(name, spec.get("skill_source", BUILTIN_SOURCE_PREFIX + name)),
        row=_row(name, spec.get("row", {})),
    )


def _seed_path(name: str, seed: Any, base: Path, root: Path) -> Path:
    if not isinstance(seed, str) or not seed.endswith(SEED_SUFFIX):
        raise ManifestError(f"{name}: 'seed' must be a {SEED_SUFFIX} path relative to the manifest")
    path = (base / seed).resolve()
    if root not in path.parents:
        raise ManifestError(f"{name}: seed {seed!r} is outside {root}")
    return path


def _source(name: str, source: Any) -> str:
    parts = Path(source).parts if isinstance(source, str) else ()
    if not parts or parts[-1] != SOURCE_FILE or Path(source).is_absolute() or ".." in parts:
        raise ManifestError(f"{name}: 'source' must be a {SOURCE_FILE} path inside automatos-skills")
    return source


def _skill_source(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name}: 'skill_source' must be a non-empty string")
    return value


def _row(name: str, row: Any) -> Mapping[str, Any]:
    if not isinstance(row, dict):
        raise ManifestError(f"{name}: 'row' must be an object")
    unknown = sorted(set(row) - set(ROW_FIELDS))
    if unknown:
        raise ManifestError(f"{name}: 'row' may pin only {list(ROW_FIELDS)}, not {unknown}")
    return MappingProxyType(dict(row))
