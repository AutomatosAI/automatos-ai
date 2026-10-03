"""
Seed the built-in skills (PRD-251 US-119)
=========================================

``core/seeds/skills/manifest.json`` lists the platform's built-in skills
(``core.builtin_skills``). This creates each missing row from its generated
seed file: a global row (``workspace_id`` NULL) with the manifest's
``skill_source``. It runs at boot for every entry (``main.py``, leader worker)
and for ``platform-management`` whenever a workspace's Auto is seeded.

An existing built-in row whose content hash differs from its seed file is
refreshed here: its body, hash and version, nothing else (F239, build 6). The
loader refreshes one too when it loads the skill
(``SkillLoader._refresh_builtin_if_stale``), but Auto's prompt never goes through
the loader: ``SkillsSection`` reads the row's ``prompt_template`` itself. So a
deployed seed (v2.3.1, F232) never reached Auto, after 29 turns on build 6.

* An entry whose seed file has not been synced yet is skipped, the way a
  Shopify agent's missing skill is (``seed_shopify_agents._lookup_skill_ids``):
  seed files arrive only from the owner's ``scripts/sync-skills.py`` run.
* A global row of the same name from anywhere else (a git import, a plugin) is
  left untouched. Global skill names are unique (``uq_skills_marketplace_name``),
  so the built-in row is not created while that row exists.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.builtin_skills import MANIFEST_PATH, ROW_FIELDS, BuiltinSkill, SeedFile, load_manifest, read_seed
from core.models.core import Skill

logger = logging.getLogger(__name__)

CREATED = "created"
REFRESHED = "refreshed"
PRESENT = "present"
NOT_SYNCED = "not_synced"
LEFT_ALONE = "left_alone"
INVALID = "invalid"
OUTCOMES = (CREATED, REFRESHED, PRESENT, NOT_SYNCED, LEFT_ALONE, INVALID)

# Frontmatter defaults: the same ones a spec-conformant import uses (skill_portability).
DEFAULT_SKILL_TYPE = "technical"
DEFAULT_CATEGORY = "general"
DEFAULT_VERSION = "1.0.0"

# PRD-191 S3: seeders on every worker serialize per skill. The key is
# 'seed:<name>', which for platform-management is the one it has always used.
_SEED_LOCK = text("SELECT pg_advisory_xact_lock(hashtext(:key))")


def seed_builtin_skills(db: Session, manifest_path: Path = MANIFEST_PATH) -> Dict[str, List[str]]:
    """Create every missing built-in skill row and refresh every stale one. Returns
    the skill names by outcome."""
    results = [(entry.name, _ensure(db, entry)[0]) for entry in load_manifest(manifest_path).values()]
    return {outcome: [name for name, got in results if got == outcome] for outcome in OUTCOMES}


def ensure_builtin_skill(db: Session, name: str, manifest_path: Path = MANIFEST_PATH) -> Optional[Skill]:
    """The built-in skill ``name``'s row, created if missing; None when it cannot be."""
    entry = load_manifest(manifest_path).get(name)
    if entry is None:
        logger.error("Built-in skill '%s' is not in %s", name, manifest_path)
        return None
    return _ensure(db, entry)[1]


def _ensure(db: Session, entry: BuiltinSkill) -> Tuple[str, Optional[Skill]]:
    _lock(db, entry.name)
    # Our own row first: beside it a git import can hold the same name (build 6:
    # row 130 'github:AutomatosAI/automatos-skills'), and an unordered first() could
    # return that one and leave ours stale.
    rows = db.query(Skill).filter(Skill.name == entry.name, Skill.workspace_id.is_(None)).order_by(Skill.id).all()
    ours = next((row for row in rows if row.skill_source == entry.skill_source), None)
    if ours is not None:
        return _refresh(ours, entry), ours
    if rows:
        logger.warning(
            "Built-in skill '%s' not seeded: the global row id=%s from '%s' has that name and is left untouched",
            entry.name, rows[0].id, rows[0].skill_source,
        )
        return LEFT_ALONE, None

    seed = read_seed(entry.seed_path)
    if seed is None:
        logger.warning("Built-in skill '%s' is not synced yet (%s is missing): skipped", entry.name, entry.seed_path)
        return NOT_SYNCED, None
    values = _row_values(entry, seed)
    if values is None:
        return INVALID, None

    skill = Skill(**values)
    try:
        with db.begin_nested():
            db.add(skill)
            db.flush()
    except IntegrityError:
        # Another worker won the insert race despite the lock: re-select and
        # return its row. Never swallow this into a silent no-seed (PRD-191 S3).
        existing = db.query(Skill).filter(
            Skill.name == entry.name,
            Skill.skill_source == entry.skill_source,
        ).first()
        logger.info("Built-in skill '%s' seeded by a concurrent worker (id=%s)", entry.name, getattr(existing, "id", None))
        return (PRESENT if existing is not None else LEFT_ALONE), existing
    logger.info("Built-in skill '%s' created (id=%s)", entry.name, skill.id)
    return CREATED, skill


def _refresh(row: Skill, entry: BuiltinSkill) -> str:
    """REFRESHED when the row's content hash differs from its seed file's, after
    taking the seed's body, hash and version; PRESENT when it matches or there is no
    seed file to compare. The row's other columns are the owner's and stay."""
    seed = read_seed(entry.seed_path)
    if seed is None or row.content_hash == seed.content_hash:
        return PRESENT
    row.prompt_template = seed.body
    row.content_hash = seed.content_hash
    row.skill_version = seed.version or row.skill_version
    logger.info("Refreshed builtin skill '%s' from disk (hash=%s…)", entry.name, seed.content_hash[:12])
    return REFRESHED


def _lock(db: Session, name: str) -> None:
    """Hold the skill's seed lock for the transaction. A database without
    advisory locks (SQLite, in the unit tests) has one writer."""
    if db.get_bind().dialect.name == "postgresql":
        db.execute(_SEED_LOCK, {"key": f"seed:{name}"})


def _row_values(entry: BuiltinSkill, seed: SeedFile) -> Optional[Dict[str, Any]]:
    """A new row's columns: the manifest's pins, else the seed's frontmatter."""
    unpinned = [field for field in ROW_FIELDS if field not in entry.row]
    meta = _frontmatter(entry, seed) if unpinned else {}
    if meta is None:
        return None
    fields = {**_frontmatter_fields(meta), **entry.row}
    if not fields["description"]:
        logger.error("Built-in skill '%s' has no description (its L1 trigger text): skipped", entry.name)
        return None
    return {
        "name": entry.name,
        "description": fields["description"],
        "skill_type": fields["skill_type"],
        "category": fields["category"],
        "tags": list(fields["tags"]),
        "skill_version": seed.version or DEFAULT_VERSION,
        "skill_source": entry.skill_source,
        "prompt_template": seed.body,
        "content_hash": seed.content_hash,
        "is_active": True,
        "workspace_id": None,
    }


def _frontmatter(entry: BuiltinSkill, seed: SeedFile) -> Optional[Mapping[str, Any]]:
    try:
        meta = yaml.safe_load(seed.frontmatter) or {}
    except yaml.YAMLError:
        logger.error("Built-in skill '%s': the seed's frontmatter is not YAML: skipped", entry.name, exc_info=True)
        return None
    if not isinstance(meta, dict):
        logger.error("Built-in skill '%s': the seed's frontmatter is not a mapping: skipped", entry.name)
        return None
    return meta


def _frontmatter_fields(meta: Mapping[str, Any]) -> Dict[str, Any]:
    tags = meta.get("tags")
    return {
        "description": str(meta.get("description") or "").strip(),
        "skill_type": meta.get("skill_type") or meta.get("type") or DEFAULT_SKILL_TYPE,
        "category": meta.get("category") or DEFAULT_CATEGORY,
        "tags": tags if isinstance(tags, list) else [],
    }
