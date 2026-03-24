#!/usr/bin/env python3
"""Seed agent_catalog_templates from SKILL.md files.

Reads every SKILL.md under automatos-skills/skills/, parses YAML frontmatter,
extracts persona from the Identity section, and upserts rows into the
agent_catalog_templates table (keyed on slug).

Usage:
    python scripts/seed_agent_catalog.py [--dry-run] [--skills-dir PATH]

Idempotent — safe to run multiple times.  Existing rows are updated, new rows
are inserted, nothing is deleted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Category → icon mapping (matches PRD-120 US-006 spec)
# ---------------------------------------------------------------------------

CATEGORY_ICONS: dict[str, str] = {
    "engineering": "\u2699\ufe0f",      # gear
    "design": "\U0001F3A8",             # palette
    "marketing": "\U0001F4C8",          # chart
    "sales": "\U0001F4B0",              # money
    "product": "\U0001F3AF",            # target
    "project-management": "\U0001F4CB", # clipboard
    "support": "\U0001F3E2",            # building
    "testing": "\U0001F50D",            # magnifier
    "paid-media": "\U0001F4E3",         # megaphone
    "specialized": "\u2B50",            # star
}

# Model slug → full OpenRouter model ID
MODEL_ID_MAP: dict[str, str] = {
    "haiku-4.5": "anthropic/claude-haiku-4-5-20251001",
    "sonnet-4.6": "anthropic/claude-sonnet-4-6",
    "opus-4.6": "anthropic/claude-opus-4-6",
}

# ---------------------------------------------------------------------------
# YAML frontmatter parser (stdlib only — no PyYAML dependency)
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> dict:
    """Parse YAML frontmatter between --- delimiters.

    Handles scalar values, flow-style lists ([a, b, c]), multi-line >- scalars,
    and nested keys at one level.  NOT a full YAML parser — tailored for the
    SKILL.md format produced by import_agency_skills.py.
    """
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return {}

    lines = m.group(1).split("\n")
    result: dict = {}
    current_key: Optional[str] = None
    multiline_buf: list[str] = []
    list_buf: list[str] = []

    def _flush() -> None:
        nonlocal current_key, multiline_buf, list_buf
        if current_key is None:
            return
        if list_buf:
            result[current_key] = list_buf
            list_buf = []
        elif multiline_buf:
            result[current_key] = " ".join(multiline_buf)
            multiline_buf = []
        current_key = None

    for line in lines:
        # Block-list item:  "  - value"
        if re.match(r"^\s+-\s+", line) and current_key:
            val = re.sub(r"^\s+-\s+", "", line).strip()
            list_buf.append(val)
            continue

        # Multi-line continuation (indented, no key)
        if line.startswith("  ") and current_key and not re.match(r"^\s+\w+:", line):
            multiline_buf.append(line.strip())
            continue

        # New key
        km = re.match(r"^(\w[\w-]*):\s*(.*)", line)
        if km:
            _flush()
            current_key = km.group(1)
            val = km.group(2).strip()

            if val == ">-" or val == ">":
                # multi-line scalar follows
                multiline_buf = []
                continue

            # Flow-style list: [a, b, c]
            flow = re.match(r"^\[(.*)\]$", val)
            if flow:
                items = [i.strip().strip("'\"") for i in flow.group(1).split(",") if i.strip()]
                result[current_key] = items
                current_key = None
                continue

            # Simple scalar
            result[current_key] = val.strip("'\"") if val else ""
            current_key = None

    _flush()
    return result


def _extract_identity(text: str) -> str:
    """Extract content under ## Identity section and build a persona string."""
    # Find the Identity section
    m = re.search(r"^## Identity\s*\n(.*?)(?=\n## |\Z)", text, re.MULTILINE | re.DOTALL)
    if not m:
        return ""
    content = m.group(1).strip()
    # Remove markdown references/instructions lines
    lines = [
        ln for ln in content.split("\n")
        if ln.strip() and not ln.strip().startswith("**Instructions Reference**")
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogEntry:
    slug: str
    name: str
    category: str
    description: str
    persona: str
    skill_slug: str
    recommended_model: str
    recommended_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    icon: str = ""
    tier: str = "free"


def parse_skill_file(skill_path: Path) -> Optional[CatalogEntry]:
    """Parse a single SKILL.md and return a CatalogEntry or None on failure."""
    text = skill_path.read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    if not fm.get("name") or not fm.get("category"):
        return None

    slug = skill_path.parent.name  # directory name = slug
    category = fm["category"]
    identity_text = _extract_identity(text)
    persona = f"You are a {fm['name']}. {identity_text}".strip() if identity_text else f"You are a {fm['name']}."

    raw_model = fm.get("recommended_model", "sonnet-4.6")
    model_id = MODEL_ID_MAP.get(raw_model, raw_model)

    tools = fm.get("recommended_tools", [])
    if isinstance(tools, str):
        tools = [tools]

    tags = fm.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]

    return CatalogEntry(
        slug=slug,
        name=fm["name"],
        category=category,
        description=fm.get("description", ""),
        persona=persona,
        skill_slug=slug,
        recommended_model=model_id,
        recommended_tools=tools,
        tags=tags,
        icon=CATEGORY_ICONS.get(category, "\u2B50"),
        tier="free",
    )


def collect_entries(skills_dir: Path) -> list[CatalogEntry]:
    """Walk skills_dir and collect CatalogEntry from every SKILL.md."""
    entries: list[CatalogEntry] = []
    for skill_md in sorted(skills_dir.rglob("SKILL.md")):
        entry = parse_skill_file(skill_md)
        if entry:
            entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Database upsert
# ---------------------------------------------------------------------------


def upsert_entries(entries: list[CatalogEntry]) -> int:
    """Upsert catalog entries into agent_catalog_templates. Returns row count."""
    # Lazy import so --dry-run works without a DB connection
    sys.path.insert(0, str(Path(__file__).parent.parent / "orchestrator"))
    from core.database.database import SessionLocal  # noqa: E402
    from core.models.core import AgentCatalogTemplate  # noqa: E402

    session = SessionLocal()
    count = 0
    try:
        for entry in entries:
            existing = (
                session.query(AgentCatalogTemplate)
                .filter(AgentCatalogTemplate.slug == entry.slug)
                .first()
            )
            if existing:
                existing.name = entry.name
                existing.category = entry.category
                existing.description = entry.description
                existing.persona = entry.persona
                existing.skill_slug = entry.skill_slug
                existing.recommended_model = entry.recommended_model
                existing.recommended_tools = entry.recommended_tools
                existing.tags = entry.tags
                existing.icon = entry.icon
                existing.tier = entry.tier
                existing.is_active = True
            else:
                row = AgentCatalogTemplate(
                    slug=entry.slug,
                    name=entry.name,
                    category=entry.category,
                    description=entry.description,
                    persona=entry.persona,
                    skill_slug=entry.skill_slug,
                    recommended_model=entry.recommended_model,
                    recommended_tools=entry.recommended_tools,
                    tags=entry.tags,
                    icon=entry.icon,
                    tier=entry.tier,
                    is_active=True,
                )
                session.add(row)
            count += 1
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    return count


# ---------------------------------------------------------------------------
# Entrypoint for Alembic data migration
# ---------------------------------------------------------------------------


def seed_from_alembic(op) -> None:
    """Called from an Alembic migration to seed rows using raw SQL.

    Uses op.execute() so it works inside the migration transaction without
    needing a SessionLocal.
    """
    import sqlalchemy as sa
    from sqlalchemy.sql import table, column

    tbl = table(
        "agent_catalog_templates",
        column("slug", sa.String),
        column("name", sa.String),
        column("category", sa.String),
        column("description", sa.Text),
        column("persona", sa.Text),
        column("skill_slug", sa.String),
        column("recommended_model", sa.String),
        column("recommended_tools", sa.JSON),
        column("tags", sa.JSON),
        column("icon", sa.String),
        column("tier", sa.String),
        column("is_active", sa.Boolean),
    )

    skills_dir = Path(__file__).parent.parent / "automatos-skills" / "skills"
    entries = collect_entries(skills_dir)

    bind = op.get_bind()
    for entry in entries:
        # Check if row exists
        exists = bind.execute(
            sa.text("SELECT 1 FROM agent_catalog_templates WHERE slug = :slug"),
            {"slug": entry.slug},
        ).fetchone()

        if exists:
            bind.execute(
                sa.text(
                    "UPDATE agent_catalog_templates SET "
                    "name = :name, category = :category, description = :description, "
                    "persona = :persona, skill_slug = :skill_slug, "
                    "recommended_model = :recommended_model, "
                    "recommended_tools = :recommended_tools, tags = :tags, "
                    "icon = :icon, tier = :tier, is_active = :is_active "
                    "WHERE slug = :slug"
                ),
                {
                    "slug": entry.slug,
                    "name": entry.name,
                    "category": entry.category,
                    "description": entry.description,
                    "persona": entry.persona,
                    "skill_slug": entry.skill_slug,
                    "recommended_model": entry.recommended_model,
                    "recommended_tools": json.dumps(entry.recommended_tools),
                    "tags": json.dumps(entry.tags),
                    "icon": entry.icon,
                    "tier": entry.tier,
                    "is_active": True,
                },
            )
        else:
            bind.execute(
                tbl.insert().values(
                    slug=entry.slug,
                    name=entry.name,
                    category=entry.category,
                    description=entry.description,
                    persona=entry.persona,
                    skill_slug=entry.skill_slug,
                    recommended_model=entry.recommended_model,
                    recommended_tools=entry.recommended_tools,
                    tags=entry.tags,
                    icon=entry.icon,
                    tier=entry.tier,
                    is_active=True,
                )
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed agent catalog templates from SKILL.md files")
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=Path(__file__).parent.parent / "automatos-skills" / "skills",
        help="Root directory containing category/slug/SKILL.md files",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print entries without writing to DB")
    args = parser.parse_args()

    entries = collect_entries(args.skills_dir)
    print(f"Found {len(entries)} SKILL.md files")

    if args.dry_run:
        for e in entries:
            print(f"  [{e.category}] {e.slug}: {e.name} (model={e.recommended_model})")
        print(f"\nDry run complete — {len(entries)} entries would be upserted.")
        return

    count = upsert_entries(entries)
    print(f"Upserted {count} rows into agent_catalog_templates.")


if __name__ == "__main__":
    main()
