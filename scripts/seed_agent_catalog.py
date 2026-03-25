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

# Category → default model slug (most skills use sonnet-4.6)
CATEGORY_MODEL_MAP: dict[str, str] = {
    "engineering": "sonnet-4.6",
    "design": "sonnet-4.6",
    "marketing": "sonnet-4.6",
    "sales": "sonnet-4.6",
    "product": "sonnet-4.6",
    "project-management": "sonnet-4.6",
    "support": "sonnet-4.6",
    "testing": "sonnet-4.6",
    "paid-media": "sonnet-4.6",
    "specialized": "sonnet-4.6",
    "agent-role": "sonnet-4.6",
    "productivity": "sonnet-4.6",
    "social-media": "sonnet-4.6",
}

# Per-skill model overrides (slug → model slug)
SLUG_MODEL_OVERRIDES: dict[str, str] = {
    # Complex reasoning → opus
    "backend-architect": "opus-4.6",
    "security-engineer": "opus-4.6",
    "software-architect": "opus-4.6",
    "incident-response-commander": "opus-4.6",
    "threat-detection-engineer": "opus-4.6",
    "deal-strategist": "opus-4.6",
    "proposal-strategist": "opus-4.6",
    "ux-architect": "opus-4.6",
    "manager": "opus-4.6",
    "project-manager-senior": "opus-4.6",
    "legal-compliance-checker": "opus-4.6",
    "compliance-auditor": "opus-4.6",
    "growth-hacker": "opus-4.6",
    # Simple/repetitive → haiku
    "support-responder": "haiku-4.5",
    "analytics-reporter": "haiku-4.5",
    "finance-tracker": "haiku-4.5",
}

# ---------------------------------------------------------------------------
# YAML frontmatter parser (stdlib only — no PyYAML dependency)
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> dict:
    """Parse YAML frontmatter between --- delimiters.

    Handles scalar values, flow-style lists ([a, b, c]), multi-line >- scalars,
    block lists of strings, and block lists of {name, description} dicts (the
    tools: format used by SKILL.md v2).
    """
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return {}

    lines = m.group(1).split("\n")
    result: dict = {}
    current_key: Optional[str] = None
    multiline_buf: list[str] = []
    list_buf: list = []
    current_dict: Optional[dict] = None

    def _flush() -> None:
        nonlocal current_key, multiline_buf, list_buf, current_dict
        if current_key is None:
            return
        if current_dict:
            list_buf.append(current_dict)
            current_dict = None
        if list_buf:
            result[current_key] = list_buf
            list_buf = []
        elif multiline_buf:
            result[current_key] = " ".join(multiline_buf)
            multiline_buf = []
        current_key = None

    for line in lines:
        # Block-list item starting with "  - "
        if re.match(r"^\s+-\s+", line) and current_key:
            # Flush any previous dict in the list
            if current_dict:
                list_buf.append(current_dict)
                current_dict = None

            val = re.sub(r"^\s+-\s+", "", line).strip()
            # Check if it's a key: value (start of a dict item)
            kv = re.match(r"^(\w[\w-]*):\s*(.*)", val)
            if kv:
                current_dict = {kv.group(1): kv.group(2).strip().strip("'\"") if kv.group(2).strip() else ""}
            else:
                list_buf.append(val)
            continue

        # Nested dict key continuation: "    key: value"
        if current_dict and re.match(r"^\s{4,}\w", line):
            kv = re.match(r"^\s+(\w[\w-]*):\s*(.*)", line)
            if kv:
                current_dict[kv.group(1)] = kv.group(2).strip().strip("'\"") if kv.group(2).strip() else ""
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
                multiline_buf = []
                continue

            # Flow-style list: [a, b, c]
            flow = re.match(r"^\[(.*)\]$", val)
            if flow:
                items = [i.strip().strip("'\"") for i in flow.group(1).split(",") if i.strip()]
                result[current_key] = items
                current_key = None
                continue

            # Simple scalar (but empty value may precede a list — keep key active)
            if val:
                result[current_key] = val.strip("'\"")
                current_key = None
            # else: keep current_key — a list or block may follow

    _flush()
    return result


def _extract_identity(text: str) -> str:
    """Extract persona text from the skill body.

    Tries in order:
    1. Text under ## Identity section (legacy format)
    2. First paragraph after the top-level # heading (new format, e.g. Sentinel)
    """
    # Try ## Identity section first
    m = re.search(r"^## Identity\s*\n(.*?)(?=\n## |\Z)", text, re.MULTILINE | re.DOTALL)
    if m:
        content = m.group(1).strip()
        lines = [
            ln for ln in content.split("\n")
            if ln.strip() and not ln.strip().startswith("**Instructions Reference**")
        ]
        return "\n".join(lines)

    # Try first paragraph after top-level heading (after frontmatter)
    body = re.sub(r"^---\n.*?\n---\s*", "", text, flags=re.DOTALL)
    m = re.search(r"^#\s+.+\n\n(.+?)(?=\n\n|\n##|\Z)", body)
    if m:
        return m.group(1).strip()

    return ""


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

    # Model: slug override > category default > sonnet-4.6
    raw_model = SLUG_MODEL_OVERRIDES.get(slug, CATEGORY_MODEL_MAP.get(category, "sonnet-4.6"))
    model_id = MODEL_ID_MAP.get(raw_model, raw_model)

    # Extract tool names from tools: [{name, description}] or fall back to
    # legacy recommended_tools: [string] format
    raw_tools = fm.get("tools", [])
    if raw_tools and isinstance(raw_tools[0], dict):
        tools = [t["name"] for t in raw_tools if isinstance(t, dict) and "name" in t]
    else:
        tools = fm.get("recommended_tools", raw_tools)
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

    # Skills live in sibling repo: automatos-skills (not inside automatos-ai)
    skills_dir = Path(__file__).parent.parent.parent / "automatos-skills"
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
        default=Path(__file__).parent.parent.parent / "automatos-skills",
        help="Root directory of automatos-skills repo (sibling to automatos-ai)",
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
