#!/usr/bin/env python3
"""Import agent skills from agency-agents repo and convert to Automatos SKILL.md format.

Usage:
    python scripts/import_agency_skills.py \
        --source-dir scripts/vendor/agency-agents \
        --output-dir automatos-skills/skills \
        --category engineering

    python scripts/import_agency_skills.py \
        --source-dir scripts/vendor/agency-agents \
        --output-dir automatos-skills/skills \
        --category all \
        --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Division -> Automatos category mapping
# ---------------------------------------------------------------------------

DIVISION_TO_CATEGORY: dict[str, str] = {
    "engineering": "engineering",
    "design": "design",
    "marketing": "marketing",
    "sales": "sales",
    "product": "product",
    "project-management": "project-management",
    "testing": "testing",
    "support": "support",
    "paid-media": "paid-media",
    "specialized": "specialized",
}

# Divisions we skip entirely
SKIP_DIVISIONS: set[str] = {"game-development", "spatial-computing"}

# Individual agents to skip (slug form after division-prefix stripping, without .md)
SKIP_AGENTS: set[str] = {
    # China-specific marketing agents (original slug variants)
    "xiaohongshu-content-strategist",
    "wechat-ecosystem-strategist",
    "baidu-seo-specialist",
    "bilibili-content-strategist",
    "douyin-growth-strategist",
    "kuaishou-commerce-specialist",
    "weibo-engagement-strategist",
    # China-specific marketing agents (actual post-strip slugs)
    "douyin-strategist",
    "kuaishou-strategist",
    "weibo-strategist",
    "xiaohongshu-specialist",
    "wechat-official-account",
    "zhihu-strategist",
    "china-ecommerce-operator",
    "private-domain-operator",
    "livestream-commerce-coach",
    # China-specific engineering agents
    "feishu-integration-developer",
    "wechat-mini-program-developer",
    # Game-dev specific that might appear in other divisions
    "game-dev-specialist",
    # Tier 2 Testing — keep only: api-tester, performance-benchmarker, accessibility-auditor
    "evidence-collector",
    "reality-checker",
    "test-results-analyzer",
    "tool-evaluator",
    "workflow-optimizer",
    # Tier 2 Paid Media — keep only: ppc-strategist, creative-strategist, auditor
    "paid-social-strategist",
    "programmatic-buyer",
    "search-query-analyst",
    "tracking-specialist",
    # Tier 2 Specialized — keep only: document-generator, compliance-auditor,
    # recruitment-specialist, supply-chain-strategist, developer-advocate
    "accounts-payable-agent",
    "agentic-identity-trust",
    "agents-orchestrator",
    "automation-governance-architect",
    "blockchain-security-auditor",
    "corporate-training-designer",
    "data-consolidation-agent",
    "government-digital-presales-consultant",
    "healthcare-marketing-compliance",
    "identity-graph-operator",
    "lsp-index-engineer",
    "report-distribution-agent",
    "sales-data-extraction-agent",
    "cultural-intelligence-strategist",
    "french-consulting-market",
    "korean-business-navigator",
    "mcp-builder",
    "model-qa",
    "salesforce-architect",
    "workflow-architect",
    "study-abroad-advisor",
    "zk-steward",
}

# Non-Automatos platform references to strip
PLATFORM_REFERENCES: list[str] = [
    "OpenClaw", "Cursor", "Qwen", "Windsurf", "Aider",
    "Continue.dev", "Cline", "Roo Code", "Copilot",
]

# ---------------------------------------------------------------------------
# Tool mapping: agency-agents generic tools -> Automatos tools
# ---------------------------------------------------------------------------

TOOL_MAPPING: dict[str, str] = {
    "read": "workspace_read_file",
    "write": "workspace_write_file",
    "edit": "workspace_write_file",
    "bash": "workspace_exec",
    "terminal": "workspace_exec",
    "shell": "workspace_exec",
    "webfetch": "workspace_exec",
    "web": "workspace_exec",
    "fetch": "workspace_exec",
    "search": "workspace_grep",
    "grep": "workspace_grep",
    "find": "workspace_list_dir",
    "ls": "workspace_list_dir",
    "list": "workspace_list_dir",
    "git": "workspace_git",
}

SERVICE_TO_TOOL: dict[str, str] = {
    "github": "GITHUB",
    "gitlab": "GITHUB",
    "slack": "SLACK",
    "google sheets": "GOOGLE_SHEETS",
    "google_sheets": "GOOGLE_SHEETS",
    "google docs": "GOOGLE_DOCS",
    "google_docs": "GOOGLE_DOCS",
    "jira": "JIRA",
    "linear": "LINEAR",
    "notion": "NOTION",
    "figma": "FIGMA",
    "trello": "TRELLO",
    "asana": "ASANA",
    "hubspot": "HUBSPOT",
    "salesforce": "SALESFORCE",
    "stripe": "STRIPE",
    "zendesk": "ZENDESK",
}

# ---------------------------------------------------------------------------
# Model assignment by category/complexity
# ---------------------------------------------------------------------------

# Categories that default to specific models
CATEGORY_MODEL_DEFAULTS: dict[str, str] = {
    "support": "haiku-4.5",
    "testing": "sonnet-4.6",
    "paid-media": "sonnet-4.6",
}

# Slugs that need opus-level reasoning
OPUS_SLUGS: set[str] = {
    "backend-architect",
    "security-engineer",
    "ai-engineer",
    "software-architect",
    "product-strategist",
    "compliance-auditor",
    "system-architect",
    "principal-engineer",
    "tech-lead",
    "staff-engineer",
    "autonomous-optimization-architect",
}

# Slugs that are simple/repetitive -> haiku
HAIKU_SLUGS: set[str] = {
    "support-responder",
    "ticket-triager",
    "faq-updater",
    "status-page-writer",
}


def assign_model(slug: str, category: str) -> str:
    """Assign recommended model based on complexity heuristics."""
    if slug in OPUS_SLUGS:
        return "opus-4.6"
    if slug in HAIKU_SLUGS:
        return "haiku-4.5"
    if category in CATEGORY_MODEL_DEFAULTS:
        return CATEGORY_MODEL_DEFAULTS[category]
    return "sonnet-4.6"


# ---------------------------------------------------------------------------
# Tag generation
# ---------------------------------------------------------------------------

CATEGORY_BASE_TAGS: dict[str, list[str]] = {
    "engineering": ["development", "software"],
    "design": ["design", "creative"],
    "marketing": ["marketing", "content"],
    "sales": ["sales", "revenue"],
    "product": ["product", "strategy"],
    "project-management": ["project-management", "coordination"],
    "testing": ["testing", "quality"],
    "support": ["support", "operations"],
    "paid-media": ["advertising", "media"],
    "specialized": ["specialist"],
}


def generate_tags(slug: str, category: str, description: str) -> list[str]:
    """Generate searchable tags from slug, category, and description."""
    tags = list(CATEGORY_BASE_TAGS.get(category, []))
    # Add slug words as tags (split on hyphens)
    slug_words = slug.split("-")
    for word in slug_words:
        if word not in tags and len(word) > 2:
            tags.append(word)
    return tags[:8]  # Cap at 8 tags


# ---------------------------------------------------------------------------
# YAML frontmatter parser (stdlib only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceFrontmatter:
    """Parsed frontmatter from an agency-agents markdown file."""
    name: str = ""
    description: str = ""
    color: str = ""
    emoji: str = ""
    vibe: str = ""
    services: list[str] = field(default_factory=list)
    raw: dict[str, str] = field(default_factory=dict)


def parse_frontmatter(content: str) -> tuple[SourceFrontmatter, str]:
    """Parse YAML frontmatter between --- delimiters. Returns (frontmatter, body)."""
    if not content.startswith("---"):
        return SourceFrontmatter(), content

    # Find closing ---
    end_idx = content.find("---", 3)
    if end_idx == -1:
        return SourceFrontmatter(), content

    yaml_block = content[3:end_idx].strip()
    body = content[end_idx + 3:].strip()

    # Simple YAML parser for flat key: value pairs
    raw: dict[str, str] = {}
    current_key: Optional[str] = None
    current_value_lines: list[str] = []

    for line in yaml_block.split("\n"):
        # Check for new key: value pair
        match = re.match(r'^(\w[\w-]*)\s*:\s*(.*)', line)
        if match:
            # Save previous key if any
            if current_key is not None:
                raw[current_key] = "\n".join(current_value_lines).strip()
            current_key = match.group(1)
            current_value_lines = [match.group(2)]
        elif current_key is not None:
            # Continuation line
            current_value_lines.append(line)

    if current_key is not None:
        raw[current_key] = "\n".join(current_value_lines).strip()

    # Clean quoted strings
    for key, val in raw.items():
        if val.startswith('"') and val.endswith('"'):
            raw[key] = val[1:-1]
        elif val.startswith("'") and val.endswith("'"):
            raw[key] = val[1:-1]

    # Parse services (may be comma-separated or YAML list)
    services: list[str] = []
    if "services" in raw:
        svc_str = raw["services"]
        if svc_str.startswith("["):
            svc_str = svc_str.strip("[]")
        services = [s.strip().strip("'\"") for s in svc_str.split(",") if s.strip()]

    return SourceFrontmatter(
        name=raw.get("name", ""),
        description=raw.get("description", ""),
        color=raw.get("color", "").strip('"'),
        emoji=raw.get("emoji", ""),
        vibe=raw.get("vibe", ""),
        services=services,
        raw=raw,
    ), body


# ---------------------------------------------------------------------------
# Tool extraction
# ---------------------------------------------------------------------------

def extract_tools(body: str, services: list[str]) -> list[str]:
    """Extract recommended Automatos tools from body content and services."""
    tools: set[str] = set()

    # Default workspace tools for all agents
    tools.add("workspace_read_file")
    tools.add("workspace_write_file")

    # Scan body for tool references
    body_lower = body.lower()
    for generic_name, automatos_tool in TOOL_MAPPING.items():
        if generic_name in body_lower:
            tools.add(automatos_tool)

    # Map services to Composio tools
    for svc in services:
        svc_lower = svc.lower().strip()
        if svc_lower in SERVICE_TO_TOOL:
            tools.add(SERVICE_TO_TOOL[svc_lower])

    # Scan body for service mentions
    for svc_name, composio_tool in SERVICE_TO_TOOL.items():
        if svc_name.lower() in body_lower:
            tools.add(composio_tool)

    return sorted(tools)


# ---------------------------------------------------------------------------
# Section extraction and body conversion
# ---------------------------------------------------------------------------

# Mapping of source section names to Automatos section names
SECTION_MAPPING: dict[str, str] = {
    "identity": "Identity",
    "identity & memory": "Identity",
    "identity and memory": "Identity",
    "who you are": "Identity",
    "role": "Identity",
    "persona": "Identity",
    "core mission": "Core Mission",
    "mission": "Core Mission",
    "purpose": "Core Mission",
    "objective": "Core Mission",
    "primary objective": "Core Mission",
    "workflow": "Workflow",
    "workflow process": "Workflow",
    "process": "Workflow",
    "methodology": "Workflow",
    "approach": "Workflow",
    "how you work": "Workflow",
    "standard operating procedure": "Workflow",
    "deliverables": "Deliverables",
    "technical deliverables": "Deliverables",
    "outputs": "Deliverables",
    "output format": "Deliverables",
    "output": "Deliverables",
    "rules": "Rules",
    "critical rules": "Rules",
    "constraints": "Rules",
    "guidelines": "Rules",
    "guardrails": "Rules",
    "boundaries": "Rules",
    "principles": "Rules",
}

# Sections from source we skip
SKIP_SECTIONS: set[str] = {
    "communication style",
    "learning & memory",
    "learning and memory",
    "success metrics",
    "advanced capabilities",
    "tools",
    "tool usage",
    "services",
    "integrations",
    "platform",
    "emoji",
}


def extract_sections(body: str) -> dict[str, str]:
    """Extract markdown sections from body, mapping to Automatos section names."""
    sections: dict[str, list[str]] = {}
    current_section: Optional[str] = None
    current_lines: list[str] = []

    for line in body.split("\n"):
        # Match ## or # headings
        heading_match = re.match(r'^#{1,3}\s+(.+)', line)
        if heading_match:
            # Save previous section
            if current_section is not None:
                sections[current_section] = current_lines
            heading_text = heading_match.group(1).strip()
            heading_lower = heading_text.lower().rstrip(":")

            # Map to Automatos section or keep raw
            if heading_lower in SKIP_SECTIONS:
                current_section = None
                current_lines = []
                continue

            mapped = SECTION_MAPPING.get(heading_lower)
            if mapped:
                current_section = mapped
            else:
                # Keep unmapped sections under closest match or Rules
                current_section = _best_section_match(heading_lower)
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    # Save last section
    if current_section is not None:
        sections[current_section] = current_lines

    # Build result dict, joining lines
    result: dict[str, str] = {}
    for section_name, lines in sections.items():
        content = "\n".join(lines).strip()
        if content:
            if section_name in result:
                result[section_name] += "\n\n" + content
            else:
                result[section_name] = content

    return result


def _best_section_match(heading: str) -> str:
    """Find the closest Automatos section for an unmapped heading."""
    # Keywords that hint at each section
    hints: dict[str, list[str]] = {
        "Identity": ["who", "about", "background", "expertise", "profile"],
        "Core Mission": ["goal", "mission", "focus", "objective", "purpose", "value"],
        "Workflow": ["step", "process", "flow", "method", "phase", "procedure", "how"],
        "Deliverables": ["output", "produce", "create", "deliver", "result", "artifact"],
        "Rules": ["rule", "must", "never", "always", "constraint", "limit", "guard", "principle", "standard"],
    }
    for section, keywords in hints.items():
        if any(kw in heading for kw in keywords):
            return section
    return "Rules"  # Default unmapped content to Rules


def strip_platform_references(text: str) -> str:
    """Remove references to non-Automatos platforms."""
    for platform in PLATFORM_REFERENCES:
        # Remove whole lines that are primarily about the platform
        text = re.sub(
            rf'^.*\b{re.escape(platform)}\b.*$',
            '',
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    # Remove double blank lines left behind
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def strip_emoji_markers(text: str) -> str:
    """Remove emoji section markers like '## 🎯 Core Mission' -> '## Core Mission'."""
    # Remove emoji at start of headings
    text = re.sub(r'^(#{1,3}\s+)[^\w\s]+ *', r'\1', text, flags=re.MULTILINE)
    return text


# ---------------------------------------------------------------------------
# SKILL.md generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConvertedSkill:
    """A converted skill ready to write."""
    slug: str
    category: str
    name: str
    description: str
    version: str
    tags: list[str]
    recommended_tools: list[str]
    recommended_model: str
    identity: str
    core_mission: str
    workflow: str
    deliverables: str
    rules: str


def _strip_division_prefix(filename_stem: str, division: str) -> str:
    """Strip the division prefix from a filename slug.

    e.g. 'engineering-backend-architect' with division 'engineering'
    becomes 'backend-architect'.
    """
    prefix = f"{division}-"
    if filename_stem.startswith(prefix):
        return filename_stem[len(prefix):]
    return filename_stem


def convert_agent_file(source_path: Path, category: str) -> Optional[ConvertedSkill]:
    """Convert a single agency-agents markdown file to Automatos skill."""
    raw_stem = source_path.stem  # filename without .md
    # Source division = parent directory name
    source_division = source_path.parent.name
    slug = _strip_division_prefix(raw_stem, source_division)

    if slug in SKIP_AGENTS:
        return None

    content = source_path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(content)

    if not fm.name:
        return None

    # Clean the body
    body = strip_emoji_markers(body)
    body = strip_platform_references(body)

    # Extract sections
    sections = extract_sections(body)

    # Build identity from first paragraph if no explicit section
    identity = sections.get("Identity", "")
    if not identity:
        # Use first non-empty paragraph from body
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        if paragraphs:
            identity = paragraphs[0]

    core_mission = sections.get("Core Mission", "")
    if not core_mission and fm.vibe:
        core_mission = fm.vibe

    workflow = sections.get("Workflow", "")
    deliverables = sections.get("Deliverables", "")
    rules = sections.get("Rules", "")

    # Use description from frontmatter, fallback to vibe
    description = fm.description or fm.vibe or f"Professional {category} agent skill."

    # Truncate description to reasonable length
    if len(description) > 300:
        description = description[:297] + "..."

    tools = extract_tools(body, fm.services)
    tags = generate_tags(slug, category, description)
    model = assign_model(slug, category)

    return ConvertedSkill(
        slug=slug,
        category=category,
        name=fm.name,
        description=description,
        version="1.0.0",
        tags=tags,
        recommended_tools=tools,
        recommended_model=model,
        identity=identity,
        core_mission=core_mission,
        workflow=workflow,
        deliverables=deliverables,
        rules=rules,
    )


def render_skill_md(skill: ConvertedSkill) -> str:
    """Render a ConvertedSkill to SKILL.md format."""
    # Build YAML frontmatter
    tools_yaml = "\n".join(f"  - {t}" for t in skill.recommended_tools)
    tags_yaml = ", ".join(skill.tags)

    # Escape description for YAML (use >- block scalar for long descriptions)
    desc_lines = textwrap.wrap(skill.description, width=78)
    if len(desc_lines) > 1:
        desc_yaml = ">-\n" + "\n".join(f"  {line}" for line in desc_lines)
    else:
        desc_yaml = skill.description

    frontmatter = f"""---
name: {skill.name}
version: {skill.version}
category: {skill.category}
tags: [{tags_yaml}]
description: {desc_yaml}
recommended_tools:
{tools_yaml}
recommended_model: {skill.recommended_model}
---"""

    # Build body sections
    sections: list[str] = []

    if skill.identity:
        sections.append(f"## Identity\n\n{skill.identity}")
    else:
        sections.append(f"## Identity\n\nYou are a {skill.name} specializing in {skill.category}.")

    if skill.core_mission:
        sections.append(f"## Core Mission\n\n{skill.core_mission}")
    else:
        sections.append(f"## Core Mission\n\n{skill.description}")

    if skill.workflow:
        sections.append(f"## Workflow\n\n{skill.workflow}")
    else:
        sections.append(
            f"## Workflow\n\n"
            f"1. Analyze the task requirements and constraints\n"
            f"2. Research relevant context and existing solutions\n"
            f"3. Develop and implement the solution iteratively\n"
            f"4. Validate output quality and completeness\n"
            f"5. Document decisions and deliver results"
        )

    if skill.deliverables:
        sections.append(f"## Deliverables\n\n{skill.deliverables}")
    else:
        sections.append(
            f"## Deliverables\n\n"
            f"- Completed work artifacts relevant to the task\n"
            f"- Documentation of approach and key decisions\n"
            f"- Summary of findings or changes made"
        )

    if skill.rules:
        sections.append(f"## Rules\n\n{skill.rules}")
    else:
        sections.append(
            f"## Rules\n\n"
            f"- Follow established best practices for {skill.category}\n"
            f"- Validate all work before marking complete\n"
            f"- Document assumptions and trade-offs\n"
            f"- Ask for clarification when requirements are ambiguous"
        )

    body = "\n\n".join(sections)

    full_content = f"{frontmatter}\n\n{body}\n"

    # Trim to 200-400 lines — if over 400, truncate workflow/deliverables/rules
    lines = full_content.split("\n")
    if len(lines) > 400:
        full_content = "\n".join(lines[:400]) + "\n"

    return full_content


# ---------------------------------------------------------------------------
# Main import logic
# ---------------------------------------------------------------------------

def discover_source_divisions(source_dir: Path) -> dict[str, list[Path]]:
    """Discover all divisions and their agent files in the source directory."""
    divisions: dict[str, list[Path]] = {}
    for child in sorted(source_dir.iterdir()):
        if not child.is_dir():
            continue
        division_name = child.name
        if division_name in SKIP_DIVISIONS:
            continue
        if division_name in ("scripts", "integrations", "examples", ".git", ".github"):
            continue
        md_files = sorted(child.glob("*.md"))
        # Filter out README, LICENSE, etc.
        agent_files = [f for f in md_files if f.stem.lower() not in ("readme", "license", "changelog")]
        if agent_files:
            divisions[division_name] = agent_files
    return divisions


def import_category(
    source_dir: Path,
    output_dir: Path,
    category: str,
    dry_run: bool = False,
) -> list[ConvertedSkill]:
    """Import all agents from a single division/category."""
    # Map category to source division name (they match 1:1)
    source_division_dir = source_dir / category
    if not source_division_dir.is_dir():
        print(f"  WARNING: Source division '{category}' not found at {source_division_dir}")
        return []

    md_files = sorted(source_division_dir.glob("*.md"))
    agent_files = [f for f in md_files if f.stem.lower() not in ("readme", "license", "changelog")]

    converted: list[ConvertedSkill] = []
    for agent_file in agent_files:
        skill = convert_agent_file(agent_file, DIVISION_TO_CATEGORY.get(category, category))
        if skill is None:
            print(f"  SKIP: {agent_file.name}")
            continue

        if dry_run:
            print(f"  WOULD CREATE: {skill.category}/{skill.slug}/SKILL.md")
            print(f"    Name: {skill.name}")
            print(f"    Model: {skill.recommended_model}")
            print(f"    Tools: {', '.join(skill.recommended_tools)}")
            print(f"    Tags: {', '.join(skill.tags)}")
        else:
            skill_dir = output_dir / skill.category / skill.slug
            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_path = skill_dir / "SKILL.md"
            content = render_skill_md(skill)
            skill_path.write_text(content, encoding="utf-8")
            line_count = len(content.split("\n"))
            print(f"  CREATED: {skill.category}/{skill.slug}/SKILL.md ({line_count} lines)")

        converted.append(skill)

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import agent skills from agency-agents repo to Automatos SKILL.md format.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Path to cloned agency-agents repository",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for Automatos skills (e.g. automatos-skills/skills)",
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Category to import (e.g. 'engineering') or 'all' for all categories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print conversion plan without writing files",
    )

    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not source_dir.is_dir():
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Determine categories to process
    if args.category == "all":
        categories = list(DIVISION_TO_CATEGORY.keys())
    else:
        categories = [args.category]

    total_converted = 0
    for category in categories:
        print(f"\n{'='*60}")
        print(f"Processing: {category}")
        print(f"{'='*60}")
        converted = import_category(source_dir, output_dir, category, dry_run=args.dry_run)
        total_converted += len(converted)
        print(f"  -> {len(converted)} skills {'planned' if args.dry_run else 'created'}")

    print(f"\n{'='*60}")
    print(f"Total: {total_converted} skills {'planned' if args.dry_run else 'created'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
