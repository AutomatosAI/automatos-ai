"""PRD-239 S1 — an agent's soul and skills, rendered for its Claude Code session.

PRD-234 S1b said the appended system prompt is "agent soul + skills + dispatch
doctrine"; the host only ever wrote the name and the session rules. This renders
the missing two thirds server-side, once per claim, in a shape that is STABLE per
agent — no ids, dates or counters — so Claude Code's prompt cache keeps holding
across a session's turns (munder's invariant, PRD-234 §Design 2).

Why full skill bodies: a session has no ``platform_load_skill`` tool, so the L1
catalogue the API path renders (``SkillsSection``) would offer skills it can
never load. Bodies are capped by ``CLI_SESSION_SKILLS_MAX_CHARS``; a skill that
does not fit keeps its name and description so the agent still knows it exists.
"""
from __future__ import annotations

from typing import Any, List, Optional

from modules.context.sections.identity import IdentitySection
from modules.context.sections.skills import SkillsSection

SKILLS_HEADER = (
    "## Skills\n"
    "These skills are attached to you. Apply the one whose description matches "
    "the work; their instructions follow."
)
OMITTED_NOTE = "_(full instructions omitted for space — ask the operator for them if you need them)_"
TRUNCATED_NOTE = "\n…(truncated for space)"
DEFAULT_SKILLS_MAX_CHARS = 24000


def _skills_max_chars() -> int:
    try:
        from config import config

        return max(0, int(getattr(config, "CLI_SESSION_SKILLS_MAX_CHARS", DEFAULT_SKILLS_MAX_CHARS)))
    except Exception:  # noqa: BLE001 — isolated unit tests import without config
        return DEFAULT_SKILLS_MAX_CHARS


def _text(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def skill_entry(skill: Any, budget: int, first: bool) -> Optional[str]:
    """One skill as a section; ``None`` when it needs more than ``budget`` chars
    and is not the first (the first always renders, truncated if it must)."""
    name = _text(getattr(skill, "name", None)) or "skill"
    desc = _text(getattr(skill, "description", None))
    body = _text(getattr(skill, "prompt_template", None))
    head = f"### {name}\n{desc}" if desc else f"### {name}"
    text = f"{head}\n\n{body}" if body else head
    if len(text) <= budget:
        return text
    if first:
        return text[: max(0, budget - len(TRUNCATED_NOTE))] + TRUNCATED_NOTE
    return None


def skills_block(agent: Any, max_chars: Optional[int] = None) -> str:
    """The ``## Skills`` section: every active skill, bodies within the cap."""
    skills = [s for s in (getattr(agent, "skills", None) or []) if getattr(s, "is_active", True)]
    skills = SkillsSection._dedup(skills)
    if not skills:
        return ""
    budget = _skills_max_chars() if max_chars is None else max(0, int(max_chars))
    entries: List[str] = []
    for index, skill in enumerate(skills):
        entry = skill_entry(skill, budget, first=(index == 0))
        if entry is None:
            name = _text(getattr(skill, "name", None)) or "skill"
            desc = _text(getattr(skill, "description", None))
            entries.append(f"### {name}\n{desc}\n{OMITTED_NOTE}" if desc else f"### {name}\n{OMITTED_NOTE}")
            continue
        entries.append(entry)
        budget = max(0, budget - len(entry))
    return SKILLS_HEADER + "\n\n" + "\n\n".join(entries)


def session_system_prompt(agent: Any, max_skill_chars: Optional[int] = None) -> str:
    """Description, persona and skills for the session's appended system prompt.

    Empty when the agent carries none of them — the host then writes exactly
    what it wrote before (name + rules).
    """
    if agent is None:
        return ""
    parts: List[str] = []
    description = _text(getattr(agent, "description", None))
    if description:
        parts.append(f"## About you\n{description}")
    persona = IdentitySection._get_persona_text(agent)
    if persona:
        parts.append(f"## Persona & Communication Style\n{persona}")
    skills = skills_block(agent, max_skill_chars)
    if skills:
        parts.append(skills)
    return "\n\n".join(parts)
