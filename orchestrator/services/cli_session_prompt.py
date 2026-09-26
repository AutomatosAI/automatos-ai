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

PRD-245 S0.6 (D8): the prompt tells the truth about tools. A ``## Tools in this
session`` block names what a ticket session can use (files inside its roots,
the Bash allowlist, the web) and says that the platform tools skill bodies call
are NOT there; each skill header names the ones it would have called.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from core.cli_presets import SESSION_BASH_VERBS
from core.cli_runtime import CONFIG_ALLOWED_TOOLS_KEY
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

# PRD-245 S0.6: the tool-name families skill bodies call that a session does not
# have (the API agents' platform tools). A name in a skill body that starts with
# one of these and is not in the session's own list is a gap the prompt names.
SESSION_UNAVAILABLE_TOOL_PREFIXES = ("composio_execute", "platform_", "search_knowledge", "scratchpad_", "workspace_")
# The Automatos tools a session DOES have — PRD-245 W1's loopback MCP bridge,
# read from the ONE definition (``services/session_tools.py``), so the prompt,
# the claim payload, the host's gate and the agent form can never disagree.
from services.session_tools import tool_names as _session_tool_names

SESSION_TOOLS_AVAILABLE: Sequence[str] = _session_tool_names()
TOOLS_HEADER = "## Tools in this session"
# F167: a session wrote that a command "needed your approval, and it went through"
# when nobody had been asked. To the session, a command that runs looks the same
# whether the host allowed it outright or the operator approved a hold.
APPROVALS_LINE = (
    "- Approvals: never write that anything needed, got or went through the operator's approval "
    "unless a question card was answered: your own `ask_human` question, with the answer in this "
    "ticket. A command that runs looks the same whether or not anyone was asked, so say that it "
    "ran, never that it was approved."
)
GAP_LINE_PREFIX = "Not available in a session: "
INSTEAD_LINE = "In a session, call {swaps}."
# The CLI's OWN tools do these jobs; the bridge has no equivalent and needs none.
CLI_NATIVE_EQUIVALENTS = {
    "workspace_read_file": "Read",
    "workspace_write_file": "Write",
    "workspace_edit_file": "Edit",
    "workspace_list_files": "Glob",
    "workspace_grep": "Grep",
}
_TOOL_NAME_RE = re.compile(r"(?<![A-Za-z0-9_])([a-z][a-z0-9_]+)(?![A-Za-z0-9_])")


def _skills_max_chars() -> int:
    try:
        from config import config

        return max(0, int(getattr(config, "CLI_SESSION_SKILLS_MAX_CHARS", DEFAULT_SKILLS_MAX_CHARS)))
    except Exception:  # noqa: BLE001 — isolated unit tests import without config
        return DEFAULT_SKILLS_MAX_CHARS


def _text(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _active_skills(agent: Any) -> List[Any]:
    skills = [s for s in (getattr(agent, "skills", None) or []) if getattr(s, "is_active", True)]
    return SkillsSection._dedup(skills)


def _is_platform_tool_name(name: str) -> bool:
    """``platform_submit_report`` yes; the bare family word ``platform_`` no."""
    for prefix in SESSION_UNAVAILABLE_TOOL_PREFIXES:
        if name.startswith(prefix) and (not prefix.endswith("_") or len(name) > len(prefix)):
            return True
    return False


def tool_names_in(body: str) -> List[str]:
    """The platform tool names a skill body mentions, first appearance, deduped."""
    out: List[str] = []
    for match in _TOOL_NAME_RE.finditer(body or ""):
        name = match.group(1)
        if name not in out and _is_platform_tool_name(name):
            out.append(name)
    return out


def session_tool_gaps(agent: Any, available: Sequence[str]) -> List[Dict[str, Any]]:
    """PRD-245 S0.6/S1.5: per active skill, the platform tools its body calls
    that the session cannot call — and, where the session has an equivalent under
    another name, what to call instead:
    ``[{"skill": name, "tools": [names], "instead": {mentioned: session_name}}]``.
    Skills with nothing to say are omitted. Pure; the agent form and the session
    prompt share it."""
    from services.session_tools import equivalent_of

    offered = set(available or ())
    # A skill that says ``workspace_read_file`` is asking to read a file, and the
    # session has a better tool for it than any bridge: the CLI's own. Naming
    # these as simply "not available" told WRITER and AUTOMATOS-DEV their file
    # tools did not exist, one line under the sentence that says they can read
    # and edit.
    native = CLI_NATIVE_EQUIVALENTS
    gaps: List[Dict[str, Any]] = []
    for skill in _active_skills(agent):
        missing: List[str] = []
        instead: Dict[str, str] = {}
        for name in tool_names_in(_text(getattr(skill, "prompt_template", None))):
            if name in offered:
                continue
            replacement = equivalent_of(name)
            if replacement and replacement in offered:
                instead[name] = replacement
            elif name in native:
                instead[name] = native[name]
            else:
                missing.append(name)
        if missing or instead:
            entry: Dict[str, Any] = {"skill": _text(getattr(skill, "name", None)) or "skill", "tools": missing}
            if instead:
                entry["instead"] = instead
            gaps.append(entry)
    return gaps


def _gap_line(gaps: Sequence[str], instead: Optional[Dict[str, str]] = None) -> str:
    """What to say under a skill whose body calls tools by the API agents' names:
    the replacement first (the agent CAN do the work, under another name), then
    the ones a session genuinely does not have."""
    parts: List[str] = []
    pairs = sorted((instead or {}).items())
    if pairs:
        swaps = ", ".join(f"`{replacement}` instead of `{mentioned}`" for mentioned, replacement in pairs)
        parts.append(INSTEAD_LINE.format(swaps=swaps))
    if gaps:
        parts.append(GAP_LINE_PREFIX + ", ".join(f"`{g}`" for g in gaps) + ".")
    return " ".join(parts)


def skill_entry(skill: Any, budget: int, first: bool, gaps: Sequence[str] = (),
                instead: Optional[Dict[str, str]] = None) -> Optional[str]:
    """One skill as a section; ``None`` when it needs more than ``budget`` chars
    and is not the first (the first always renders, truncated if it must).
    ``gaps`` — the tools this skill names that the session cannot call — ride
    the header, so they survive a truncated body."""
    name = _text(getattr(skill, "name", None)) or "skill"
    desc = _text(getattr(skill, "description", None))
    body = _text(getattr(skill, "prompt_template", None))
    head = "\n".join(line for line in (f"### {name}", desc, _gap_line(gaps, instead)) if line)
    text = f"{head}\n\n{body}" if body else head
    if len(text) <= budget:
        return text
    if first:
        return text[: max(0, budget - len(TRUNCATED_NOTE))] + TRUNCATED_NOTE
    return None


def _omitted_entry(skill: Any, gaps: Sequence[str], instead: Optional[Dict[str, str]] = None) -> str:
    name = _text(getattr(skill, "name", None)) or "skill"
    desc = _text(getattr(skill, "description", None))
    return "\n".join(line for line in (f"### {name}", desc, _gap_line(gaps, instead), OMITTED_NOTE) if line)


def skills_block(agent: Any, max_chars: Optional[int] = None) -> str:
    """The ``## Skills`` section: every active skill, bodies within the cap."""
    skills = _active_skills(agent)
    if not skills:
        return ""
    by_skill = {g["skill"]: g for g in session_tool_gaps(agent, SESSION_TOOLS_AVAILABLE)}
    budget = _skills_max_chars() if max_chars is None else max(0, int(max_chars))
    entries: List[str] = []
    for index, skill in enumerate(skills):
        found = by_skill.get(_text(getattr(skill, "name", None)) or "skill") or {}
        gaps, instead = found.get("tools") or (), found.get("instead") or {}
        entry = skill_entry(skill, budget, first=(index == 0), gaps=gaps, instead=instead)
        if entry is None:
            entries.append(_omitted_entry(skill, gaps, instead))
            continue
        entries.append(entry)
        budget = max(0, budget - len(entry))
    return SKILLS_HEADER + "\n\n" + "\n\n".join(entries)


def _configured_bash_extras(agent: Any) -> List[str]:
    """The agent's own Bash prefixes (``configuration.allowed_tools``) — the host
    adds them on top of the defaults (``bash_allowlist_from_config``)."""
    cfg = getattr(agent, "configuration", None)
    raw = cfg.get(CONFIG_ALLOWED_TOOLS_KEY) if isinstance(cfg, dict) else None
    return [c.strip() for c in (raw or []) if isinstance(c, str) and c.strip()]


def _family_label(prefix: str) -> str:
    return f"`{prefix}*`" if prefix.endswith("_") else f"`{prefix}`"


def _unavailable_families(available: Sequence[str]) -> List[str]:
    """The families to warn about, minus any whose exact name the session HAS —
    naming `search_knowledge` as unavailable one line under the list that offers
    it is how a prompt loses the agent's trust."""
    offered = set(available or ())
    return [p for p in SESSION_UNAVAILABLE_TOOL_PREFIXES if p.endswith("_") or p not in offered]


def tools_block(agent: Any, available: Sequence[str] = SESSION_TOOLS_AVAILABLE) -> str:
    """PRD-245 S0.6: what a ticket session can use — stable per agent, no ids."""
    verbs = ", ".join(f"`{v}`" for v in SESSION_BASH_VERBS)
    extras = _configured_bash_extras(agent)
    bash = f"- Bash: these commands run without asking — {verbs}."
    if extras:
        bash += " This agent's own allowlist adds " + ", ".join(f"`{e}`" for e in extras) + "."
    bash += (
        # F167: "may": a host run with ``--unlisted-bash allow`` runs them without asking.
        " Any other command may be HELD until the operator allows or denies it (in the Questions tab, "
        "on the ticket, or by Telegram); no answer in time means denied. Pushing, publishing and escalating never run."
    )
    lines = [
        TOOLS_HEADER,
        "- Files: read, search and edit inside the folder you were started in, the ticket's own "
        "session folder, and the Deliverables folder your ticket names; anything else is refused.",
        bash,
        APPROVALS_LINE,
        "- Web: the CLI's own web fetch and web search tools.",
    ]
    if available:
        # "if they are wired": a host older than the claim that carries them
        # writes no MCP config, and then every one of these is refused at the
        # gate. Saying so costs one clause and keeps the prompt honest in the
        # one case where the backend cannot know.
        lines.append(
            "- Automatos: " + ", ".join(f"`{n}`" for n in available)
            + ". Call one and read what comes back; if they are not wired on this machine the call is "
            "refused and says so — then carry on without them and say so in your final message."
        )
    lines.append(
        "- NOT available in a session: every other platform tool your skills name — "
        + ", ".join(_family_label(p) for p in _unavailable_families(available))
        + ". Do not call them and do not wait for them: do the work with what is listed here, "
        "and say in your final message what you could not do."
    )
    return "\n".join(lines)


def session_system_prompt(agent: Any, max_skill_chars: Optional[int] = None, *, ticket_session: bool = True) -> str:
    """Description, persona, the session's tools and skills for the appended
    system prompt.

    ``ticket_session`` — a host-run ticket is policy-gated, so the tools block
    always renders (even for an agent with no soul: the gate is worth knowing);
    the operator's own Canvas terminal has no hooks (PRD-239 S7 v2), so it gets
    the soul only and is empty when there is none — the host then writes exactly
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
    if ticket_session:
        parts.append(tools_block(agent))
    skills = skills_block(agent, max_skill_chars)
    if skills:
        parts.append(skills)
    return "\n\n".join(parts)
