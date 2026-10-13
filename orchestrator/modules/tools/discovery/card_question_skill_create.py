"""PRD-256 P256-FIX-N14-7 (F421, Decision D1 amended): a skill made for the workspace waits
for the owner's click, and its card says which skill and what it does.

Night 14: "give the WRITER a skill for writing product descriptions" made workspace skill
product-description-writer (#361) with no card: ``platform_create_workspace_skill`` was not
owner-only, and only the assignment after it asked. A skill made is an agent setting in
waiting (every agent of the workspace can be given it, and it is enabled at once), so it
asks as a skill given or edited does (``owner_only.OWNER_ONLY_ACTIONS``).

The card names the skill, what it does (the call's description, else its SKILL.md
frontmatter's, as the tool saves it), the first line of its instructions, and when the call
saves over the security scanner's warnings. A call the tool would refuse on the click (no
name or content, a name longer than the column holds or one this workspace already has, or
content the security scanner blocks or warns about with no ``acknowledge_warnings``) is
refused before any grant: nothing is asked about what the click could not make (F091).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from modules.tools.discovery.card_question_skills import OVERRIDES_THE_SCAN
from modules.tools.discovery.card_question_text import said_line, value_line

CREATE_SKILL = "platform_create_workspace_skill"
NAME, CONTENT, DESCRIPTION = "name", "content", "description"
NEW_SKILL = "skill: '{name}', new: this workspace's own, enabled for it and given to no agent yet"
WHAT_IT_DOES = "what it does"
FIRST_LINE = "its instructions start"
NEEDS_BOTH = ("A skill needs a name and its SKILL.md content: nothing was asked or done. Call "
              f"{CREATE_SKILL} again with both.")
TOO_LONG = "A skill's name is at most {limit} characters: nothing was asked or done. Call again with a shorter one."
TAKEN = ("This workspace already has a skill named '{name}' (skill #{id}): nothing was asked or done. "
         "Edit it with platform_update_skill, or call again with another name.")
# What the scanner's verdict (handlers_skills._scan) means before the card: the click would be refused.
SCANNED = {
    "blocked": ("The security scanner blocks this skill (critical findings): nothing was asked or done. Fix the "
                "flagged patterns, then call again."),
    "warnings": ("The security scanner flags this skill (high-severity findings): nothing was asked or done. Review "
                 "them, then call again with acknowledge_warnings=true if it should be saved anyway."),
}
SCAN_OK = "ok"


def refused_before_the_create(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The refusal for a create the click could not make, or None, by the handler's own
    rules (``handlers_skills.create_workspace_skill``): no name or content, a name too long
    or one an active skill of this workspace carries, or content the scanner refuses."""
    from core.models.core import Skill

    name, content = str(params.get(NAME) or "").strip(), params.get(CONTENT)
    if not name or not content:
        return {"success": False, "error": NEEDS_BOTH}
    limit = Skill.__table__.c.name.type.length
    if len(name) > limit:
        return {"success": False, "error": TOO_LONG.format(limit=limit)}
    taken = (db.query(Skill.id).filter(Skill.workspace_id == workspace_id, Skill.name == name,
                                       Skill.is_active.is_(True)).first())
    if taken:
        return {"success": False, "error": TAKEN.format(name=name, id=taken.id)}
    return _scanned(name, str(content), bool(params.get("acknowledge_warnings")))


def _scanned(name: str, content: str, acknowledged: bool) -> Optional[Dict[str, Any]]:
    """The scanner's refusal of the content, with its findings, as the click would meet it; None when it passes."""
    from modules.tools.discovery.handlers_skills import _scan

    scan = _scan(content, filename=f"{name}.md", acknowledge_warnings=acknowledged)
    if scan["status"] == SCAN_OK:
        return None
    return {"success": False, "status": scan["status"], "findings": scan["findings"], "error": SCANNED[scan["status"]]}


def create_skill_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """The new skill by name, what it does, the first line of its instructions, and when
    the call overrides the scanner's warnings."""
    from modules.agents.services.skill_loader import parse_yaml_frontmatter

    content = str(params.get(CONTENT) or "")
    frontmatter, body = parse_yaml_frontmatter(content)
    frontmatter = frontmatter if isinstance(frontmatter, dict) else {}
    does = params.get(DESCRIPTION) or frontmatter.get(DESCRIPTION)
    lines = [value_line(NEW_SKILL.format(name=str(params.get(NAME) or "").strip())),
             said_line(WHAT_IT_DOES, does), said_line(FIRST_LINE, _first_line(body))]
    return [*lines, value_line(OVERRIDES_THE_SCAN)] if params.get("acknowledge_warnings") else lines


def _first_line(body: str) -> str:
    """The first line of the instructions with words on it."""
    return next((line.strip() for line in body.splitlines() if line.strip()), "")


__all__ = ["CREATE_SKILL", "create_skill_lines", "refused_before_the_create"]
