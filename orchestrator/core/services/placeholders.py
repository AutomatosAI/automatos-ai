"""F202 (night 6): work that still holds a template's placeholders is not finished.

At 06:09 Auto filed a staff-meeting summary to Reports, and the bell announced
it. It read "Analysis … revealed [Insert insights from Task 1148 here]". A
placeholder is a slot the writer left for itself: "[Insert …]", "[… here]",
"[TBD]", "{{name}}". A deliberate blank for a person to fill in is not one: the
printable checklist's "[Current Monday's Date]". Neither is a markdown link, a
checkbox or a reference.

F248 (night 7): mission step #0126.3 was marked verified with "[Number]" and
"[Your Name/Company Name]" in it. A slot for the writer's own details ("[Your …]"),
a name slot ("[Member Name]", "[Company Name]") and a bare figure slot ("[Number]",
"[Amount]") are placeholders too. A dated blank like the checklist's stays a blank.
"""
from __future__ import annotations

import re
from typing import List

_PLACEHOLDER = re.compile(
    r"\[(?:insert|add|include|paste|fill in|enter|put|type)\b[^\]\n]{0,160}\](?!\()"
    r"|\[[^\]\n]{0,160}\bhere\](?!\()"
    r"|\[(?:tbd|tbc|todo|placeholder|to be (?:added|confirmed|completed|written))\b[^\]\n]{0,80}\](?!\()"
    r"|\{\{\s*[\w.\-]+\s*\}\}"
    r"|<(?:insert|add)\b[^>\n]{0,160}>"
    r"|\[your\s[^\]\n]{1,80}\](?!\()"
    r"|\[(?:[a-z][a-z']{0,20} ){0,3}name\](?!\()"
    r"|\[(?:number|amount|figure|total|price|quantity|value|percentage)\](?!\()",
    re.IGNORECASE,
)
# How a step that still holds them is told so (mission verification, F248).
UNFINISHED = "It still has placeholders where its content belongs: "


def template_placeholders(text: object) -> List[str]:
    """The template placeholders ``text`` still holds, in order, each once."""
    found: List[str] = []
    for match in _PLACEHOLDER.finditer(str(text or "")):
        if match.group(0) not in found:
            found.append(match.group(0))
    return found
