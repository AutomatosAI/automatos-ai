"""F202 (night 6): work that still holds a template's placeholders is not finished.

At 06:09 Auto filed a staff-meeting summary to Reports, and the bell announced
it. It read "Analysis … revealed [Insert insights from Task 1148 here]". A
placeholder is a slot the writer left for itself: "[Insert …]", "[… here]",
"[TBD]", "{{name}}". A deliberate blank for a person to fill in is not one: the
printable checklist's "[Current Monday's Date]". Neither is a markdown link, a
checkbox or a reference.
"""
from __future__ import annotations

import re
from typing import List

_PLACEHOLDER = re.compile(
    r"\[(?:insert|add|include|paste|fill in|enter|put|type)\b[^\]\n]{0,160}\](?!\()"
    r"|\[[^\]\n]{0,160}\bhere\](?!\()"
    r"|\[(?:tbd|tbc|todo|placeholder|to be (?:added|confirmed|completed|written))\b[^\]\n]{0,80}\](?!\()"
    r"|\{\{\s*[\w.\-]+\s*\}\}"
    r"|<(?:insert|add)\b[^>\n]{0,160}>",
    re.IGNORECASE,
)


def template_placeholders(text: object) -> List[str]:
    """The template placeholders ``text`` still holds, in order, each once."""
    found: List[str] = []
    for match in _PLACEHOLDER.finditer(str(text or "")):
        if match.group(0) not in found:
            found.append(match.group(0))
    return found
