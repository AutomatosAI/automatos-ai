"""F205 (night 6): when a call fails, Auto tells the owner what they can do, in
their words, not the platform's.

From 04:46 to 06:04 Auto's replies read "I missed a required parameter for the
create_blog_post tool" and "the document_id needs to be an integer, not the
filename". The owner never saw those names and can do nothing with them.

A reply that follows a failed call in its turn is checked for the platform's own
names: an offered tool's, a platform action's with or without its "platform_"
prefix, or one of their parameters (snake_case only, so ordinary words never
count). A name the owner used this turn is theirs to hear. A reply that names
one is re-prompted once, with no tools, to say it in the owner's words. What
still names one is logged.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set

from core.llm.owner_words_stream import offered_names

_SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
OWNER_WORDS_NUDGE = (
    "Your reply names {names}: the platform's own tool and parameter names, which the owner never sees and "
    "cannot use. Say what went wrong and what they can do, in their words, without those names."
)


def internal_vocabulary(tools: Optional[Iterable[Dict[str, Any]]]) -> Set[str]:
    """The platform's own names: the offered tools, every registered action,
    and their snake_case parameters (core.llm.owner_words_stream, which also
    says them in plain words as Auto's reply streams, F264)."""
    return offered_names(tools)


def internal_names(answer: object, vocabulary: Set[str], owner_text: object = "") -> List[str]:
    """The platform names ``answer`` uses that the owner did not, in order."""
    owner = set(_SNAKE.findall(str(owner_text or "").lower()))
    found: List[str] = []
    for token in _SNAKE.findall(str(answer or "")):
        if token in vocabulary and token not in owner and token not in found:
            found.append(token)
    return found


def owner_words_nudge(names: List[str]) -> str:
    return OWNER_WORDS_NUDGE.format(names=", ".join(names[:5]))
