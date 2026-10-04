"""F320 (night 9b): a card's answer is the work, not the agent's working.

An agent run's result is its last reply (``AgentFactory``: ``response.content``; the
rounds before it are not joined in). Claude writes that reply straight after its last
tool result, and on night 9b it opened with the working: "Perfect! Now I have all the
information I need. Let me compile the answer…" (#0045, #0046, #0054, #0063, #0065,
#0070, #0132), "Now let me generate the report in the correct format. Based on the
previous step results and current database query, here's the stock report:" (#0069),
"Now I have all the information needed. Let me write the brief as requested:" (#1994).
The owner: "I'd have to trim it before sending anything on."

``without_the_working`` takes those sentences off the top: interjections ("Perfect!"),
sentences about the agent's own process ("Now I have…", "Let me…", "I'll draft…",
"Based on my analysis…, I can now…", "…, here's the stock report:"). It reads only the
leading paragraphs, and only when the answer opens with such a sentence; the work
below is never touched. A process sentence that carries a fact (a figure, a name, a
quote) the rest of the answer does not repeat stays: #1981's first run wrote the free
-delivery draft the owner asked for, and only its working said "it would normally cost
£8.50". The owner: "the one thing I needed to know was in the working".

``sources_out_of_the_draft``: a source line inside a draft (after its greeting) is
moved above the greeting. On #0065 the draft to Rosa ended "*Source:
wholesale-terms-2026.md and brand-voice.md*" after the sign-off, on #0002 inside the
body: text the owner would have to cut before sending. The source itself is kept, not
dropped: the owner marks answers down when a figure has no source (#1970 run 2,
"industry 18%") and up when every figure has one (#1970 run 3, #1966).
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Set, Tuple

_VERBS = (r"(?:draft|write|compile|calculate|check|search|look|query|create|prepare|put together|provide|"
          r"present|summari[sz]e|format|generate|analy[sz]e|compare|get|find|pull|fetch|gather|start|begin|"
          r"proceed|identify|show|give|answer|verify|confirm|review|read|use|follow|work out|break|rank|fix|update|explain|list)")
_INTENT = (r"(?:let me|i'?ll|i will|i'?m going to|i am going to|i need to|i can now|i'?m now going to)\s+"
           r"(?:now\s+|also\s+|then\s+|first\s+)?" + _VERBS + r"\b")
_STATE = (r"(?:now\s+i\s+(?:have|can|understand|see|know)\b|i\s+(?:now\s+)?(?:can\s+)?see\b|i\s+also\s+have\b"
          r"|i(?:'ve|\s+have)?\s+found\b|i(?:'ve|\s+have)\s+(?:now\s+)?(?:got|gathered|collected)\b"
          r"|i\s+(?:now\s+)?have\s+(?:all|enough|everything|the\s+(?:information|data|details|figures)"
          r"|what\s+i\s+need)\b|i\s+(?:now\s+)?understand\b)")
_LEAD_IN = r"(?:here(?:'s|’s|\s+is|\s+are)\b|looking\s+at\b).*:$"
_PREFIX = r"^(?:(?:now|so|ok(?:ay)?|great|right),?\s+|based on\b.*?,\s*)?"
_PROCESS = re.compile(rf"{_PREFIX}(?:{_INTENT}|{_STATE})", re.IGNORECASE)
_LEAD = re.compile(rf"{_PREFIX}{_LEAD_IN}", re.IGNORECASE)
_INTERJECTION = re.compile(r"^(?:perfect|great|excellent|good|ok(?:ay)?|alright|all right|got it|understood)"
                           r"\s*[!.]*$", re.IGNORECASE)
_NOT_WORKING = re.compile(r"^let me know\b", re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_NAME = re.compile(r"\b[A-Z][A-Za-z]*(?:['’]s)?\b")
_QUOTE = re.compile(r"[\"“]([^\"”]{3,})[\"”]")
_NOT_A_NAME = frozenset({"I", "I'll", "I've", "I'm", "I'd"})

_GREETING = re.compile(r"^(?:hi|hello|dear|hey)\b[^\n]{0,60},\s*$", re.IGNORECASE)
_SOURCE_LINE = re.compile(r"^[*_\s]*sources?\b[^:\n]{0,40}:", re.IGNORECASE)
_LIST_ITEM = re.compile(r"^\s*(?:[-•*]|\d+\.)\s+")

Lines = List[List[str]]  # one paragraph: its lines, each split into sentences
Spot = Tuple[int, int, int]  # (paragraph, line, sentence)


def is_working(sentence: str, *, opening: bool = False) -> bool:
    """A sentence about the agent's own process ("Perfect!", "Now I have all the
    information I need.", "Let me write the brief as requested:"), not about the work.
    A lead-in ("Based on your wholesale terms, here's what you charge for delivery:")
    is working after other working (#0069), never as the answer's ``opening``: there it
    is often the only place the answer names its source (#0042, #0043, the night's best)."""
    text = sentence.strip()
    if not text or _NOT_WORKING.match(text):
        return False
    if _INTERJECTION.match(text) or _PROCESS.match(text):
        return True
    return not opening and bool(_LEAD.match(text))


def _number(token: str) -> str:
    """A figure as a comparable value: '£8.50' and '8.5' match, '1,470.1' and '1470.1' too."""
    try:
        return str(Decimal(token.replace(",", "")).normalize())
    except InvalidOperation:
        return token


def _numbers(text: str) -> Set[str]:
    return {_number(m) for m in _NUMBER.findall(text or "")}


def _names(sentence: str) -> Set[str]:
    """Capitalised words after the first (that one is the sentence's own capital)."""
    found = [m.group(0) for m in _NAME.finditer(sentence)][1:]
    return {re.sub(r"['’]s$", "", word) for word in found if word not in _NOT_A_NAME}


def carries_a_fact(sentence: str, rest: str) -> bool:
    """The sentence holds a figure, a name or a quote that ``rest`` (the answer
    without the working) does not: #1981's "it would normally cost £8.50"."""
    if _numbers(sentence) - _numbers(rest):
        return True
    if any(not re.search(rf"\b{re.escape(name)}\b", rest, re.IGNORECASE) for name in _names(sentence)):
        return True
    return any(quote.lower() not in rest.lower() for quote in _QUOTE.findall(sentence))


def _split(paragraph: str) -> Lines:
    return [[s for s in _SENTENCE_BREAK.split(line.strip()) if s] for line in paragraph.splitlines()]


def _first_sentence(paragraph: str) -> str:
    lines = [s for line in _split(paragraph) for s in line]
    return lines[0] if lines else ""


def _leading(paragraphs: List[str]) -> int:
    """How many paragraphs at the top open with the agent's working."""
    count = 0
    while count < len(paragraphs) and is_working(_first_sentence(paragraphs[count]), opening=count == 0):
        count += 1
    return count


def _assemble(split: List[Lines], dropped: Set[Spot]) -> List[str]:
    """The leading paragraphs without the ``dropped`` sentences; empty ones go."""
    paragraphs = []
    for pi, lines in enumerate(split):
        kept_lines = [" ".join(s for si, s in enumerate(line) if (pi, li, si) not in dropped)
                      for li, line in enumerate(lines)]
        text = "\n".join(line for line in kept_lines if line.strip())
        if text.strip():
            paragraphs.append(text)
    return paragraphs


def _without_leading_rules(paragraphs: List[str]) -> List[str]:
    """A "---" left on top once the working above it is gone (#0095's draft)."""
    start = 0
    while start < len(paragraphs) and _RULE.match(paragraphs[start]):
        start += 1
    return paragraphs[start:]


def without_the_working(text: str) -> str:
    """``text`` without the agent's working on top; unchanged when it does not open
    with any, or when nothing but working would be left."""
    paragraphs = _PARAGRAPH_BREAK.split((text or "").strip())
    lead = _leading(paragraphs)
    if not lead:
        return text
    split = [_split(p) for p in paragraphs[:lead]]
    working = {(pi, li, si) for pi, lines in enumerate(split) for li, line in enumerate(lines)
               for si, s in enumerate(line) if is_working(s)}
    rest = "\n\n".join(_assemble(split, working) + paragraphs[lead:])
    dropped = {spot for spot in working if not carries_a_fact(split[spot[0]][spot[1]][spot[2]], rest)}
    top = _assemble(split, dropped)
    below = paragraphs[lead:] if top else _without_leading_rules(paragraphs[lead:])
    answer = "\n\n".join(top + below).strip()
    return answer or text


def _greeting_at(lines: List[str]) -> Optional[int]:
    return next((i for i, line in enumerate(lines) if _GREETING.match(line.strip())), None)


def _source_lines_after(lines: List[str], greeting: int) -> List[int]:
    """Source lines below the greeting, each with the list items straight under it."""
    found: List[int] = []
    in_block = False
    for i in range(greeting + 1, len(lines)):
        in_block = bool(_SOURCE_LINE.match(lines[i]) or (in_block and _LIST_ITEM.match(lines[i])))
        if in_block:
            found.append(i)
    return found


def sources_out_of_the_draft(text: str) -> str:
    """A draft's source lines moved from inside it to just above its greeting, so the
    draft runs clean from greeting to sign-off; nothing is dropped."""
    lines = (text or "").splitlines()
    greeting = _greeting_at(lines)
    moved = _source_lines_after(lines, greeting) if greeting is not None else []
    if not moved:
        return text
    block = [lines[i].strip() for i in moved]
    body = [line for i, line in enumerate(lines) if i not in set(moved)]
    above = body[:greeting]
    spacer = [""] if above and above[-1].strip() else []
    out = "\n".join(above + spacer + block + [""] + body[greeting:])
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def the_answer_itself(text: str) -> str:
    """F320: an agent's result as the card shows it: no working on top, and a draft
    with nothing inside it but the draft."""
    return sources_out_of_the_draft(without_the_working(text))


__all__ = ["carries_a_fact", "is_working", "sources_out_of_the_draft", "the_answer_itself",
           "without_the_working"]
