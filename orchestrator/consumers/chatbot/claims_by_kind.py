"""P256-FIX-RVW-37: four claims a verb's family alone cannot read, each backed by its own kind of write.

FX-007 retired the regex claim families and inverted four of their tests instead of replacing
them. Each was a wrong statement the owner acts on, so the receipts rule (``claims_backed``)
reads them again, from the turn's done writes, with no family module back:

- F351 (night 10b): "I've created the letter to Maya and saved it to your Deliverables." over a
  card made: created, generated or saved with a document noun (a letter, a document, a PDF, a
  docx, a report; saved to Deliverables) needs a write that makes a document (``DOCUMENT_WRITES``).
- F308 (night 9): "Each step will pause for your approval." over a mission whose answer said its
  steps run unchecked: a claim that each step waits needs a receipt that says so (``EACH_STEP_WAITS``).
- F324 (night 9b): "I've stored it in my memory so that all agents know.": a claim that the team or
  the agents know needs a write other than Auto's memory (the agents read documents and cards).
- F261-A (night 7b, #0181): "I will now send this updated brief to the agent." ending a reply with
  nothing sent: the last sentence announcing a family's work now needs a write of that family; the
  tool loop gives it the announced-step nudge (``nudges.announced_step``).

A question, an offer, a denial, a plan's purpose ("so every agent knows") or a condition ("once I
post the note") is no claim. Stdlib only, like ``claims_backed``, which imports this module.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from modules.tools.execution.call_effects import DOCUMENT_MAKES

Receipt = Dict[str, Any]
Backs = Callable[[Receipt], bool]

# ── F351: a document said to be made ───────────────────────────────────────
DOCUMENT_VERBS = frozenset({"created", "generated", "saved"})
DOCUMENT_WRITES = DOCUMENT_MAKES + ("create_document", "upload_document", "save_document", "submit_report")
# … or a write whose name pairs a writing verb with a document word: NOTION_CREATE_PAGE, GOOGLEDRIVE_UPLOAD_FILE,
# platform_update_document, workspace_edit_file.
_WRITES_WORDS = frozenset({"create", "upload", "save", "write", "insert", "add", "update", "edit", "append", "copy",
                           "generate", "export"})
_DOCUMENT_WORDS = frozenset({"document", "documents", "doc", "docs", "page", "pages", "file", "files", "pdf",
                             "docx", "report", "reports", "letter"})
_DOC_NOUN = r"(?:letters?|documents?|pdfs?|docx|word\s+(?:files?|docs?)|reports?)"
# The noun is the thing made, not what a card or a setting is about: "the report task", "report preferences".
_NOT_ITS_KIND = (r"(?![-’'])(?!\s+(?:tasks?|cards?|tickets?|missions?|playbooks?|steps?|reminders?|templates?|"
                 r"agents?|schedules?|runs?|prefs?|preferences|settings|requests?|items?)\b)")
_STOP = (r"(?:for|to|of|with|about|from|into|in|on|at|by|and|or|that|which|who|whose|where|as|using|so|"
         r"called|named|titled)")
# … and the head of its phrase: "an agent called REPORT GENERATOR" made an agent.
_HEAD = r"(?=\s*(?:$|[.,;:!?)—–]|\s(?:to|for|in|into|on|at|with|as|about|from|and|which|that|so|you)\b))"
_MADE_THING = re.compile(rf"\s+(?:(?:the|a|an|your|that|this|these|those|its|two|three)\s+)?"
                         rf"(?:(?!{_STOP}\b)[^\s.!?,;:]+\s+){{0,5}}?{_DOC_NOUN}\b{_NOT_ITS_KIND}{_HEAD}", re.I)
_TO_DELIVERABLES = re.compile(r"(?:\s+(?:it|them|this|that|a copy))?\s+(?:to|in|into|under)\s+(?:your\s+|the\s+)?"
                              r"deliverables\b", re.I)
# A passive's subject: the noun, or the noun and its own phrase ("The letter to Maya … has been generated"); a
# clause after it ("The report shows … and the task has been created") is another subject's.
_OF_ITS_OWN = r"(?:\s+(?:to|for|about|on|from|of|with|in|at|regarding)\b(?:(?!\b(?:and|or|but)\b)[^,;:—–]){0,100}?)?"
_MADE_SUBJECT = re.compile(rf"\b{_DOC_NOUN}\b{_NOT_ITS_KIND}{_OF_ITS_OWN}\s+(?:has|have|is|are|was|were)\s+"
                           rf"(?:[\w'’]+\s+){{0,2}}$", re.I)
_IN_MEMORY = re.compile(r"\bmemory\b", re.I)

# ── F308: each step said to wait for the owner ─────────────────────────────
EACH_STEP_WAITS = "each step waits for your OK"      # a mission write's effect (receipts._SAID)
STEPS_LABEL = "set to wait for the owner's check of each step"
_STEPS_WAIT = re.compile(
    r"\b(?:each|every)\s+step\b[^.!?\n]{0,40}\b(?:will|would|is going to|is set to|are set to)\s+(?:\w+\s+){0,2}?"
    r"(?:pause|wait|stop)\b|\b(?:pause|wait|stop)s?\s+(?:after|before|at|for)\s+(?:each|every)\s+step\b", re.I)

# ── F324: the team said to know ────────────────────────────────────────────
TEAM_LABEL = "told to the whole team"
_GROUP = (r"(?:(?:all|every|each)(?: (?:of )?(?:the|your|my))? (?:agents?|helpers?|team members?)"
          r"|(?:the|your|my) (?:whole |entire )?team|(?:the|your|my) agents|everyone|everybody)")
# A claim says they were told, are now aware, or that Auto saw to it ("so that all agents know"); "Everyone knows
# the price" or "Your agents know how to read documents" says what they know, not that this reply told them.
_TOLD = re.compile(
    r"\b" + _GROUP + r"\b[^.!?\n]{0,60}?\b(?:(?:are|is|will be|'re|’re)\s+(?:now\s+|all\s+|fully\s+)*"
    r"(?:aware|informed|briefed|updated|up to date|in the loop)|(?:now|will|all) knows?|"
    r"(?:has|have) been (?:told|informed|briefed|updated|notified))\b"
    r"|\bso(?: that)? " + _GROUP + r"\b[^.!?\n]{0,30}?\b(?:knows?|aware|informed)\b"
    r"|\bi(?:'ve|’ve| have)(?: (?:just|now|already|also))* (?:told|informed|briefed|notified|let)\b[^.!?\n]{0,40}"
    r"\b(?:team|agents?|everyone|everybody)\b"
    r"|\b(?:make|made|making) sure (?:that )?" + _GROUP + r"\b[^.!?\n]{0,30}\b(?:knows?|aware|informed)\b"
    r"|\bensures? (?:that )?" + _GROUP + r"\b[^.!?\n]{0,60}?\b(?:knows?|aware|informed)\b",
    re.I)
_PLAN = re.compile(r"\b(?:i'll|i’ll|i will|i'm going to|i’m going to|i am going to|let me)\b", re.I)
_PURPOSE = re.compile(r"\bso(?: that)?\s+" + _GROUP + r"\b", re.I)
_CONDITION = re.compile(r"\b(?:once|if|when|after|as soon as|until)\s+(?:i|you|we)\b", re.I)
_MEMORY_WRITE = ("memory", "remember")

# ── F261-A: work announced as being done now ───────────────────────────────
_NOW = re.compile(r"^\s*(?:(?:ok(?:ay)?|right|alright|great|sure)\s*[,!]?\s+)?(?:i(?:'ll|’ll| will)|"
                  r"i(?:'m|’m| am) going to)\s+now\s+(?:go ahead and\s+)?(?P<verb>[a-z]+)\b", re.I)
_DONE_AS = {"send": "sent", "run": "started", "make": "created", "build": "built", "set": "set up"}

# ── what makes a sentence no claim ─────────────────────────────────────────
_ASKING = re.compile(r"\?|\b(?:if you|would you|do you want|shall i|should i|want me to|could you|can you|"
                     r"i can|i could)\b", re.I)
_NEGATED = re.compile(r"\b(?:not|never|no longer)\b|n['’]t\b", re.I)
_LATER = re.compile(r"^\W*(?:once|when|after|if|as soon as|until)\b", re.I)
# Work announced for later ("I will now send it once you approve") is a plan.
_ONCE = re.compile(r"\b(?:if|once|when|after|as soon as|until)\s+(?:you|it|the|they|we|i)\b", re.I)

# What the not-done line says for a claim with no verb of its own.
OWN_SAID = {
    STEPS_LABEL: "nothing in this reply set the mission's steps to wait for your OK",
    TEAM_LABEL: ("your agents haven't been told: I only keep this in my own memory, and they read your documents "
                 "and their cards, not my memory"),
}


def _stem(r: Receipt) -> str:
    return str(r.get("action") or "").lower().removeprefix("platform_")


def _makes_a_document(r: Receipt) -> bool:
    stem = _stem(r)
    words = set(stem.split("_"))
    return (any(make in stem for make in DOCUMENT_WRITES) or stem.startswith("upload_")
            or bool(words & _WRITES_WORDS and words & _DOCUMENT_WORDS))


def _steps_wait(r: Receipt) -> bool:
    return EACH_STEP_WAITS in str(r.get("effect") or "")


def _beyond_memory(r: Receipt) -> bool:
    return not any(word in _stem(r) for word in _MEMORY_WRITE)


def document_backs(verb: str, sentence: str) -> Optional[Backs]:
    """What backs ``verb`` when its sentence names a document as what it made (F351): a write that makes
    one; None when the claim names no document ("I've created the report task", "saved it to memory")."""
    if verb not in DOCUMENT_VERBS or (verb == "saved" and _IN_MEMORY.search(sentence)):
        return None
    for found in re.finditer(rf"\b{verb[:-1]}d?\b", sentence, re.I):       # "created", or "create" announced now
        after, before = sentence[found.end():], sentence[: found.start()]
        if _MADE_THING.match(after) or _MADE_SUBJECT.search(before) or (
                verb == "saved" and _TO_DELIVERABLES.match(after)):
            return _makes_a_document
    return None


def _no_claim(sentence: str) -> bool:
    return bool(_ASKING.search(sentence) or _NEGATED.search(sentence) or _LATER.match(sentence))


def _says_the_team_knows(sentence: str) -> bool:
    if not _TOLD.search(sentence) or _CONDITION.search(sentence):
        return False
    return not (_PLAN.search(sentence) and _PURPOSE.search(sentence))


def said_in(sentence: str) -> List[Tuple[str, Backs]]:
    """The sentence's claims that each step waits (F308) or that the team knows (F324): (label, what backs it)."""
    if _no_claim(sentence):
        return []
    said: List[Tuple[str, Backs]] = []
    if _STEPS_WAIT.search(sentence):
        said.append((STEPS_LABEL, _steps_wait))
    if _says_the_team_knows(sentence):
        said.append((TEAM_LABEL, _beyond_memory))
    return said


def promised_now(sentence: str) -> List[str]:
    """The participles of the verb the sentence says is being done now ("I will now send …": "sent"),
    for the caller to find its family; [] when it announces nothing (a question, a condition)."""
    found = _NOW.match(sentence)
    if not found or _ASKING.search(sentence) or _ONCE.search(sentence):
        return []
    base = found.group("verb").lower()
    doubled = f"{base}{base[-1]}ed"
    return [_DONE_AS.get(base, ""), f"{base}ed", f"{base}d", f"{base[:-1]}ied", doubled]


__all__ = ["DOCUMENT_VERBS", "DOCUMENT_WRITES", "EACH_STEP_WAITS", "OWN_SAID", "STEPS_LABEL", "TEAM_LABEL",
           "document_backs", "promised_now", "said_in"]
