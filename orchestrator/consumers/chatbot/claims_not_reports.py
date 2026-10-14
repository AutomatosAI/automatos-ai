"""P256-FIX-N15-1 (F422, night 15): read-only reports the completed-action pattern read as claims.

The completed-action pattern (``claims_backed.COMPLETED_ACTION``) was run over every sentence of
night 15's 458 delivered replies and the 129 eval replies of 13 Oct. It fired 152 times; next to
none was a claim of the turn's own work. Each shape below is a report of something read, said or
not done, and is no claim:

- A denial in the claim's clause: "Nothing has been sent.", "Neither card has been approved or
  cancelled.", "…, and no newsletter or board post has gone out.", "my last reply didn't say anything
  was removed", "so no approvals, emails, or DONE reply have gone out". The first person's own denial follows its verb: "I've changed nothing on the board.", "I made no tool
  calls this turn".
- A question asked inside the sentence, or a relative clause: "I'll report back which ones were
  closed", "the count I closed", "I can't confirm any agent I created tonight."
- A condition: "whether the 250g bag is posted or collected", "£4.12 if it's posted.", "They may have
  been removed", "The email only goes out after you've approved the card and I've sent it".
- Words reported, or quoted: "My last reply said both cards were finished", "tell you it was done",
  'The "has been fixed" wording came from HARBOURLINE-HOUSE-FACTS.'
- A habit or a process, not an act: "Club orders are posted on Wednesdays from November", "the pulp
  and sticky mucilage are removed before drying."

- A participle used as an adjective: "Done cards are finished work".

And two that the turn's receipts decide (``claims_backed.unbacked_where``): a turn that read Auto's own
record (platform_my_actions) and wrote nothing is reporting that record ("- Created a Gmail draft to
Quay Coffee House…", "All three were assigned to agents and started."); a turn that read one thing's
work or state (a card, a report, a schedule: ``THING_READS``) and wrote nothing reports, in a stative
passive, the state it found ("Yes, 1595 is closed as cancelled.", "It's scheduled for 9am, not 8am.",
"Its one-line wholesale note is saved in your session 2867 deliverables folder"). A list of cards is no
such read: "The post is scheduled for 5pm." after one is still a claim (P256-FIX-RVW-25).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Sequence

Receipt = Dict[str, Any]

AUTOMATIC_READS = "automatic_reads"            # the turn's own reads before the model's first call, folded
RECORD_READS = frozenset({"platform_my_actions"})   # Auto's own record of what it did (N14-3)
# Reads of one thing's work or state: a card, a run, a deliverable, a report, a file, a schedule, an agent's set-up,
# the connections.
THING_READS = frozenset({
    "platform_get_task", "platform_get_mission", "platform_get_deliverable", "platform_get_latest_report",
    "platform_get_agent", "platform_list_agents",
    "platform_browse_reports", "platform_read_document", "workspace_read_file", "platform_get_agent_heartbeat",
    "platform_get_schedule", "platform_list_scheduled_tasks", "platform_list_connected_apps"})

_FIRST_PERSON = re.compile(r"I(?:'ve|’ve| have)?\s", re.I)
# Where a clause starts: a mark, or a conjunction that opens a new one.
_CLAUSE_BREAK = re.compile(r"[,;:—–(]|\b(?:and|but|so|then|while|because)\b", re.I)
_NEGATION = re.compile(r"\b(?:nothing|none|nobody|no\s+one|neither|nor|no|not|never)\b|n['’]t\b", re.I)
# … or a denial that lists what was not done, commas and all: "so no approvals, emails, or DONE reply have gone out".
_DENIED_LIST = re.compile(r"\b(?:no|nothing|none|neither)\b[^;:—–.!?]*\b(?:or|nor)\b[^,;:—–]*$", re.I)
_LIST_BREAK = re.compile(r"[;:—–(]|\b(?:and|but|so|then|while|because)\b", re.I)
# A participle used as an adjective before a noun of work: "Done cards are finished work".
_ATTRIBUTIVE = re.compile(r"\s+(?:work|items?|cards?|tasks?|jobs?|tickets?|posts?|orders?)\b", re.I)
_NOTHING_AFTER = re.compile(r"\s+(?:nothing|none|nobody|no)\b", re.I)
_ASKED_INSIDE = re.compile(r"\b(?:which|what|whether|who|how\s+many|how\s+much)\b(?:\s+[\w'’-]+){0,3}\s+$", re.I)
# A first-person simple past after "any", "no", "which" … is a relative clause: "any agent I created", "the count I
# closed". After a definite noun it still says the writer did it ("The PDF I generated is still not checked").
_RELATIVE = re.compile(r"\b(?:any|no|every|each|which|what|whichever|whatever|the\s+(?:count|number|total))\b"
                       r"(?:\s+[\w'’-]+){0,2}\s+$", re.I)
_CONDITION = re.compile(r"\b(?:whether|if|unless|depending\s+on|depends\s+on)\b", re.I)
_MODAL = re.compile(r"\b(?:may|might|could|would|should|must|can|will)(?:\s+not|n['’]t)?\s+$", re.I)
_GOVERNED = re.compile(r"\b(?:once|when|after|until|before|as\s+soon\s+as)\s+(?:you|i|we|it|they|he|she|the|that|"
                       r"this|your)\b[^;:—–]*$", re.I)
_REPORTED_WORDS = re.compile(r"\b(?:said|says|say|saying|told|claimed|claims?|shows?|showed|reads)\b"
                             r"(?:\s+(?:that|me|you|us))?(?:\s+[\w'’-]+){0,6}\s+$", re.I)
# What someone asked to be said or done: "asked me to approve card 0930 and tell you it was done".
_INSTRUCTED = re.compile(r"\b(?:asked|told|instructed|wants?|wanted)\s+(?:me|you|us|auto)\s+to\b", re.I)
_TIMES = r"(?:day|week|month|morning|evening|night|hour|weekday|weekend|(?:mon|tues|wednes|thurs|fri|satur|sun)day)"
_HABIT = re.compile(rf"\b(?:on\s+(?:mon|tues|wednes|thurs|fri|satur|sun)days|(?:every|each)\s+{_TIMES}|"
                    r"daily|weekly|monthly|"
                    r"(?:before|after|during|while)\s+\w+ing)\b", re.I)


def _clause(text: str, breaking: "re.Pattern[str]" = _CLAUSE_BREAK) -> str:
    """The last clause of ``text`` (after its last clause break)."""
    breaks = [found.end() for found in breaking.finditer(text)]
    return text[breaks[-1]:] if breaks else text


def _quoted(before: str) -> bool:
    """Whether a claim that follows ``before`` sits inside quotation marks."""
    return before.count('"') % 2 == 1 or before.count("“") > before.count("”")


def _relative(match: "re.Match[str]", before: str) -> bool:
    """A first-person simple past in a relative clause: "any agent I created", "the count I closed"."""
    return bool(match.group("mine")) and bool(_RELATIVE.search(before))


def _conditional(sentence: str, match: "re.Match[str]", before: str) -> bool:
    """Under a condition: "whether …", "if it's posted", "may have been", "after you've approved … and I've sent"."""
    return bool(_CONDITION.search(_clause(sentence[: match.end()])) or _MODAL.search(before)
                or _GOVERNED.search(before))


def _not_its_own(sentence: str, match: "re.Match[str]", before: str) -> bool:
    """Not the writer's own report: denied in its clause, words reported, or a habit."""
    if _FIRST_PERSON.match(match.group(0)):
        return False
    denied = _NEGATION.search(_clause(before)) or _DENIED_LIST.search(_clause(before, _LIST_BREAK))
    return bool(denied or _REPORTED_WORDS.search(before) or _INSTRUCTED.search(before)
                or _HABIT.search(sentence, match.start()))


def _found_state(sentence: str, match: "re.Match[str]", looked_up: bool) -> bool:
    """A stative passive used as an adjective ("are finished work"), or, with ``looked_up``, the state
    the turn's read of that thing found ("1595 is closed as cancelled")."""
    return bool(match.group("state")) and bool(looked_up or _ATTRIBUTIVE.match(sentence, match.end()))


def _not_said_done(sentence: str, match: "re.Match[str]", before: str) -> bool:
    """Denied after its verb, quoted, a relative clause, or a question asked inside the sentence ("which
    ones were closed"; "which version of PIL is installed" still says one is)."""
    asked = bool(_ASKED_INSIDE.search(before)) and not match.group("state")
    return bool(_NOTHING_AFTER.match(sentence, match.end()) or _quoted(before) or _relative(match, before) or asked)


def not_a_report(sentence: str, match: "re.Match[str]", looked_up: bool = False) -> bool:
    """Whether the claim shape ``match`` in ``sentence`` reports no work of this turn: denied, asked,
    conditional, reported, quoted or a habit; a stative passive used as an adjective, or, with
    ``looked_up`` (the turn read one thing's work or state and wrote nothing), the state it found."""
    before = sentence[: match.start()]
    return (_found_state(sentence, match, looked_up) or _not_said_done(sentence, match, before)
            or _conditional(sentence, match, before) or _not_its_own(sentence, match, before))


def recalled(reads: Sequence[Receipt]) -> bool:
    """Whether a done read of the turn was Auto's own record of what it did."""
    return any(r.get("action") in RECORD_READS for r in reads)


def looked_something_up(reads: Sequence[Receipt]) -> bool:
    """Whether a done read of the turn read one thing's work or state (``THING_READS``). A list of cards
    is no such read: "The post is scheduled for 5pm." after one stays a claim (P256-FIX-RVW-25)."""
    return any(r.get("action") in THING_READS for r in reads)


__all__ = ["AUTOMATIC_READS", "RECORD_READS", "THING_READS", "looked_something_up", "not_a_report", "recalled"]
