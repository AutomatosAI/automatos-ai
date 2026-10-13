"""P256-FIX-N13-7 (F414, night 13): Auto never writes down an instruction the owner did not give.

Night 13, a brief Auto wrote said "as you asked" about something the owner had never asked
for, and the card, its agent and the memory kept it as the owner's instruction. A brief, a
card note, a memory or a mission's goal that Auto writes from a person's chat now keeps an
attribution ("as you asked", "per your instruction") only when one of the owner's own
messages in this chat said it: the sentence it stands in shares most of its words with one
of those messages (``SAID_SHARE``). Otherwise the attribution is taken off and the rest of
the sentence is kept as it was written: "Draft the reply, as you asked." becomes "Draft the
reply." Nothing else in the text changes.

The owner's messages are this chat's user turns (``mission_owner_words.chat_of_the_call``,
read by ``handlers_board_task_review.owner_words``), never what the model says they said. The
check runs on the call before the owner-only gate (``owner_only.asks_the_owner_first``), so a
card shows the text that will be written, and its click (which runs the call again) finds
nothing left to take off.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, FrozenSet, List, Sequence

logger = logging.getLogger(__name__)

# The calls that write Auto's words down, and the fields that hold them.
WRITTEN_FIELDS: Dict[str, tuple] = {
    "platform_create_task": ("description", "raw_prompt"),
    "platform_update_task": ("description", "note"),
    "platform_update_task_status": ("note",),
    "platform_store_memory": ("content",),
    "platform_create_mission": ("goal",),
    "platform_approve_mission": ("note",),
    "platform_reject_mission": ("reason",),
}
# "as you asked", "per your instruction": with the comma that sets it off, either side.
ATTRIBUTION = re.compile(
    r"(?:,\s*)?\b(?:(?:just\s+)?(?:as|like)\s+you\s+(?:asked|requested|instructed|wanted|said)"
    r"(?:\s+(?:me|us))?(?:\s+(?:to|for)\b)?"
    r"|per\s+your\s+(?:instructions?|requests?|ask)|at\s+your\s+request|as\s+(?:instructed|requested))\b(?:\s*,)?",
    re.I)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")
_WORD = re.compile(r"[a-z0-9£$€][a-z0-9'£$€.]*", re.I)
MIN_WORD_CHARS = 4
# P256-FIX-N14-4: the words that turn a sentence round, weighed as one (``in_their_words``).
NEGATION = "not"
NEGATIONS: FrozenSet[str] = frozenset({"not", "no", "never", "none", "nothing", "nobody", "neither", "nor",
                                       "without", "cannot"})
# The share of a sentence's words one of the owner's messages must hold for it to be theirs.
SAID_SHARE = 0.5
# This chat's user turns read for the check: every one a working chat holds.
CHAT_TURNS_READ = 50
COMMON_WORDS: FrozenSet[str] = frozenset({
    "about", "after", "again", "also", "been", "before", "being", "could", "does", "done", "each", "from",
    "have", "here", "into", "just", "make", "more", "once", "only", "over", "please", "should", "some",
    "than", "that", "their", "them", "then", "there", "these", "they", "this", "those", "very", "want",
    "were", "what", "when", "where", "which", "while", "will", "with", "would", "your", "yours",
})


def as_the_owner_said(db: Any, workspace_id: Any, action: str, params: Any, caller_context: Any) -> Any:
    """The call as it goes on record: in a person's chat, a write call's text attributes to
    the owner only what this chat's owner messages said (a new dict when anything changed).
    ``owner_only.asks_the_owner_first`` reads every call through it, before its gate."""
    if not _checked(action, params, caller_context):
        return params
    return _as_said(db, workspace_id, action, params)


def _checked(action: str, params: Any, caller_context: Any) -> bool:
    """A write call from a person's chat whose text claims an instruction."""
    from modules.tools.discovery.owner_only import human_driven

    if action not in WRITTEN_FIELDS or not isinstance(params, dict) or not human_driven(caller_context):
        return False
    return any(isinstance(params.get(field), str) and ATTRIBUTION.search(params[field])
               for field in WRITTEN_FIELDS[action])


def _as_said(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """The call with each written field's unsaid attributions taken off (a new dict)."""
    said = chat_owner_turns(db, workspace_id, params)
    written = {field: without_unsaid_attributions(params[field], said)
               for field in WRITTEN_FIELDS[action] if isinstance(params.get(field), str)}
    out = {**params, **written}
    if out != params:
        logger.info("[said_instructions] %s: took off an instruction the owner did not give", action)
    return out


def chat_owner_turns(db: Any, workspace_id: Any, params: Dict[str, Any]) -> List[str]:
    """The owner's messages in the chat the call came from, newest first; [] outside one."""
    from modules.tools.discovery.handlers_board_task_review import owner_words
    from modules.tools.discovery.mission_owner_words import chat_of_the_call

    if db is None:
        return []
    return [words for words in owner_words(db, workspace_id, chat_of_the_call(params), limit=CHAT_TURNS_READ)
            if words]


def without_unsaid_attributions(text: str, owner_turns: Sequence[str]) -> str:
    """``text`` with each attribution taken off whose sentence none of ``owner_turns`` said."""
    if not ATTRIBUTION.search(text or ""):
        return text
    turns = [_words(turn) for turn in owner_turns]
    return _SENTENCE.sub(lambda match: _sentence_as_said(match.group(0), turns), text)


def _sentence_as_said(sentence: str, turns: List[FrozenSet[str]]) -> str:
    if not ATTRIBUTION.search(sentence) or _owner_said(ATTRIBUTION.sub(" ", sentence), turns):
        return sentence
    lead = sentence[: len(sentence) - len(sentence.lstrip())]
    rest = ATTRIBUTION.sub("", sentence.strip())
    rest = re.sub(r"\s+([,.!?;:])", r"\1", re.sub(r"\s{2,}", " ", rest)).strip(" ,;")
    trail = " " if sentence.endswith((" ", "\t")) else ""
    return f"{lead}{rest[:1].upper()}{rest[1:]}{trail}" if rest else lead


def _owner_said(sentence: str, turns: List[FrozenSet[str]]) -> bool:
    """Whether one of the owner's messages holds most of ``sentence``'s words."""
    words = _words(sentence)
    if not words:
        return False
    return any(len(words & turn) / len(words) >= SAID_SHARE for turn in turns)


def in_their_words(text: str, owner_turns: Sequence[str], share: float) -> bool:
    """P256-FIX-N14-4: whether each sentence of ``text`` is one of the owner's own: a sentence of
    ``owner_turns`` holds ``share`` of its words in the same order, and says no where it says no
    ('Tuesdays, not Mondays' is never 'Mondays, not Tuesdays'; 'we do not post' never 'we post').
    A sentence with no word to weigh ('OK.') is passed over; False when none is left."""
    said = [_sequence(sentence) for turn in owner_turns for sentence in _SENTENCE.findall(turn or "")]
    stored = [seq for seq in map(_sequence, _SENTENCE.findall(text or "")) if seq]
    return bool(stored) and all(any(_held(seq, theirs, share) for theirs in said) for seq in stored)


def _sequence(sentence: str) -> List[str]:
    """The sentence's words in order (``_words``' rule), each negation as one word, NEGATION."""
    found = (word.strip("'.").lower() for word in _WORD.findall((sentence or "").replace("\u2019", "'")))
    return [NEGATION if _negates(word) else word for word in found
            if _negates(word) or (len(word) >= MIN_WORD_CHARS and word not in COMMON_WORDS)]


def _negates(word: str) -> bool:
    return word in NEGATIONS or word.endswith("n't")


def _held(seq: List[str], theirs: List[str], share: float) -> bool:
    """``theirs`` holds ``share`` of ``seq`` in order (a common subsequence), and both or neither say no."""
    if (NEGATION in seq) != (NEGATION in theirs):
        return False
    return _in_order(seq, theirs) / len(seq) >= share


def _in_order(a: List[str], b: List[str]) -> int:
    """The longest common subsequence of two word lists (a sentence is a few dozen words)."""
    row = [0] * (len(b) + 1)
    for word in a:
        diagonal = 0
        for j, other in enumerate(b, 1):
            diagonal, row[j] = row[j], diagonal + 1 if word == other else max(row[j], row[j - 1])
    return row[-1]


def _words(text: str) -> FrozenSet[str]:
    found = (word.strip("'.").lower() for word in _WORD.findall(text or ""))
    return frozenset(word for word in found if len(word) >= MIN_WORD_CHARS and word not in COMMON_WORDS)


__all__ = ["ATTRIBUTION", "WRITTEN_FIELDS", "as_the_owner_said", "chat_owner_turns", "in_their_words",
           "without_unsaid_attributions"]
