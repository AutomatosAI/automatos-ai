"""P256-FIX-N14-4 (night 14, F418): is a memory in the owner's own words?

A standing rule rides every turn as the owner's instruction, so a memory is one only when it is
what they said: each of its sentences is one of the owner's own sentences, word for word in their
order (Auto may leave words out, never add or change one), keeping at least ``coverage`` of that
sentence, and saying no exactly as often as they did. Every word counts (numbers, 'before' and
'after', 'on' and 'off'), so 'post at 10' is never 'post at 11', 'Mondays, not Tuesdays' never
'Tuesdays, not Mondays', and a dropped 'not' never passes. A question the owner asked ('do we post
on Tuesdays?') is no rule. Sentences past ``MAX_RULE_WORDS`` are not weighed: a rule is short, and
the comparison stays a few thousand steps.

Both sides are read alike in any script (NFKC, casefold, Unicode words), and a stored sentence
holding anything that is neither a word nor punctuation (an emoji, a sign) is no rule: what the
check cannot weigh never passes for the owner's. A word that sets a condition ('only', 'unless',
'ask', 'before') in the owner's sentence is kept in the rule, or it is not theirs. The negations
and conditions known are English: an owner's sentence with any other word ('café', 'لا', 'नहीं')
is the rule only word for word, none left out.

P256-FIX-N17-4 (nights 16-17, F447): the owner's lead-in ('New standing rule, remember this one:',
'The only one that counts:') sits inside their sentence, so Auto's shorter copy of the rule keeps
too little of it and is no rule. ``their_sentence`` names the owner's own sentence that holds what
Auto stored, which Auto may store as it stands: the check itself (``in_their_words``) is unchanged.

F460 (night 18): "Two things to remember, please: from November we post on Tuesdays. And Quay…" was
stored as "From November we post on Tuesdays." (6 of 11 words), so it was a memory, not a rule, and
the Thursday rule it replaced stayed. A lead-in that announces several things heads a list: it is
not part of each item. So the clause after its ':' is weighed as a sentence of its own too, but only
when the lead-in names several things (``SEVERAL``) and holds no negation, no condition and no scope
word (``SCOPES``: 'Two rules for the Falmouth shop:' keeps its shop). Any other lead-in stays inside
its sentence, as N17-4 has it. The owner's words stay verbatim: nothing is added or changed.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

NEGATION = "not"
NEGATIONS = frozenset({"not", "no", "never", "none", "nothing", "nobody", "neither", "nor", "without"})
# Contractions whose stem is not the word before "n't" ("cannot" is said either way).
CONTRACTED = {"can't": "can", "won't": "will", "shan't": "shall", "ain't": "is", "cannot": "can"}
# The words that make a rule conditional: dropping one changes what the owner asked.
CONDITIONS = frozenset({"if", "unless", "except", "until", "only", "before", "after", "ask", "confirm", "approve",
                        "approval", "first", "when", "once", "while", "but"})
MAX_RULE_WORDS = 80
# F460: a lead-in naming several things ('Two things to remember, please:') heads a list.
SEVERAL = frozenset({"two", "three", "four", "five", "several", "few", "couple", "both", "things", "rules", "points"})
# Words in a lead-in that may set where or for whom its items hold: such a lead-in is never left out.
SCOPES = frozenset({"for", "at", "in", "on"})
_LEAD_IN = re.compile(r"^(?P<lead>[^:]+):\s+(?P<rest>\S.*)$", re.S)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")
_WORD = re.compile(r"[\w£$€%][\w'£$€%.,:/-]*")
_TRAILING = "'.,:/-"
# What surrounds a sentence the owner quoted, taken off the copy ``their_sentence`` gives back.
_QUOTES = "\"'\u201c\u201d\u2018\u2019\u00ab\u00bb"
_END_MARKS = ".!?"
_FULL_STOP = "."


def in_their_words(text: str, owner_turns: Sequence[str], coverage: float) -> bool:
    """Whether each sentence of ``text`` is one of ``owner_turns``' own sentences (above);
    False when it has none, or the owner said nothing."""
    said = [seq for turn in owner_turns for seq in _statements(turn)]
    stored = [_weighed(sentence) for sentence in _SENTENCE.findall(_normal(text))]
    if any(seq is None or len(seq) > MAX_RULE_WORDS for seq in stored):
        return False
    stored = [seq for seq in stored if seq]
    return bool(stored) and all(any(_held(seq, theirs, coverage) for theirs in said) for seq in stored)


def their_sentence(text: str, owner_turns: Sequence[str], coverage: float) -> Optional[str]:
    """``text`` with each sentence the owner's messages do not hold (``_held``) put back as the
    owner's own sentence that holds all its words in their order: the shortest in their newest
    message that has one, never a question, in their casing (NFKC), its surrounding quotes off,
    a full stop added when it has no end mark. A scope or a 'not' Auto left out comes back with
    their sentence. None when a sentence cannot be weighed, is past the cap, or holds a word no
    sentence of theirs holds in order (an added 'not', 'Noted.')."""
    said = [_their_sentences(turn) for turn in owner_turns]
    pieces: List[str] = []
    for sentence in _SENTENCE.findall(unicodedata.normalize("NFKC", text or "")):
        seq = _weighed(_normal(sentence))
        if seq is None or len(seq) > MAX_RULE_WORDS:
            return None
        piece = _as_held(sentence, seq, said, coverage) if seq else ""
        if piece is None:
            return None
        pieces.append(piece)
    return " ".join(dict.fromkeys(piece for piece in pieces if piece)) or None


def _their_sentences(turn: str) -> List[Tuple[str, List[str]]]:
    """Each sentence of a message that is no question, nor past the cap: its text (NFKC) and its
    words (``_statements``' sentences, with the text they were read from)."""
    sentences = (s for s in _listed(_SENTENCE.findall(unicodedata.normalize("NFKC", turn or ""))) if "?" not in s)
    pairs = [(sentence, _words(_normal(sentence))) for sentence in sentences]
    return [(sentence, seq) for sentence, seq in pairs if seq and len(seq) <= MAX_RULE_WORDS]


def _as_held(sentence: str, seq: List[str], said: List[List[Tuple[str, List[str]]]],
             coverage: float) -> Optional[str]:
    """The stored sentence as stored when a sentence of theirs holds it; else their sentence that
    holds all its words in order, from their newest message with one; None when none does."""
    if any(_held(seq, theirs, coverage) for message in said for _text, theirs in message):
        return sentence.strip()
    for message in said:
        holding = [(len(theirs), text) for text, theirs in message
                   if set(seq) <= set(theirs) and _in_order(seq, theirs) == len(seq)]
        if holding:
            return _as_said(min(holding, key=lambda pair: pair[0])[1])
    return None


def _as_said(sentence: str) -> str:
    """The owner's sentence as a rule is stored: trimmed, its surrounding quotes off, ended."""
    trimmed = sentence.strip().strip(_QUOTES).strip()
    return trimmed if trimmed.endswith(tuple(_END_MARKS)) else f"{trimmed}{_FULL_STOP}"


def words(text: Optional[str]) -> List[str]:
    """Every word of ``text`` in order, read as a rule is weighed (normalised, punctuation off, a
    negation as NEGATION), across its sentences."""
    return _words(_normal(text))


def _normal(text: Optional[str]) -> str:
    return unicodedata.normalize("NFKC", text or "").replace("\u2019", "'").casefold()


def _statements(turn: str) -> List[List[str]]:
    """The words of each of a message's sentences that is no question, nor past the cap."""
    sentences = (s for s in _listed(_SENTENCE.findall(_normal(turn))) if "?" not in s)
    return [seq for seq in map(_words, sentences) if seq and len(seq) <= MAX_RULE_WORDS]


def _listed(sentences: Sequence[str]) -> List[str]:
    """Each sentence, and after it the clause past its lead-in's ':' when that lead-in heads a list
    (F460, above)."""
    out: List[str] = []
    for sentence in sentences:
        out.append(sentence)
        found = _LEAD_IN.match(sentence.strip())
        if found and _heads_a_list(found.group("lead")):
            out.append(found.group("rest"))
    return out


def _heads_a_list(lead: str) -> bool:
    """A lead-in that names several things and holds no negation, condition or scope word."""
    said = set(_words(_normal(lead)))
    return bool(said & SEVERAL) and NEGATION not in said and not said & (CONDITIONS | SCOPES)


def _weighed(sentence: str) -> Optional[List[str]]:
    """The stored sentence's words, or None when it holds something that is neither a word (its
    marks, a Devanagari vowel sign say, included), punctuation nor space (the check cannot weigh
    it, so it is never the owner's)."""
    rest = _WORD.sub("", sentence)
    if any(not (c.isspace() or unicodedata.category(c)[0] in "PM") for c in rest):
        return None
    return _words(sentence)


def _words(sentence: str) -> List[str]:
    """Every word of the (normalised) sentence in order, its punctuation off; a negation is
    NEGATION and a contraction its stem and NEGATION ("don't" → "do", "not")."""
    out: List[str] = []
    for raw in _WORD.findall(sentence):
        word = raw.rstrip(_TRAILING)
        if word.endswith("n't") or word in CONTRACTED:
            out.extend([CONTRACTED.get(word, word[:-3]), NEGATION])
        elif word:
            out.append(NEGATION if word in NEGATIONS else word)
    return out


def _held(seq: List[str], theirs: List[str], coverage: float) -> bool:
    """``seq`` is all ``theirs``' words in their order, keeps ``coverage`` of them, and says no as often."""
    if seq.count(NEGATION) != theirs.count(NEGATION) or not set(seq) <= set(theirs):
        return False
    if not CONDITIONS.intersection(theirs) <= set(seq):
        return False
    if not all(word.isascii() for word in theirs):    # a no or an if this check cannot read: every word
        return seq == theirs
    return _in_order(seq, theirs) == len(seq) and len(seq) / len(theirs) >= coverage


def _in_order(a: List[str], b: List[str]) -> int:
    """The longest common subsequence of two word lists (each at most MAX_RULE_WORDS)."""
    row = [0] * (len(b) + 1)
    for word in a:
        diagonal = 0
        for j, other in enumerate(b, 1):
            diagonal, row[j] = row[j], diagonal + 1 if word == other else max(row[j], row[j - 1])
    return row[-1]


__all__ = ["MAX_RULE_WORDS", "NEGATION", "in_their_words", "their_sentence", "words"]
