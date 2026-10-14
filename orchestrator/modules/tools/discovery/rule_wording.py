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
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Sequence

NEGATION = "not"
NEGATIONS = frozenset({"not", "no", "never", "none", "nothing", "nobody", "neither", "nor", "without"})
# Contractions whose stem is not the word before "n't" ("cannot" is said either way).
CONTRACTED = {"can't": "can", "won't": "will", "shan't": "shall", "ain't": "is", "cannot": "can"}
# The words that make a rule conditional: dropping one changes what the owner asked.
CONDITIONS = frozenset({"if", "unless", "except", "until", "only", "before", "after", "ask", "confirm", "approve",
                        "approval", "first", "when", "once", "while", "but"})
MAX_RULE_WORDS = 80
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")
_WORD = re.compile(r"[\w£$€%][\w'£$€%.,:/-]*")
_TRAILING = "'.,:/-"


def in_their_words(text: str, owner_turns: Sequence[str], coverage: float) -> bool:
    """Whether each sentence of ``text`` is one of ``owner_turns``' own sentences (above);
    False when it has none, or the owner said nothing."""
    said = [seq for turn in owner_turns for seq in _statements(turn)]
    stored = [_weighed(sentence) for sentence in _SENTENCE.findall(_normal(text))]
    if any(seq is None or len(seq) > MAX_RULE_WORDS for seq in stored):
        return False
    stored = [seq for seq in stored if seq]
    return bool(stored) and all(any(_held(seq, theirs, coverage) for theirs in said) for seq in stored)


def words(text: Optional[str]) -> List[str]:
    """Every word of ``text`` in order, read as a rule is weighed (normalised, punctuation off, a
    negation as NEGATION), across its sentences."""
    return _words(_normal(text))


def _normal(text: Optional[str]) -> str:
    return unicodedata.normalize("NFKC", text or "").replace("\u2019", "'").casefold()


def _statements(turn: str) -> List[List[str]]:
    """The words of each of a message's sentences that is no question, nor past the cap."""
    sentences = (s for s in _SENTENCE.findall(_normal(turn)) if "?" not in s)
    return [seq for seq in map(_words, sentences) if seq and len(seq) <= MAX_RULE_WORDS]


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


__all__ = ["MAX_RULE_WORDS", "NEGATION", "in_their_words", "words"]
