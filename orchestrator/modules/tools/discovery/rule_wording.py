"""P256-FIX-N14-4 (night 14, F418): is a memory in the owner's own words?

A standing rule rides every turn as the owner's instruction, so a memory is one only when it is
what they said: each of its sentences is one of the owner's own sentences, word for word in their
order (Auto may leave words out, never add or change one), keeping at least ``coverage`` of that
sentence, and saying no exactly as often as they did. Every word counts (numbers, 'before' and
'after', 'on' and 'off'), so 'post at 10' is never 'post at 11', 'Mondays, not Tuesdays' never
'Tuesdays, not Mondays', and a dropped 'not' never passes. A question the owner asked ('do we post
on Tuesdays?') is no rule. Sentences past ``MAX_RULE_WORDS`` are not weighed: a rule is short, and
the comparison stays a few thousand steps.
"""
from __future__ import annotations

import re
from typing import List, Sequence

NEGATION = "not"
NEGATIONS = frozenset({"not", "no", "never", "none", "nothing", "nobody", "neither", "nor", "without"})
# Contractions whose stem is not the word before "n't" ("cannot" is said either way).
CONTRACTED = {"can't": "can", "won't": "will", "shan't": "shall", "ain't": "is", "cannot": "can"}
MAX_RULE_WORDS = 80
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")
_WORD = re.compile(r"[a-z0-9£$€%][a-z0-9'£$€%.,:/-]*", re.I)
_TRAILING = "'.,:/-"


def in_their_words(text: str, owner_turns: Sequence[str], coverage: float) -> bool:
    """Whether each sentence of ``text`` is one of ``owner_turns``' own sentences (above);
    False when it has none, or the owner said nothing."""
    said = [seq for turn in owner_turns for seq in _statements(turn)]
    stored = [_words(sentence) for sentence in _SENTENCE.findall(text or "")]
    stored = [seq for seq in stored if seq]
    if not stored or any(len(seq) > MAX_RULE_WORDS for seq in stored):
        return False
    return all(any(_held(seq, theirs, coverage) for theirs in said) for seq in stored)


def _statements(turn: str) -> List[List[str]]:
    """The words of each of a message's sentences that is no question, nor past the cap."""
    sentences = (s for s in _SENTENCE.findall(turn or "") if not s.rstrip().endswith("?"))
    return [seq for seq in map(_words, sentences) if seq and len(seq) <= MAX_RULE_WORDS]


def _words(sentence: str) -> List[str]:
    """Every word of the sentence in order, lower case, its punctuation off; a negation is
    NEGATION and a contraction its stem and NEGATION ("don't" → "do", "not")."""
    out: List[str] = []
    for raw in _WORD.findall((sentence or "").replace("\u2019", "'")):
        word = raw.lower().rstrip(_TRAILING)
        if word.endswith("n't") or word in CONTRACTED:
            out.extend([CONTRACTED.get(word, word[:-3]), NEGATION])
        elif word:
            out.append(NEGATION if word in NEGATIONS else word)
    return out


def _held(seq: List[str], theirs: List[str], coverage: float) -> bool:
    """``seq`` is all ``theirs``' words in their order, keeps ``coverage`` of them, and says no as often."""
    if seq.count(NEGATION) != theirs.count(NEGATION) or not set(seq) <= set(theirs):
        return False
    return _in_order(seq, theirs) == len(seq) and len(seq) / len(theirs) >= coverage


def _in_order(a: List[str], b: List[str]) -> int:
    """The longest common subsequence of two word lists (each at most MAX_RULE_WORDS)."""
    row = [0] * (len(b) + 1)
    for word in a:
        diagonal = 0
        for j, other in enumerate(b, 1):
            diagonal, row[j] = row[j], diagonal + 1 if word == other else max(row[j], row[j - 1])
    return row[-1]


__all__ = ["MAX_RULE_WORDS", "NEGATION", "in_their_words"]
