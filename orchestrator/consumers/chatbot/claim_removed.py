"""P256-FIX-N15-1 (F422, night 15): an unbacked claim is taken out of the reply, never nudged.

The loop's final-reply check (``ToolLoopExecutor._recover_claimed_action``, F108) answered a reply
that claimed work no call did with a nudge in the owner's turn ("Your previous reply says something
was sent …") and returned the retry IN PLACE of the reply. The retry answered the nudge ("Nothing has
been sent. My last reply said…") and dropped what the turn's reads had fetched: about one board or
roster reply in four on night 15 (~100 nudges on ~372 asks) came back as commentary on a reply the
owner never saw.

Now the reply stands as it was composed. Each sentence that makes a claim no done write backs (the
receipts' rule, ``receipts.unbacked_sentences``) is taken out, and one plain line takes the first
one's place: the not-done wording, naming the claim ("Just to be clear: I haven't sent anything in
this reply. …"). A write that waits for the owner's click, or was refused, is said in that same line
(what the nudge's notes said, US-006 and P256-FIX-RVW-4). There is no second model call. The reply
streamed live, so the screen is told it stands corrected (``screen_watch.corrected``): the reply is
taken off and the corrected copy is sent in its place.

N15-CI: a claim the rule reads only in Auto's voice (an agent's run is held to its first-person claims)
goes with the claim it continues on its line: "I've approved the mission. It's now running." left "It's
now running." after the line that denied it. A reply that was nothing but its claims, whose line would
deny a write the turn did ("Task #0422 has been moved to 'cancelled'." after a move to Done: "I haven't
moved anything" is false), says so (``Removed.denies_done``): Auto's turn answers it with the account
of its calls instead (F264-b, ``turn_account.account_of``).
"""
from __future__ import annotations

import copy
import dataclasses
import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from consumers.chatbot.claim_check import NOT_DONE, NOTHING_DONE_SAID
from consumers.chatbot.claims_backed import FAMILIES, SAYS_DONE, SENTENCES, not_said, own_lines, without_marks
from consumers.chatbot.receipts import DONE, REFUSED, WRITE, receipt, tried, unbacked_sentences
from core.llm.screen_watch import corrected
from modules.tools.execution.card_raised import act_of, has_a_card, the_ask
from modules.tools.execution.turn_account import auto_speaks

Outcome = Tuple[str, Dict[str, Any], Any]

ASKED = "I asked for your {how} to {act}, so it hasn't happened yet"
CLICK, OK = "click", "OK"
TRIED = "I tried to {what} and it didn't go through"
SENTENCE_LOGGED_CHARS = 200    # the loop's warning names the sentence, cut to this
_LEADING_SPACE = re.compile(r"^\s*")
_OPENING_MARKS = re.compile(r"^\s*([*_`]+)")


@dataclass(frozen=True)
class Removed:
    """A reply with its unbacked claims taken out: the text the owner gets, the first claim's verb
    ("done" for a bare claim) and the sentence that made it (its first ``SENTENCE_LOGGED_CHARS``, for the
    loop's warning: a night's log shows which sentences fire); ``denies_done``: nothing of the reply is
    left but the line, and the line denies a write the turn did (N15-CI, F264-b)."""

    text: str
    family: str
    sentence: str
    denies_done: bool = False


def _asks(outcomes: Iterable[Outcome]) -> List[str]:
    """The turn's asks for the owner's click (or OK), each once, in the card's words."""
    asks = []
    for action, _params, result in outcomes:
        ask = the_ask(result)
        if ask is not None:
            asks.append(ASKED.format(how=CLICK if has_a_card(ask) else OK, act=act_of(ask, action)))
    return list(dict.fromkeys(asks))


def _refused(outcomes: Iterable[Outcome]) -> List[str]:
    """The turn's refused writes that no later call of the same action made good, each once."""
    writes = [r for r in (receipt(*outcome) for outcome in outcomes) if r["kind"] == WRITE]
    done = {r["action"] for r in writes if r["status"] == DONE}
    refused: Dict[str, Dict[str, Any]] = {}
    for r in writes:
        if r["status"] == REFUSED and r["action"] not in done:
            refused.setdefault(r["action"], r)
    return list(dict.fromkeys(TRIED.format(what=tried(r)) for r in refused.values()))


def not_done_in_place(verbs: Iterable[str], outcomes: List[Outcome]) -> str:
    """The one line in place of the claims: what was not done, naming the claims' verbs (the plain
    not-done words for a bare "Done."), then each ask that waits and each write that was refused."""
    named = list(dict.fromkeys(verb for verb in verbs if verb not in SAYS_DONE))
    said = [not_said(named) if named else NOTHING_DONE_SAID, *_asks(outcomes), *_refused(outcomes)]
    return NOT_DONE.format(said="; ".join(said))


def _line_without(line: str, claimed: FrozenSet[str], put: str) -> Tuple[str, bool]:
    """``line`` with its claimed sentences taken out, ``put`` in the first one's place when given;
    and whether a sentence was taken out. What only closes a taken-out sentence's marks goes with it."""
    kept, at, removed, marks = [], 0, False, ""
    for found in SENTENCES.finditer(line):
        sentence = found.group(0)
        kept.append(line[at:found.start()])
        at = found.end()
        if without_marks(sentence).strip() in claimed:
            kept.append(f"{_LEADING_SPACE.match(sentence).group(0)}{put}" if put and not removed else "")
            opening = _OPENING_MARKS.match(sentence)
            removed, marks = True, opening.group(1) if opening else ""
            continue
        if marks and sentence.startswith(marks):    # the marks that closed the sentence taken out ("**")
            sentence = sentence[len(marks):]
        kept.append(sentence)
        marks = ""
    kept.append(line[at:])
    return "".join(kept), removed


def _replaced(answer: str, claimed: FrozenSet[str], line: str) -> str:
    """``answer`` with each of its own sentences in ``claimed`` taken out and ``line`` in the first
    one's place; quoted lines are never touched. A line left empty goes; ``line`` ends a reply whose
    sentences could not be found."""
    lines = answer.split("\n")
    kept, placed = [], False
    for text, mine in zip(lines, own_lines(lines)):
        if not mine:
            kept.append(text)
            continue
        cut, removed = _line_without(text, claimed, "" if placed else line)
        placed = placed or removed
        if not (removed and not cut.strip()):
            kept.append(cut)
    if not placed:
        kept.extend(["", line])
    return "\n".join(kept)


def _going_on(answer: str, claimed: FrozenSet[str], voiced: FrozenSet[str]) -> FrozenSet[str]:
    """The claims read in Auto's voice (``voiced``) that go on, on its line, from a claim taken out
    ("I've approved the mission. It's now running." in an agent's run, held to its first-person claims):
    they go with it, or the reply would go on to say what the line denies. Each is one of ``voiced`` the
    rule did not take out, right after a sentence taken out (or another that goes on from one)."""
    more = voiced - claimed
    if not more:
        return frozenset()
    going = set()
    for text in answer.split("\n"):
        after_one = False
        for found in SENTENCES.finditer(text):
            said = without_marks(found.group(0)).strip()
            after_one = said in claimed or (after_one and said in more)
            if after_one and said in more:
                going.add(said)
    return frozenset(going)


def _denies_a_done_write(verbs: Iterable[str], outcomes: List[Outcome]) -> bool:
    """Whether the not-done line would deny work a write of the turn did: a claim's verb whose family a
    done write is in, though not as the claim said it (F264-b, night 8: #0422 moved to Done, the reply
    said to Cancelled; "I haven't moved anything" is false)."""
    named = set(verbs)
    done = [r for r in (receipt(*outcome) for outcome in outcomes) if r["kind"] == WRITE and r["status"] == DONE]
    families = [backs for family, backs in FAMILIES if named & family]
    return any(backs(r) for backs in families for r in done)


def without_unbacked_claims(answer: str, outcomes: Iterable[Outcome],
                            promises: Optional[bool] = True) -> Optional[Removed]:
    """The reply with the sentences that make an unbacked claim taken out and the not-done line in
    the first one's place, or None when every claim is backed (``promises``: whose voice the reply
    is, as ``receipts.unbacked_claim`` reads it). The claims in Auto's voice that go on from one taken
    out go with it (``_going_on``)."""
    ran = list(outcomes)
    found = unbacked_sentences(answer, ran, promises)
    if not found:
        return None
    verbs = [verb for verb, _sentence in found]
    line = not_done_in_place(verbs, ran)
    claimed = frozenset(sentence.strip() for _verb, sentence in found)
    voiced = claimed if auto_speaks(promises) else frozenset(
        sentence.strip() for _verb, sentence in unbacked_sentences(answer, ran, True))
    text = _replaced(answer, claimed | _going_on(answer, claimed, voiced), line)
    return Removed(text=text, family=found[0][0], sentence=found[0][1].strip()[:SENTENCE_LOGGED_CHARS],
                   denies_done=text.strip() == line and _denies_a_done_write(verbs, ran))


def in_place_of(response: Any, text: str) -> Any:
    """A copy of ``response`` holding ``text``, marked not streamed so the chat sends it; the screen
    is told it stands corrected, so the streamed reply comes off. ``response`` is never changed."""
    if dataclasses.is_dataclass(response):
        names = {f.name for f in dataclasses.fields(response)}
        fresh = dataclasses.replace(response, content=text, **({"streamed": False} if "streamed" in names else {}))
    else:
        fresh = copy.copy(response)
        fresh.content = text
        if hasattr(fresh, "streamed"):
            fresh.streamed = False
    corrected(response, fresh)
    return fresh


__all__ = ["ASKED", "Removed", "SENTENCE_LOGGED_CHARS", "TRIED", "in_place_of", "not_done_in_place", "without_unbacked_claims"]
