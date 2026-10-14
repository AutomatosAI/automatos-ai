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

N15-CI:

- The line is Auto's, to the owner: an agent's text (a customer's email among them) never carries
  it. In an agent's voice the claim is taken out with nothing in its place, and so is a claim that
  goes on from it on its line ("I've approved the mission. It's now running."); the owner is told
  beside the answer (``Removed.taken_out``: the card's note, services/claims_taken_out.py). Only when
  nothing of the agent's text is left is the line the whole answer.
- The line never denies a write that went through: a claim whose verb's family a done write to the
  very card it names is in, said wrong ("Task #0422 has been moved to 'cancelled'." after #0422 moved to
  Done), is replaced by that write's account ("#0422 was moved to Done."). A claim about anything else
  (a letter over an unrelated card made) still takes the line. A reply that was nothing but such claims says so
  (``Removed.denies_done``): Auto's turn answers with the account of its calls (F264-b).
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
WENT_THROUGH = "What went through: {effect}{subject}."
_DID_TO = ("moved to", "sent back", "started", "marked")   # a receipt's effect that reads "<card> was …"
_LINE = "\0the not-done line"     # in a sentence's place: the line, said once
_CARD_KEYS = ("task_id", "task_number", "number")      # a write's params that name its card
_NUMBER = re.compile(r"(?<![\w.#])#?(\d{1,7}(?:\.\d+)?)(?!\w)")
_LEADING_SPACE = re.compile(r"^\s*")
_OPENING_MARKS = re.compile(r"^\s*([*_`]+)")


@dataclass(frozen=True)
class Removed:
    """A reply with its unbacked claims taken out: the text the owner gets, the first claim's verb
    ("done" for a bare claim) and the sentence that made it (its first ``SENTENCE_LOGGED_CHARS``, for the
    loop's warning: a night's log shows which sentences fire). N15-CI: ``denies_done``, Auto's reply was
    nothing but claims and one of them said a write that went through wrong (F264-b: the loop answers
    with the turn's account); ``taken_out``, the sentences taken out of an agent's text with nothing in
    their place, for the owner to be told beside it (``claims_in_place.TAKEN_OUT``)."""

    text: str
    family: str
    sentence: str
    denies_done: bool = False
    taken_out: Tuple[str, ...] = ()


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


def _instead_of(said: str, put: Dict[str, str], line: str, placed: bool) -> Tuple[str, bool]:
    """What goes in place of the sentence ``said`` (one of ``put``), and whether the line is said now:
    the line once, at the first sentence it stands for; a write's own account; or nothing."""
    instead = put[said]
    if instead != _LINE:
        return instead, placed
    return ("" if placed else line), True


def _line_without(text: str, put: Dict[str, str], line: str, placed: bool) -> Tuple[str, bool, bool]:
    """``text`` (one line) with its sentences in ``put`` taken out, each with what ``put`` says goes in
    its place; whether a sentence was taken out, and whether the line is said by now. What only
    closes a taken-out sentence's marks goes with it."""
    kept, at, removed, marks = [], 0, False, ""
    for found in SENTENCES.finditer(text):
        sentence = found.group(0)
        kept.append(text[at:found.start()])
        at = found.end()
        said = without_marks(sentence).strip()
        if said in put:
            instead, placed = _instead_of(said, put, line, placed)
            kept.append(f"{_LEADING_SPACE.match(sentence).group(0)}{instead}" if instead else "")
            opening = _OPENING_MARKS.match(sentence)
            removed, marks = True, opening.group(1) if opening else ""
            continue
        if marks and sentence.startswith(marks):    # the marks that closed the sentence taken out ("**")
            sentence = sentence[len(marks):]
        kept.append(sentence)
        marks = ""
    kept.append(text[at:])
    return "".join(kept), removed, placed


def _tidied(text: str, cut: str) -> str:
    """A line a sentence was taken out of with nothing in its place: its indent kept, its words single-spaced."""
    return f"{_LEADING_SPACE.match(text).group(0)}{' '.join(cut.split())}"


def _replaced(answer: str, put: Dict[str, str], line: str, tidy: bool = False) -> str:
    """``answer`` with each of its own sentences in ``put`` taken out and what ``put`` says in its
    place (``_LINE``: the not-done line, once, at the first); quoted lines are never touched. A line
    left empty goes; the line ends a reply whose sentences could not be found. ``tidy``: a line a
    sentence left with nothing in its place is single-spaced."""
    lines = answer.split("\n")
    kept, placed = [], False
    for text, mine in zip(lines, own_lines(lines)):
        if not mine:
            kept.append(text)
            continue
        cut, removed, placed = _line_without(text, put, line, placed)
        cut = _tidied(text, cut) if removed and tidy else cut
        if not (removed and not cut.strip()):
            kept.append(cut)
    if not placed and _LINE in put.values():
        kept.extend(["", line])
    return "\n".join(kept)


def _going_on(answer: str, claimed: FrozenSet[str], voiced: FrozenSet[str]) -> Tuple[str, ...]:
    """The claims read in Auto's voice (``voiced``) that go on, on its line, from a claim taken out
    ("I've approved the mission. It's now running." in an agent's run, held to its first-person claims):
    they go with it, or the reply would go on to say what was taken out. Each is one of ``voiced`` the
    rule did not take out, right after a sentence taken out (or another that goes on from one)."""
    more = voiced - claimed
    if not more:
        return ()
    going: List[str] = []
    for text in answer.split("\n"):
        after_one = False
        for found in SENTENCES.finditer(text):
            said = without_marks(found.group(0)).strip()
            after_one = said in claimed or (after_one and said in more)
            if after_one and said in more:
                going.append(said)
    return tuple(dict.fromkeys(going))


def _named(text: str) -> Dict[str, str]:
    """The numbers ``text`` names (a card's "#0422", "0422" or 422), each as read ("422") to as written."""
    return {number.lstrip("0") or "0": number for number in _NUMBER.findall(text)}


def _done_writes(outcomes: List[Outcome]) -> List[Tuple[Dict[str, Any], FrozenSet[str]]]:
    """The turn's done writes, each with the numbers of what it wrote to (its card's number or id,
    from its params and its receipt's subject)."""
    done = []
    for action, params, result in outcomes:
        r = receipt(action, params, result)
        if r["kind"] != WRITE or r["status"] != DONE:
            continue
        given = [str(params.get(key)) for key in _CARD_KEYS if isinstance(params, dict) and params.get(key) is not None]
        done.append((r, frozenset(number for text in (*given, str(r.get("subject") or "")) for number in _named(text))))
    return done


def _of_its_family(verb: str, r: Dict[str, Any]) -> bool:
    """Whether the done write ``r`` is of the family of the claim's verb, however the claim said it
    ("moved": any card moved, to whichever column; "installed" is no create)."""
    return any(verb in family and backs(r) for family, backs in FAMILIES)


def _what_it_did(r: Dict[str, Any], written: str) -> str:
    """A done write, in a sentence: "#0422 was moved to Done."; "What went through: Card made (#0231)."
    ``written``: its number as the claim wrote it, when its receipt names none."""
    subject, effect = str(r.get("subject") or "") or f"#{written}", str(r.get("effect") or "")
    if effect.startswith(_DID_TO):
        return f"{subject} was {effect}."
    return WENT_THROUGH.format(effect=effect[:1].upper() + effect[1:], subject=f" ({subject})")


def _own_instead(found: List[Tuple[str, str]], outcomes: List[Outcome]) -> Dict[str, str]:
    """In Auto's voice, what goes in place of each claim: the account of the done writes of its verb's
    family to the very thing the claim names ("Task #0422 has been moved to 'cancelled'." over #0422
    moved to Done: "I haven't moved anything" would be false), else the not-done line (``_LINE``). A
    claim that names no number a write was to (a letter, a card it doesn't name) takes the line."""
    done = _done_writes(outcomes)
    put: Dict[str, str] = {}
    for verb, sentence in found:
        named = _named(sentence)
        did = [_what_it_did(r, named[next(iter(cards & named.keys()))]) for r, cards in done
               if cards & named.keys() and _of_its_family(verb, r)]
        put.setdefault(sentence.strip(), " ".join(dict.fromkeys(did)) if did else _LINE)
    return put


def _dropped(answer: str, found: List[Tuple[str, str]], outcomes: List[Outcome]) -> Dict[str, str]:
    """In an agent's voice (its text may be for a customer), each claim and each claim in Auto's voice
    that goes on from one is taken out with nothing in its place."""
    claimed = frozenset(sentence.strip() for _verb, sentence in found)
    voiced = frozenset(sentence.strip() for _verb, sentence in unbacked_sentences(answer, outcomes, True))
    return dict.fromkeys((*claimed, *_going_on(answer, claimed, voiced)), "")


def without_unbacked_claims(answer: str, outcomes: Iterable[Outcome],
                            promises: Optional[bool] = True) -> Optional[Removed]:
    """The reply with the sentences that make an unbacked claim taken out, or None when every claim
    is backed (``promises``: whose voice the reply is, as ``receipts.unbacked_claim`` reads it). In
    Auto's voice, the not-done line takes the first one's place, or the account of a write the claim
    said wrong takes its; in an agent's, nothing does (``Removed.taken_out`` tells the owner), unless
    nothing of its text is left, when the line is the whole answer."""
    ran = list(outcomes)
    found = unbacked_sentences(answer, ran, promises)
    if not found:
        return None
    own = auto_speaks(promises)
    put = _own_instead(found, ran) if own else _dropped(answer, found, ran)
    line = not_done_in_place([verb for verb, sentence in found if not own or put[sentence.strip()] == _LINE], ran)
    only = not _replaced(answer, dict.fromkeys(put, ""), line).strip()
    text = line if only and not own else _replaced(answer, put, line, tidy=not own)
    return Removed(text=text, family=found[0][0], sentence=found[0][1].strip()[:SENTENCE_LOGGED_CHARS],
                   denies_done=only and own and any(said != _LINE for said in put.values()),
                   taken_out=() if own or only else tuple(put))


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
