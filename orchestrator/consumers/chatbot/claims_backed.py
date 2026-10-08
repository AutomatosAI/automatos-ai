"""PRD-256 FX-006: each claim of the answer is matched to the turn's done writes.

US-002's rule fired the not-done line only when NO write went through, so one successful
write silenced it for every other claim of the reply: night 12 (A697) said "I've reverted
the heartbeat" with no call at all, beside a refused playbook write and a saved memory, and
no line was shown. Now each completed-action claim of the answer (``COMPLETED_ACTION``) is
read for its verb and matched to the turn's done write receipts by the verb's family
(``FAMILIES``: created → a create, sent → a send, saved → a memory or a thing saved,
reverted → an update, installed → an install, …). A claim no done write of its family backs is not done, whatever
else succeeded in the turn. A claim whose verb is in no family ("Done.", "it's now on your
board", "I've set up…") says no more than that work happened: any done write backs it.

The line names what was not done when another write went through ("Just to be clear: I
haven't reverted anything in this reply…"); when nothing went through it stays the plain
not-done line ("…nothing has changed").

The shapes read as a report of work done: "I've <verb>", "has/have been <verb>" (a state like
"has been based on" is not a report), "it's been <verb>", "is now <verb>", "<Subject> launched ✅",
"you should now see", and a sentence that is only "Done.". A simple-past passive ("Ticket #1110
was completed at 03:04", "the order was placed last week") reports history, not this turn's own
work (P256-FIX-RVW-1, F186): it is no claim.
What the answer quotes ("> " lines, a fenced block) is a draft in its writer's voice, never a
claim. A plan word exempts only the claim it introduces ("Once I've sent it…"), never a claim
further on ("I'll just confirm that the card has been approved" is a claim); the reply's own
content ("I've drafted the email below") and a past time ("was approved yesterday", "as I've
noted before") are not reports of this turn's work, nor is the owner heard ("I've noted that
you're happy to…") or a read begun ("I've started reading…").
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from consumers.chatbot.claim_check import NOT_DONE, NOTHING_DONE
from modules.tools.execution.composio_action import is_slug, slug_stems

Receipt = Dict[str, Any]
Backs = Callable[[Receipt], bool]

NOT_DONE_PREFIX = NOT_DONE.split("{", 1)[0]          # "Just to be clear: "
NAMED_SAID = "I haven't {verbs} anything in this reply"

# Looking things up is not work done: "I've checked the board" claims no change.
_LOOKING = ("looked|checked|read|reviewed|found|searched|seen|noticed|pulled|gone|been|had|got|heard|understood|"
            "asked|tried|confirmed|verified|explored|listed|counted|compared|considered|included|outlined|"
            "summari[sz]ed|explained|mentioned|attached")
# A passive that says how a thing is, not what was done to it: "was based on", "were meant to".
_STATES = ("based|called|named|supposed|meant|designed|intended|used|related|interested|concerned|involved|"
           "expected|required|needed|caused|limited|located|pleased|surprised|worried|confused|excited|"
           "planned|proposed|suggested|requested|recommended|allowed|linked|connected|labell?ed|titled|told|informed")
_DONE_VERB = rf"(?!(?:{_LOOKING})\b)(?:[a-z]+ed|sent|made|set|put|given|done|written|built|run|begun|kept|told|" \
             rf"taken|brought|chosen|paid|sold|cut|shut|drawn|thrown)\b"
_PASSIVE_VERB = rf"(?!(?:{_STATES})\b){_DONE_VERB}"
_ADVERBS = r"(?:(?:now|just|already|also|successfully|all)\s+)*"
# The ONE completed-action pattern. A match ends on the claim's verb where the shape has one.
COMPLETED_ACTION = re.compile(
    rf"\bI(?:'ve|’ve| have)\s+(?:(?:now|just|already|also|successfully|correctly|finally|actually|gone ahead and)"
    rf"\s+)*{_DONE_VERB}"
    rf"|\b(?:has|have)\s+{_ADVERBS}been\s+{_PASSIVE_VERB}"
    rf"|\b(?:it|that|this|everything|they)(?:'s|’s|'ve|’ve)\s+{_ADVERBS}been\s+{_PASSIVE_VERB}"
    rf"|\b(?:it'?s|it’s|they'?re|they’re|is|are)\s+now\s+(?:(?:on|in)\s+(?:your|the)\b|running\b|live\b|"
    rf"{_DONE_VERB})"
    rf"|\b{_DONE_VERB}(?=\s*[!.]?\s*[✅✔☑])"
    rf"|\byou(?:'ll|’ll| will| should)\s+now\s+see\b|\byou should see (?:it|this|them)\b"
    rf"|^\s*(?:all\s+)?(?:done|sorted|set)\s*[.!,]", re.I)
# A plan word introduces the claim right after it ("once I've sent it", "when the card has been
# approved"): that claim is not a report. Further on, it governs nothing.
_PLANNED = re.compile(r"\b(?:i'?ll|i’ll|i will|i'?m going to|i am going to|once|when|after|if|until|before)"
                      r"\s+(?:[\w'’]+\s+){0,2}$", re.I)
# Not a report of this turn's work: the reply's own content ("below"), or a past time ("as I've
# noted before," / "earlier": FX-007 keeps the families' back-references; "before Friday" is no past).
_NOT_THIS_TURN = re.compile(r"\b(?:below|here(?:'s|’s| is| are)|the following|yesterday|ago|previously|earlier|"
                            r"last (?:week|month|night|time|turn)|before(?=\s*(?:[,.;:!?)]|$)))\b", re.I)
# What follows the verb says it was no work (FX-007 keeps the families' two, F187 night 6): the
# owner heard ("I've noted that you're happy to…"), or a read begun ("I've started reading…").
_NO_WORK_AFTER = re.compile(
    r"\s+(?:(?:that\s+)?you(?:'re|’re| are|'d|’d| would| want| wish| have|'ve|’ve|'ll|’ll| will)\b|"
    r"to\s+(?:read|review|look|check|analy[sz]e|process|search|dig)\b|"
    r"(?:reading|reviewing|looking|checking|analy[sz]ing|processing|searching|digging|going\s+through)\b)", re.I)
_SENTENCES = re.compile(r"[^.!?\n]+[.!?]?")
_MARKS = re.compile(r"[*_`#>]")
_WORD = re.compile(r"[a-z]+", re.I)
_SENT_BACK = re.compile(r"\bsent\b.*\bback\b", re.I)


def _stem(r: Receipt) -> str:
    return str(r.get("action") or "").lower().removeprefix("platform_")


def _composio(r: Receipt) -> bool:
    return is_slug(str(r.get("action") or ""))


def _stems(r: Receipt) -> Tuple[str, ...]:
    """What a receipt's action is read by: a platform call's name, or each word of the Composio
    action that ran (P256-FIX-RVW-5: HUBSPOT_CREATE_CONTACT is "create_", "contact_"; a send
    word is "send_"), so a write backs only the claims its own verbs say."""
    return slug_stems(str(r["action"])) if _composio(r) else (_stem(r),)


def _effect(r: Receipt) -> str:
    return str(r.get("effect") or "").lower()


def _starts(*prefixes: str) -> Backs:
    return lambda r: any(stem.startswith(prefixes) for stem in _stems(r))


def _has(*words: str) -> Backs:
    """A word in the call's name; in a Composio slug, a whole word ("email" is no "mail")."""
    return lambda r: any(f"{word}_" in _stems(r) if _composio(r) else word in _stem(r) for word in words)


def _says(*effects: str) -> Backs:
    return lambda r: any(effect in _effect(r) for effect in effects)


def _any(*tests: Backs) -> Backs:
    return lambda r: any(test(r) for test in tests)


_CARD_MOVES = ("update_task_status", "update_task", "send_back")


_MOVE_EFFECTS = ("moved to", "sent back", "started", "marked blocked")
# FX-007 (F261, night 8): "Task #0422 has been moved to 'cancelled'" after a move to done. A
# claim that names the column needs the move there (the receipt's effect: receipts._MOVES).
_MOVED_TO = re.compile(r"\bmoved\b.*?\bto\s+(?:the\s+)?[\"'“‘]?(done|cancell?ed|review|inbox|blocked)\b", re.I)
_WHERE: Dict[str, Backs] = {
    "done": _any(_starts("approve_"), _says("moved to done")), "cancelled": _says("moved to cancelled"),
    "canceled": _says("moved to cancelled"), "review": _says("moved to review"),
    "inbox": _says("moved to the inbox"), "blocked": _says("marked blocked")}


def _moved(r: Receipt) -> bool:
    """A card the board moved (any column, or sent back): an edit that moved nothing is no move."""
    return _stem(r) in _CARD_MOVES and any(word in _effect(r) for word in _MOVE_EFFECTS)


_MEMORY = _has("memory", "remember", "note")
# A draft post is not a send: a post is sent when it is submitted, published or posted by its channel.
_SENDS = _has("send", "publish", "mail", "tweet", "notify", "reply", "forward", "broadcast", "submit_social_post",
              "create_post", "linked_in_post")
_BACK = _any(_says("sent back"), _starts("send_back", "reject"))

# The verb families: the claim's verb → what done write backs it. A verb in none says only
# that work happened; any done write backs it.
FAMILIES: Tuple[Tuple[FrozenSet[str], Backs], ...] = (
    (frozenset({"created", "made", "built", "generated"}), _starts("create_", "generate_", "make_", "build_")),
    (frozenset({"approved", "closed", "completed", "finished"}), _any(_starts("approve_"), _says("moved to done"))),
    (frozenset({"started", "launched", "kicked", "resumed", "initiated", "triggered"}),
     _any(_starts("start_", "launch_", "execute_", "run_", "resume_", "trigger_", "approve_mission",
                  "create_mission"), _says("started"))),
    (frozenset({"sent", "emailed", "mailed", "posted", "published", "messaged", "tweeted", "forwarded",
                "notified", "submitted"}), _any(_SENDS, _BACK)),
    # F363 (night 10c, FX-007 keeps it): a decision written onto the card the turn made is noted.
    (frozenset({"noted", "remembered", "stored", "memorised", "memorized"}),
     _any(_MEMORY, _starts("create_task", "update_task"))),
    (frozenset({"saved"}), _any(_MEMORY, _starts("create_", "generate_", "update_", "upload_", "submit_", "save_"))),
    (frozenset({"updated", "changed", "renamed", "reverted", "switched", "edited", "amended", "modified",
                "adjusted", "replaced", "fixed", "configured"}),
     _starts("update_", "configure_", "set_", "edit_", "rename_", "patch_", "revert_")),
    (frozenset({"deleted", "cancelled", "canceled", "archived", "unassigned"}),
     _any(_starts("delete_", "unassign_", "remove_", "cancel_", "archive_"), _says("moved to cancelled"))),
    # F379 (FX-007 keeps it): "I've removed the tasting notes from the carousel" is an edit of the post.
    (frozenset({"removed"}), _starts("delete_", "unassign_", "remove_", "update_", "edit_")),
    (frozenset({"scheduled"}), _starts("schedule_", "create_playbook", "create_schedule", "update_schedule")),
    (frozenset({"assigned"}), _any(_starts("assign_"), _says("agent set", "sent back to its agent"))),
    # F222 (FX-007 keeps it): an empty copy of a marketplace playbook is a create, never an install.
    (frozenset({"installed"}), _starts("install_")),
    (frozenset({"moved"}), _moved),
)


def _family(verb: str, sentence: str) -> Optional[Backs]:
    """What backs a claim of ``verb``: "sent … back" is a card sent back, not a send; "moved to
    <column>" is the move to that column."""
    if verb == "sent" and _SENT_BACK.search(sentence):
        return _any(_BACK, _moved)
    where = _MOVED_TO.search(sentence) if verb == "moved" else None
    if where:
        return _WHERE[where.group(1).lower()]
    return next((backs for verbs, backs in FAMILIES if verb in verbs), None)


def _claims_in(sentence: str) -> List[str]:
    """The verbs of the sentence's reports of work done ("" for a shape with no verb)."""
    if _NOT_THIS_TURN.search(sentence):
        return []
    verbs = []
    for match in COMPLETED_ACTION.finditer(sentence):
        if _PLANNED.search(sentence[: match.start()]) or _NO_WORK_AFTER.match(sentence, match.end()):
            continue
        words = _WORD.findall(match.group(0))
        verbs.append(words[-1].lower() if words else "")
    return verbs


def _own_words(text: str) -> str:
    """``text`` without what it quotes ("> " lines, fenced blocks): a draft written for the owner
    speaks in its writer's voice (FX-007 keeps the families' rule)."""
    kept, fenced = [], False
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            fenced = not fenced
        elif not fenced and not stripped.startswith(">"):
            kept.append(line)
    if fenced:  # a fence never closed quotes nothing: only the "> " lines are left out
        kept = [line for line in text.splitlines() if not line.lstrip().startswith(">")]
    return "\n".join(kept)


def claims(answer: str) -> List[Tuple[str, str]]:
    """Each report of work done in Auto's own words: (its verb, its sentence)."""
    text = _MARKS.sub("", _own_words(answer or ""))
    return [(verb, sentence) for sentence in _SENTENCES.findall(text) for verb in _claims_in(sentence)]


def claims_work_done(answer: str) -> bool:
    """Whether the answer reports, in the first person or as an outcome, work as done."""
    return bool(claims(answer))


def unbacked_claims(answer: str, done_writes: Sequence[Receipt]) -> List[Tuple[str, bool]]:
    """The answer's claims no done write backs: (the verb, whether its family is known). A
    claim of a known family needs a done write of that family (a Composio action's by the
    words of its slug); any other needs any done write."""
    unbacked = []
    for verb, sentence in claims(answer):
        backs = _family(verb, sentence)
        backed = any(backs(r) for r in done_writes) if backs else bool(done_writes)
        if not backed:
            unbacked.append((verb, backs is not None))
    return unbacked


def _listed(verbs: Sequence[str]) -> str:
    return verbs[0] if len(verbs) == 1 else f"{', '.join(verbs[:-1])} or {verbs[-1]}"


def not_done_line(answer: str, done_writes: Sequence[Receipt]) -> Optional[str]:
    """The one not-done line for the answer, or None when every claim is backed. It names what
    was not done when another write went through; else it is the plain not-done line."""
    unbacked = unbacked_claims(answer, done_writes)
    if not unbacked:
        return None
    named = list(dict.fromkeys(verb for verb, known in unbacked if known))
    if not done_writes or not named:
        return NOTHING_DONE
    return NOT_DONE.format(said=NAMED_SAID.format(verbs=_listed(named)))


def is_not_done_line(line: str) -> bool:
    """Whether a line above the answer is the not-done line (plain or naming its claim)."""
    return line.startswith(NOT_DONE_PREFIX)


__all__ = ["COMPLETED_ACTION", "FAMILIES", "NOT_DONE_PREFIX", "claims", "claims_work_done", "is_not_done_line",
           "not_done_line", "unbacked_claims"]
