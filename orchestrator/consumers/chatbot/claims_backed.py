"""PRD-256 FX-006: each claim of the answer is matched to the turn's done writes.

US-002's rule fired the not-done line only when NO write went through, so one successful
write silenced it for every other claim of the reply: night 12 (A697) said "I've reverted
the heartbeat" with no call at all, beside a refused playbook write and a saved memory, and
no line was shown. Now each completed-action claim of the answer (``COMPLETED_ACTION``) is
read for its verb and matched to the turn's done write receipts by the verb's family
(``FAMILIES``: created → a create, sent → a send, saved → a memory or a thing saved,
reverted → an update, installed → an install, …). A claim no done write of its family backs is not done, whatever
else succeeded in the turn. A claim whose verb is in no family ("Done.", "it's now on your
board", "I've sorted it") says no more than that work happened: any done write backs it.
P256-FIX-RVW-6: added, paused, enabled, "set up", "turned off", uploaded, booked, paid,
connected, … have families (the deleted families asked an update for paused or "set up"), a
participle coordinated with a claim is a claim of its own ("I've created the task and sent it
to Declan" is created and sent), and the reply's own content exempts only its clause ("Here's
the update: the card has been approved." is a claim).

The line names what was not done when another write went through ("Just to be clear: I
haven't reverted anything in this reply…"); when nothing went through it stays the plain
not-done line ("…nothing has changed").

The shapes read as a report of work done: "I've <verb>", "has/have been <verb>" (a state like
"has been based on" is not a report), "it's been <verb>", "is now <verb>", "<Subject> launched ✅",
"you should now see", "was/were <verb>", and a sentence that is only "Done." (a shape with no
verb of its own: its verb is ""). P256-FIX-RVW-7: the first person ("I've <verb>", and what it
coordinates) is the writer's own claim; an agent's run reads only that (``first_person``): its
passives are its customer draft's voice. A simple-past passive with a past time in its sentence
("Ticket #1110 was completed at 03:04", "were approved this morning", "the order was placed last
week") reports history, not this turn's own work (P256-FIX-RVW-1, F186): it is no claim; without
one ("The card was approved.") it is (P256-FIX-RVW-20), unless it denies ("Nothing was done
yet.", "no step was added"), describes ("the coffees that were identified") or says a refusal
("was refused"). Replied, texted and dm'd are sends; shared and invited need a share/invite
write or a send, refunded a refund or a payment write: a mail draft or a saved memory backs none.
What the answer quotes ("> " lines, a fenced block) is a draft in its writer's voice, never a
claim. A plan word exempts only the claim it introduces ("Once I've sent it…"), never a claim
further on ("I'll just confirm that the card has been approved" is a claim); the reply's own
content ("I've drafted the email below", in its clause) and a past time ("was approved
yesterday", "as I've noted before", in its sentence) are not reports of this turn's work, nor is
the owner heard ("I've noted that you're happy to…") or a read begun ("I've started reading…").
P256-FIX-RVW-25: with the families gone this rule is the only guard, and a passive escaped it. A
stative passive ("The post is published.", "The task is done.", "is based on" a state), "got/went"
("The post got sent.", "It went out."), the first person's simple past ("I sent the email to
Declan.") and a sentence that is only "<Noun> <participle>." ("Email sent.", "Posted!") are claims,
under the same plan-word, reply-content, history and denial exemptions, and a stative one that
says what a read found ("I found that the boxes are posted on Monday") is none; a participle that
continues the claim's list after a comma is a claim of its own ("I've created the ticket, emailed Sam.").
P256-FIX-RVW-30: a turn that raised a card for the owner's click (a waiting receipt) says so in the
first person ("I've raised a card to change Scout's model", "I've sent you an approval card"): a
claim of raising, requesting, sending, submitting, putting or queuing whose clause names the ask (a
card, an approval, a click, an OK, a sign-off) is the ask, backed by it (``waiting``). A claim of the
change itself ("I've changed Scout's model.") never is, nor a numbered card sent or a card sent back.
P256-FIX-RVW-34: "made" is in no family of its own: "I've made a note of that." is noted, "I've made Scout
the owner of #0931." assigned, "I've made a carousel" a create, "I've made the changes" only that work
happened. A calendar write (GOOGLECALENDAR_CREATE_EVENT) backs "scheduled" and "booked". A question asks,
it reports nothing ("Can you confirm the invoice was paid?"), unless it opens on no question word and its
claim comes before a clause break ("I've sent it to Declan, want me to chase?"). "Mission #0365 has been
cancelled." after a read of #0365 says what the read found: a has-been or was claim about a number a
done read of the turn names is no claim.
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
           "planned|proposed|suggested|requested|recommended|allowed|linked|connected|labell?ed|titled|told|informed|"
           "refused|denied|failed|skipped")
_DONE_VERB = rf"(?!(?:{_LOOKING})\b)(?:[a-z]+ed|dm['’]d|sent|made|set|put|given|done|written|built|run|begun|" \
             rf"kept|told|taken|brought|chosen|paid|sold|cut|shut|drawn|thrown)\b"
_PASSIVE_VERB = rf"(?!(?:{_STATES})\b){_DONE_VERB}"
_ADVERBS = r"(?:(?:now|just|already|also|successfully|all)\s+)*"
_FIRST_ADVERBS = r"(?:(?:now|just|already|also|then|successfully|correctly|finally|actually|gone ahead and)\s+)*"
# The shapes the completed-action pattern (``COMPLETED_ACTION``, below the families) reads. A match
# ends on the claim's verb where the shape has one; a simple past is the group ``past``.
_SHAPES = (
    rf"\bI(?:'ve|’ve| have)\s+{_FIRST_ADVERBS}{_DONE_VERB}"
    rf"|\b(?:has|have)\s+{_ADVERBS}been\s+{_PASSIVE_VERB}"
    rf"|\b(?:it|that|this|everything|they)(?:'s|’s|'ve|’ve)\s+{_ADVERBS}been\s+{_PASSIVE_VERB}"
    rf"|(?P<past>\b(?:was|were)\s+{_ADVERBS}{_PASSIVE_VERB})"
    rf"|\b(?:it'?s|it’s|they'?re|they’re|is|are)\s+now\s+(?:(?:on|in)\s+(?:your|the)\b|running\b|live\b|"
    rf"{_DONE_VERB})"
    rf"|\b{_DONE_VERB}(?=\s*[!.]?\s*[✅✔☑])"
    rf"|\byou(?:'ll|’ll| will| should)\s+now\s+see\b|\byou should see (?:it|this|them)\b"
    rf"|^\s*(?:all\s+)?(?:done|sorted|set)\s*[.!,]")
# P256-FIX-RVW-25: what a stative passive or "got/went" says was done to a thing ("The post is
# published.", "The post got sent."); a state of it ("is based on", "is assigned to Scout") is none.
_OUTCOMES = (r"(?:done|sorted|fixed|published|posted|scheduled|created|sent|emailed|submitted|approved|closed|"
             r"completed|finished|cancell?ed|deleted|archived|removed|saved|installed|uploaded|booked|ordered|"
             r"purchased|paid|refunded|launched|added|shared|set(?=\s+up\b))\b")
# A plan word introduces the claim right after it ("once I've sent it", "when the card has been
# approved", "as soon as the agent is installed"): that claim is not a report. Further on, it governs nothing.
_PLANNED = re.compile(r"\b(?:i'?ll|i’ll|i will|i'?m going to|i am going to|once|when|after|if|until|before|"
                      r"as soon as)"
                      r"\s+(?:[\w'’]+\s+){0,2}$", re.I)
# Not a report of this turn's work: a past time exempts its sentence ("as I've noted before," /
# "earlier": FX-007 keeps the families' back-references; "before Friday" is no past) …
_PAST = re.compile(r"\b(?:yesterday|ago|previously|earlier|last (?:week|month|night|time|turn)|"
                   r"before(?=\s*(?:[,.;:!?)]|$)))\b", re.I)
# … a simple past is history when its sentence has a past time (P256-FIX-RVW-1, F186: "Ticket #1110
# was completed at 03:04", RVW-25: "It went out this morning."); "The card was approved." alone is a
# claim (P256-FIX-RVW-20) … and so is a denial ("Nothing was done yet.", "no step was added": Auto's
# own refusal lines, "No post is scheduled") or a relative clause ("the coffees that were identified").
# RVW-25: a stative passive or "went out" after a read says what the read found ("I found that the boxes are
# posted on Monday", "the board shows the post is published"), not this turn's work.
_READ_SAYS = re.compile(rf"\b(?:{_LOOKING}|see|sees|shows?|says?)\b(?:\s+(?:that|and))?(?:\s+[\w#'’-]+){{0,6}}\s+$",
                        re.I)
_DENIED = re.compile(r"\b(?:nothing(?:\s+(?:new|else|more))?|none|nobody|no\s+one|no(?:\s+[\w'’-]+){1,4}|"
                     r"that|which|who)\s+$", re.I)
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_HISTORY = re.compile(
    rf"\b(?:at\s+\d{{1,2}}(?:[:.]\d{{2}}\s*(?:am|pm)?|\s*(?:am|pm))|this\s+(?:morning|afternoon|evening)|"
    rf"last\s+(?:week|night)|ago|earlier|yesterday|on\s+(?:(?:mon|tues|wednes|thurs|fri|satur|sun)day|"
    rf"(?:the\s+)?\d{{1,2}}(?:st|nd|rd|th)|\d{{1,2}}\s+(?:of\s+)?{_MONTH}|{_MONTH}\s+\d{{1,2}}|"
    rf"\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}))\b", re.I)
# … the reply's own content only its clause (P256-FIX-RVW-6): "Here's the update: the card has
# been approved." is a claim after the colon; "I've drafted the email below" is none.
_REPLY_CONTENT = re.compile(r"\b(?:below|here(?:'s|’s| is| are)|the following)\b", re.I)
_CLAUSE_BREAK = re.compile(r"[:;—–]")
# A participle coordinated with a claim is a claim (P256-FIX-RVW-6): "I've created the task and sent it";
# so is one that continues its list after a comma (RVW-25: "I've created the ticket, emailed Sam."), not
# a state ("…, called 'Wholesale'") nor a new clause with its own subject ("…, Sam emailed").
_AND_THEN = re.compile(rf"\b(?:and|then)\s+(?:(?:then|also|just|now)\s+)?({_DONE_VERB})"
                       rf"|[,;]\s+(?:(?:and|then|also|just|now)\s+)*({_PASSIVE_VERB})", re.I)
# A phrasal claim's particle: "set up" and "turned off" are read whole when a family names them.
_PARTICLE = re.compile(r"\s+(up|off|on)\b", re.I)
# What follows the verb says it was no work (FX-007 keeps the families' two, F187 night 6): the
# owner heard ("I've noted that you're happy to…"), a read begun ("I've started reading…"), or
# (RVW-25) a slip owned ("I made a mistake in the call").
_NO_WORK_AFTER = re.compile(
    r"\s+(?:an?\s+(?:mistake|error|typo)\b|"
    r"(?:that\s+)?you(?:'re|’re| are|'d|’d| would| want| wish| have|'ve|’ve|'ll|’ll| will)\b|"
    r"to\s+(?:read|review|look|check|analy[sz]e|process|search|dig)\b|"
    r"(?:reading|reviewing|looking|checking|analy[sz]ing|processing|searching|digging|going\s+through)\b)", re.I)
_SENTENCES = re.compile(r"[^.!?\n]+[.!?]?")
_MARKS = re.compile(r"[*_`#>]")
_SENT_BACK = re.compile(r"\bsent\b.*\bback\b", re.I)
# P256-FIX-RVW-30: the verbs that say an ask was raised, and the words that name the ask in their clause
# ("Approve", "an approval card", "your OK"); "card #0422" (its "#" read off) is a card, never the ask's.
_ASKED = frozenset({"raised", "requested", "sent", "submitted", "put", "queued"})
_THE_ASK = re.compile(r"\b(?:approv(?:al|e)\b|click(?:s|ing)?\b|(?-i:OK)\b|sign[- ]?off\b|cards?\b(?!\s*#?\d))", re.I)
# P256-FIX-RVW-34: a question reports nothing: one that opens on its question word ("Should I tell
# Declan it's been sent?", "How many boxes went out on Monday, October 5th?"), or a claim with no clause
# break before its "?" ("…the invoice was paid?"); "I've sent it to Declan, want me to chase?" is a claim.
_QUESTION = re.compile(r"^\s*(?:can|could|would|should|shall|will|do|does|did|is|are|was|were|has|have|had|how|"
                       r"what|when|where|why|who|which|may|might)\b[^?]*\?\s*$", re.I)
_ASKS = re.compile(r"[^:;—–,]*\?\s*$")
# … nor a has-been or was claim about a number a read of the turn named ("#0365 has been cancelled").
_REPORTED = re.compile(r"\b(?:been|was|were)\b", re.I)
_CARD_NUMBER = re.compile(r"#\s?(\d+)")
# A shape that ends on its verb ("I've sent", "has been approved", "launched ✅"); "Done.", "is now live"
# and "you should now see" have none (P256-FIX-RVW-7).
_ON_ITS_VERB = re.compile(rf"\b{_DONE_VERB}$", re.I)
_FIRST_PERSON = re.compile(r"I(?:'ve|’ve| have)?\s", re.I)
# "It went out." is a send (RVW-25); "went live", like "is now live", says only that work happened.
_GONE_OUT = re.compile(r"\bout$", re.I)
# The claims that say only that work was done ("Done.", "it's been done", "is now live"): the nudge's
# "done". Any other verb in no family ("I've prepared a summary") is the line's alone.
SAYS_DONE = frozenset({"", "done"})


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
# P256-FIX-RVW-6: a setting switched or set up is an update, a configure, a schedule or a create.
_SETTING = ("update_", "configure_", "set_", "pause_", "resume_", "schedule_", "create_")


def _named(*words: str) -> Backs:
    """A word anywhere in the call's name or its slug's words (GOOGLECALENDAR is a calendar)."""
    return lambda r: any(word in stem for stem in _stems(r) for word in words)


# P256-FIX-RVW-34: a write whose slug or name says event, meeting, calendar or booking schedules or books.
_CALENDAR = _named("event", "meeting", "calendar", "booking")


def _placed(*words: str) -> Backs:
    """An order, a booking or a payment: a send, or a write that names it (SHOPIFY_CREATE_ORDER)."""
    return _any(_SENDS, _has(*words))


# The verb families: the claim's verb → what done write backs it. A verb in none says only
# that work happened; any done write backs it.
FAMILIES: Tuple[Tuple[FrozenSet[str], Backs], ...] = (
    (frozenset({"created", "built", "generated"}), _starts("create_", "generate_", "make_", "build_")),
    (frozenset({"approved", "closed", "completed", "finished"}), _any(_starts("approve_"), _says("moved to done"))),
    (frozenset({"started", "launched", "kicked", "resumed", "initiated", "triggered"}),
     _any(_starts("start_", "launch_", "execute_", "run_", "resume_", "trigger_", "approve_mission",
                  "create_mission"), _says("started"))),
    (frozenset({"sent", "emailed", "mailed", "posted", "published", "messaged", "tweeted", "forwarded",
                "notified", "submitted", "replied", "texted", "dm'd", "dmed"}), _any(_SENDS, _BACK)),
    # P256-FIX-RVW-20: a share, an invite or a refund is said only by its own write (a draft backs none).
    (frozenset({"shared"}), _any(_SENDS, _has("share", "shares", "sharing"))),
    (frozenset({"invited"}), _any(_SENDS, _has("invite", "invites", "invitation", "invitations"))),
    (frozenset({"refunded"}), _has("refund", "refunds", "payment", "payments")),
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
    (frozenset({"scheduled"}),
     _any(_starts("schedule_", "create_playbook", "create_schedule", "update_schedule"), _CALENDAR)),
    (frozenset({"assigned"}), _any(_starts("assign_"), _says("agent set", "sent back to its agent"))),
    # F222 (FX-007 keeps it): an empty copy of a marketplace playbook is a create, never an install.
    (frozenset({"installed"}), _starts("install_")),
    (frozenset({"moved"}), _moved),
    # P256-FIX-RVW-6: the verbs that said only "work happened" (A697's shape) now need their own kind.
    (frozenset({"added"}), _starts("add_", "assign_", "create_", "install_")),
    (frozenset({"paused", "disabled", "enabled", "activated", "deactivated", "turned off", "turned on"}),
     _starts(*_SETTING)),
    # F337 (FX-007 keeps it): "I've set up an invoice template" over a generate_document.
    (frozenset({"set up"}), _starts(*_SETTING, "generate_")),
    (frozenset({"uploaded"}), _starts("upload_", "create_document", "save_")),
    (frozenset({"booked"}), _any(_placed("book", "booking", "bookings"), _CALENDAR)),
    (frozenset({"ordered", "purchased"}), _placed("order", "orders", "purchase", "purchases")),
    (frozenset({"paid"}), _placed("pay", "payment", "payments", "charge", "charges")),
    (frozenset({"connected", "linked"}), _starts("connect_", "link_", "install_", "update_")),
)
# The families' two-word verbs: a claim's particle is read with its verb only for these.
_PHRASAL = frozenset(verb for verbs, _ in FAMILIES for verb in verbs if " " in verb)
# P256-FIX-RVW-34: "made" says what it made by what follows it; the family it takes ("made the changes": none).
_MADE = "made"
_MADE_AS = (
    (re.compile(r"\bmade\s+a\s+note\b", re.I), "noted"),
    (re.compile(r"\bmade\s+(?:[\w'’-]+\s+){1,3}(?:the\s+)?(?:owner|assignee)\b", re.I), "assigned"),
    (re.compile(r"\bmade\s+an?\s+(?!(?:change|edit|update|tweak|fix|correction|adjustment|start|decision)s?\b)\w",
                re.I), "created"),
)


def _said_as(verb: str) -> str:
    """A family verb as the pattern reads it: a phrasal one's first word before its particle."""
    word, _, particle = verb.partition(" ")
    word = re.escape(word).replace("'", "['’]")
    return rf"{word}(?=\s+{particle}\b)" if particle else rf"{word}\b"


# P256-FIX-RVW-25: the families' verbs, for the shapes that read no other ("I sent the email."); a
# "<Noun> <participle>." has no auxiliary ("Rosa has ordered." is hers, "is paused" a state).
_AUXILIARY = r"(?:is|are|was|were|be|been|being|has|have|had)\b"
_DID = "(?:" + "|".join(sorted({_said_as(verb) for verbs, _ in FAMILIES for verb in verbs} | {_said_as(_MADE)},
                               reverse=True)) + ")"
# The ONE completed-action pattern: the shapes above and, since RVW-25, a stative passive ("The post is
# published."), "got/went/gone" ("The post got sent.", "It went out."), the first-person simple past
# ("I sent the email to Declan.") and a sentence that is only "<Noun> <participle>." ("Email sent.",
# "Posted!", not "Not sent yet.").
COMPLETED_ACTION = re.compile(
    rf"{_SHAPES}"
    rf"|(?P<state>\b(?:is|are|(?:it|that|this|everything|they)['’](?:s|re))\s+{_ADVERBS}{_OUTCOMES})"
    rf"|(?P<said>\b(?:got|went|gone)\s+{_ADVERBS}(?:{_OUTCOMES}|out\b|live\b))"
    rf"|(?P<mine>\bI\s+{_FIRST_ADVERBS}{_DID})"
    rf"|(?P<bare>^\s*(?!.*(?:\b(?:not|never|nothing|no|none|yet)\b|n['’]t\b))"
    rf"(?:(?!{_AUXILIARY})[\w#'’-]+\s+){{0,3}}(?:{_DID}|done\b)"
    rf"(?=(?:\s+(?:up|off|on))?\s*[.!]+\s*$))", re.I)


def _family(verb: str, sentence: str) -> Optional[Backs]:
    """What backs a claim of ``verb``: "sent … back" is a card sent back, not a send; "moved to
    <column>" is the move to that column."""
    if verb == "sent" and _SENT_BACK.search(sentence):
        return _any(_BACK, _moved)
    where = _MOVED_TO.search(sentence) if verb == "moved" else None
    if where:
        return _WHERE[where.group(1).lower()]
    if verb == _MADE:
        said = next((family for pattern, family in _MADE_AS if pattern.search(sentence)), None)
        return _of(said) if said else None
    return _of(verb)


def _of(verb: str) -> Optional[Backs]:
    """The family of ``verb``: what done write backs it, None when it is in none."""
    return next((backs for verbs, backs in FAMILIES if verb in verbs), None)


def _verb(sentence: str, said: str, end: int) -> str:
    """The claim's verb ("" for a shape with none), with its particle when a family names the
    pair ("I've set up the report" is "set up"; "kicked off" stays "kicked")."""
    found = _ON_ITS_VERB.search(said)
    if not found:
        return "sent" if _GONE_OUT.search(said) else ""
    verb = found.group(0).lower().replace("’", "'")
    particle = _PARTICLE.match(sentence, end)
    phrasal = f"{verb} {particle.group(1).lower()}" if particle else ""
    return phrasal if phrasal in _PHRASAL else verb


def _clause(sentence: str, start: int, end: int) -> str:
    """The clause of the sentence that holds sentence[start:end] (between ":", ";" or a dash)."""
    breaks = [found.end() for found in _CLAUSE_BREAK.finditer(sentence, 0, start)]
    after = _CLAUSE_BREAK.search(sentence, end)
    return sentence[(breaks[-1] if breaks else 0): (after.start() if after else len(sentence))]


def _in_reply_content(sentence: str, start: int, end: int) -> bool:
    """Whether the clause holding sentence[start:end] is the reply's own content ("below", "here's")."""
    return bool(_REPLY_CONTENT.search(_clause(sentence, start, end)))


def _asked(verb: str, sentence: str) -> bool:
    """Whether a claim of ``verb`` reports the ask for the owner's click (P256-FIX-RVW-30): a card
    raised, an approval requested, an approval card sent; never a card sent back."""
    if verb not in _ASKED or (verb == "sent" and _SENT_BACK.search(sentence)):
        return False
    found = re.search(rf"\b{verb}\b", sentence, re.I)
    return bool(found and _THE_ASK.search(_clause(sentence, found.start(), found.end())))


def _reported(sentence: str, start: int, end: int) -> bool:
    """Whether the claim shape at sentence[start:end] reports work: not the reply's own content,
    not the owner heard nor a read begun."""
    return not (_in_reply_content(sentence, start, end) or _NO_WORK_AFTER.match(sentence, end))


def _coordinated(sentence: str, start: int, end: int) -> List[str]:
    """The participles coordinated with a claim, up to the next claim ("…and sent it to Declan")."""
    return [_verb(sentence, found.group(found.lastindex), found.end())
            for found in _AND_THEN.finditer(sentence, start, end)
            if _reported(sentence, found.start(found.lastindex), found.end())]


def _history_or_denial(sentence: str, match: re.Match) -> bool:
    """Whether a simple past is history ("at 03:04", "this morning"); a stative passive or "went out"
    says what a read found ("I found that the boxes are posted"); or a simple past or a stative passive
    follows a denial ("Nothing was", "No post is") or a relative pronoun ("that were identified")."""
    before = sentence[: match.start()]
    past = any(match.group(shape) for shape in ("past", "said", "mine", "bare"))
    if past and _HISTORY.search(sentence):
        return True
    if (match.group("state") or match.group("said")) and _READ_SAYS.search(before):
        return True
    return bool((past or match.group("state")) and _DENIED.search(before))


def _found_by_a_read(sentence: str, match: re.Match, read: FrozenSet[str]) -> bool:
    """Whether a has-been or was claim names a number a done read of the turn named (P256-FIX-RVW-34):
    "Mission 0365 has been cancelled." after a read of #0365 is what the read found. "I've cancelled
    0365" is the writer's own work."""
    if not read or _FIRST_PERSON.match(match.group(0)) or not _REPORTED.search(match.group(0)):
        return False
    return any(re.search(rf"(?<!\d){re.escape(number)}(?!\d)", sentence) for number in read)


def _counts(sentence: str, match: re.Match, first_person: bool, read: FrozenSet[str]) -> bool:
    """Whether a claim shape reports this turn's work: no plan word before it, not the reply's own
    content, nor a simple past's history, a read's finding or a denial, nor a question or what a read
    of the turn found (RVW-34); with ``first_person``, only the writer's own "I've <verb>"
    (P256-FIX-RVW-7) or "I <verb>" (RVW-25)."""
    if first_person and not _FIRST_PERSON.match(match.group(0)):
        return False
    if _history_or_denial(sentence, match) or _QUESTION.match(sentence) or _ASKS.match(sentence, match.end()):
        return False
    if _found_by_a_read(sentence, match, read):
        return False
    return not _PLANNED.search(sentence[: match.start()]) and _reported(sentence, match.start(), match.end())


def _claims_in(sentence: str, first_person: bool, read: FrozenSet[str]) -> List[str]:
    """The verbs of the sentence's reports of work done ("" for a shape with no verb)."""
    if _PAST.search(sentence):
        return []
    matches = list(COMPLETED_ACTION.finditer(sentence))
    ends = [found.start() for found in matches[1:]] + [len(sentence)]
    verbs = []
    for match, upto in zip(matches, ends):
        if not _counts(sentence, match, first_person, read):
            continue
        verbs.append(_verb(sentence, match.group(0), match.end()))
        verbs.extend(_coordinated(sentence, match.end(), upto))
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


def claims(answer: str, first_person: bool = False, read: FrozenSet[str] = frozenset()) -> List[Tuple[str, str]]:
    """Each report of work done in the writer's own words: (its verb, its sentence); with
    ``first_person``, only its "I've <verb>" claims; ``read``: the numbers the turn's reads named."""
    text = _MARKS.sub("", _own_words(answer or ""))
    return [(verb, sentence) for sentence in _SENTENCES.findall(text)
            for verb in _claims_in(sentence, first_person, read)]


def numbers_read(reads: Sequence[Receipt]) -> FrozenSet[str]:
    """The numbers the done reads' subjects name ("#0365" is "0365"), P256-FIX-RVW-34."""
    return frozenset(number for r in reads for number in _CARD_NUMBER.findall(str(r.get("subject") or "")))


def claims_work_done(answer: str) -> bool:
    """Whether the answer reports, in the first person or as an outcome, work as done."""
    return bool(claims(answer))


def unbacked_claims(answer: str, done_writes: Sequence[Receipt], first_person: bool = False,
                    waiting: bool = False, reads: Sequence[Receipt] = ()) -> List[Tuple[str, bool]]:
    """The answer's claims no done write backs: (the verb, whether its family is known). A
    claim of a known family needs a done write of that family (a Composio action's by the
    words of its slug); any other needs any done write. With ``waiting`` (the turn raised a card
    for the owner's click), a claim that reports that ask is backed by it (P256-FIX-RVW-30); a
    has-been or was claim about a number one of the turn's done ``reads`` names is none (RVW-34)."""
    unbacked = []
    for verb, sentence in claims(answer, first_person, numbers_read(reads)):
        if waiting and _asked(verb, sentence):
            continue
        backs = _family(verb, sentence)
        backed = any(backs(r) for r in done_writes) if backs else bool(done_writes)
        if not backed:
            unbacked.append((verb, backs is not None))
    return unbacked


def _listed(verbs: Sequence[str]) -> str:
    return verbs[0] if len(verbs) == 1 else f"{', '.join(verbs[:-1])} or {verbs[-1]}"


def not_done_line(answer: str, done_writes: Sequence[Receipt], waiting: bool = False,
                  reads: Sequence[Receipt] = ()) -> Optional[str]:
    """The one not-done line for the answer, or None when every claim is backed (``waiting``: by
    the card the turn raised, too; ``reads``: the turn's done reads). It names what was not done
    when another write went through; else it is the plain not-done line."""
    unbacked = unbacked_claims(answer, done_writes, waiting=waiting, reads=reads)
    if not unbacked:
        return None
    named = list(dict.fromkeys(verb for verb, known in unbacked if known))
    if not done_writes or not named:
        return NOTHING_DONE
    return NOT_DONE.format(said=NAMED_SAID.format(verbs=_listed(named)))


def is_not_done_line(line: str) -> bool:
    """Whether a line above the answer is the not-done line (plain or naming its claim)."""
    return line.startswith(NOT_DONE_PREFIX)


__all__ = ["COMPLETED_ACTION", "FAMILIES", "NOT_DONE_PREFIX", "SAYS_DONE", "claims", "claims_work_done",
           "is_not_done_line", "not_done_line", "numbers_read", "unbacked_claims"]
