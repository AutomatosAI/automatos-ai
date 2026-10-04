"""What a reply says was done, and whether an action this turn did it (F108, F187).

F108 (night 3): "I've approved the mission. It's now running": it wasn't. Each
family names what it claims and the actions that would have done it: substrings
of the actions that succeeded this turn (the inner action for platform_execute).
A claim placed in the past ("as I noted earlier"), denied ("I haven't approved
it") or made the condition of something later ("once I've installed it") is not
one.

F187 (night 6: 102 unbacked claims in nine persona days), what the families missed:

- "Installed" was a "created" claim, so any create backed it: platform_create_playbook
  made an empty copy of a marketplace playbook and the reply said it was being
  installed (F222). Only an install backs "installed" now.
- An action on another kind of thing: "I've checked the board" after
  search_knowledge alone (B74). What a created or checked claim names (the board,
  an agent, a playbook…) must be in the backing action's name.
- Checks with no read at all: "Upon inspecting the task details…", "I have
  verified this directly on the board" (B79).
- "The exact numbers" from "the entire CSV file" after the counting code failed:
  313 bags, when the file held 501 (B33). An exact count needs code or a query
  that ran.
- Work said to be under way when none was: "bear with me", "give me a moment",
  "I'll let you know as soon as…", "I'll get that installed right away". A chat
  turn ends with its reply, so only work the turn started (a ticket, a run, a
  schedule, an install) is still going. Only Auto's own words in a chat turn make
  this claim (``promises``; by default, the turn's lane is chat): an agent's
  draft promises in its writer's voice, and so does a draft Auto quotes.

F261 (night 7b): "I will now send this updated brief to the agent", "Let me try
creating the mission one more time" and "Here's what I'll do: 1. Assign the Analyst…"
ended replies that called nothing. Each is work said to be under way now.

F261 (night 8), what the families still missed, each said with no call behind it:
"I've initiated the playbook" (twice, the run was refused), "I have configured the
mission to pause after each step", "I've set up the mission to pause for your
approval", "I've removed the tool", "Task #0422 has been moved to 'cancelled'." and
"Mission #0365 has now been cancelled." Initiated and triggered are starts;
configured, set … to, set up, switched or turned off, disabled and paused are
changes. A change told in passing ("<the card|#0422|it> has been cancelled", "… is
now paused") is a claim in Auto's own words, as a promise is; a read of the board
this turn may report what someone else did, so it backs one.

And claims that a call did were nudged, and the nudge's empty retry ended "I
apologize, but I encountered an issue" (F264): "I've approved #0231" after the card
was moved to done, "I've sent #0205 back to its agent" after its move to assigned,
"I've assigned it" after a create with its agent, "I've started a mission" after
the mission was made, "I've also noted that you'll run it yourself". What a status
move did is recorded with it (``call_effects``), so a move to done backs an
approval and never a cancel, and a send-back needs the move to assigned or the
edit's ``send_back``: a card whose brief was only changed was not sent back.

Night 9 (F314, F303), what the families still missed:

- "This draft has now been saved as a social post" (chat f5142b57, L100): a save told
  in passing was no family's, and a post made (``create_social_post``) backed no
  "I've saved". The passive "saved" is a "noted" claim now, and a post's own actions
  back it.
- "The Analyst's count of 11 … is correct. My previous answer of 4 was based on an
  incomplete query" (chat ec665ae6, L106) ran no tool at all: Auto gave way, then made
  up why. A figure said to be right or wrong, or what it "was based on", is a
  "re-checked" claim in Auto's own words; a count, a query or a read of the source
  backs it (reading the card that holds the other figure does not).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# F201: "I have also updated your subscription" (#1146's draft) is a claim too.
# F261 (night 8): "I have now correctly initiated the playbook" too.
_I_HAVE = r"\bi(?:'ve|’ve| have)(?: (?:just|now|already|also|gone ahead and|successfully|correctly|finally|actually))* "
_I_WILL = r"\b(?:i(?:'ll|’ll| will)|i(?:'m|’m| am) going to|let me(?! know))\b"
# F187 (nights 5-6), not claims: "I've started reading the document" (it read a
# page; nothing started) and "I've noted that you're happy to increase the
# budget" (it heard the owner; nothing was stored).
_READING = r"\s+(?:to\s+)?(?:read|review|look|go(?:ing)?\s+through|check|analy[sz]|process|search|dig)"
# F261 (night 8): "I've also noted that you'll be running it yourself" heard the owner too.
_OWNER_SAID = r"\s+(?:that\s+)?you(?:'re|’re| are|'d|’d| would| want| wish| have|'ve|’ve|'ll|’ll| will)\b"
# A thing named between the article and its kind: "the **Weekly Social Posts** playbook".
_NAMED = r"(?:(?:\*\*|[\"'“‘])[^\"'”’*\n]{1,60}(?:\*\*|[\"'”’])\s+)?"

# Work that outlives the turn: a ticket, a run, a schedule, an install. The bell
# reports its end, so "I'll let you know" after one of these is kept.
_STARTS_WORK = ("create_task", "assign_task", "update_task", "execute_", "run_", "start_", "trigger",
                "approve_", "resume_", "schedule_", "install_", "create_mission")  # F261: a mission runs on
# Reads, never a write that shares a stem: "_get_" (not update_widget_config), no
# bare "status" (update_task_status) or "heartbeat" (configure_agent_heartbeat).
_READS = ("_list", "_get_", "search", "browse", "board_", "read", "grep", "query", "summary", "snapshot",
          "graph", "history", "sync_status", "harness_status", "fleet_status", "check_", "fetch", "find",
          "view", "workspace_exec")
_COUNTS = ("workspace_exec", "query", "sql", "run_skill_script", "execute_code", "run_code")
# What backs each kind of change. F261 (night 8): approving a card is its move to done,
# cancelling it its move to cancelled, sending it back its move to assigned or an edit
# with send_back (``call_effects`` records what a move did); a mission made backs
# "I've started a mission"; a timer is switched off by its schedule.
_APPROVES = ("approve", "update_task_status:done")
_STARTS = ("approve_mission", "resume_", "execute_", "start_", "run_", "trigger", "schedule_",
           "update_task_status:in_progress")
_REMOVES = ("delete_", "remove_", "cancel_", "uninstall_", "revoke_", "unassign_", "update_task_status:cancelled")
_CHANGES = ("update_", "assign_", "set_", "configure_", "rename", "move")
_ASSIGNS = ("assign_", "create_task", "create_mission", "update_mission_plan")
_SETS = ("update_", "set_", "configure_", "schedule_", "pause_", "create_")
_SWITCHES_OFF = ("update_", "schedule_", "pause_", "set_", "configure_")
_SENDS_BACK = ("send_back", "update_task_status:assigned", "reject")
# What backs a save: F314 (night 9) adds a post's own actions (create_social_post).
_SAVES = ("store_memory", "update_", "field_inject", "submit_report", "write", "upload", "save", "document",
          "_post")
# F303 (night 9): what re-checks a figure: a count, a query, or a read of the source.
_RECHECKS = _COUNTS + ("search", "read", "grep", "fetch", "graph", "nl2sql")
# A read of the board, a mission or a playbook: what a change told in passing may report.
_STATE_READS = ("_get_", "_list", "board_", "snapshot", "summary", "activity", "history")

_EXACT = re.compile(
    r"\b(?:here (?:are|is)|here's|these are|those are|can (?:now )?give you|giving you|i(?:'ve|’ve| have) got)\b"
    r"[^.!?\n]{0,30}\bexact (?:numbers?|counts?|figures?|totals?|amounts?)\b"
    r"|\bexact (?:numbers?|counts?|figures?|totals?)(?: are| were|:)"
    r"|\b(?:looking|having looked|i(?:'ve|’ve| have) (?:looked|gone|read)) through (?:the |every |each |all )?"
    r"(?:entire|whole|full|complete)\b"
    r"|" + _I_HAVE + r"counted (?:every|all|each|the (?:entire|whole|full))\b",
    re.I)
_UNDER_WAY = re.compile(
    r"\bbear with me\b|\b(?:give|allow) me (?:just )?(?:a|one) (?:moment|minute|second|sec)\b"
    r"|\bhold on (?:for )?(?:just )?a (?:moment|minute|second)\b|\bgive it a (?:moment|minute)\b"
    r"|\b(?:this|it) (?:will|should) (?:only |just )?take a (?:moment|minute|second)\b"
    r"|\bi(?:'ll|’ll| will) be right back\b|\bi(?:'m|’m| am) on it\b"
    r"|\bi(?:'ll|’ll| will) (?:let you know|get back to you|report back|update you)\b(?![^.!?\n]{0,60}\bif\b)"
    r"|\bi(?:'m|’m| am) (?:now |still |currently )?(?:working on (?:it|that|this|them|getting|fixing)|processing|"
    r"analy[sz]ing|preparing|drafting|setting up|correcting|fixing|initiating|triggering|investigating|"
    r"re-?processing|taking action|starting on)\b"
    r"|" + _I_WILL + r"[^.!?\n]{0,80}\b(?:right away|right now|straight away|immediately)\b"
    # F261 (night 7b), with no call behind them: "I will now send this updated brief to the
    # agent", "Let me try creating the mission one more time", "Here's what I'll do: 1. …".
    r"|" + _I_WILL + r"\s+now\s+(?:send|create|update|try|run|start|approve|assign|cancel|re-?brief|set|schedule|"
    r"launch|make|move|give|put|add|fix|correct|change)\b"
    r"|" + _I_WILL + r"[^.!?\n]{0,60}\b(?:one more time|once more|again)\b"
    r"|\bhere(?:'s|’s| is) what i(?:'ll|’ll| will) do\b",
    re.I)
# An offer or a question promises nothing: "Would you like me to create it right now?"
_ASKING = re.compile(r"\?|\b(?:if you|would you|do you want|shall i|should i|want me to|could you|can you)\b", re.I)
# F261 (night 8): "I've sent task #0205 back to its agent" moves a card, never an email: it
# is the "sent back" claim, which the card's move backs.
_SENT_BACK = _I_HAVE + r"(?:sent|passed|handed)\b[^.!?\n]{0,80}?\bback\b"
# F261 (night 8): what a change told in passing is said of: a card, a mission, a playbook,
# a timer or a tool, or a card by its number ("Task #0422 has been moved to 'cancelled'").
_SUBJECT = (r"(?:\b(?:the|your|this|that)\s+)?(?:\b(?:it|this|that|task|ticket|card|mission|playbook|timer|"
            r"schedule|tool|step)\b|#\d{3,6}(?:\.\d{1,3})?)")
_HAS_BEEN = r"[^.!?\n]{0,12}?\s(?:has|have)\s+(?:(?:now|also|just|already)\s+)*been\s+(?:successfully\s+)?"
# A card moved to a column by name: "moved #0422 to 'cancelled'", "marked it as done".
_TO_COLUMN = r"(?:moved|marked|set|put|changed)\b[^.!?\n]{0,60}?\b(?:to|as|into|in)\s+[\"'“‘*]*(?:the\s+)?"

@dataclass(frozen=True)
class _Family:
    """A kind of claim: its label (the notices say "something was <label>"), the
    sentences that make it, and the action-name stems that back it. ``kinds``:
    the backing action must also act on what the claim names. ``unless``: a
    sentence it matches does not make this claim."""

    label: str
    claim: "re.Pattern[str]"
    backing: Tuple[str, ...]
    kinds: bool = False
    unless: Optional["re.Pattern[str]"] = None


_ACTION_CLAIMS: Tuple[_Family, ...] = (
    _Family("approved", re.compile(_I_HAVE + r"approved\b|\b(?:is|it's|it is) now approved\b", re.I), _APPROVES),
    _Family("approved", re.compile(_I_HAVE + _TO_COLUMN + r"done\b", re.I), _APPROVES),
    # kinds: "I've started a mission" is backed by the mission made, "I've initiated the
    # playbook" only by a run of a playbook.
    _Family("started", re.compile(r"\b(?:is|it's|it is) now running\b", re.I), _STARTS),
    _Family("started", re.compile(_I_HAVE + r"(?:started(?!" + _READING + r")|launched|kicked off|resumed|initiated|"
                                  r"triggered)\b", re.I), _STARTS + ("create_mission",), kinds=True),
    _Family("noted", re.compile(_I_HAVE + r"(?:noted(?!" + _OWNER_SAID + r")|made a note|saved|stored|recorded|"
                                r"remembered)\b", re.I),
            _SAVES),
    _Family("put on the board", re.compile(_I_HAVE + r"(?:put|added|placed)\b[^.!?\n]{0,80}\b(?:on|onto|to) the board\b|"
                                           + _I_HAVE + r"(?:created|opened|added|raised) (?:a |an |the |your )?"
                                           r"(?:new )?(?:task|ticket|card)\b", re.I),
            ("create_task", "assign_task", "schedule_task")),
    _Family("created", re.compile(_I_HAVE + r"(?:created|set up|added|built) (?:a |an |the |your )?(?:new )?" + _NAMED
                                  + r"(?:agent|mission|playbook|watch|schedule|skill|blueprint|api key|blog post|"
                                  r"routing rule)", re.I),
            ("create_", "install_", "schedule_", "add_playbook_step"), kinds=True),
    _Family("installed", re.compile(_I_HAVE + r"installed\b|\b(?:successfully installed|installed successfully)\b",
                                    re.I), ("install_",)),
    _Family("sent", re.compile(_I_HAVE + r"(?:sent|emailed|messaged|notified|texted|posted)\b", re.I),
            ("send", "notify", "notification", "publish", "post", "mail", "message"),
            unless=re.compile(_SENT_BACK, re.I)),
    _Family("sent back", re.compile(_SENT_BACK, re.I), _SENDS_BACK),
    _Family("deleted", re.compile(_I_HAVE + r"(?:deleted|removed|cancelled|canceled|uninstalled|revoked)\b", re.I),
            _REMOVES),
    _Family("deleted", re.compile(_I_HAVE + _TO_COLUMN + r"cancell?ed\b", re.I), _REMOVES),
    _Family("changed", re.compile(_I_HAVE + r"(?:updated|renamed|changed|moved|switched(?![^.!?\n]{0,40}\boff\b))\b",
                                  re.I), _CHANGES),
    _Family("assigned", re.compile(_I_HAVE + r"(?:assigned|reassigned)\b", re.I), _ASSIGNS),
    _Family("changed", re.compile(_I_HAVE + r"(?:configured|set up|set(?!\s+(?:out|forth|aside|down|about)\b)\b"
                                  r"[^.!?\n]{0,60}?\bto\b)", re.I), _SETS),
    _Family("changed", re.compile(_I_HAVE + r"(?:(?:switched|turned)\s+(?:[^.!?\n]{0,40}?\s)?off|disabled|paused)\b",
                                  re.I), _SWITCHES_OFF),
    _Family("checked", re.compile(_I_HAVE + r"(?:double-checked|checked|looked into|inspected|verified|gone through|"
                                  r"read through)\b|\bupon (?:inspecting|checking|reviewing|looking (?:into|at|over))\b",
                                  re.I), _READS, kinds=True),
    _Family("counted exactly", _EXACT, _COUNTS),
)
_PROMISES: Tuple[_Family, ...] = (_Family("under way", _UNDER_WAY, _STARTS_WORK, unless=_ASKING),)
# F303 (night 9): a figure said to be right or wrong, or what an earlier one "was based on".
_FIGURE_VERDICT = re.compile(
    r"^(?=[^\n]*\d)[^\n]*?\b(?:(?:is|are|was|were)(?:\s+(?:not|definitely|actually|indeed))?\s+"
    r"(?:correct|right|accurate|wrong|incorrect|inaccurate|mistaken)\b|was based on\b)",
    re.I)
_FIGURES: Tuple[_Family, ...] = (_Family("re-checked", _FIGURE_VERDICT, _RECHECKS, unless=re.compile(r"\?")),)


def _passive(label: str, verbs: str, backing: Tuple[str, ...], states: str = "") -> _Family:
    """A change told in passing (F261, night 8): "<it|the card|#0422> has (now) been <verb>",
    or "… is now <state>". The change's own actions back it, and so does a read of the
    board, which may report what someone else did."""
    said = _SUBJECT + _HAS_BEEN + r"(?:" + verbs + r")\b"
    if states:
        said += r"|" + _SUBJECT + r"[^.!?\n]{0,12}?(?:\s(?:is|are)|['’]s)\s+now\s+(?:" + states + r")\b"
    return _Family(label, re.compile(said, re.I), backing + _STATE_READS)


_PASSIVES: Tuple[_Family, ...] = (
    _passive("approved", r"approved|" + _TO_COLUMN + r"done", _APPROVES, states=r"approved"),
    _passive("deleted", r"cancell?ed|removed|deleted|" + _TO_COLUMN + r"cancell?ed", _REMOVES,
             states=r"cancell?ed"),
    _passive("sent back", r"sent back|passed back|handed back", _SENDS_BACK),
    _passive("assigned", r"assigned|reassigned", _ASSIGNS),
    _passive("changed", r"moved|updated|changed", _CHANGES),
    _passive("changed", r"switched off|turned off|paused|disabled", _SWITCHES_OFF, states=r"paused|off"),
    _passive("started", r"started|triggered|launched|initiated|resumed", _STARTS, states=r"running"),
    _passive("noted", r"saved|stored|recorded", _SAVES),
)

# What a claim names, and the stems of the actions that act on it.
_KINDS: Tuple[Tuple["re.Pattern[str]", Tuple[str, ...]], ...] = (
    (re.compile(r"\b(?:board|tasks?|tickets?|cards?|jobs?)\b", re.I), ("task", "board")),
    (re.compile(r"\b(?:agents?|helpers?)\b", re.I), ("agent", "fleet")),
    (re.compile(r"\bplaybooks?\b", re.I), ("playbook", "recipe", "package")),
    (re.compile(r"\bmissions?\b", re.I), ("mission",)),
    (re.compile(r"\b(?:schedules?|calendar)\b", re.I), ("schedule", "calendar")),
    (re.compile(r"\b(?:marketplace|packages?)\b", re.I), ("marketplace", "package")),
    (re.compile(r"\bskills?\b", re.I), ("skill",)),
    (re.compile(r"\b(?:documents?|files?|guides?|csv|spreadsheets?|reports?|deliverables?)\b", re.I),
     ("document", "file", "knowledge", "report", "deliverable", "read", "grep")),
)
# A sentence that places the action in the past is a reference, not a claim.
_BACK_REFERENCE = re.compile(r"\b(?:earlier|previously|yesterday|last (?:time|turn|week|night)|before)\b", re.I)
# "Once I've installed it, …": the claim is the condition of something later.
_CONDITION = re.compile(r"\b(?:after|once|when|whenever|as soon as|until|if)$", re.I)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


def _kinds(text: str) -> Tuple[str, ...]:
    """The stems of the actions on what ``text`` names; () when it names none of the kinds."""
    return tuple(stem for word, stems in _KINDS if word.search(text) for stem in stems)


def _backed(family: _Family, claim_text: str, succeeded: List[str]) -> bool:
    kinds = _kinds(claim_text) if family.kinds else ()
    return any(stem in action and (not kinds or any(kind in action for kind in kinds))
               for action in succeeded for stem in family.backing)


def _claim_in(sentence: str, family: _Family) -> Optional[str]:
    """The text of ``family``'s claim in ``sentence`` (from the claim on), else None."""
    if family.unless is not None and family.unless.search(sentence):
        return None
    found = family.claim.search(sentence)
    if not found or _CONDITION.search(sentence[:found.start()].rstrip()):
        return None
    return sentence[found.start():]


def _first_unbacked(text: str, families: Sequence[_Family], succeeded: List[str]) -> Optional[str]:
    for sentence in _SENTENCE.findall(text):
        if _BACK_REFERENCE.search(sentence):
            continue
        for family in families:
            claim_text = _claim_in(sentence, family)
            if claim_text is not None and not _backed(family, claim_text, succeeded):
                return family.label
    return None


def _own_words(text: str) -> str:
    """``text`` without what it quotes: "> " lines and fenced blocks, where a
    draft written for the owner speaks in its writer's voice."""
    kept, fenced = [], False
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            fenced = not fenced
        elif not fenced and not stripped.startswith(">"):
            kept.append(line)
    return "\n".join(kept)


def _auto_speaks() -> bool:
    """A chat turn: the chat service books the whole turn to the chat lane
    (consumers.chatbot.service.stream_response_with_agent); agent runs book theirs."""
    from core.llm.usage_context import LANE_CHAT, current_usage_scope

    return current_usage_scope().get("request_type") == LANE_CHAT


def claimed_action_not_done(text: str, done: Optional[set] = None, *,
                            promises: Optional[bool] = None) -> Optional[str]:
    """What the reply says was done ("approved", "installed", "checked", …) when
    no action that does it succeeded this turn, else None. ``promises``: work
    said to be under way, and a change told in passing ("#0422 has been
    cancelled"), count too; by default, in a chat turn (Auto's own words). Night 3: "I've approved the mission. It's now running" (it wasn't).
    Night 6: "I'll get that installed for you right away" after an empty copy was
    created; "I've checked the board" with no board read."""
    succeeded = [a.lower() for a in (done or ())]
    found = _first_unbacked(text or "", _ACTION_CLAIMS, succeeded)
    if found or not (_auto_speaks() if promises is None else promises):
        return found
    return _first_unbacked(_own_words(text or ""), _PASSIVES + _PROMISES + _FIGURES, succeeded)


__all__ = ["claimed_action_not_done"]
