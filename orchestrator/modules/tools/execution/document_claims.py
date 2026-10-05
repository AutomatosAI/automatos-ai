"""F351 / F337a (nights 10 and 10b): a document Auto said it made, and no call made it.

Night 10b, each with no generate_document that succeeded behind it:

- "I've generated the letter to Maya Osei … and saved it to Deliverables. You can download the
  PDF here." (chat 8ac5cf3a): generate_document had answered DATA_BAD_JSON.
- "I've generated the Quay Coffee House letter for you in PDF format … The download link will be
  available in your Deliverables shortly." (b6db1b8a), "It's on your Harbourline Letter template
  … It's ready to go." (433eaf26), "I've drafted the wholesale supply agreement … as a Word
  document" (09a2afb5), then "the agreement has been drafted".
- "I've tried to generate the letter again" (f367a9d6, turn 2) with no call at all.

Night 10: "you can find … .docx in your Deliverables" (none there), "I've set up an invoice
template" (a placeholder page).

No family in ``action_claims`` read a document as made, so none of these was nudged or
corrected. Now, in a chat turn (Auto's own words; a draft it quotes is left out), and never
for a question or an offer:

- a document said to be made ("I've generated / created / exported the letter, the PDF, the
  invoice…"; "drafted", "written" or "prepared" only beside a file word: a PDF, a Word document,
  a template, Deliverables; "the agreement has been drafted"; "it's on your … template") needs a
  call that makes one (``DOCUMENT_MAKES``, or a social post's render for a flyer or a card) to have
  succeeded this turn. "I've drafted that letter for you" alone is left alone: Auto writes letters
  into its reply.
- a document said to be in Deliverables ("saved it to Deliverables", "it's available in your
  Deliverables", "you can download the PDF here", "here's the link to your document") needs one
  made, or Deliverables, documents or the board's cards read, this turn: a card's answer may say
  where its agent saved the file (night 10, chat f4b2ae7e).
- a document said to be on its way ("will be available in your Deliverables shortly") needs one
  made, or work started that outlives the turn (a card, a run).
- "I've tried to generate it again" needs a document call this turn, succeeded or refused.
- a template said to be made needs a call that writes a template; none of Auto's does yet.

"I used the create blueprint tool" (night 10, chat db3bef60) is not checked here: it told of an
earlier turn, and "I used the query data tool to count the subscribers" (night 9b, chat 2fe5384a)
has the same shape and was true. Only the calls of this turn are known to the check.

When the call that makes the document was refused this turn (``call_effects.MAKE_REFUSED``), the
claim is told as that: the owner reads that Auto tried and it didn't go through.
"""
from __future__ import annotations

import functools
import re
from typing import Callable, List, Optional

from .call_effects import DOCUMENT_MAKES, MAKE_REFUSED

DOCUMENT_MADE = "made into a document"
DOCUMENT_REFUSED = "made into a document when the call to make it failed"
DOCUMENT_THERE = "put in your Deliverables"
DOCUMENT_COMING = "on its way to your Deliverables"
TRIED_AGAIN = "tried again"
TEMPLATE_MADE = "set up as a template"
LABELS = (DOCUMENT_MADE, DOCUMENT_REFUSED, DOCUMENT_THERE, DOCUMENT_COMING, TRIED_AGAIN, TEMPLATE_MADE)
# A refused make turns these into DOCUMENT_REFUSED: the owner hears it was tried.
_REFUSABLE = (DOCUMENT_MADE, DOCUMENT_THERE, DOCUMENT_COMING)

# What a document can be called: a file's kind, or the business paper it is.
_DOC_NOUN = (r"(?:pdfs?|word (?:documents?|files?|docs?)|docx|xlsx|csv|excel (?:files?|spreadsheets?|sheets?)|"
             r"spreadsheets?|documents?|files?|letters?|invoices?|quotes?|quotations?|proposals?|agreements?|"
             r"contracts?|price ?lists?|data ?sheets?|one-pagers?|flyers?|brochures?|receipts?)")
# The noun is the thing made, not what a card or a step is about: "the invoice task", "a
# letter-writing agent" made no document.
_NOT_ITS_KIND = (r"(?![-’'])(?!\s+(?:tasks?|cards?|tickets?|missions?|playbooks?|steps?|reminders?|emails?|templates?|"
                 r"agents?|watch(?:es)?|schedules?|runs?)\b)")
_MADE_THING = _DOC_NOUN + r"\b" + _NOT_ITS_KIND
# "I've tried to generate the letter again": a try at a document, not at a card or a mission.
_ABOUT_A_DOCUMENT = r"(?=[^\n]*\b(?:" + _DOC_NOUN + r"|generat\w*)\b)"
# A file word beside a softer verb: "drafted the agreement … as a Word document".
_FILE_CUE = r"(?=[^\n]*\b(?:pdf|word document|word file|docx|xlsx|spreadsheet|template|deliverables|download)\b)"
# What the made thing is, before its noun: "the Quay Coffee House letter", "that PDF". A word that
# starts another clause ("a task for the Analyst to draft the invoice") ends it.
_ARTICLE = r"(?:(?:the|a|an|your|that|this|these|those|its|two|three)\s+)?"
_WORDS = (r"(?:(?!(?:for|to|of|with|about|from|into|in|on|at|by|and|or|that|which|who|whose|where|as|using|so)\b)"
          r"[^\s.!?,;:]+\s+){0,5}?")
_OBJECT = _ARTICLE + _WORDS
_TRIED_AGAIN = (r"(?:tried|attempted)\b[^.!?\n]{0,40}?\b(?:generat|creat|mak|render|export|produc|draft|writ|redo|"
                r"run)\w*\b[^.!?\n]{0,60}?\b(?:again|once more|one more time)\b")
_MADE_VERBS = r"(?:generated|regenerated|created|recreated|produced|rendered|re-rendered|exported|built|designed)"
_SOFT_VERBS = r"(?:drafted|written|prepared|put together|made|finished|completed)"
_BEEN_MADE = r"(?:generated|created|drafted|saved|exported|rendered|produced|made|prepared|written)"
_DELIVERABLES = r"(?:your\s+|the\s+)?deliverables\b"
_FILE_FORM = r"(?:pdf|word(?: document| file| doc)?|docx|xlsx|csv|excel(?: file| spreadsheet)?|spreadsheet|file)"

# Not a claim: a question, an offer or a proposal ("I can tell you…" offers nothing); a document
# said to be somewhere else, or there only once something else is done ("Once it's made, you can
# download it"), or only meant ("I'll put it in your Deliverables"); a doing made the condition of
# another ("it'll be in your Deliverables once the Analyst finishes").
_ASKS = (r"\?|\b(?:would you like|do you want|shall i|should i|want me to|if you(?:'d|’d| would)? like)\b"
         r"|\bi (?:can|could)\b(?!\s+(?:tell|say|see|confirm)\b)|\blet(?:'s|’s| us)\b")
_OFFER = re.compile(_ASKS, re.I)
_ELSEWHERE = re.compile(_ASKS + r"|\b(?:task|ticket|card|mission|playbook|board|agent)s?\b|#\d{3,6}"
                        r"|^\W*(?:once|when|after|as soon as)\b"
                        r"|\b(?:i(?:'ll|’ll| will)|i(?:'m|’m| am) going to|let me)\b", re.I)
_LATER = re.compile(_ASKS + r"|\b(?:once|when|after|as soon as|if|until)\b", re.I)

# What backs each: a document made (a flyer or a card may be a social post's render), Deliverables,
# documents or cards read, work started that outlives the turn, a template written.
_MADE_BY = DOCUMENT_MAKES + ("social_post",)
_DOCUMENT_READS = ("deliverable", "list_documents", "get_document", "read_document", "search_documents",
                   "workspace_list", "workspace_read", "read_file", "list_directory", "list_tasks", "get_task",
                   "board_")
_MAKES_LATER = ("create_task", "assign_task", "update_task", "execute_", "run_", "start_", "trigger", "schedule_",
                "create_mission", "approve_", "resume_")
# No tool of Auto's writes a template today (night 10: "set it up as your default invoice template"
# was a feature that doesn't exist), so a template said to be made is corrected until one does.
_TEMPLATE_WRITES = ("create_template", "save_template", "update_template")

# Where a document is said to be: kept there, there, to be found there, linked, or on its way.
_KEPT_THERE = (r"\b(?:saved|stored|filed|put|placed|added|uploaded|dropped)\b(?:\s+(?:it|them|this|that|a copy))?"
               r"\s+(?:to|in|into|under)\s+" + _DELIVERABLES)
_IS_THERE = (r"(?:\bis|\bare|['’]s)\s+(?:now\s+)?(?:(?:available|saved|waiting|ready|sitting|stored)\s+)?"
             r"(?:as\s+an?\s+[\w-]+(?:\s+[\w-]+)?\s+)?(?:in|under)\s+" + _DELIVERABLES)
_FIND_IT_THERE = (r"\byou(?:'ll|’ll| will| can| should| may)\s+(?:now\s+|also\s+)?(?:find|see|open|view|download|"
                  r"access|get|grab)(?:\s+(?:and|or)\s+(?:download|open|view|print|share))?\s+(?:it|them|"
                  r"(?:the|your)\s+(?:[^\s.!?,;:]+\s+){0,4}?" + _DOC_NOUN + r")\b"
                  r"(?=[^\n]*\b(?:deliverables|here|below|link)\b)")
_HERE_IS_THE_LINK = (r"\bhere(?:'s|’s| is| are)\s+(?:the|your)\s+(?:download\s+)?links?\b"
                     r"(?=[^\n]*\b(?:deliverables|" + _DOC_NOUN + r")\b)")
_COMING_THERE = (r"\b(?:will|['’]ll|should)\s+(?:be\s+|appear\b|show\s+up\b|land\b|arrive\b|turn\s+up\b)"
                 r"[^.!?\n]{0,50}?\b" + _DELIVERABLES)
# A link is not a sentence: its "?" asks nothing, and its "deliverables" says nothing.
_LINK = re.compile(r"https?://\S+")
_A_LINK = "<link>"


def _made(family, i_have: str) -> tuple:
    """The families of a document said to be made (all backed by a make)."""
    return (
        family(DOCUMENT_MADE, re.compile(i_have + _MADE_VERBS + r"\s+" + _OBJECT + _MADE_THING, re.I), _MADE_BY),
        family(DOCUMENT_MADE, re.compile(i_have + _SOFT_VERBS + r"\s+" + _OBJECT + _MADE_THING + _FILE_CUE,
                                         re.I), _MADE_BY),
        family(DOCUMENT_MADE, re.compile(i_have + r"(?:exported|saved|converted|turned)\b[^.!?\n]{0,60}?\b"
                                         r"(?:as|to|into)\s+(?:an?\s+)?" + _FILE_FORM + r"\b", re.I), DOCUMENT_MAKES),
        family(DOCUMENT_MADE, re.compile(r"\b" + _DOC_NOUN + r"\b[^.!?\n]{0,60}?\b(?:has|have)\s+(?:(?:now|just|also|"
                                         r"already)\s+)*been\s+(?:successfully\s+)?" + _BEEN_MADE + r"\b", re.I),
               _MADE_BY, unless=_ELSEWHERE),
        family(DOCUMENT_MADE, re.compile(r"\bit(?:'s|’s| is)\s+(?:now\s+)?(?:on|using|built on|laid out on)\s+"
                                         r"(?:your|the)\s+(?:[^\s.!?,;:]+\s+){0,4}?templates?\b", re.I),
               _MADE_BY),
    )


def _there(family) -> tuple:
    """The families of a document said to be in Deliverables, or on its way there."""
    found = DOCUMENT_MAKES + _DOCUMENT_READS
    return (
        family(DOCUMENT_THERE, re.compile(_KEPT_THERE, re.I), found, unless=_ELSEWHERE),
        family(DOCUMENT_THERE, re.compile(_IS_THERE, re.I), found, unless=_ELSEWHERE),
        family(DOCUMENT_THERE, re.compile(_FIND_IT_THERE, re.I), found, unless=_ELSEWHERE),
        family(DOCUMENT_THERE, re.compile(_HERE_IS_THE_LINK, re.I), found, unless=_ELSEWHERE),
        family(DOCUMENT_COMING, re.compile(_COMING_THERE, re.I), DOCUMENT_MAKES + _MAKES_LATER, unless=_LATER),
    )


@functools.lru_cache(maxsize=1)
def _families() -> tuple:
    """Every family above, as ``action_claims`` reads them (built on first use: that module
    imports this one)."""
    from .action_claims import _I_HAVE, _Family

    def family(label, claim, backing, unless=_OFFER):
        return _Family(label, claim, tuple(backing), unless=unless)

    tried = family(TRIED_AGAIN, re.compile(_ABOUT_A_DOCUMENT + _I_HAVE + _TRIED_AGAIN, re.I),
                   DOCUMENT_MAKES + (MAKE_REFUSED,))
    template = family(TEMPLATE_MADE, re.compile(_I_HAVE + r"(?:created|made|built|designed|set up|added)\s+" + _ARTICLE
                                                + r"(?:new\s+)?" + _WORDS + r"templates?\b", re.I), _TEMPLATE_WRITES)
    return _made(family, _I_HAVE) + (tried, template) + _there(family)


def unbacked_document_claim(text: str, succeeded: List[str]) -> Optional[str]:
    """The label of the first document claim in Auto's own words ``text`` that no call this turn
    backs, else None. ``succeeded``: the turn's actions, lowercased."""
    from .action_claims import _MID_NUMBER, _IN_A_NUMBER, _first_unbacked, _own_words

    said = _own_words(_MID_NUMBER.sub(_IN_A_NUMBER, _LINK.sub(_A_LINK, text or "")))
    found = _first_unbacked(said, _families(), succeeded)
    if found in _REFUSABLE and MAKE_REFUSED in succeeded:
        return DOCUMENT_REFUSED
    return found


Check = Callable[..., Optional[str]]


def also_checks_documents(check: Check) -> Check:
    """Wrap ``action_claims.claimed_action_not_done``: when it finds nothing, a chat turn's reply
    is checked for a document or a template that no call backs (see the module)."""
    @functools.wraps(check)
    def wrapped(text: str, done: Optional[set] = None, *, promises: Optional[bool] = None) -> Optional[str]:
        found = check(text, done, promises=promises)
        if found or not _autos_words(promises):
            return found
        return unbacked_document_claim(text or "", [a.lower() for a in (done or ())])
    return wrapped


def _autos_words(promises: Optional[bool]) -> bool:
    if promises is not None:
        return promises
    from .action_claims import _auto_speaks

    return _auto_speaks()


__all__ = ["DOCUMENT_COMING", "DOCUMENT_MADE", "DOCUMENT_REFUSED", "DOCUMENT_THERE", "LABELS", "TEMPLATE_MADE",
           "TRIED_AGAIN", "also_checks_documents", "unbacked_document_claim"]
