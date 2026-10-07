"""F379 (night 11, 7 Oct): a social post Auto says it made or changed, and no post call did it.

Night 11, each with no platform_create_social_post or platform_update_social_post that succeeded
behind it:

- "I've drafted the three social media posts" (chat 508b4e05, B4): none of them exists;
- "I've updated the carousel … and removed any placeholder tasting notes" (post 39ee6ae3, B-i3-3):
  the notes were still there;
- "It's waiting in the Socials tab" (ticket 2160, B-i3-2): the agent had made a PNG, not a post.

No family in ``action_claims`` read a post as drafted, and the ones that did match backed the
claim with the wrong calls: "removed" was a "deleted" claim (a delete or a cancel backs it, never
a post edit), and "I've saved it" was backed by any action with "_post" in its name, the reads
of the posts included. So, before those families (every reply, as they are; a draft Auto quotes
is left out):

- a social post said to be drafted, created, made, prepared, written or rendered ("I've drafted
  the three Instagram posts", "I've made a carousel", never a blog post) needs a create or an
  update of a social post that succeeded this turn;
- a change said to be made to a post ("I've removed the tasting notes from the carousel", "I've
  updated the post") needs the same;
- a post said to be waiting in the Socials tab, or for the owner's approval ("It's waiting in the
  Socials tab", "You'll find them in your Socials tab") needs a create, an update or a submit.

Reads never back them: listing or reading posts makes and changes nothing. A sentence that makes
one of these claims is judged here only, and the rest of the reply goes on to the other families.
"""
from __future__ import annotations

import functools
import re
from typing import Callable, List, Optional, Tuple

POST_MADE = "made into a social post"
POST_CHANGED = "changed in the social post"
POST_WAITING = "put in your Socials tab"
LABELS = (POST_MADE, POST_CHANGED, POST_WAITING)
NOT_DONE_LINE = "Just to be clear: {said}. Ask me again if you want it done."   # claim_check.NOT_DONE's shape
LINES = {
    POST_MADE: "I didn't make that post in this reply, so nothing new is waiting for your approval in the Socials tab",
    POST_CHANGED: "I didn't change that post in this reply, so it still says what it said before",
    POST_WAITING: "I didn't make or change a post in this reply, so I can't say it's waiting in your Socials tab",
}

# What backs each: a post made or changed this turn (a submit puts one in the approval queue too).
_MAKES = ("create_social_post", "update_social_post")
_WAITS = _MAKES + ("submit_social_post",)

# What a post is called: its channel or kind, then the post (never a blog post), and not what a
# playbook, a template or a card is about ("the Weekly Instagram posts playbook").
_KIND = (r"(?:(?:social(?:\s+media)?|socials|instagram|insta|ig|linkedin|facebook|tiktok|twitter|x|youtube|story|"
         r"image|photo|video|quote|fact|carousel|draft|new)\s+){0,3}")
_NOT_ITS_KIND = (r"(?![\"'”’*]*\s+(?:playbooks?|templates?|plans?|schedules?|timers?|tasks?|tickets?|cards?|missions?|"
                 r"agents?|calendar|ideas?)\b)")
_POST = _KIND + r"(?<!blog )(?:posts?|carousels?|reels?|stories)\b" + _NOT_ITS_KIND
_ITS_PARTS = _KIND + r"(?<!blog )(?:posts?|carousels?|reels?|captions?|slides?|stories)\b" + _NOT_ITS_KIND
# Up to five words between the doing and the post, none of them a preposition or a card: "I've
# created a ticket for the posts" made a ticket, not a post.
_GAP = (r"(?:(?!(?:for|to|of|with|about|from|into|in|on|at|by|and|or|that|which|who|where|as|so|task|ticket|card)\b)"
        r"[^\s.!?,;:]+\s+){0,5}?")
_MADE_VERBS = (r"(?:drafted|created|made|prepared|put together|written|rendered|re-rendered|built|designed|"
               r"generated|produced|set up|lined up|queued(?: up)?)")
_CHANGE_VERBS = (r"(?:removed|deleted|taken out|cut|dropped|changed|updated|replaced|swapped|fixed|edited|added|"
                 r"rewritten|reworded|corrected|shortened|tweaked|adjusted|moved)")
_THE = r"(?:the|your|that|this|these|those|each|both|all|its)\s+"
_POST_SUBJECT = _THE + r"(?:[\w-]+\s+){0,3}?(?<!blog )(?:posts?|carousels?|reels?)\b"
# "It's waiting in the Socials tab" names the tab; "it's waiting for your approval" alone may be a mission's.
_SUBJECT = r"(?:it|they|them|this|that|these|those|" + _POST_SUBJECT + r")"
_IN_THE_TAB = r"(?:in|on|under)\s+(?:the|your)\s+socials?\s+tab\b"
_IS = r"(?:['’]s|['’]re|\s+is|\s+are)\s+(?:now\s+|all\s+)?"


@functools.lru_cache(maxsize=1)
def _families() -> tuple:
    """The three families, as ``action_claims`` reads them (built on first use: that module imports this one)."""
    from .action_claims import _ASKING, _I_HAVE, _Family

    made = _Family(POST_MADE, re.compile(_I_HAVE + _MADE_VERBS + r"\s+" + _GAP + _POST, re.I), _MAKES,
                   unless=_ASKING)
    changed = (
        _Family(POST_CHANGED, re.compile(_I_HAVE + _CHANGE_VERBS + r"\b[^.!?\n]{0,80}?\b(?:in|from|on|to|of)\s+"
                                         + _THE + _ITS_PARTS, re.I), _MAKES, unless=_ASKING),
        _Family(POST_CHANGED, re.compile(_I_HAVE + _CHANGE_VERBS + r"\s+" + _THE + _ITS_PARTS, re.I), _MAKES,
                unless=_ASKING),
    )
    waiting = (
        _Family(POST_WAITING, re.compile(r"\b" + _SUBJECT + _IS + r"(?:(?:waiting|ready|sitting|saved|available)\s+)?"
                                         r"(?:for\s+(?:you|your\s+(?:approval|review))\s+)?" + _IN_THE_TAB, re.I),
                _WAITS, unless=_ASKING),
        _Family(POST_WAITING, re.compile(r"\b" + _POST_SUBJECT + _IS + r"waiting\s+for\s+(?:your\s+)?(?:approval|review)\b",
                                         re.I), _WAITS, unless=_ASKING),
        _Family(POST_WAITING, re.compile(r"\byou(?:['’]ll| will| can| should)\s+(?:now\s+)?(?:find|see|review|approve|"
                                         r"check)\s+" + _SUBJECT + r"\s+" + _IN_THE_TAB, re.I), _WAITS, unless=_ASKING),
    )
    return (made, *changed, *waiting)


def judged_here(text: str, succeeded: List[str]) -> Tuple[Optional[str], str]:
    """The label of the first post claim in Auto's own words ``text`` that no call this turn backs
    (else None), and ``text`` without the sentences that make a post claim, for the other families."""
    from .action_claims import _BACK_REFERENCE, _IN_A_NUMBER, _MID_NUMBER, _SENTENCE, _backed, _claim_in, _own_words

    said = _MID_NUMBER.sub(_IN_A_NUMBER, text or "")
    rest, found = said, None
    for sentence in _SENTENCE.findall(_own_words(said)):
        if _BACK_REFERENCE.search(sentence):
            continue
        for family in _families():
            claim_text = _claim_in(sentence, family)
            if claim_text is None:
                continue
            rest = rest.replace(sentence, "", 1)
            if found is None and not _backed(family, claim_text, succeeded):
                found = family.label
            break
    return found, rest


Check = Callable[..., Optional[str]]


def also_checks_social_posts(check: Check) -> Check:
    """Wrap ``action_claims.claimed_action_not_done``: a post said to be made, changed or waiting is
    judged first, by the post calls of this turn; the rest of the reply goes on to the other families."""
    @functools.wraps(check)
    def wrapped(text: str, done: Optional[set] = None, *, promises: Optional[bool] = None) -> Optional[str]:
        found, rest = judged_here(text or "", [action.lower() for action in (done or ())])
        return found or check(rest, done, promises=promises)
    return wrapped


Line = Callable[[str], str]


def says_it_for_social_posts(not_done: Line) -> Line:
    """Wrap ``claim_check.not_done``: the owner's line for a post claim, in Auto's words."""
    @functools.wraps(not_done)
    def wrapped(claim: str) -> str:
        said = LINES.get(claim)
        return NOT_DONE_LINE.format(said=said) if said else not_done(claim)
    return wrapped


__all__ = ["LABELS", "LINES", "POST_CHANGED", "POST_MADE", "POST_WAITING", "also_checks_social_posts",
           "judged_here", "says_it_for_social_posts"]
